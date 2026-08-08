use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, sys as cublas_sys};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use half::{bf16, f16};
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use std::io::{self, Write};
use std::panic;
use std::sync::Arc;
use std::time::{Duration, Instant};

const DEFAULT_GPU_MAX_C: u32 = 80;
const DEFAULT_STATUS_INTERVAL: Duration = Duration::from_secs(2);
const DEFAULT_CALIBRATE: Duration = Duration::from_secs(4);
const CONTROL_SLEEP: Duration = Duration::from_millis(500);

/// Square matrix sizes the autotuner sweeps. Small sizes saturate tiny GPUs; large ones
/// fill the SMs on big cards. The winner is whichever draws the most watts on THIS GPU.
const SWEEP_SIZES: &[usize] = &[1024, 2048, 4096];
/// Fraction of free VRAM the burn buffers may claim. Conservative to survive fragmentation.
const VRAM_BUDGET_FRAC: f64 = 0.4;
/// Config used when there is no NVML to autotune against and the user pinned nothing.
const FALLBACK: RunConfig = RunConfig {
    math: Math::F16,
    n: 2048,
    batch: 8,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cfg = Config::from_args()?;

    let count = device_count()?;
    if count == 0 {
        return Err("No CUDA devices found.".to_string());
    }
    let mut workers = Vec::with_capacity(count);
    for ordinal in 0..count {
        workers.push(GpuWorker::new(ordinal)?);
    }

    let sensors = Sensors::new();

    // Per-GPU calibration: find the (math, size, batch) that pulls the most watts on each card.
    eprintln!(
        "Calibrating {} GPU(s) for peak power draw...",
        workers.len()
    );
    for worker in &mut workers {
        let chosen = calibrate(worker, &sensors, &cfg)?;
        worker.reconfigure(chosen)?;
        eprintln!("GPU{}: {}", worker.ordinal, chosen.describe());
    }

    eprintln!("{}", cfg.thermostat.describe());
    eprintln!("Press Ctrl+C to stop.");

    let mut loads = vec![1.0f32; workers.len()];
    let mut credit = vec![0.0f32; workers.len()];
    let mut next_status = Instant::now();
    loop {
        if Instant::now() >= next_status {
            let readings: Vec<Option<GpuStatus>> = workers
                .iter()
                .map(|w| sensors.gpu_status(w.ordinal))
                .collect();
            loads = readings
                .iter()
                .map(|r| cfg.thermostat.gpu_load(r.as_ref()))
                .collect();
            print_status(&readings, &loads);
            // Interval from *now*, not from the last deadline: one launch can outlast the
            // interval, and `+=` would then burst a catch-up status on every tick.
            next_status = Instant::now() + cfg.status_interval;
        }

        // Per-GPU duty cycle: each GPU accrues its own `load` in credit per tick and
        // launches a batch when it reaches 1.0, so a card at 0.45 runs ~45% of the ticks
        // a full-load card runs. Loop cadence is set by whichever GPUs actually launch, so
        // full-load cards stay saturated no matter how throttled their neighbours are.
        let mut launched = Vec::new();
        for i in 0..workers.len() {
            if tick_credit(&mut credit[i], loads[i]) {
                workers[i].launch()?;
                launched.push(i);
            }
        }
        if launched.is_empty() {
            // ponytail: a lone lightly-throttled GPU runs cooler than its target here (this
            // idle sleep stretches its cadence). Per-GPU threads if you ever need it exact.
            std::thread::sleep(CONTROL_SLEEP);
            continue;
        }
        for &i in &launched {
            workers[i]
                .dev
                .synchronize()
                .map_err(|err| format!("CUDA synchronize failed on GPU {i}: {err:?}"))?;
        }
    }
}

/// Advances one GPU's duty-cycle credit by `load` and reports whether it should launch
/// this tick. Credit stays in `[0, 1)`, so launch frequency converges to `load`.
fn tick_credit(credit: &mut f32, load: f32) -> bool {
    if load <= 0.0 {
        *credit = 0.0;
        return false;
    }
    *credit += load;
    if *credit >= 1.0 {
        *credit -= 1.0;
        true
    } else {
        false
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Math {
    F32,
    F16,
    Bf16,
}

impl Math {
    const ALL: &'static [Math] = &[Math::F32, Math::F16, Math::Bf16];

    fn label(self) -> &'static str {
        match self {
            Math::F32 => "f32",
            Math::F16 => "f16",
            Math::Bf16 => "bf16",
        }
    }

    /// Bytes per element on the GPU. f16/bf16 halve the buffers vs f32.
    fn bytes(self) -> usize {
        match self {
            Math::F32 => 4,
            Math::F16 | Math::Bf16 => 2,
        }
    }
}

#[derive(Clone, Copy)]
struct RunConfig {
    math: Math,
    n: usize,
    batch: usize,
}

impl RunConfig {
    fn matrix_elements(&self) -> usize {
        self.n * self.n
    }

    fn total_elements(&self) -> usize {
        self.matrix_elements() * self.batch
    }

    fn vram_bytes(&self) -> usize {
        self.total_elements() * self.math.bytes() * 3
    }

    fn describe(&self) -> String {
        format!(
            "{} {}x{} SGEMM x{} (~{} MiB VRAM)",
            self.math.label(),
            self.n,
            self.n,
            self.batch,
            self.vram_bytes() / 1024 / 1024
        )
    }
}

/// Picks the batch that fills `budget_bytes` for a given math+size, at least 1.
/// Larger cards get a bigger budget (more free VRAM) and thus more concurrent work.
fn batch_for(math: Math, n: usize, budget_bytes: usize) -> usize {
    let per_batch = math.bytes() * n * n * 3; // a, b, c for one batch element
    (budget_bytes / per_batch.max(1)).clamp(1, i32::MAX as usize)
}

/// Sweeps the candidate configs (any axis the user pinned is fixed, the rest swept) and
/// returns the one that measured the highest watts. Falls back gracefully without NVML.
fn calibrate(worker: &mut GpuWorker, sensors: &Sensors, cfg: &Config) -> Result<RunConfig, String> {
    let budget = sensors
        .free_vram(worker.ordinal)
        .map(|free| (free as f64 * VRAM_BUDGET_FRAC) as usize);

    // No power feedback (no NVML) or fully-pinned config -> skip the sweep.
    let maths: Vec<Math> = cfg
        .math
        .map(|m| vec![m])
        .unwrap_or_else(|| Math::ALL.to_vec());
    let sizes: Vec<usize> = cfg
        .size
        .map(|s| vec![s])
        .unwrap_or_else(|| SWEEP_SIZES.to_vec());
    let fully_pinned = maths.len() == 1 && sizes.len() == 1 && cfg.batch.is_some();

    if budget.is_none() || fully_pinned {
        let chosen = RunConfig {
            math: maths[0],
            n: sizes[0],
            batch: cfg.batch.unwrap_or(FALLBACK.batch),
        };
        if budget.is_none() && !fully_pinned {
            eprintln!(
                "GPU{}: NVML unavailable, cannot autotune - using fallback.",
                worker.ordinal
            );
            return Ok(if cfg.size.is_some() || cfg.math.is_some() {
                chosen
            } else {
                FALLBACK
            });
        }
        return Ok(chosen);
    }
    let budget = budget.unwrap();

    let mut best: Option<(RunConfig, f32)> = None;
    for &math in &maths {
        for &n in &sizes {
            let candidate = RunConfig {
                math,
                n,
                batch: cfg.batch.unwrap_or_else(|| batch_for(math, n, budget)),
            };
            match measure(worker, candidate, cfg.calibrate, sensors) {
                Some(watts) => {
                    eprintln!(
                        "  GPU{} {:<22} -> {:.0} W",
                        worker.ordinal,
                        candidate.describe(),
                        watts
                    );
                    if best.as_ref().map(|(_, w)| watts > *w).unwrap_or(true) {
                        best = Some((candidate, watts));
                    }
                }
                None => eprintln!(
                    "  GPU{} {:<22} -> skipped (unsupported or out of memory)",
                    worker.ordinal,
                    candidate.describe()
                ),
            }
        }
    }

    best.map(|(c, _)| c).ok_or_else(|| {
        format!(
            "GPU{}: no burn config was runnable during calibration",
            worker.ordinal
        )
    })
}

/// Runs `candidate` for `dur`, sampling NVML power, and returns the average watts over the
/// second half (the first half is boost-clock ramp). Returns None if the config can't run.
fn measure(
    worker: &mut GpuWorker,
    candidate: RunConfig,
    dur: Duration,
    sensors: &Sensors,
) -> Option<f32> {
    worker.reconfigure(candidate).ok()?;
    // Prime once: surfaces an unsupported math (e.g. tensor GEMM on a card without it).
    worker.launch().ok()?;
    worker.dev.synchronize().ok()?;

    let start = Instant::now();
    let midpoint = start + dur / 2;
    let deadline = start + dur;
    let mut sum = 0.0f32;
    let mut samples = 0u32;
    while Instant::now() < deadline {
        worker.launch().ok()?;
        worker.dev.synchronize().ok()?;
        if Instant::now() >= midpoint
            && let Some(w) = sensors.power(worker.ordinal)
        {
            sum += w;
            samples += 1;
        }
    }
    (samples > 0).then(|| sum / samples as f32)
}

enum Buffers {
    F32(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>),
    F16(CudaSlice<f16>, CudaSlice<f16>, CudaSlice<f16>),
    Bf16(CudaSlice<bf16>, CudaSlice<bf16>, CudaSlice<bf16>),
}

struct GpuWorker {
    ordinal: usize,
    dev: Arc<CudaDevice>,
    cublas: CudaBlas,
    cfg: RunConfig,
    buffers: Option<Buffers>,
}

impl GpuWorker {
    fn new(ordinal: usize) -> Result<Self, String> {
        let dev = cuda_device(ordinal)?;
        let cublas = CudaBlas::new(dev.clone())
            .map_err(|err| format!("Failed to initialize cuBLAS on GPU {ordinal}: {err:?}"))?;
        Ok(Self {
            ordinal,
            dev,
            cublas,
            cfg: FALLBACK,
            buffers: None,
        })
    }

    /// Reallocates the burn buffers for `cfg`. Frees the old buffers first so the sweep never
    /// holds two configs' worth of VRAM at once.
    fn reconfigure(&mut self, cfg: RunConfig) -> Result<(), String> {
        self.buffers = None; // drop old allocation before requesting new VRAM
        let total = cfg.total_elements();
        let ord = self.ordinal;
        self.buffers = Some(match cfg.math {
            Math::F32 => {
                let (a, b, c) = alloc3::<f32>(&self.dev, total, ord)?;
                Buffers::F32(a, b, c)
            }
            Math::F16 => {
                let (a, b, c) = alloc3::<f16>(&self.dev, total, ord)?;
                Buffers::F16(a, b, c)
            }
            Math::Bf16 => {
                let (a, b, c) = alloc3::<bf16>(&self.dev, total, ord)?;
                Buffers::Bf16(a, b, c)
            }
        });
        self.cfg = cfg;
        Ok(())
    }

    fn launch(&mut self) -> Result<(), String> {
        let cfg = self.cfg;
        let ord = self.ordinal;
        let cublas = &self.cublas;
        match self.buffers.as_mut().expect("launch before reconfigure") {
            Buffers::F32(a, b, c) => launch_typed(cublas, &cfg, a, b, c, 1.0f32, 0.0f32, ord),
            Buffers::F16(a, b, c) => launch_typed(
                cublas,
                &cfg,
                a,
                b,
                c,
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                ord,
            ),
            Buffers::Bf16(a, b, c) => launch_typed(
                cublas,
                &cfg,
                a,
                b,
                c,
                bf16::from_f32(1.0),
                bf16::from_f32(0.0),
                ord,
            ),
        }
    }
}

#[allow(clippy::type_complexity)]
fn alloc3<T: DeviceRepr + ValidAsZeroBits>(
    dev: &Arc<CudaDevice>,
    len: usize,
    ordinal: usize,
) -> Result<(CudaSlice<T>, CudaSlice<T>, CudaSlice<T>), String> {
    let mk = |name: &str| {
        dev.alloc_zeros::<T>(len)
            .map_err(|err| format!("Failed to allocate matrix {name} on GPU {ordinal}: {err:?}"))
    };
    Ok((mk("A")?, mk("B")?, mk("C")?))
}

#[allow(clippy::too_many_arguments)]
fn launch_typed<T>(
    cublas: &CudaBlas,
    cfg: &RunConfig,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
    alpha: T,
    beta: T,
    ordinal: usize,
) -> Result<(), String>
where
    CudaBlas: Gemm<T>,
{
    let n = cfg.n as i32;
    let stride = cfg.matrix_elements() as i64;
    unsafe {
        cublas
            .gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        m: n,
                        n,
                        k: n,
                        alpha,
                        lda: n,
                        ldb: n,
                        beta,
                        ldc: n,
                    },
                    batch_size: cfg.batch as i32,
                    stride_a: stride,
                    stride_b: stride,
                    stride_c: stride,
                },
                a,
                b,
                c,
            )
            .map_err(|err| format!("cuBLAS GEMM failed on GPU {ordinal}: {err:?}"))
    }
}

struct Sensors {
    nvml: Option<Nvml>,
}

impl Sensors {
    fn new() -> Self {
        Self {
            nvml: Nvml::init().ok(),
        }
    }

    fn gpu_status(&self, ordinal: usize) -> Option<GpuStatus> {
        let device = self.nvml.as_ref()?.device_by_index(ordinal as u32).ok()?;
        Some(GpuStatus {
            ordinal,
            temperature_c: device.temperature(TemperatureSensor::Gpu).ok()?,
            power_w: device.power_usage().ok()? as f32 / 1000.0,
        })
    }

    fn power(&self, ordinal: usize) -> Option<f32> {
        let device = self.nvml.as_ref()?.device_by_index(ordinal as u32).ok()?;
        Some(device.power_usage().ok()? as f32 / 1000.0)
    }

    fn free_vram(&self, ordinal: usize) -> Option<usize> {
        let device = self.nvml.as_ref()?.device_by_index(ordinal as u32).ok()?;
        Some(device.memory_info().ok()?.free as usize)
    }
}

fn print_status(readings: &[Option<GpuStatus>], loads: &[f32]) {
    let mut fields = Vec::new();

    for (reading, load) in readings.iter().zip(loads) {
        match reading {
            Some(gpu) => fields.push(format!(
                "GPU{} {}C {:.0}W {:.0}%",
                gpu.ordinal,
                gpu.temperature_c,
                gpu.power_w,
                load * 100.0
            )),
            None => fields.push("GPU ?".to_string()),
        }
    }

    if readings.iter().all(Option::is_none) {
        fields.insert(0, "NVML unavailable, running uncapped".to_string());
    }

    print_status_line(&fields.join(" | "));
}

fn print_status_line(line: &str) {
    eprint!("\r\x1b[2K{line}");
    let _ = io::stderr().flush();
}

struct GpuStatus {
    ordinal: usize,
    temperature_c: u32,
    power_w: f32,
}

struct Config {
    math: Option<Math>,
    size: Option<usize>,
    batch: Option<usize>,
    thermostat: Thermostat,
    status_interval: Duration,
    calibrate: Duration,
}

struct Thermostat {
    gpu_max_c: u32,
}

impl Config {
    fn from_args() -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut thermostat = Thermostat {
            gpu_max_c: DEFAULT_GPU_MAX_C,
        };
        let mut status_interval = DEFAULT_STATUS_INTERVAL;
        let mut calibrate = DEFAULT_CALIBRATE;
        let mut math = None;

        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                "-V" | "--version" => {
                    println!("thermalbloater {}", env!("CARGO_PKG_VERSION"));
                    std::process::exit(0);
                }
                "--gpu-max" => {
                    let value = next_arg(&mut args, "--gpu-max")?;
                    thermostat.gpu_max_c = parse_gpu_max(&value)?;
                }
                "--status-interval" => {
                    let value = next_arg(&mut args, "--status-interval")?;
                    status_interval =
                        Duration::from_secs_f32(parse_positive_f32(&value, "--status-interval")?);
                }
                "--calibrate-secs" => {
                    let value = next_arg(&mut args, "--calibrate-secs")?;
                    calibrate =
                        Duration::from_secs_f32(parse_positive_f32(&value, "--calibrate-secs")?);
                }
                "--math" => {
                    let value = next_arg(&mut args, "--math")?;
                    math = Some(parse_math(&value)?);
                }
                flag if flag.starts_with("--") => {
                    return Err(format!("unknown option `{flag}`\n{}", usage()));
                }
                value => positional.push(value.to_string()),
            }
        }

        if positional.len() > 2 {
            return Err(usage());
        }

        let size = positional
            .first()
            .map(|value| parse_positive_usize(value, "matrix size"))
            .transpose()?;
        let batch = positional
            .get(1)
            .map(|value| parse_positive_usize(value, "batch size"))
            .transpose()?;

        if let Some(s) = size
            && s > i32::MAX as usize
        {
            return Err("matrix size must fit in i32".to_string());
        }
        if let Some(b) = batch
            && b > i32::MAX as usize
        {
            return Err("batch size must fit in i32".to_string());
        }
        Ok(Self {
            math,
            size,
            batch,
            thermostat,
            status_interval,
            calibrate,
        })
    }
}

impl Thermostat {
    fn describe(&self) -> String {
        format!("Thermostat: GPU max {}C (per GPU).", self.gpu_max_c)
    }

    fn gpu_load(&self, gpu: Option<&GpuStatus>) -> f32 {
        let Some(gpu) = gpu else {
            return 1.0;
        };
        let gpu_temp = gpu.temperature_c;
        if gpu_temp >= self.gpu_max_c {
            0.0
        } else {
            let headroom = (self.gpu_max_c - gpu_temp) as f32;
            (headroom / 5.0).clamp(0.15, 1.0)
        }
    }
}

fn next_arg(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    name: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value\n{}", usage()))
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_positive_f32(value: &str, name: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("{name} must be a positive number"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_gpu_max(value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| "--gpu-max must be a whole-number Celsius temperature".to_string())?;
    if parsed < 40 {
        return Err("--gpu-max must be at least 40C".to_string());
    }
    Ok(parsed)
}

fn parse_math(value: &str) -> Result<Math, String> {
    match value.to_ascii_lowercase().as_str() {
        "f32" | "sgemm" => Ok(Math::F32),
        "f16" | "fp16" | "half" => Ok(Math::F16),
        "bf16" | "bfloat16" => Ok(Math::Bf16),
        _ => Err("--math must be one of: f32, f16, bf16".to_string()),
    }
}

fn usage() -> String {
    "Usage: thermalbloater [matrix-size] [batch-size] [--math f32|f16|bf16] \
     [--gpu-max C] [--calibrate-secs S] [--status-interval S] [--version]\n\
     Autotunes per-GPU for peak watts unless you pin size/batch/math."
        .to_string()
}

fn device_count() -> Result<usize, String> {
    match guarded(CudaDevice::count) {
        Ok(Ok(count)) => Ok(count.max(0) as usize),
        Ok(Err(err)) => Err(format!(
            "Failed to query CUDA device count: {err:?}\n{}",
            cuda_load_hint()
        )),
        Err(payload) => Err(format!(
            "Failed to load the CUDA driver library: {}\n{}",
            panic_message(payload.as_ref()),
            cuda_load_hint()
        )),
    }
}

fn cuda_device(ordinal: usize) -> Result<Arc<CudaDevice>, String> {
    match guarded(|| CudaDevice::new(ordinal)) {
        Ok(Ok(dev)) => Ok(dev),
        Ok(Err(err)) => Err(format!(
            "Failed to initialize CUDA device {ordinal}: {err:?}\n{}",
            cuda_load_hint()
        )),
        Err(payload) => Err(format!(
            "Failed to load the CUDA driver library: {}\n{}",
            panic_message(payload.as_ref()),
            cuda_load_hint()
        )),
    }
}

/// Runs `f`, swallowing any panic (e.g. cudarc aborting when the CUDA libraries are missing)
/// so it surfaces as an `Err` instead of killing the process.
fn guarded<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> std::thread::Result<T> {
    let old_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(old_hook);
    result
}

fn cuda_load_hint() -> &'static str {
    if cfg!(windows) {
        "Install the NVIDIA driver (and CUDA runtime), and make sure the folder holding \
         cublas64_*.dll is on your PATH. Any CUDA 11/12/13 install works."
    } else {
        "Ensure the NVIDIA driver and cuBLAS are installed and reachable (any CUDA 11/12/13). \
         On WSL, check `nvidia-smi` works and set \
         LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH."
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        message.to_string()
    } else {
        "unknown panic while loading CUDA".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{Math, batch_for, tick_credit};

    #[test]
    fn launch_fraction_converges_to_load() {
        for &load in &[0.15f32, 0.45, 0.5, 1.0] {
            let mut credit = 0.0;
            let ticks = 100_000;
            let launches = (0..ticks)
                .filter(|_| tick_credit(&mut credit, load))
                .count();
            let frac = launches as f32 / ticks as f32;
            assert!((frac - load).abs() < 0.001, "load {load}: got {frac}");
        }
    }

    #[test]
    fn zero_load_never_launches_and_resets_credit() {
        let mut credit = 0.7;
        assert!(!tick_credit(&mut credit, 0.0));
        assert_eq!(credit, 0.0);
    }

    #[test]
    fn batch_fills_budget_without_exceeding_it() {
        let budget = 4 * 1024 * 1024 * 1024; // 4 GiB
        for math in [Math::F32, Math::F16, Math::Bf16] {
            for &n in super::SWEEP_SIZES {
                let batch = batch_for(math, n, budget);
                assert!(batch >= 1);
                let used = batch * math.bytes() * n * n * 3;
                assert!(used <= budget, "math {math:?} n {n}: {used} > {budget}");
            }
        }
    }

    #[test]
    fn batch_is_at_least_one_even_when_matrix_exceeds_budget() {
        // A 4096 f32 triple-buffer is ~768 MiB; a 100-byte budget still yields batch 1.
        assert_eq!(batch_for(Math::F32, 4096, 100), 1);
    }
}
