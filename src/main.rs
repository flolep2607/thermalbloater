use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, sys as cublas_sys};
use cudarc::driver::{CudaDevice, CudaSlice};
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use std::io::{self, Write};
use std::panic;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::Components;

const DEFAULT_MATRIX_SIZE: usize = 512;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_GPU_MAX_C: u32 = 80;
const DEFAULT_STATUS_INTERVAL: Duration = Duration::from_secs(2);
const CONTROL_SLEEP: Duration = Duration::from_millis(500);

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cfg = Config::from_args()?;
    let n = cfg.matrix_size;
    let batch_size = cfg.batch_size;
    let matrix_elements = n
        .checked_mul(n)
        .ok_or_else(|| "matrix size is too large".to_string())?;
    let total_elements = matrix_elements
        .checked_mul(batch_size)
        .ok_or_else(|| "batch size is too large".to_string())?;

    let count = device_count()?;
    let mut workers = Vec::with_capacity(count);
    for ordinal in 0..count {
        workers.push(GpuWorker::new(ordinal, total_elements)?);
    }
    if workers.is_empty() {
        return Err("No CUDA devices found.".to_string());
    }

    let bytes = total_elements * std::mem::size_of::<f32>() * 3;
    eprintln!(
        "Frying {} GPU(s): {batch_size} batched {n}x{n} SGEMMs each (~{} MiB VRAM/GPU). Press Ctrl+C to stop.",
        workers.len(),
        bytes / 1024 / 1024
    );
    eprintln!("{}", cfg.thermostat.describe());

    let mut sensors = Sensors::new();
    let mut loads = vec![1.0f32; workers.len()];
    let mut next_status = Instant::now();
    loop {
        if Instant::now() >= next_status {
            let readings: Vec<Option<GpuStatus>> =
                workers.iter().map(|w| sensors.gpu_status(w.ordinal)).collect();
            loads = readings
                .iter()
                .map(|r| cfg.thermostat.gpu_load(r.as_ref()))
                .collect();
            let cpu = sensors.cpu_temp();
            print_status(&readings, &loads, cpu.as_ref());
            next_status += cfg.status_interval;
        }

        let active: Vec<usize> = (0..workers.len()).filter(|&i| loads[i] > 0.0).collect();
        if active.is_empty() {
            std::thread::sleep(CONTROL_SLEEP);
            continue;
        }

        let started = Instant::now();
        for &i in &active {
            workers[i].launch(n, batch_size, matrix_elements)?;
        }
        for &i in &active {
            workers[i]
                .dev
                .synchronize()
                .map_err(|err| format!("CUDA synchronize failed on GPU {i}: {err:?}"))?;
        }
        let min_load = active.iter().map(|&i| loads[i]).fold(1.0f32, f32::min);
        sleep_for_load(min_load, started.elapsed());
    }
}

fn sleep_for_load(load: f32, work_time: Duration) {
    if load >= 0.999 {
        return;
    }

    let cycle = work_time.mul_f32(1.0 / load);
    if cycle > work_time {
        std::thread::sleep(cycle - work_time);
    }
}

struct GpuWorker {
    ordinal: usize,
    dev: Arc<CudaDevice>,
    cublas: CudaBlas,
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    c: CudaSlice<f32>,
}

impl GpuWorker {
    fn new(ordinal: usize, total_elements: usize) -> Result<Self, String> {
        let dev = cuda_device(ordinal)?;
        let a = dev
            .alloc_zeros::<f32>(total_elements)
            .map_err(|err| format!("Failed to allocate matrix A on GPU {ordinal}: {err:?}"))?;
        let b = dev
            .alloc_zeros::<f32>(total_elements)
            .map_err(|err| format!("Failed to allocate matrix B on GPU {ordinal}: {err:?}"))?;
        let c = dev
            .alloc_zeros::<f32>(total_elements)
            .map_err(|err| format!("Failed to allocate matrix C on GPU {ordinal}: {err:?}"))?;
        let cublas = CudaBlas::new(dev.clone())
            .map_err(|err| format!("Failed to initialize cuBLAS on GPU {ordinal}: {err:?}"))?;
        Ok(Self {
            ordinal,
            dev,
            cublas,
            a,
            b,
            c,
        })
    }

    fn launch(&mut self, n: usize, batch_size: usize, matrix_elements: usize) -> Result<(), String> {
        unsafe {
            self.cublas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            m: n as i32,
                            n: n as i32,
                            k: n as i32,
                            alpha: 1.0f32,
                            lda: n as i32,
                            ldb: n as i32,
                            beta: 0.0f32,
                            ldc: n as i32,
                        },
                        batch_size: batch_size as i32,
                        stride_a: matrix_elements as i64,
                        stride_b: matrix_elements as i64,
                        stride_c: matrix_elements as i64,
                    },
                    &self.a,
                    &self.b,
                    &mut self.c,
                )
                .map_err(|err| format!("cuBLAS GEMM failed on GPU {}: {err:?}", self.ordinal))?;
        }
        Ok(())
    }
}

struct Sensors {
    nvml: Option<Nvml>,
    components: Components,
}

impl Sensors {
    fn new() -> Self {
        Self {
            nvml: Nvml::init().ok(),
            components: Components::new_with_refreshed_list(),
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

    fn cpu_temp(&mut self) -> Option<TemperatureReading> {
        self.components.refresh(false);
        let sensors = self.temperature_sensors();
        hottest_cpu_sensor(&sensors).or_else(|| hottest_sensor(&sensors))
    }

    fn temperature_sensors(&self) -> Vec<TemperatureReading> {
        self.components
            .iter()
            .filter_map(|component| {
                let temperature_c = component.temperature()?;
                if temperature_c.is_nan() {
                    return None;
                }
                Some(TemperatureReading {
                    label: component.label().to_string(),
                    temperature_c,
                })
            })
            .collect()
    }
}

fn print_status(readings: &[Option<GpuStatus>], loads: &[f32], cpu: Option<&TemperatureReading>) {
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

    if let Some(cpu) = cpu {
        fields.push(format!("CPU {:.0}C ({})", cpu.temperature_c, cpu.label));
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

#[derive(Clone)]
struct TemperatureReading {
    label: String,
    temperature_c: f32,
}

fn hottest_cpu_sensor(sensors: &[TemperatureReading]) -> Option<TemperatureReading> {
    hottest_matching_sensor(sensors, |sensor| {
        let text = sensor.label.to_lowercase();
        text.contains("cpu")
            || text.contains("core")
            || text.contains("package")
            || text.contains("k10temp")
            || text.contains("zenpower")
    })
}

fn hottest_sensor(sensors: &[TemperatureReading]) -> Option<TemperatureReading> {
    hottest_matching_sensor(sensors, |_| true)
}

fn hottest_matching_sensor(
    sensors: &[TemperatureReading],
    matches: impl Fn(&TemperatureReading) -> bool,
) -> Option<TemperatureReading> {
    sensors
        .iter()
        .filter(|sensor| matches(sensor))
        .max_by(|a, b| a.temperature_c.total_cmp(&b.temperature_c))
        .cloned()
}

struct Config {
    matrix_size: usize,
    batch_size: usize,
    thermostat: Thermostat,
    status_interval: Duration,
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

        let mut args = std::env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => return Err(usage()),
                "--gpu-max" => {
                    let value = next_arg(&mut args, "--gpu-max")?;
                    thermostat.gpu_max_c = parse_gpu_max(&value)?;
                }
                "--status-interval" => {
                    let value = next_arg(&mut args, "--status-interval")?;
                    status_interval =
                        Duration::from_secs_f32(parse_positive_f32(&value, "--status-interval")?);
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

        let matrix_size = positional
            .first()
            .map(|value| parse_positive_usize(value, "matrix size"))
            .transpose()?
            .unwrap_or(DEFAULT_MATRIX_SIZE);
        let batch_size = positional
            .get(1)
            .map(|value| parse_positive_usize(value, "batch size"))
            .transpose()?
            .unwrap_or(DEFAULT_BATCH_SIZE);

        if matrix_size > i32::MAX as usize || batch_size > i32::MAX as usize {
            return Err("matrix size and batch size must fit in i32".to_string());
        }
        Ok(Self {
            matrix_size,
            batch_size,
            thermostat,
            status_interval,
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

fn usage() -> String {
    "Usage: thermalbloater [matrix-size] [batch-size] [--gpu-max C] [--status-interval SECONDS]"
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
