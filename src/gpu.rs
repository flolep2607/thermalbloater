use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, sys as cublas_sys};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use half::{bf16, f16};
use std::panic;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::sensors::Sensors;

pub const SWEEP_SIZES: &[usize] = &[1024, 2048, 4096];
pub const VRAM_BUDGET_FRAC: f64 = 0.4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Math {
    F32,
    F16,
    Bf16,
}

impl Math {
    pub const ALL: &'static [Math] = &[Math::F32, Math::F16, Math::Bf16];

    pub fn label(self) -> &'static str {
        match self {
            Math::F32 => "f32",
            Math::F16 => "f16",
            Math::Bf16 => "bf16",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Math::F32 => 4,
            Math::F16 | Math::Bf16 => 2,
        }
    }
}

#[derive(Clone, Copy)]
pub struct RunConfig {
    pub math: Math,
    pub n: usize,
    pub batch: usize,
}

impl RunConfig {
    pub fn matrix_elements(&self) -> usize {
        self.n * self.n
    }

    pub fn total_elements(&self) -> usize {
        self.matrix_elements() * self.batch
    }

    pub fn vram_bytes(&self) -> usize {
        self.total_elements() * self.math.bytes() * 3
    }

    pub fn describe(&self) -> String {
        format!(
            "{} {}x{} GEMM x{} (~{} MiB VRAM)",
            self.math.label(),
            self.n,
            self.n,
            self.batch,
            self.vram_bytes() / 1024 / 1024
        )
    }
}

pub const FALLBACK: RunConfig = RunConfig {
    math: Math::F16,
    n: 2048,
    batch: 8,
};

pub fn batch_for(math: Math, n: usize, budget_bytes: usize) -> usize {
    let per_batch = math.bytes() * n * n * 3;
    (budget_bytes / per_batch.max(1)).clamp(1, i32::MAX as usize)
}

pub fn calibrate(
    worker: &mut GpuWorker,
    sensors: &Sensors,
    cfg: &Config,
) -> Result<RunConfig, String> {
    let budget = sensors
        .free_vram(worker.ordinal)
        .map(|free| (free as f64 * VRAM_BUDGET_FRAC) as usize);

    let maths: Vec<Math> = cfg
        .math
        .map(|m| vec![m])
        .unwrap_or_else(|| Math::ALL.to_vec());
    let sizes: Vec<usize> = cfg
        .size
        .map(|s| vec![s])
        .unwrap_or_else(|| SWEEP_SIZES.to_vec());
    let fully_pinned = maths.len() == 1 && sizes.len() == 1 && cfg.batch.is_some();
    let pinned = RunConfig {
        math: cfg.math.unwrap_or(FALLBACK.math),
        n: cfg.size.unwrap_or(FALLBACK.n),
        batch: cfg.batch.unwrap_or(FALLBACK.batch),
    };
    // Autotuning needs both a VRAM budget and a power reading to rank candidates.
    let budget = budget.filter(|_| sensors.power(worker.ordinal).is_some());
    let Some(budget) = budget.filter(|_| !fully_pinned) else {
        if !fully_pinned {
            eprintln!(
                "GPU{}: NVML power/memory unavailable, cannot autotune - using fallback.",
                worker.ordinal
            );
        }
        return Ok(pinned);
    };

    let mut best: Option<(RunConfig, f32)> = None;
    for &math in &maths {
        for &n in &sizes {
            if crate::shutting_down() {
                return Ok(pinned);
            }
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
                None if crate::shutting_down() => return Ok(pinned),
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

fn measure(
    worker: &mut GpuWorker,
    candidate: RunConfig,
    dur: Duration,
    sensors: &Sensors,
) -> Option<f32> {
    worker.reconfigure(candidate).ok()?;
    worker.launch().ok()?;
    worker.dev.synchronize().ok()?;

    let start = Instant::now();
    let midpoint = start + dur / 2;
    let deadline = start + dur;
    let mut sum = 0.0f32;
    let mut samples = 0u32;
    while Instant::now() < deadline && !crate::shutting_down() {
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

pub struct GpuWorker {
    pub ordinal: usize,
    pub dev: Arc<CudaDevice>,
    cublas: CudaBlas,
    cfg: RunConfig,
    buffers: Option<Buffers>,
}

impl GpuWorker {
    pub fn new(ordinal: usize) -> Result<Self, String> {
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

    pub fn reconfigure(&mut self, cfg: RunConfig) -> Result<(), String> {
        self.buffers = None;
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

    pub fn launch(&mut self) -> Result<(), String> {
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

    /// Hands queued work to the GPU now. WDDM/WSL drivers batch launches and otherwise
    /// only submit on synchronize(), which would leave the GPU idle while we sleep.
    pub fn flush(&self) {
        // NOT_READY is the expected answer; we only want the side effect.
        let _ = unsafe { cudarc::driver::sys::lib().cuStreamQuery(*self.dev.cu_stream()) };
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

pub fn device_count() -> Result<usize, String> {
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

fn guarded<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> std::thread::Result<T> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().unwrap();
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
    if let Some(m) = payload.downcast_ref::<String>() {
        m.clone()
    } else if let Some(m) = payload.downcast_ref::<&str>() {
        m.to_string()
    } else {
        "unknown panic while loading CUDA".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_fills_budget_without_exceeding_it() {
        let budget = 4 * 1024 * 1024 * 1024;
        for math in [Math::F32, Math::F16, Math::Bf16] {
            for &n in SWEEP_SIZES {
                let batch = batch_for(math, n, budget);
                assert!(batch >= 1);
                let used = batch * math.bytes() * n * n * 3;
                assert!(used <= budget, "math {math:?} n {n}: {used} > {budget}");
            }
        }
    }

    #[test]
    fn batch_is_at_least_one_even_when_matrix_exceeds_budget() {
        assert_eq!(batch_for(Math::F32, 4096, 100), 1);
    }
}
