mod config;
mod gpu;
mod sensors;
mod thermostat;

use config::Config;
use gpu::{GpuWorker, calibrate};
use sensors::Sensors;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use thermostat::Thermostat;

const CONTROL_SLEEP: Duration = Duration::from_millis(500);

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn shutting_down() -> bool {
    SHUTDOWN.load(Ordering::SeqCst)
}

/// Sleeps for `dur`, waking early on Ctrl+C.
fn nap(dur: Duration) {
    let deadline = Instant::now() + dur;
    while !shutting_down() {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        thread::sleep(left.min(Duration::from_millis(100)));
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cfg = Config::from_args()?;
    let count = gpu::device_count()?;
    if count == 0 {
        return Err("No CUDA devices found.".to_string());
    }

    let ordinals: Vec<usize> = match &cfg.gpu_filter {
        Some(list) => {
            for &i in list {
                if i >= count {
                    return Err(format!(
                        "GPU index {i} out of range (found {count} device(s))"
                    ));
                }
            }
            list.clone()
        }
        None => (0..count).collect(),
    };

    if ordinals.is_empty() {
        return Err("No GPUs selected.".to_string());
    }

    let sensors = Arc::new(Sensors::new());
    ctrlc::set_handler(|| SHUTDOWN.store(true, Ordering::SeqCst))
        .map_err(|e| format!("Failed to install signal handler: {e}"))?;

    eprintln!(
        "Calibrating {} GPU(s) for peak power draw...",
        ordinals.len()
    );
    // All GPUs calibrate at once; each log line is prefixed with its GPU.
    let workers: Vec<GpuWorker> = thread::scope(|s| {
        let handles: Vec<_> = ordinals
            .iter()
            .map(|&ordinal| {
                let (sensors, cfg) = (&sensors, &cfg);
                // One failed GPU aborts the run, so stop the others calibrating.
                s.spawn(move || {
                    prepare_worker(ordinal, sensors, cfg)
                        .inspect_err(|_| SHUTDOWN.store(true, Ordering::SeqCst))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("calibration thread panicked".to_string()))
            })
            .collect::<Result<_, String>>()
    })?;
    if shutting_down() {
        return Ok(());
    }

    eprintln!("{}", cfg.thermostat.describe());
    eprintln!("Press Ctrl+C to stop.");

    let th = cfg.thermostat;
    let handles: Vec<_> = workers
        .into_iter()
        .map(|worker| {
            let sensors = sensors.clone();
            thread::spawn(move || worker_loop(worker, &sensors, th))
        })
        .collect();

    while !shutting_down() && !handles.iter().any(|h| h.is_finished()) {
        let readings: Vec<_> = ordinals.iter().map(|&o| sensors.gpu_status(o)).collect();
        print_status(&readings, th);
        nap(cfg.status_interval);
    }

    eprintln!();
    SHUTDOWN.store(true, Ordering::SeqCst);
    let mut worker_err = None;
    for h in handles {
        let res = h
            .join()
            .unwrap_or_else(|_| Err("worker thread panicked".to_string()));
        if let Err(e) = res {
            worker_err.get_or_insert(e);
        }
    }
    worker_err.map_or(Ok(()), Err)
}

fn prepare_worker(ordinal: usize, sensors: &Sensors, cfg: &Config) -> Result<GpuWorker, String> {
    let mut w = GpuWorker::new(ordinal)?;
    let chosen = calibrate(&mut w, sensors, cfg)?;
    if shutting_down() {
        return Ok(w);
    }
    w.reconfigure(chosen)?;
    eprintln!("GPU{ordinal}: {}", chosen.describe());
    Ok(w)
}

/// Runs launches back to back, idling between them so the busy fraction tracks the thermostat.
fn worker_loop(mut worker: GpuWorker, sensors: &Sensors, th: Thermostat) -> Result<(), String> {
    while !shutting_down() {
        let load = th.gpu_load(sensors.gpu_status(worker.ordinal).as_ref());
        if load <= 0.0 {
            nap(CONTROL_SLEEP);
            continue;
        }
        let start = Instant::now();
        worker.launch()?;
        worker
            .dev
            .synchronize()
            .map_err(|e| format!("GPU{}: CUDA synchronize failed: {e:?}", worker.ordinal))?;
        nap(thermostat::idle_after(start.elapsed(), load));
    }
    Ok(())
}

fn print_status(readings: &[Option<sensors::GpuStatus>], th: Thermostat) {
    let mut fields: Vec<String> = readings
        .iter()
        .map(|r| match r {
            Some(g) => format!(
                "GPU{} {}C {} {:.0}%",
                g.ordinal,
                g.temperature_c,
                g.power_w.map_or("?W".to_string(), |w| format!("{w:.0}W")),
                th.gpu_load(Some(g)) * 100.0
            ),
            None => "GPU ?".to_string(),
        })
        .collect();
    if readings.iter().all(Option::is_none) {
        fields.insert(0, "NVML unavailable, running uncapped".to_string());
    }
    eprint!("\r\x1b[2K{}", fields.join(" | "));
    let _ = io::stderr().flush();
}
