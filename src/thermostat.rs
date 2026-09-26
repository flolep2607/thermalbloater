use crate::sensors::GpuStatus;
use std::time::Duration;

#[derive(Clone, Copy)]
pub struct Thermostat {
    pub gpu_max_c: u32,
}

impl Thermostat {
    pub fn describe(&self) -> String {
        format!("Thermostat: GPU max {}C (per GPU).", self.gpu_max_c)
    }

    pub fn gpu_load(&self, gpu: Option<&GpuStatus>) -> f32 {
        let Some(gpu) = gpu else {
            return 1.0;
        };
        let t = gpu.temperature_c;
        if t >= self.gpu_max_c {
            0.0
        } else {
            let headroom = (self.gpu_max_c - t) as f32;
            (headroom / 5.0).clamp(0.15, 1.0)
        }
    }
}

/// Idle time after a launch that took `busy`, so busy / (busy + idle) == load. `load` must be > 0.
pub fn idle_after(busy: Duration, load: f32) -> Duration {
    busy.mul_f32((1.0 - load) / load)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensors::GpuStatus;

    fn status(temp: u32) -> GpuStatus {
        GpuStatus {
            ordinal: 0,
            temperature_c: temp,
            power_w: Some(100.0),
        }
    }

    #[test]
    fn at_or_above_max_is_zero() {
        let th = Thermostat { gpu_max_c: 80 };
        assert_eq!(th.gpu_load(Some(&status(80))), 0.0);
        assert_eq!(th.gpu_load(Some(&status(90))), 0.0);
    }

    #[test]
    fn far_below_max_is_one() {
        let th = Thermostat { gpu_max_c: 80 };
        assert_eq!(th.gpu_load(Some(&status(60))), 1.0);
        assert_eq!(th.gpu_load(Some(&status(0))), 1.0);
    }

    #[test]
    fn close_to_max_clamps_to_min() {
        let th = Thermostat { gpu_max_c: 80 };
        assert!((th.gpu_load(Some(&status(79))) - 0.2).abs() < 1e-6);
        assert!((th.gpu_load(Some(&status(78))) - 0.4).abs() < 1e-6);
        assert!((th.gpu_load(Some(&status(76))) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn idle_matches_duty_cycle() {
        let busy = Duration::from_millis(100);
        assert_eq!(idle_after(busy, 1.0), Duration::ZERO);
        assert_eq!(idle_after(busy, 0.5), busy);
        assert_eq!(idle_after(busy, 0.2), Duration::from_millis(400));
    }

    #[test]
    fn no_sensor_runs_full() {
        let th = Thermostat { gpu_max_c: 80 };
        assert_eq!(th.gpu_load(None), 1.0);
    }
}
