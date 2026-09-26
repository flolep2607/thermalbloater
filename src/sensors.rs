use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;

#[derive(Clone, Debug)]
pub struct GpuStatus {
    pub ordinal: usize,
    pub temperature_c: u32,
    pub power_w: Option<f32>,
}

pub struct Sensors {
    nvml: Option<Nvml>,
}

impl Sensors {
    pub fn new() -> Self {
        Self {
            nvml: Nvml::init().ok(),
        }
    }

    pub fn gpu_status(&self, ordinal: usize) -> Option<GpuStatus> {
        let device = self.nvml.as_ref()?.device_by_index(ordinal as u32).ok()?;
        Some(GpuStatus {
            ordinal,
            temperature_c: device.temperature(TemperatureSensor::Gpu).ok()?,
            // Temperature is what the thermostat needs; power is optional on some cards.
            power_w: device.power_usage().ok().map(|mw| mw as f32 / 1000.0),
        })
    }

    pub fn power(&self, ordinal: usize) -> Option<f32> {
        let device = self.nvml.as_ref()?.device_by_index(ordinal as u32).ok()?;
        Some(device.power_usage().ok()? as f32 / 1000.0)
    }

    pub fn free_vram(&self, ordinal: usize) -> Option<usize> {
        let device = self.nvml.as_ref()?.device_by_index(ordinal as u32).ok()?;
        Some(device.memory_info().ok()?.free as usize)
    }
}
