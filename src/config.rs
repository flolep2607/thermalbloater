use crate::gpu::Math;
use crate::thermostat::Thermostat;
use std::time::Duration;
pub const DEFAULT_GPU_MAX_C: u32 = 80;
pub const DEFAULT_STATUS_INTERVAL: Duration = Duration::from_secs(2);
pub const DEFAULT_CALIBRATE: Duration = Duration::from_secs(4);

pub struct Config {
    pub math: Option<Math>,
    pub size: Option<usize>,
    pub batch: Option<usize>,
    pub thermostat: Thermostat,
    pub status_interval: Duration,
    pub calibrate: Duration,
    pub gpu_filter: Option<Vec<usize>>,
}

impl Config {
    pub fn from_args() -> Result<Self, String> {
        Self::from_iter(std::env::args().skip(1))
    }

    pub(crate) fn from_iter(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut thermostat = Thermostat {
            gpu_max_c: DEFAULT_GPU_MAX_C,
        };
        let mut status_interval = DEFAULT_STATUS_INTERVAL;
        let mut calibrate = DEFAULT_CALIBRATE;
        let mut math = None;
        let mut gpu_filter = None::<Vec<usize>>;

        let mut args = args;
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
                    let v = next_arg(&mut args, "--gpu-max")?;
                    thermostat.gpu_max_c = parse_gpu_max(&v)?;
                }
                "--status-interval" => {
                    let v = next_arg(&mut args, "--status-interval")?;
                    status_interval =
                        Duration::from_secs_f32(parse_positive_f32(&v, "--status-interval")?);
                }
                "--calibrate-secs" => {
                    let v = next_arg(&mut args, "--calibrate-secs")?;
                    calibrate =
                        Duration::from_secs_f32(parse_positive_f32(&v, "--calibrate-secs")?);
                }
                "--math" => {
                    let v = next_arg(&mut args, "--math")?;
                    math = Some(parse_math(&v)?);
                }
                "--gpus" => {
                    let v = next_arg(&mut args, "--gpus")?;
                    gpu_filter = Some(parse_gpu_list(&v)?);
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
            .map(|v| parse_positive_usize(v, "matrix size"))
            .transpose()?;
        let batch = positional
            .get(1)
            .map(|v| parse_positive_usize(v, "batch size"))
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
            gpu_filter,
        })
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value\n{}", usage()))
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize, String> {
    let p = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer"))?;
    if p == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(p)
}

fn parse_positive_f32(value: &str, name: &str) -> Result<f32, String> {
    let p = value
        .parse::<f32>()
        .map_err(|_| format!("{name} must be a positive number"))?;
    if !p.is_finite() || p <= 0.0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(p)
}

fn parse_gpu_max(value: &str) -> Result<u32, String> {
    let p = value
        .parse::<u32>()
        .map_err(|_| "--gpu-max must be a whole-number Celsius temperature".to_string())?;
    if p < 40 {
        return Err("--gpu-max must be at least 40C".to_string());
    }
    Ok(p)
}

fn parse_math(value: &str) -> Result<Math, String> {
    match value.to_ascii_lowercase().as_str() {
        "f32" | "sgemm" => Ok(Math::F32),
        "f16" | "fp16" | "half" => Ok(Math::F16),
        "bf16" | "bfloat16" => Ok(Math::Bf16),
        _ => Err("--math must be one of: f32, f16, bf16".to_string()),
    }
}

fn parse_gpu_list(value: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in value.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let n = p
            .parse::<usize>()
            .map_err(|_| format!("invalid GPU index `{p}` in --gpus"))?;
        out.push(n);
    }
    if out.is_empty() {
        return Err("--gpus requires at least one index".to_string());
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

pub fn usage() -> String {
    "Usage: thermalbloater [matrix-size] [batch-size] \
     [--math f32|f16|bf16] [--gpus 0,1] [--gpu-max C] \
     [--calibrate-secs S] [--status-interval S] [--version]\n\
     Autotunes per-GPU for peak watts unless you pin size/batch/math."
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(|x| x.to_string()).collect()
    }

    #[test]
    fn default_config() {
        let c = Config::from_iter(Vec::<String>::new().into_iter()).unwrap();
        assert_eq!(c.thermostat.gpu_max_c, DEFAULT_GPU_MAX_C);
        assert!(c.math.is_none());
        assert!(c.size.is_none());
        assert!(c.gpu_filter.is_none());
    }

    #[test]
    fn gpus_filter_parsing() {
        let c = Config::from_iter(args("--gpus 0,2").into_iter()).unwrap();
        assert_eq!(c.gpu_filter, Some(vec![0, 2]));
    }

    #[test]
    fn gpus_dedup_sorted() {
        let c = Config::from_iter(args("--gpus 2,0,2").into_iter()).unwrap();
        assert_eq!(c.gpu_filter, Some(vec![0, 2]));
    }

    #[test]
    fn invalid_gpu_rejected() {
        assert!(Config::from_iter(args("--gpus foo").into_iter()).is_err());
    }

    #[test]
    fn unknown_flag_rejected() {
        assert!(Config::from_iter(args("--unknown").into_iter()).is_err());
    }

    #[test]
    fn math_parsing() {
        assert!(parse_math("f32").is_ok());
        assert!(parse_math("bf16").is_ok());
        assert!(parse_math("nope").is_err());
    }
}
