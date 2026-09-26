# 🔥 thermalbloater

> The chef recommends the GPU, lightly seared and served at a steady 80 °C.

![Jensen preheating the oven](https://i.pinimg.com/originals/29/39/13/293913aaf9399fe5952b36a912f75ad6.jpg)

Your GPU probably costs more than your fridge, yet it sits at 3% utilization while you doomscroll. That is a tragic waste of a perfectly capable space heater.

**thermalbloater** keeps every NVIDIA GPU in your machine busy with batched cuBLAS matrix multiplications, then automatically adjusts the workload to keep each card below a temperature limit you choose.

The room gets warmer. The power bill gets more interesting. Your GPU finally gets the workout it never requested.

It is essentially the mischievous cousin of [gpu-fryer](https://github.com/huggingface/gpu-fryer): the same general idea of hammering the GPU with SGEMM workloads, but with a very different objective. `gpu-fryer` tries to expose unstable hardware. `thermalbloater` is mostly here to keep you cozy.

## What it does

* 🍳 Runs batched GEMM workloads on **every CUDA-capable GPU** it finds. Install four GPUs, heat the room with all four.
* 🔥 **Autotunes each GPU for peak power draw.** On startup it sweeps math type (`f32` / `f16` / `bf16` — CUDA cores vs. tensor cores) and matrix size, measures the actual watts each config pulls, and keeps the hottest one. A GTX 1060 and an H100 want completely different workloads to redline; the tool finds each card's on its own instead of guessing.
* 🌡️ Uses a separate thermostat for each GPU. A hot card reduces its duty cycle while cooler cards continue running at full load.
* 🎛️ Supports **CUDA 11, 12, and 13** with a single binary. CUDA libraries are loaded dynamically at runtime, so you do not need the CUDA toolkit to build the project or an exact toolkit version match to run it.
* 🐧 🪟 Provides prebuilt Linux and Windows binaries with every release.

## Installation

Download the appropriate binary from [Releases](../../releases), then run it.

You need:

* An NVIDIA GPU
* A working NVIDIA driver
* A CUDA 11, 12, or 13 runtime containing cuBLAS

As a practical rule, if `nvidia-smi` works and cuBLAS is installed, you should be ready to start wasting electricity productively.

### Build from source

No CUDA toolkit is required. You only need Rust:

```sh
cargo build --release
```

### Heat the room on a schedule (Linux)

Electricity is cheapest — and rooms are coldest — in the evening. One line downloads the latest release, installs it, and schedules a daily heating window (default **17:00 → 22:00**):

```sh
curl -fsSL https://raw.githubusercontent.com/flolep2607/thermalbloater/main/packaging/install.sh | sudo sh

# Custom window and options
curl -fsSL .../install.sh | sudo START=20 END=23 ARGS="--gpu-max 70 --math f16" sh

# Or from a clone, using your own build
cargo build --release && sudo ./packaging/install.sh
```

It uses **systemd** when systemd is running (a timer starts the run, `RuntimeMaxSec` hard-stops it at the end hour) and falls back to **crontab** when it is not — WSL and containers included. Either way it also installs a daily self-update job that pulls the latest release binary; the new version takes effect at the next start, never mid-run.

Change the window by re-running the installer with different `START`/`END`. Skip the updater with `NO_UPDATE=1`. Remove everything — units, cron lines, both binaries:

```sh
sudo UNINSTALL=1 ./packaging/install.sh
```

Booting *inside* the window skips that evening rather than starting a run that would burn past the end hour.

On Windows, the same idea in one line — Task Scheduler starts it, `RuntimeMaxSec` becomes a matching stop task:

```powershell
schtasks /create /tn thermalbloater /tr "C:\path\thermalbloater.exe" /sc daily /st 17:00
schtasks /create /tn thermalbloater-stop /tr "taskkill /im thermalbloater.exe /f" /sc daily /st 22:00
```

## Usage

```text
thermalbloater [matrix-size] [batch-size] [--math f32|f16|bf16] [--gpus 0,1] [--gpu-max C] [--calibrate-secs S] [--status-interval SECONDS]
```

By default, with no positional arguments, thermalbloater **autotunes**: it briefly runs each candidate workload on every GPU, reads the power draw from NVML, and locks in whichever config pulls the most watts on that specific card. Pin any axis yourself and it drops out of the sweep — pin all three (size, batch, math) and it skips autotuning entirely.

### Examples

```sh
# Autotune every GPU for peak watts, keep each below 80 °C
thermalbloater

# Autotune, but only ever use tensor-core FP16, simmering at 70 °C
thermalbloater --math f16 --gpu-max 70

# Fully manual: fixed size, batch, and math — no autotuning
thermalbloater 1024 64 --math bf16 --gpu-max 85
```

A typical run looks like this:

```text
Calibrating 1 GPU(s) for peak power draw...
  GPU0 f32 4096x4096 GEMM x11 (~2112 MiB VRAM) -> 231 W
  GPU0 f16 2048x2048 GEMM x93 (~2232 MiB VRAM) -> 199 W
  ...
GPU0: f32 4096x4096 GEMM x11 (~2112 MiB VRAM)
Thermostat: GPU max 80C (per GPU).
GPU0 78C 234W 100%
```

(On this RTX 3070, plain FP32 out-draws the tensor-core paths — on a bigger card the winner will often be `f16` or `bf16`. That's the whole point of measuring instead of assuming.)

Press `Ctrl+C` when the room is warm enough, or when your electricity provider begins asking personal questions.

## Options

| Option                | Default | Description                                                                                                  |
| --------------------- | ------: | ------------------------------------------------------------------------------------------------------------ |
| `matrix-size`         |  autotuned | Matrix dimension `N` for each `N × N` multiplication. Pinning it skips the size sweep. |
| `batch-size`          |  autotuned | Number of matrix multiplications submitted per batch. Pinning it skips VRAM-based batch sizing. |
| `--math f32\|f16\|bf16` | autotuned | Arithmetic type: `f32` (CUDA cores) or `f16`/`bf16` (tensor cores). Pinning it skips the math sweep. |
| `--gpus 0,1`          |     all | Comma-separated GPU indices to heat. The others are left alone. |
| `--gpu-max C`         |    `80` | Maximum target temperature for each GPU. The workload is reduced as the card approaches this limit.          |
| `--calibrate-secs S`  |     `4` | Seconds spent measuring each candidate config during autotuning. |
| `--status-interval S` |     `2` | Number of seconds between status updates.                                                                    |

## Releasing

CI (`.github/workflows/ci.yml`) runs `fmt`, `clippy -D warnings`, tests and a release build on Linux and Windows for every push and PR. Cutting a release is one command:

```sh
cargo install cargo-release   # once
cargo release patch --execute # or minor / major
```

That bumps the version, commits, tags `vX.Y.Z` and pushes. The tag push triggers `release.yml`, which builds the Linux and Windows binaries and attaches them to a GitHub Release.

## Safety and limitations

This tool is designed to **deliberately run your own hardware under sustained load**. That is not an accidental side effect. It is the whole ridiculous idea.

Only use it on hardware you own, with adequate cooling, a reliable power supply, and electricity you are prepared to pay for.

* `thermalbloater` attempts to keep each GPU below `--gpu-max`, but sustained operation near that temperature still places the hardware under significant thermal and electrical load.
* Dust, poor airflow, failing fans, unstable overclocks, and questionable power supplies can still cause problems. Software cannot negotiate with physics, despite decades of human effort.
* Autotuning needs NVML to read power draw. Without it, thermalbloater cannot measure watts and falls back to a safe default config (or whatever you pin manually).
* If NVML cannot report a GPU's temperature, that GPU cannot be temperature-limited and will run at full load. The status output will display:

  ```text
  NVML unavailable, running uncapped
  ```

  Take that warning seriously.
* Temperature readings and throttling behaviour depend on the GPU, driver, firmware, cooling system, and operating environment.
* This software is provided without warranty. There is no guarantee of fitness for any purpose, including heating a room, validating hardware, or cooking an egg directly on the backplate.

Use common sense, monitor your hardware, and stop the program if anything looks, smells, or sounds wrong.

Stay warm. 🔥
