# 🔥 thermalbloater

> The chef recommends the GPU, lightly seared and served at a steady 80 °C.

![Jensen preheating the oven](https://i.pinimg.com/originals/29/39/13/293913aaf9399fe5952b36a912f75ad6.jpg)

Your GPU probably costs more than your fridge, yet it sits at 3% utilization while you doomscroll. That is a tragic waste of a perfectly capable space heater.

**thermalbloater** keeps every NVIDIA GPU in your machine busy with batched cuBLAS matrix multiplications, then automatically adjusts the workload to keep each card below a temperature limit you choose.

The room gets warmer. The power bill gets more interesting. Your GPU finally gets the workout it never requested.

It is essentially the mischievous cousin of [gpu-fryer](https://github.com/huggingface/gpu-fryer): the same general idea of hammering the GPU with SGEMM workloads, but with a very different objective. `gpu-fryer` tries to expose unstable hardware. `thermalbloater` is mostly here to keep you cozy.

## What it does

* 🍳 Runs batched SGEMM workloads on **every CUDA-capable GPU** it finds. Install four GPUs, heat the room with all four.
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

## Usage

```text
thermalbloater [matrix-size] [batch-size] [--gpu-max C] [--status-interval SECONDS]
```

### Examples

```sh
# Preheat the room and keep every GPU below 80 °C
thermalbloater

# Gentle simmer at 70 °C
thermalbloater --gpu-max 70

# Larger matrices, higher temperature limit, and a chunkier VRAM footprint
thermalbloater 1024 64 --gpu-max 85
```

A typical status display looks like this:

```text
Frying 2 GPU(s): 32 batched 512x512 SGEMMs each (~96 MiB VRAM/GPU). Press Ctrl+C to stop.
Thermostat: GPU max 80C (per GPU).
GPU0 78C 310W 45% | GPU1 71C 285W 100% | CPU 62C (Package)
```

Press `Ctrl+C` when the room is warm enough, or when your electricity provider begins asking personal questions.

## Options

| Option                | Default | Description                                                                                                  |
| --------------------- | ------: | ------------------------------------------------------------------------------------------------------------ |
| `matrix-size`         |   `512` | Matrix dimension `N` for each `N × N` multiplication. Larger values use more VRAM and may produce more heat. |
| `batch-size`          |    `32` | Number of matrix multiplications submitted per batch.                                                        |
| `--gpu-max C`         |    `80` | Maximum target temperature for each GPU. The workload is reduced as the card approaches this limit.          |
| `--status-interval S` |     `2` | Number of seconds between status updates.                                                                    |

## Safety and limitations

This tool is designed to **deliberately run your own hardware under sustained load**. That is not an accidental side effect. It is the whole ridiculous idea.

Only use it on hardware you own, with adequate cooling, a reliable power supply, and electricity you are prepared to pay for.

* `thermalbloater` attempts to keep each GPU below `--gpu-max`, but sustained operation near that temperature still places the hardware under significant thermal and electrical load.
* Dust, poor airflow, failing fans, unstable overclocks, and questionable power supplies can still cause problems. Software cannot negotiate with physics, despite decades of human effort.
* If NVML cannot report a GPU's temperature, that GPU cannot be temperature-limited and will run at full load. The status output will display:

  ```text
  NVML unavailable, running uncapped
  ```

  Take that warning seriously.
* Temperature readings and throttling behaviour depend on the GPU, driver, firmware, cooling system, and operating environment.
* This software is provided without warranty. There is no guarantee of fitness for any purpose, including heating a room, validating hardware, or cooking an egg directly on the backplate.

Use common sense, monitor your hardware, and stop the program if anything looks, smells, or sounds wrong.

Stay warm. 🔥
