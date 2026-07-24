# 🔥 thermalbloater

> The chef recommends the GPU, lightly seared, served at a rolling 80 °C.

![Jensen preheating the oven](https://i.pinimg.com/originals/29/39/13/293913aaf9399fe5952b36a912f75ad6.jpg)

Your GPU cost more than your fridge. It idles at 3% while you doomscroll. That's
a waste of a perfectly good space heater.

**thermalbloater** pins every NVIDIA GPU in the machine at full tilt with batched
cuBLAS matrix multiplies, then backs off just enough to hold each card under a
temperature you pick. The room gets warm. The electricity bill gets exciting.
The GPU gets a cardio workout it never asked for.

It is basically [gpu-fryer](https://github.com/huggingface/gpu-fryer)'s
mischievous cousin: same "hammer the card with SGEMM" engine, opposite intent.
gpu-fryer wants to *catch* a weak GPU. thermalbloater just wants to keep you cozy.

## What it actually does

- 🍳 Runs a batched SGEMM burn load on **every** CUDA device it finds — plug in
  four GPUs, fry four GPUs.
- 🌡️ Per-GPU thermostat: each card throttles its own duty cycle to stay under
  `--gpu-max` (default 80 °C). Hot card slows down, cool card keeps cooking.
- 🎛️ One binary runs on **any CUDA 11 / 12 / 13** install. The CUDA libraries are
  loaded at runtime, so you don't need a toolkit to build it and you don't need a
  matching version to run it.
- 🐧🪟 Prebuilt binaries for Linux and Windows on every release.

## Install

Grab a binary from [Releases](../../releases) and run it. That's it.

You need an NVIDIA GPU, a working driver, and cuBLAS somewhere on your system
(any CUDA 11/12/13 runtime provides it). If `nvidia-smi` works, you're good.

Or build it yourself — no CUDA toolkit required, just Rust:

```sh
cargo build --release
```

## Usage

```
thermalbloater [matrix-size] [batch-size] [--gpu-max C] [--status-interval SECONDS]
```

```sh
# Preheat the room, keep every GPU under 80 °C
thermalbloater

# Gentle simmer — cap at 70 °C
thermalbloater --gpu-max 70

# Full send — bigger matrices, cap at 85 °C, chunky VRAM footprint
thermalbloater 1024 64 --gpu-max 85
```

Live status looks like:

```
Frying 2 GPU(s): 32 batched 512x512 SGEMMs each (~96 MiB VRAM/GPU). Press Ctrl+C to stop.
Thermostat: GPU max 80C (per GPU).
GPU0 78C 310W 45% | GPU1 71C 285W 100% | CPU 62C (Package)
```

Press `Ctrl+C` when the room is warm enough (or when the electricity bill texts you).

### Flags

| Flag | Default | Does |
|------|---------|------|
| `matrix-size` (positional) | `512` | N for the N×N matmuls. Bigger = more heat & VRAM. |
| `batch-size` (positional) | `32` | How many matmuls per batch. |
| `--gpu-max C` | `80` | Temperature ceiling per GPU. Card eases off as it approaches. |
| `--status-interval S` | `2` | Seconds between status updates. |

## The fine print (please read this one)

This is a tool for **deliberately running your own hardware hot**. That's the
entire point. Only run it on GPUs you own, on power you're paying for, in a case
with cooling that actually works.

- It will not exceed `--gpu-max`, but "not overheating" still means "running hot
  for a long time." Dust, bad airflow, and sketchy PSUs do not care about your
  thermostat.
- If NVML can't read your GPU's temperature, there's **no cap** and it runs
  flat-out. The status line will yell `NVML unavailable, running uncapped` at you.
  Believe it.
- No warranty, express or implied, including but not limited to fitness for
  cooking an actual egg on your GPU. If you brick something, you got to keep both
  halves.

Stay warm. 🔥
