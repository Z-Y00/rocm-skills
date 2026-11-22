---
name: primus-projection
description: Project training step time and memory with Primus (`primus projection performance|memory`), calibrate it against a real GPU (benchmark mode, hardware calibration, the FlyDSL matrix-core clock), plug in an out-of-tree GEMM backend or hardware profile, model architectures the projector does not know, and report a measured-vs-projected "roofline ratio" for a training proxy. Use when projecting a model onto a cluster, comparing a measured run against its projection, adding a GPU without a built-in profile, or writing a weekly performance report.
---

# Primus performance projection

`primus projection` predicts step time (performance mode) and per-GPU memory (memory mode) for a
Megatron training config, without the target hardware. The authoritative reference is
`docs/02-user-guide/projection.md` in Primus; this skill is the working method around it.

## Modes and the knobs that matter

```bash
primus-cli direct -- projection performance --config <exp.yaml> \
    --profiling-mode simulate --gemm-backend <name> --gpu-arch <profile> \
    --hardware-config <cluster.yaml> --target-nodes N  [Megatron overrides, e.g. --num_layers 4]
primus-cli direct -- projection memory --config <exp.yaml>
```

| Knob | What it does |
|---|---|
| `--profiling-mode simulate` | Price every layer with the simulation backends; no GPU needed |
| benchmark mode (default) | Time the layers on the local GPU, then extrapolate to `--target-nodes` |
| `--gemm-backend` / `PRIMUS_GEMM_BACKEND` | GEMM cost model: `origami` built in; others register through `register_gemm_backend` |
| `PRIMUS_GEMM_BACKEND_PLUGINS=<module>` | Import an out-of-tree backend module (it calls `register_gemm_backend`) |
| `--gpu-arch` / `PRIMUS_GPU_ARCH`, `PRIMUS_HW_PROFILE_DIR` | Name the hardware profile the backend prices with, and where profiles live |
| `--hardware-config <yaml>` | Cluster: GPUs per node, network bandwidth / efficiency, topology |
| `--gpu-clock-mhz`, `--hbm-bandwidth-gbps` | Override the clock and HBM bandwidth (e.g. with calibrated values) |
| `--save-benchmark` / `--load-benchmark` | Reuse a benchmark-mode measurement |
| `PRIMUS_PRINT_LAYER_BREAKDOWN=1` | Per-layer fwd / bwd split in the log |

Unknown flags are passed through as Megatron overrides, so a proxy is one command away:
`--num_layers 4 --num_experts 64` with `GPUS_PER_NODE=1 PRIMUS_EP=1 PRIMUS_PP=1 MBS=1 GBS=2`.

## Unreleased or unrecognised hardware

- **Never put unreleased hardware specs in Primus** (code, tests, docs, commit messages, PR text):
  no peak FLOPs, clocks, bandwidths, CU counts. They belong in an out-of-tree hardware profile
  and GEMM backend plugin, loaded with `PRIMUS_GEMM_BACKEND_PLUGINS` / `PRIMUS_HW_PROFILE_DIR`.
  Before pushing, grep the diff for spec-like numbers and product names.
- A backend can supply the matrix-core rates through the optional hook
  `GEMMSimulationBackend.matrix_flops_per_cu_per_clk()`; calibration then works on a part with no
  built-in table. `--gpu-arch` names the profile only when the device is not recognised.

## Architectures the projector does not model

Patch at runtime instead of editing Primus: a runner script imports the projector, wraps the
pieces it gets wrong, then calls the normal CLI. Keep each correction switchable by env
(e.g. `MODEL_DISABLE=latent,a2a,swa,extras`) so its effect can be shown one at a time. Typical
corrections:

- Experts of a non-standard width written as residual-width equivalents with the same FLOPs and
  bytes (e.g. latent `4096 x 4096` experts as `8192 x 2048`).
- All-to-all priced at the width the dispatcher really sends (after a latent projection).
- Attention priced as a blend of sliding-window and full layers (window FLOPs ≈ causal FLOPs ×
  the covered fraction of the triangle).
- Extra GEMMs (latent projections, output gates) and extra memory-bound norms / gates at an
  assumed fraction of HBM bandwidth.

Document every correction next to the numbers it produced.

## Hardware calibration (benchmark mode)

Before the model is built, each bench GPU measures its effective compute clock and HBM
bandwidth; the projection then prices with them. Skip with `--skip-hw-calibration` or
`PRIMUS_BENCH_SKIP_HW_CALIB=1`.

- **Compute clock**, in order of preference:
  1. model inversion: the clock at which the GEMM backend's predicted time of the workload's
     largest GEMMs equals their measured time;
  2. live SCLK sampled during those GEMMs (amdsmi, mean of all XCDs; sysfs as fallback);
  3. the matrix-core loop's cycle-counter clock (below);
  4. the matrix-core-peak lower bound (achieved FLOP rate at 100 % utilisation).
- **Do not invert with a GEMM that runs slower than its bf16 counterpart.** A library's fp8 / MX
  kernel can be slower than bf16 on a new part (wrong tile, missing tuning); inverting a model
  that assumes a proper low-precision kernel gives a meaningless clock. Check the kernel trace
  (`rocprofv3 --kernel-trace`) before trusting a per-precision clock.
- **HBM bandwidth**: STREAM copy / scale / add / triad, best kernel, derated by
  `PRIMUS_HBM_STREAM_DERATE` for the memory-bound layer kernels. Size the arrays well past the
  last-level cache, and remember `torch` copies may run as runtime blits: a hand-written
  vectorized load kernel is the honest roofline.
- **Matrix-core loop** (`hardware_calibration/matrix_core_loop.py`, FlyDSL, where supported):
  every wave issues back-to-back WMMAs on register operands; lane 0 reads the shader-cycle
  counter (`llvm.readcyclecounter`) and the constant-rate real-time counter
  (`llvm.readsteadycounter`) around the loop. Clock = cycles / ticks × the real-time rate, the
  rate itself measured against host time over the kernel's span, so no hardware constant is
  needed. It agreed with the driver's SCLK to well under 1 %. Two lessons:
  - settle on the in-kernel counter (consecutive launches within 1 %), not the driver's clock
    metric: the driver lags a load switch and read several % high right after it;
  - a matrix-only load runs at a **higher** clock than a real GEMM (less power), so use the loop
    as a cross-check and fallback, not as the workload clock.
- amdsmi on multi-die parts: SCLK comes per XCD (`current_gfxclks`), and UCLK may be reported
  only per AID (`current_uclk_aid`) with `current_uclk` unset; read both.

## Measured vs projected: the roofline ratio

For a weekly report on a training proxy, **roofline ratio = projected step time / measured step
time**, always with a per-category breakdown.

1. **Measure.** Run the proxy config with the PyTorch profiler on 2 late steps
   (`--profile True --use_pytorch_profiler True --profile_step_start 15 --profile_step_end 17
   --torch_profiler_with_stack False`). Step time = median of the **unprofiled** steady
   iterations (drop the first ~3 and the profiled ones, which are inflated).
2. **Project** the identical config (same layers, experts, EP / PP / MBS / GBS, precision,
   sliding-window fraction) in simulate mode.
3. **Break down** the trace by category (dense GEMM, grouped GEMM, attention fwd / bwd,
   optimizer, norms, copies / casts, host-to-device copies, idle gaps) and set each against the
   projection's layer fwd / bwd, output layer and optimizer terms.
4. **Report both the headline and the split.** The headline can hide opposite errors: e.g. an
   optimizer projected ~2x slower than a fused Adam really runs, offsetting forward / backward
   compute that is far from its projection. The split says where the next work is.
5. State the exact stack (Primus and Primus-Turbo commits, extra cherry-picks, config path,
   env) so next week's number is comparable.
