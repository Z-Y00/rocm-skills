# Triton on ROCm — Backend Orientation

Read this first when `target_lang` / `target_backend` is **Triton** on an AMD GPU. It covers what the
AMD backend does with a `@triton.jit` kernel, how to read what it compiled (the step that answers
"why is this kernel fast or slow"), and the launch knobs. Techniques to try are in
[`optimization-directions.md`](optimization-directions.md); version and interpreter traps are in
[`pitfalls.md`](pitfalls.md). Hardware budgets (VGPR / LDS quanta, occupancy) are in
[`../../hardware/gfx950/overview.md`](../../hardware/gfx950/overview.md).

## Compilation model

A Triton kernel is traced once per specialization (constexpr values, dtypes, divisibility of integer
arguments, launch options) and lowered:

```text
Python -> TTIR (tt.*, layout-free) -> TTGIR (ttg.* / amdg.*, every tensor has a layout)
       -> LLVM IR (.llir) -> AMDGCN (.amdgcn) -> code object (.hsaco)
```

The decisions that matter for performance are made in **TTGIR**, by the compiler, not the author:

- **Layouts.** Each tensor gets a distributed layout: `#ttg.blocked` (elementwise / loads),
  `#ttg.amd_mfma` (an MFMA result; `instrShape` is the MFMA size, `warpsPerCTA` how warps tile it),
  `#ttg.dot_op` (an MFMA operand) and `#ttg.swizzled_shared` / padded shared (LDS). A
  `ttg.convert_layout` between distributed layouts usually goes through LDS.
- **Loads.** A `tt.load` becomes `amdg.buffer_load` when the base pointer is uniform and the offsets
  provably fit 32 bits (int64 offsets fall back to global loads). `cache_modifier=".cg"` and similar
  show up as `cachePolicy` attributes.
- **Dot operands.** `tt.dot` needs operands in `dot_op` layout. A loaded tile reaches it either
  directly (when the load layout already matches) or through LDS: `ttg.local_alloc` (write) then
  `ttg.local_load` (read in MFMA layout). One reused LDS tile per operand is the common shape.
- **Pipelining.** `num_stages` drives the AMD stream pipeliner on `scf.for` loops: 1 means no
  software pipeline (load, then compute, per iteration); 2+ hoists the next iteration's loads
  (register-staged, or direct-to-LDS async copies where the target and pattern allow) and allocates
  that many LDS buffers. `shared` in the metadata grows with it.
- **MFMA shape.** `matrix_instr_nonkdim` (16 or 32) selects 16x16xK vs 32x32xK MFMAs for `tt.dot`;
  small-M dots (decode: a few rows) fit 16x16 with 4x less wasted work. Different MFMA shapes may
  sum a dot product in a different order: do not mix them across paths that must match bitwise.

## Reading what Triton compiled

Set `TRITON_CACHE_DIR=<dir>` (fresh) and run the kernel once. Each specialization gets
`<dir>/<hash>/` with:

| File | Use |
|---|---|
| `<kernel>.json` | Launch metadata: `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `kpack`, and **`shared`** (LDS bytes per workgroup, allocated at launch) |
| `<kernel>.ttgir` | Layouts, `amdg.buffer_load`, `local_alloc` / `local_load`, `tt.dot`, loop structure: how data moves |
| `<kernel>.amdgcn` | ISA plus a header: `; NumVgprs`, `; Occupancy` (waves per SIMD), `.vgpr_spill_count`, `.sgpr_count`, `.max_flat_workgroup_size` |
| `<kernel>.llir`, `.ttir`, `.source` | Lower / higher levels when a TTGIR decision needs explaining |

- The `.amdgcn` header's `LDSByteSize` is compile-time only and reads 0 for Triton kernels; the
  real LDS use is `shared` in the JSON.
- Occupancy per CU is the minimum of the VGPR limit (`; Occupancy`, per SIMD), the LDS limit
  (160 KiB / `shared` rounded to 1280 B on gfx950) and `waves_per_eu`; workgroups per CU =
  occupancy x 4 SIMDs / waves per workgroup (`num_warps`).
- A quick instruction census of `.amdgcn` (`buffer_load_dwordx4`, `ds_write_*`, `ds_read_*`,
  `v_mfma_*` by shape, `s_waitcnt`, `s_barrier`) shows the load path and the MFMA shape at a glance.
- Other switches: `TRITON_ALWAYS_COMPILE=1` (ignore the cache), `MLIR_ENABLE_DUMP=1` (IR after every
  pass), `AMDGCN_ENABLE_DUMP=1` (ISA to stderr).

## Launch knobs (AMD)

| Knob | Effect |
|---|---|
| `num_warps` | Waves per workgroup; with the layout it decides how warps tile a dot (rows, columns or both) |
| `num_stages` | Pipeline depth (see above); 1 = none |
| `waves_per_eu` | Occupancy target passed to LLVM (`amdgpu-waves-per-eu`); trades registers for waves |
| `matrix_instr_nonkdim` | MFMA M/N size (16 / 32) |
| `kpack` | K packing of MFMA operands (1 / 2) |

Launch knobs can change bits: if warps tile both rows and columns of a dot, the reduction
splits differently (see [`../../numerics/batch-invariance.md`](../../numerics/batch-invariance.md)).

## Worked example: why a Triton decode beat a hand-scheduled FlyDSL one 2x

Paged GQA decode, Llama-3-8B (32 / 8 heads, D = 128), 8 x 32k context, bf16, MI355X, split-KV. Same
math, both bitwise equal to their own training forward. Triton 195 us, FlyDSL dual-wave 373 us.

| | Triton (`num_warps` 2, `num_stages` 1) | FlyDSL dual-wave |
|---|---|---|
| Rows per workgroup | 16 (4 real: 1 token x 4 heads) | 64 (4 real) |
| MFMA | `16x16x32` | `32x32x16` |
| K / V path | `buffer_load` -> VGPR -> `ds_write` -> `ds_read` (MFMA layout) | `buffer_load ... lds` straight to LDS, double buffered |
| LDS per workgroup | 16 KiB (one tile, reused for K then V) | 68 KiB |
| VGPR / occupancy | 204 / 2 waves per SIMD | 312 + 56 AGPR / 1 wave per SIMD |
| Workgroups per CU | 4 | 2 (LDS bound) |
| K / V bytes in flight per CU | ~4 x 32 KiB | ~2 x 32 KiB |

The static comparison suggests bytes in flight per CU (Little's law) explain the gap. Measured,
they did not: a FlyDSL build with one 34 KiB buffer (4 workgroups per CU, loads overlapping compute)
was no faster. Per-kernel GPU times (`torch.profiler`) showed most of the gap in the second launch:
the FlyDSL merge of the split partials took 102 us against Triton's 9 us reduce, because each lane
loaded one segment, merged it, then loaded the next (32 dependent round trips), while Triton's reduce
issues every segment's loads at once and merges in registers. The streaming split programs were only
1.3x apart (232 vs 183 us). Merging one accumulator chunk per program with 8 segments' loads in
flight took the FlyDSL merge to 29 us.

Lesson: compare per-kernel GPU times before reasoning from compiled resources; a multi-launch op's
end-to-end time hides which launch is slow.
