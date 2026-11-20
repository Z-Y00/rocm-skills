# Triton on ROCm — Optimization Directions

Directions worth trying on Triton kernels for AMD GPUs, as Triton expresses them. Read
[`overview.md`](overview.md) first (compilation model, how to read the compiled kernel), and the op
catalogs ([`../../ops/attention/optimization-directions.md`](../../ops/attention/optimization-directions.md),
[`../../ops/gemm/optimization-directions.md`](../../ops/gemm/optimization-directions.md)) for the
algorithm-level reasons. One hypothesis per round; read the compiled kernel before and after.

## 1. Size the footprint for the bottleneck

- **Memory-streaming kernels (decode, elementwise, reductions over long axes):** bandwidth follows
  bytes in flight per CU. Prefer small workgroups (2-4 warps), `num_stages` 1-2, one reused LDS tile
  per operand, and enough programs for several workgroups per CU. Check `shared` and `; Occupancy`,
  then compute workgroups per CU.
- **Compute-bound kernels (prefill, training forward):** larger tiles and `num_stages` 2 feed the
  MFMAs; occupancy matters less. Re-check the block size per head dim: in a masked attention forward
  256-row blocks were 1.4-2.9x slower than 128 at D = 128 (but fastest at D = 64).
- **A deep pipeline is not free:** each stage is another LDS buffer, and LDS is what caps workgroups
  per CU on a 160 KiB gfx950 CU.

## 2. Load path

- **Buffer loads.** Keep offsets 32-bit (cast with a constexpr guard when tensors have < 2^31
  elements) so loads lower to `amdg.buffer_load`; check the TTGIR.
- **Masks.** Load interior tiles unmasked and mask only the edge tile; per-element masks on every
  tile cost VALU and block vectorization.
- **Cache modifiers.** `.cg` on streamed-once data (KV cache in decode) avoids polluting L2-resident
  data; measure.
- **Issue order.** Loading K and V at the top of the loop body (before the QK dot) keeps the V load
  in flight under QK even with `num_stages=1`.

## 3. Dot shape and warps

- **`matrix_instr_nonkdim=16`** for small-M dots (decode rows, GQA groups): 16x16 MFMAs waste 4x less
  than 32x32 when only a few rows are real.
- **Rows-only warp tiling** when bits must not depend on the launch: keep `num_warps <= BLOCK_M / 16`.
- **`kpack`** 1 vs 2 only matters for some operand layouts; sweep it last.

## 4. Occupancy

- `waves_per_eu` asks LLVM for an occupancy; it helps only when registers are the limit. If LDS
  (`shared`) is the limit, it does nothing; shrink the LDS footprint instead.
- Spills (`.vgpr_spill_count`) at a high `waves_per_eu` usually cost more than the extra wave.

## 5. Split work across programs

- **Split-KV / split-K** when programs < ~2 x CUs: fixed segment boundaries and an in-order merge keep
  bits independent of the split (see [`../../numerics/batch-invariance.md`](../../numerics/batch-invariance.md)).
- **Count the work actually read.** A sliding window reads at most two segments; splitting over the
  whole context launches programs with no keys (1.7x slower in a measured decode).
