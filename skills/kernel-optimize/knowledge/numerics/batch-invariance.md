# Batch Invariance and Train/Infer Bitwise Match

Read this when a kernel must give **bitwise-identical** results across batch composition, or between
two stacks (for example an RL rollout engine and a trainer computing the same forward). It applies
to any reduction kernel: GEMM, grouped GEMM, attention, norms, softmax. It complements
[`../ops/attention/math-optimizations.md`](../ops/attention/math-optimizations.md) (deterministic
reductions) with the stronger requirement that the reduction order must not depend on the batch.

Background: Thinking Machines, "Defeating Nondeterminism in LLM Inference"; SGLang's deterministic
inference mode (batch-invariant ops, fixed split tile sizes). The findings below were measured on
MI355X (gfx950), Triton 3.8, ROCm 7.x/10.x, in the OneTensor project.

## The rule

Floating-point addition is not associative, so the bits of a reduction are fixed by the **order** of
its additions. Run-to-run determinism (no atomics) is necessary but not enough: most production
kernels are deterministic yet not batch-invariant, because the **host-side config choice** picks a
different reduction order for a different batch shape.

- Everything that sets a reduction order is a function of the model only (N, K, head dim, GQA ratio,
  window, dtype), never of M, batch size, sequence lengths, or prefill vs decode:
  `BLOCK_K`, split-K boundaries, attention `TILE_SIZE` (keys per inner step), KV segment boundaries,
  the MFMA instruction (`matrix_instr_nonkdim`, `kpack`), and the order partials are combined.
- Everything that only schedules work may depend on shape: grid size, persistent-loop striding,
  tile swizzle, which CU gets which tile, cache modifiers, `num_stages`, `waves_per_eu`, and (with
  the caveat below) `BLOCK_M` / `num_warps`.

Typical violations found in production libraries:

| Library entry | What changes with batch shape |
|---|---|
| Primus-Turbo Triton GEMM | `BLOCK_K` 32 vs 64 by `min(M, N)` and K; `BLOCK_M/N` from an analytical model |
| Primus-Turbo grouped GEMM | config from `avg_m = M_total // G` |
| aiter `gemm_a16w16` | per-M JSON buckets change `BLOCK_K`, and some shapes use split-K |
| aiter `gmm` | config by average rows per group |
| aiter `unified_attention` | 2D vs 3D (split-KV) kernel by batch; `NUM_SEGMENTS` from the batch; segments carved relative to each sequence |

## Splitting is allowed if the split points are aligned

Split-K and split-KV (flash-decoding) do not break invariance by themselves; choosing the split from
the batch does. Fix split points per model: K split `i` covers `[i * S_K, (i + 1) * S_K)`, KV segment
`s` covers keys `[s * S_KV, (s + 1) * S_KV)` at **absolute** key positions. Combine partials in a
fixed order (sequential fp32 left fold for K partials, LSE merge in increasing segment order). Never
atomics; never a reduction tree whose shape depends on how many partials exist. Stream-K and "pick
the split count to fill the CUs" are not allowed (their split points depend on M).

The same bits can then be produced two ways, chosen freely by shape:

- **Split mode** (decode / small M): one program per split writes fp32 partials, a combine kernel folds them.
- **Fused mode** (training forward / large M): one program walks the same splits in order with fresh
  per-split state and folds into a running total. No partial buffer, no extra HBM traffic.

To make both modes bit-identical, write the per-tile step, the merge and the epilogue **once** as
shared `@triton.jit` helpers and call them from every kernel. Verified bitwise on MI355X for
attention: fused 2D kernel with a segment loop == 3D split kernel + reduce kernel.

## Online softmax: exact no-ops

For per-row results to be independent of which tiles a program visits, tiles that hold none of a
row's keys must be **exact** no-ops for that row:

- **Do not reset the running max from `-inf` to 0.** vLLM/aiter's kernels do
  `m = where(m > -inf, m, 0)` after a fully masked tile. With a sliding window, whether a fully
  masked *leading* tile is visited depends on how query rows are grouped into a block, so the same
  row in prefill and in decode rounds differently. Keep `M = -inf` and use a 0 stand-in only inside
  `exp2`: `m_safe = where(m_new > -inf, m_new, 0); p = exp2(s - m_safe); alpha = exp2(M - m_safe)`.
  Then a leading masked tile leaves `(M, L, acc) = (-inf, 0, 0)` and a trailing one multiplies by
  `alpha = exp2(0) = 1` and adds 0.
- The LSE merge `m = max(m_a, m_b); l = l_a * exp2(m_a - m) + l_b * exp2(m_b - m)` (same for acc) is
  an exact no-op for an empty segment `(-inf, 0, 0)` on either side of the real ones, with the same
  `m_safe` trick. So a combine kernel may skip segments a row does not touch while fused mode visits
  them, and both agree.
- Tile boundaries must be absolute (`j * TILE_SIZE`), not relative to the program's first query.

## Masked loads

- Always mask key loads by `seq_len`, including the `TILE_SIZE == page_size` case that upstream
  skips. Unused page slots hold stale data; `0 * NaN` in `P @ V` would poison the row, and the
  masked/unmasked variants are different code paths.
- Skipping the mask on interior tiles (all keys `< seq_len`) and masking only the last tile is fine:
  loads change, arithmetic does not.
- Contiguous K/V (training) can be read through the paged kernel with a page size of 1
  (`row = cu_seqlens_k[s] + key`); the page size only enters address arithmetic. Verified bitwise
  for page sizes 1, 16, 64.

## Launch layout can change bits (measured)

Assumption to **test, not trust**: "BLOCK_M and num_warps only schedule work." On MI355X / Triton
3.8, for the attention forward (`S = Q @ K^T`, online softmax, `acc = P @ V`, MFMA 16x16):

- Every launch whose warps tile only the **rows** of the score tile (`num_warps <= BLOCK_M / 16`), or
  only the **columns** (`BLOCK_M == 16`), gave identical output bits: e.g. (16, 1..8), (32, 1..2),
  (64, 1..4), (128, 1..8).
- Launches whose warps tile **both** axes, e.g. `(BLOCK_M, num_warps)` = (32, 4), (32, 8), (64, 8),
  gave different output bits on some head configs.
- The LSE (row max and row sum) was identical in all launches, so the difference is in the
  `P @ V` / `Q @ K^T` path, not in `tl.sum` / `tl.max`.

Method that found it (reuse it for any kernel): run the same inputs through a grid of launch
configs, compare outputs bitwise, and print equivalence classes; compare secondary outputs (LSE)
separately to localize the difference. Then enforce the rule in the config code.

## Pin the MFMA instruction shape

The MFMA instruction (16x16xK vs 32x32xK) sets the per-element accumulation granularity, so it must
be the same in every mode. Rows per program (`BLOCK_M`) may vary; the instruction may not. A wide
program can be built from many slim instructions, but a slim program cannot use a wide instruction
without wasting rows, so pick the slimmest MFMA that prefill still runs near peak and vary `BLOCK_M`.

- **Triton picks it from the tile shape unless told.** `chooseMfmaInstruction`
  (`third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`, Triton main, 2026-11) uses
  32x32 when `min(M, N) >= 32`, 16x16 when `>= 16`, else 64x4 / 4x64, unless
  `matrix_instr_nonkdim` is set. A decode launch with 16 rows and a prefill launch with 64+ rows
  therefore get different instructions. Pass `matrix_instr_nonkdim` (and `kpack`) on every launch.
- **aiter `unified_attention` on gfx950 defaults to its Gluon kernels**, which take `MFMA_DIM` from
  `configs/gfx950/gluon/attention/unified_attention/DEFAULT.json` per bucket (aiter `892e5d0`):
  plain decode (3D `any`) 16; two or more query tokens (3D `Q_GEQ_2`) 32; prefill (2D `any`) 32.
  With `backend="triton"` the configs carry no `matrix_instr_nonkdim`, so the heuristic above applies.
- SGLang's Triton attention (`extend_attention_fwd_unified`, `decode_attention_fwd`) pins
  `matrix_instr_nonkdim=16, kpack=2` on HIP (sglang `sglang-miles` branch `3f7c4f02`). OneTensor
  pins `matrix_instr_nonkdim=16` on every attention launch.
- The k-width variant (e.g. 16x16x32 vs 16x16x16 for bf16) is still Triton's choice; keep `kpack`
  and the Triton version identical on both sides, and check the ISA when it matters.

## Wave-uniform lazy rescale breaks prefill/decode match

FlashAttention kernels that skip the accumulator rescale until the running max grows past a margin
must make that decision **per row**. The gfx950 FlyDSL dual-wave forward (FlyDSL
`kernels/attention/flash_attn_gfx950.py`, and its copies in Primus-Turbo
`primus_turbo/flydsl/attention/flash_attn_fwd.py` and MSLK `mslk/attention/flydsl/flash_attn_gfx950.py`)
fires the rebase **wave-uniformly** (`_wave_any` / ballot over the 32 rows of a wave), then rebases
every row of the wave. A row's reference, and so its exp2 inputs and QK seed, depends on which rows
share its wave: 31 neighbors in a full prefill, none in a decode step, others in a chunked prefill.
Fix: per-row trigger, unconditional accumulator multiply (factor exactly 1.0 when the row did not
fire). MSLK's `dualwave_swp_lazy_rescale=False` path (`rescale_o` every tile) is a ready per-row
variant to measure the cost against.

## KV blocks must be accumulated in increasing absolute key position

A row's result depends on the **order** its key blocks are added, not only on the set. Two ways this
breaks even when every visible block is visited:

- **Partial-before-full ordering.** PyTorch FlexAttention (torch 2.8, `torch/_inductor/kernel/flex_attention.py`)
  walks the block mask's partial blocks first, then its full blocks (each list ascending). Partial vs
  full is decided for the whole 128-row query block (`SPARSE_Q_BLOCK_SIZE`), not per row, so a block
  fully visible to a row can be "partial" in a training query block and "full" when that token is
  decoded alone. With pure causal masks the only partial block is the diagonal in both cases (likely
  consistent, untested); with sliding window, document or custom interval masks the order differs
  and the bits differ, even with the kernel and config pinned.
- **Compacted keys vs masked keys.** Masking keys outside `[i, j]` on the full absolute grid is not the
  same as physically keeping only keys `i..j` and renumbering them from 0: the tile boundaries move.
  Both sides must use the same absolute positions.

## Library audit (what breaks invariance, by library)

| Library / kernel | Status | Findings |
|---|---|---|
| aiter `unified_attention` (Triton and Gluon) | Measured on MI355X (aiter in `rocm/primus:v26.5`) | Decode alone vs batched 8/64: differs (up to 2e-3, bf16); prefill vs decode and chunked prefill: differ (~1e-3 / 4e-3); with a window, batched decode matched (always 2D). Causes: 2D vs split-KV 3D chosen by `max_seqlen_k <= 512` or program count vs 4x CUs; `NUM_SEGMENTS` from the batch with segments relative to each sequence; per-bucket tile and MFMA; running max reset to 0 after a fully masked tile |
| FlexAttention | Code-read, torch 2.8.0 | Query length < 128 switches to the flex-decoding kernel; its `SPLIT_KV = num_CUs // (batch * kv_heads) * 2`, split length `cdiv(KV_LEN, SPLIT_KV)`; `max_autotune` tunes block sizes per shape; partial-before-full block order (above). Keeps the running max at -inf (good). Can be made invariant across other sequences with `kernel_options` (`FORCE_USE_FLEX_ATTENTION=True`, fixed block sizes, no autotune), at the cost of decode speed |
| FlyDSL gfx950 dual-wave forward (Primus-Turbo `fce86ab`) | Code-read | Wave-uniform lazy rescale (above); D <= 64 score bound computed with `k_sq_norm_max(k, 8)` over the whole packed K tensor (depends on other sequences); `cross_seqlen` build flag chosen from batch max lengths; dispatcher only selects FlyDSL for uniform lengths (`_uniform_cu_seqlens`), else aiter; no split-K in the forward |
| MSLK FlyDSL (`0ccfd55`) | Code-read | Same lazy rescale; prefill `num_kv_splits` from a CK-ported batch/CU heuristic with splits relative to each sequence; in-workgroup `merge_wave_splitk`; dense routing between dual-wave and generic kernels by batch and length (`_dense_routes_to_dualwave`); decode kernels use 16x16x32 MFMA and 32-key tiles (prefill: 32x32x16, 64-key) with `auto_split_k_hp` from batch/CUs and a wave-butterfly merge whose order depends on the partition count |
| SGLang Triton backend, deterministic mode | Code-read, `sglang-miles` `3f7c4f02` | Extend/prefill uses the 1-stage `extend_attention_fwd_unified`; decode uses `decode_attention_fwd` with fixed-tile split-KV (`num_kv_splits = ceil(seq_len / split_tile_size)`). Deterministic and batch-invariant, but decode is a different kernel from prefill |

## Two contracts for train/infer match

- **Decode must match training** (OneTensor): every mode runs the same per-row arithmetic; decode
  uses fixed absolute KV segments that training walks in-kernel. No extra work at rollout.
- **Prefill recompute** (Miles true-on-policy, `radixark/miles` `11fd95f`): sample with any
  deterministic decode, then recompute the rollout log-probs with a deterministic prefill
  (`--recompute-logprobs-via-prefill`, which forces SGLang's prefill-only deterministic mode). Only
  prefill must match the training forward; costs one extra prefill pass per rollout. Miles pins FA3
  on both sides (SGLang `fa3` backend from `sgl_kernel`, built from `sgl-project/sgl-flash-attn`;
  trainer `flash_attention_3` through transformers, from the image's `flash_attn_3` wheel) with
  `num_splits = 1`. Its only contract is Qwen3 dense; the Megatron backend rejects
  `--true-on-policy-mode` at this commit (use FSDP). On ROCm, FSDP `--attn-implementation triton`
  bridges the trainer's attention forward to SGLang's `extend_attention_fwd_unified`, matching
  `--sglang-attention-backend triton` prefill, but no script wires it into the contract.

## Backward

The backward never has to match inference, only be deterministic. Still note:

- Primus-Turbo's FlyDSL backward is bitwise deterministic except its "a16" dQ path (atomics, dense
  square SBHD full-causal only), which `deterministic=True` disables. Its dQ split-K band count
  (`ceil(Skv / block_kv)`) and dK/dV `q_split` (up to 4; 1 with a window; dispatch-width based on the
  ragged path) depend on shape, so gradients for a sequence can differ in the last bits when it is
  packed differently. Make them per-model constants if packing-invariant gradients are required.
- MSLK's FlyDSL backward is deterministic on gfx950 under `torch.use_deterministic_algorithms`
  (per-tile dQ workspace, fixed-order reduction); it supports varlen block-diagonal masks but no
  attention sink and no paged/gappy backward.
- Shared-prefix (paged) training needs a backward that sums the prompt's dK/dV over N responses in a
  fixed order, never with atomics. None of the libraries above has it.

## Toolchain

Bitwise match across two stacks also requires the same Triton version and the same ROCm/LLVM, since
different compilers may contract FMAs or order instructions differently. Vendor the kernel source
into one library both stacks import, record the Triton version and a hash of the kernel source in
debug dumps, and refuse or warn on mismatches.

## Test matrix that catches the failures above

All comparisons bitwise (`torch.equal` on the int16 view), bf16 and fp16, with and without sliding
window, GQA ratios 1/3/4/8, head dims 64/128/256:

1. Paged (page 1/16/64, shuffled physical blocks, garbage in unused slots) vs contiguous.
2. Split mode vs fused mode, with short segments so several segments are exercised.
3. Launch configs (`BLOCK_M` x `num_warps`) allowed by the rule.
4. A sequence alone vs packed with other sequences of various lengths.
5. Rollout vs training: paged prefill of a prompt, then one decode step per token next to another
   sequence, vs one varlen forward over the whole sequence; every token must match.
6. Backward: two runs bitwise equal (determinism), plus tolerance vs an fp32 reference.
7. Masks with partial blocks off the diagonal (sliding window, document, interval masks): the same
   token decoded alone vs inside a training query block.
8. Chunked prefill with chunk sizes that are not multiples of the query block, including chunks
   short enough to trigger a library's decode kernel (e.g. < 128 queries for FlexAttention).
