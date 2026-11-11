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
