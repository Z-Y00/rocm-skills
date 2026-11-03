# Attention — Math Optimizations

Algebraic rewrites that remove work, memory traffic or a kernel launch from attention
forward/backward without changing the result (beyond rounding order). Each entry gives the
identity, the condition it needs, and what it bought in the gfx1250 sparse-MLA campaign
([AMD-AGI/Primus-Turbo#560](https://github.com/AMD-AGI/Primus-Turbo/pull/560)). The ideas are
backend- and architecture-independent.

Notation: per query row, `S = scale · Q Kᵀ`, `P = exp(S − lse)`, `O = P V`, `dP = dO Vᵀ`,
`delta = rowsum(O ∘ dO)`, `dS = P ∘ (dP − delta)`. In MLA the same latent row serves as key and
value (`K = [kv_nope, rope]`, `V = kv_nope`).

---

## 1. Backward: `delta = rowsum(P ∘ dP)` Instead of `rowsum(O ∘ dO)`

- **Identity.** `rowsum(O ∘ dO) = Σ_d Σ_j P_j V_jd dO_d = Σ_j P_j dP_j`. Exact in real arithmetic.
- **Attention sink.** A sink adds probability mass to the softmax denominator but no value row,
  so it contributes nothing to `O` and nothing to `Σ P dP`; the identity still holds with `P`
  normalized by the sink-inclusive `lse`.
- **Condition.** Every key of a query row must be visited before any `dS` is needed, i.e. the
  whole row is processed in one S/dP pass (window / short top-k kernels). Not for tiled long rows
  where the first tile's `dS` is needed before the last tile's `dP`.
- **Gain.** No O read, no delta pre-pass, no O prefetch registers: window backward -14..-16%.
  Implement the row sum as a 16-lane DPP xor butterfly plus one LDS exchange between waves that
  split the slots, in a fixed order so all waves hold an identical delta.
- **Numerics.** Better, not worse: a reference that forms delta from bf16 `O` carries O's
  rounding. dsink against fp32 went from 55 to 127 dB.

## 2. Forward: Lazy Rescale of the Online Softmax

- **Identity.** With running max `m`, `O = Σ exp(s − m) V / Σ exp(s − m)` holds for any `m`; the
  max only bounds the exponent. Keep a stale `m` while every new tile max exceeds it by ≤ τ.
- **Choice of τ.** With exp2 and fp32 accumulators, τ = 8 (factor 2⁸ = 256 headroom) is safe.
  The test is wave-uniform (`ballot`), and the O rescale sits under an `scf.if`.
- **Gain.** -3% forward with O^T at 512 VGPRs/wave, once other costs had been removed. An earlier
  thresholded version was neutral and "never rescale" was only 1.3% faster: with many heads per
  wave the softmax work is amortized over more WMMAs, so measure the bound first.
- **Numerics.** Changes rounding order. Judge against an fp32 reference and the bf16-output
  ceiling, not against a bf16 oracle that rounds P relative to the same eager max (SNR vs the
  oracle fell 57-83 → 52-62 dB; vs fp32 it moved 53.8 → 53.4 against a 55.6 dB ceiling).

## 3. exp2 With Constants Folded

- `exp(scale · s − lse) = exp2(s · (scale · log2 e) − lse · log2 e)`: fold `scale · log2 e` into one
  compile-time constant applied to the raw `Q Kᵀ` accumulator, and pre-multiply `lse` by `log2 e`
  once per row. One FMA plus `v_exp_f32` per element.
- **Empty rows.** A row with no visible key and no sink has `lse = −∞`. Map it to `+∞` before
  the subtraction so `P = exp2(s − ∞) = 0` instead of NaN, with no per-element branch.

## 4. Backward: Fold the Softmax Scale Into dS

- Compute `S` without the scale (scale is inside the exp2 constant), and apply it once in
  `dS = P (dP − delta) · scale`. Then `dQ = dS K` and `dK = dSᵀ Q` need no extra pass.

## 5. MLA: One dKV Accumulator for the Shared Latent

- When the latent row is both key and value, `dKV_nope = dSᵀ Q_nope + Pᵀ dO` is a single sum:
  accumulate `C = Qᵀ dS + dOᵀ P` into the same fp32 tile (two WMMA chains, one accumulator).
  (In the PR #560 contract the rope columns of dKV are zero.)
- One accumulator tile and one output stream per kv row instead of separate dK and dV.

## 6. Closed-Form Band Masks and Reductions (Sliding Window, Compressed Pools)

- **Window.** For a 128-key sliding window, slot `s` of token `t` is kv row `t − 127 + s`. Validity
  is `0 ≤ t − 127 + s` (plus causal by construction); no index tensor, no mask tensor.
- **Compressed pool.** With compression ratio `cr`, pool row `p` covers tokens `[cr·p, cr·(p+1))`
  and is visible to token `t` iff `t ≥ cr·(p + 1) − 1`. One comparison per lane.
- **Window as a shifted read.** Tokens of a block share most of their window. Gather the block's
  window once into LDS (row `r` = kv row `t0 − 127 + r`) and give each token a per-lane row shift
  in the fragment address; out-of-range lanes point at a zero row. Forward -11..-13% (the gather
  share), and no per-token index loads.
- **Reduce without an inverse index.** With per-block dKV partials, kv row `j` is window row
  `j + 127 − b·block_t` of every block `b` whose window holds it (a fixed, small set). A closed-form
  per-row gather replaces sort + CSR inverse + gather.

## 7. Accumulate dKV Across a Token Block On Chip

- For banded key sets, neighbouring tokens hit the same kv rows. A workgroup that owns a block of
  consecutive tokens (e.g. 16) and fewer heads can keep window-wide fp32 dKV tiles in registers
  across the block and emit one partial per block, instead of a `[T, topk, d]` per-token
  intermediate. Backward -40..-53%, and more accurate (fp32 partials instead of a bf16 intermediate).
- **Head split is not a register lever.** The dKV accumulators are summed over heads, so they
  cover `rows × d` whatever the head partition: per lane `rows × d / (waves × 32)`.
- Rows shared by every token (a compressed pool) get their own per-workgroup partials, summed by
  column-slice workgroups in the same reduce launch (no extra torch ops).

## 8. Deterministic Scatter-to-Gather (No Atomics)

- Invert the top-k map `[T, topk] → kv rows` with one **stable sort** and a **searchsorted** row
  pointer (`torch.searchsorted(sorted_keys, arange(n+1))`): identical CSR to a histogram, no host sync.
- **Narrow keys.** Cast sort keys to int16 when the range allows: radix sort runs half the passes.
- **Composite keys.** Several CSR inverses over slices of one map come from one sort of
  `slice · num_rows + row`; each slice's row pointers are a view of one searchsorted.
- **Skewed rows.** Split only heavy rows into fixed blocks of the sorted list, write per-block
  partials and add them in block order; keep light rows one program per row in the same launch.
- **Skip empty rows** in a per-block row loop (jump to the next non-empty row): the loop is
  O(row span), not O(entries), once inputs are chunked.
- Any reduction (partials, dsink, pool rows) is summed in a fixed order so the result is
  bit-identical run to run.

## 9. Attention-Sink Gradient From delta and lse

- With a per-head sink logit, `P_sink[t] = exp(sink − lse[t])` and
  `dsink = −Σ_t exp(sink − lse[t]) · delta[t]`. It needs only the saved `lse` and the `delta` the
  backward already forms: a tiny per-head reduction, no extra pass over Q/K/V.

## 10. Work Proportional to Valid Slots

- Real indexers (`topk` with masked picks set to −1) leave invalid slots trailing; at 4K tokens
  ~45% of slots were empty. Bound the tile loop by the last valid slot: lane `c` tests tile `c`,
  then `ballot` + `ctlz` gives a wave-uniform count with no barrier and no extra launch. Keep the
  trip count static and skip with a wave-uniform `if` near the register cap (a dynamic bound
  spilled). Backward -35% on real indices.
- Benchmark with indices built the way the producer builds them (causal mask, sort order, −1
  placement). Random fully valid indices hide this entirely.

## 11. Numerics Checklist for Any Rewrite

- Compare against an fp32 (or fp64) reference, and know the bf16-output ceiling (the SNR of the
  reference itself rounded to bf16). A drop in agreement with another bf16 kernel that shares
  the removed rounding error is not a regression.
- Rewrites that change summation order (lazy max, block partials, P∘dP delta) move SNR vs a bf16
  oracle by several dB while moving SNR vs fp32 by < 0.5 dB (or improving it).
- Check gradients of every output (dq, dkv, dsink): a delta change shows up most in dsink.
