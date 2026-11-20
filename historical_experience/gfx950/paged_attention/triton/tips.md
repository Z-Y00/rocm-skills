# Tips — batch-invariant paged GQA attention (unified attention), Triton, gfx950

Kernel: OneTensor `onetensor/kernels/attention_fwd.py`, derived from aiter's
`kernel_unified_attention_2d/3d` + `reduce_segments`, with fixed absolute KV segments and a fused
mode (2D kernel with a segment loop). Shapes below: Hq=32, Hkv=8, D=128, bf16, page 16, MI355X,
Triton 3.8. Baseline "aiter" = `aiter.ops.triton.attention.unified_attention.unified_attention` with
its own backend choice. Numerical rules: [`knowledge/numerics/batch-invariance.md`](../../../../skills/kernel-optimize/knowledge/numerics/batch-invariance.md).

### 2026-11-10 — first-pass defaults vs aiter
- Context: decode (q_len 1), prefill (paged, long context), training forward (varlen) with first-pass
  launch settings: TILE 64, segment 2048, decode BLOCK_M 16 / 2 warps, prefill BLOCK_M 64 / 4 warps / 1 stage.
- Signal: decode 1x4k 52 us vs aiter 47; 8x32k 258 vs 186; 32x32k 1371 vs 673. Prefill 1x2048 on 16k
  context 1594 vs 638. Training fwd 8x4096 3.06 ms vs Turbo (CK/asm fmha v3) 1.11 ms; fwd+bwd 11.6 vs 5.3 ms.
- Takeaway: correctness first, but the launch settings left 2x on the table; the training gap vs
  hand-written asm is expected for Triton.
- Applicability: baseline for later rounds.

### 2026-11-10 — prefill launch sweep (does not change bits)
- Context: sweep BLOCK_M {64,128,256} x warps {2,4,8,16} (rows-only rule) x stages {1,2} x waves {1,2}
  for each global reduce order (TILE 32/64/128, segment 1024/2048/4096).
- Signal: TILE 64 best overall. 4x(512 q on 4k ctx): 540 -> 191 us (aiter 180) at BLOCK_M 256, 8
  warps, 2 stages. 1x(2048 q on 16k): 1602 -> 711 us (aiter 638) at the same config. 8x(2048 on 2k):
  879 -> 479 us at BLOCK_M 64, 4 warps, 2 stages.
- Takeaway: for prefill, larger BLOCK_M (128-256) with 8 warps and num_stages=2 is the main lever
  (~2.2x); TILE 128 is not better than 64 here.
- Applicability: gfx950 Triton flash-attention-style forward with D=128.

### 2026-11-10 — decode launch sweep (does not change bits)
- Context: BLOCK_M 16, warps {1,2,4,8} x stages {1,2,3} x waves {1,2,4}, split and fused mode.
- Signal: 64x4k 258 -> 232 us (aiter 180); 32x32k 1372 -> 794 us (aiter 673) with split mode,
  4 warps, 3 stages; small batches gain 5-10%. Split mode wins when q_blocks x kv_heads < ~2x CUs and
  the context spans several segments; fused wins for large batches. Segment 1024 helps small-batch
  long-context slightly; 4096 hurts small batches (too few programs).
- Takeaway: launch tuning alone leaves decode at ~60% of HBM bandwidth vs aiter's ~75-80%; the
  remaining gap is in the load path, not the schedule.
- Applicability: gfx950 Triton paged decode.

### 2026-11-10 — decode load path (does not change bits) — ACCEPTED
- Context: after launch tuning, decode sat at ~60% of HBM bandwidth vs aiter's ~75-80%.
- Change: (1) 32-bit K/V offsets when the cache has < 2^31 elements (int64 `.to()` on the block
  index blocks buffer loads on AMD); (2) interior KV tiles (all keys < seq_len) load without a key
  mask, only the last tile is masked; (3) `.cg` cache modifier on K/V for decode (q_len 1).
- Signal: with retuned launches, 64x4k 232 -> 175 us (aiter 180); 256x2k 414 -> 345 (aiter 392);
  8x32k 237 -> 179 (aiter 186); 32x32k 794 -> 688 (aiter 673); 1x4k 40 -> 47 at segment 2048 but
  44 at segment 1024. All bitwise tests still pass.
- Takeaway: for memory-bound paged decode the load path (buffer loads, no per-key masks on interior
  tiles) is worth ~25%; schedule tuning alone was ~10%.
- Applicability: any gfx950 Triton kernel streaming a paged cache with a per-key block-table gather.

### 2026-11-10 — split-KV segment size and auto-mode
- Signal: segment 1024 vs 2048 (TILE 64): 1x32k 53 -> 40 us, 8x4k 50 -> 44, 32x32k 688 -> 709,
  prefill ~3% slower. With BLOCK_M 128-256 for prefill, the "few programs -> split" heuristic wrongly
  picked split for prefill (1x2048 on 16k: 3012 us split vs 879 fused).
- Takeaway: decode-leaning default is segment 1024; only choose split-KV for decode-length queries
  (q_len <= 16); for split with many programs (> 512) use fewer warps / 1 stage (32x32k: 887 -> 738 us).
- Applicability: choosing split-KV vs single pass in batch-invariant attention.

### 2026-11-18 — decode stays bandwidth-bound at any batch size
- Context: whether larger decode batches make attention compute-bound.
- Signal: batch 256 x 2k context decode ran 347 us for ~2.15 GB of K/V (256 x 2048 tokens x 8 KV
  heads x 128 dims x 2 x 2 bytes), ~6.2 TB/s, close to HBM peak, with 25-50 % MFMA row use.
- Takeaway: each sequence has its own KV, so batch adds programs, not reuse per KV tile; rows per KV
  tile stay GQA group x query tokens. Head packing (read each KV tile once per KV head) and enough
  programs matter far more than MFMA row utilization. The exception is a shared prompt (GRPO), where
  N responses reuse the prompt's KV.
- Applicability: choosing decode tiling and MFMA shape on gfx950.

### 2026-11-10 — launch layout changes bits
- Context: bitwise comparison across (BLOCK_M, num_warps).
- Signal: warps tiling both rows and columns of the score tile, e.g. (32, 4), (32, 8), (64, 8),
  gave different output bits; rows-only or columns-only launches all agreed; LSE always agreed.
- Takeaway: enforce `num_warps <= BLOCK_M / 16` when `BLOCK_M > 16` if decode and training must match.
- Applicability: any Triton attention forward that must be batch-invariant on gfx950.

### 2026-11-19 — visibility intervals, sinks, pruned backward
- Context: per-token visibility [lo, hi] in global key indices (kernel subtracts the sequence's first
  key), per-head sinks folded after all keys, Hq=32, Hkv=8, D=128, bf16, Triton 3.9.
- Signal: decode with a causal mask given as intervals: +0-5% vs the built-in causal path, once the
  paged key starts (exclusive cumsum of seq_lens) are cached on the mask. Computing them per call
  added two small launches, +14% at batch 1 x 4k (57 -> 65 us). Training fwd with intervals +25%
  vs built-in causal (8x4096: 2.57 -> 3.21 ms). Pruning the backward (per KV tile first/last query
  row for dK/dV, per query block key range for dQ) took fwd+bwd from 2.3x to 1.2x of built-in.
- Rejected: splitting the tile loop to skip the interval mask on tiles every row fully sees (exact
  identity, same bits) made the training forward about 2x slower (3.21 -> 4.93 ms), likely from
  four inlined copies of the tile body. One masked loop plus a masked last tile is faster.
- Takeaway: per-step mask metadata belongs on the mask object (one computation per step, reused by
  all layers); do not add loop variants for mask elision in Triton attention without measuring.
- Applicability: masked/varlen Triton attention on gfx950 that must stay batch invariant.

### 2026-11-19 — FlyDSL dual-wave forward (Turbo gfx950): what breaks invariance, and the fix
- Context: Primus-Turbo's FlyDSL dual-wave software-pipelined forward (flash_attn_fwd_stock.py),
  Hq=32/64, Hkv=8, D=128/64, bf16, MI355X, FlyDSL 0.3.4.1. Probed bitwise: alone vs packed ragged,
  block_m 64 vs 128, and chunked prefill (512/37 tokens) and single-row decode vs the training forward.
- Signal: batch composition and launch shape were already invariant. With random inputs every
  check passed too, because the lazy rescale (reference = row max + 96) never fired. With key norms
  growing along the sequence (so it fires), D=128 chunked prefill / decode differed from training
  by 100-2000 elements per chunk; with a 128-key window it differed even on random data (leading
  masked tiles). Cause: the rebase branch is wave-uniform and inside it every row with a tile max
  within 96 of its reference rebased, so a row's bits depended on which rows shared its wave.
- Fix (accepted): keep the wave-uniform branch but select per row: rows whose own tile max did not
  outrun their own reference take shift 0 / rescale 1 / reference kept (exact identities). All
  chunk/decode diffs went to 0 for D=64/128 with and without window; forward time -1.8 to +2.4%.
  Also turn off the D<=64 Cauchy-Schwarz score bound: it is computed over the whole packed K tensor
  and floors masked prologue rows, so it can make a row depend on other sequences.
- Accuracy note: with large logits (hundreds) the FlyDSL forward's max error vs fp32 was 0.15-1.2
  while the Triton kernel stayed at 0.015 (both 0.009 on normal-scale data). FlyDSL pre-scales Q by
  log2(e)/sqrt(D) and rounds it to bf16 before the QK MFMA, which perturbs large logits by units.
  It does not affect train/infer match (both sides round the same) but is worth knowing.
- Applicability: any lazily rescaled online softmax with wave- or block-uniform triggers.

### 2026-11-19 — FlyDSL RL forward: split-KV with in-kernel segment merge
- Context: bitwise-invariant split-KV decode needs the training forward to reduce the same fixed
  absolute KV segments and merge them in the same order. Implemented in the FlyDSL dual-wave fork as
  an outer (reference, row sum, accumulator) state carried through the pipelined loop, merged at
  segment boundaries; decode runs one program per segment plus a merge launch of the same kernel.
- Signal: correct on the first try once (a) the segment's first tile is scored without the previous
  reference shift (as a fresh program's prologue does) and (b) the merge uses plain IEEE ops (no
  fast-math), so two kernels cannot fuse it differently. Cost is register pressure, not merges:
  D=128 training fwd +68% at 2 waves/SIMD + 128-row blocks, +32% at 1 wave/SIMD + 64-row blocks;
  D=64 +11%; segment length (16 vs 32 tiles) changed nothing. Packed decode with segments: +100% at
  128/2, ~+5% at 64/1. Long prefill +53%.
- Gain: Llama-3-8B decode 1x32k 1010 -> 125 us, 1x4k 133 -> 60 us (aiter 67).
- Also: per-call host work (cumsum, interval and group-bound kernels) dominated short decode kernels
  (GPT-OSS decode 71 -> 14 us after caching it per step on tensor identity + version).
- Takeaway: for invariant split-KV, budget the outer accumulator's registers up front; try 1 wave/SIMD
  and smaller blocks before restructuring; keep segment merges in exact arithmetic.

### 2026-11-19 — FlyDSL: read register use from the compiler, not from timings
- How: `FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_DIR=<dir>` (both needed), then
  `.vgpr_count`, `.agpr_count`, `.vgpr_spill_count`, `.private_segment_fixed_size`,
  `.group_segment_fixed_size` in `<dir>/**/15_final_isa.s`. Compile-only (`module.compile(...)`) is
  enough; use a fresh `FLYDSL_RUNTIME_CACHE_DIR` so it actually compiles. rocprofv3 kernel trace
  aborted in rocm/primus:v26.9 with PyTorch loaded ("api registration ... outside of valid
  rocprofiler configuration period").
- Signal (RL fork, D=128): stock 230 VGPR; fp32 score scaling 250 (no spill, same 2 waves);
  segments at 2 waves: 256 + 46 spilled (+68%); at 1 wave / 64-row blocks: 300 VGPR + 44 AGPR, no
  spill (+32%, from occupancy). D=64: 150 / 170 / 200, never spills. LDS: 68 KB (D=128), 34 KB (D=64).
- Correction: an earlier tip assumed the D=128 kernel used 34 KB LDS; it is 68 KB.
