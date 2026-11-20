# Tips — batch-invariant paged GQA attention (RL attention), FlyDSL, gfx950

Kernel: Primus-Turbo's RL fork of the FlyDSL dual-wave software-pipelined gfx950 forward
(`primus_turbo/flydsl/attention/rl_attention_fwd.py`), MI355X, bf16, FlyDSL 0.3.4.1. Triton-side tips
for the same op: [`../triton/tips.md`](../triton/tips.md). Numerical rules:
[`knowledge/numerics/batch-invariance.md`](../../../../skills/kernel-optimize/knowledge/numerics/batch-invariance.md).

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

### 2026-11-20 — correction: no per-step metadata cache on tensor identity
- The 2026-11-19 entry above credits a per-step cache on tensor identity + version for GPT-OSS decode
  71 -> 14 us. That cache is keyed on activation tensors (forbidden by the iteration rules, W3
  pattern); it was replaced by one fused Triton launch (`rl_metadata_kernel`) writing key starts,
  intervals and group bounds. Honest GPT-OSS decode: 44-50 us, bound by that launch's host cost.
- Takeaway: measure decode with the metadata recomputed per call; a cache hit hides host cost.

### 2026-11-20 — split decode: occupancy, not program count; split threshold
- Context: segmented D=128 build at 1 wave/SIMD (300 VGPR + 44 AGPR), packed GQA decode, 1024-key
  segments, MI355X (256 CUs).
- Signal: (1) letting a split program walk K consecutive segments (8x32k: 2048 -> 512 programs) did
  not help (386 -> 387 us; tuned per shape, K = 1 best or tied everywhere). Dropping the outer state
  from split programs (fewer registers, same 1 wave) was +1% worse. (2) MODE=auto split only below
  1 program per CU, so 32x32k (exactly 256 programs) ran fused; splitting below 2 per CU took it
  1190 -> 1123 us, while 64x4k (512 programs) stays fused, where split was not faster. (3) Windowed
  decode (gpt-oss, window 128) under auto split over the whole context: fused 64 -> 48 us.
- Takeaway: split decode here is latency-bound per workgroup at ~2.8 TB/s (aiter ~5.8); program
  count and per-program overhead are not the limit, waves in flight are. Use < 2 x CUs as the split
  threshold, and count the segments a window actually reads.
- Noise: GPT-OSS decode rows (~45-55 us, host-bound) moved +-15% between nodes for unchanged code;
  re-run before a decision that hinges on them.
- Applicability: FlyDSL / any gfx950 split-KV decode with a heavy per-row state.
