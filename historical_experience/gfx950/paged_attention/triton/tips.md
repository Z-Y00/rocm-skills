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
