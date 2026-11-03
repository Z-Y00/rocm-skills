# Tips — sparse MLA fwd/bwd, FlyDSL, gfx1250

### 2026-10-04 12:14 round-2 — ACCEPTED
- Context: CSR inverse of the top-k map for the deterministic dkv gather.
- Signal: torch profiler showed a DtoH memcpy + histogram per backward; `torch.bincount` on GPU syncs to find max.
- Takeaway: build `inv_ptr` with `torch.searchsorted(sorted_keys, arange(n+1), out_int32=True)` from `torch.sort(stable=True)` values; identical CSR, no host sync. +3.7% step on DSV4 shapes.
- Applicability: any CSR / segment build on GPU where bincount/unique would otherwise be used inside the training step.

### 2026-10-04 12:17 round-3 — ACCEPTED
- Context: stable torch.sort of T*topk kv indices (CSR build).
- Signal: rocprim radix sort 0.133 ms for 524K int32 keys.
- Takeaway: cast keys to int16 when the key range allows; the radix sort runs half the passes (0.049 ms) and the stable permutation is identical.
- Applicability: any torch.sort/argsort on small-range integer keys on ROCm.

### 2026-10-04 12:25 round-4 — ACCEPTED
- Context: sparse-MLA forward (gather KV rows to LDS, 16 heads/wave, transposed S^T / O^T).
- Signal: kernel-only sweep: 128-head WG (8 waves) bn64 2349 us vs bn32 1414 us at topk 1152; 64-head WG prefers bn64.
- Takeaway: the right KV tile depends on waves/WG — sweep block_k per heads_per_wg instead of "bigger tile when divisible".
- Applicability: FlyDSL gfx1250 attention-style kernels with gather prefetch in registers.

### 2026-10-04 12:31 round-5 — ACCEPTED
- Context: per-token attention backward with ~1 MB/token traffic.
- Signal: removing 128 KB/token of reads gave +3-6% on H128 shapes, ~0 on H64.
- Takeaway: measure the copy bandwidth first (with a tuned stream kernel, see the roofline entry below) — gfx1250 kernels with per-token staging are bandwidth-sensitive at short top-k; never re-read from HBM what is already staged in LDS. H64 (4 waves/CU) did not benefit: it is latency-bound, fix occupancy first.
- Applicability: per-token / per-row fused backward kernels on gfx1250.

### 2026-10-04 12:37 round-6 — ACCEPTED
- Context: per-token fused attention backward, one workgroup per token, LDS-limited to 1 WG per CU.
- Signal: H64 per-tile time ≈ H128 per-tile time at half the FLOPs; 580 VGPRs, 4 waves/CU.
- Takeaway: when LDS pins 1 WG/CU, keep 8 waves per WG regardless of head count — split the per-head-group work (slot sub-tiles, dQ d-range) across 2 waves; it also halves accumulator VGPRs. +10-12% on H64 bwd.
- Applicability: gfx1250 kernels where a WG's wave count follows a problem dim (heads) that can be small.

### 2026-10-04 12:46 round-7 — ACCEPTED
- Context: scattered-row gather (CSR) over a multi-GB bf16 intermediate on gfx1250.
- Signal: the same compiled gather ran 2.7x faster when the buffer it reads spans < ~2 GB; 2.3 GB chunks got no speedup. Copy bandwidth also drops once a working set outgrows the last-level cache; sweep the buffer size to find the knee.
- Takeaway: keep randomly-accessed buffers under ~1.5-2 GB per kernel launch (chunk the work); also size per-token-workgroup chunks as multiples of the CU count when LDS pins 1 WG/CU — a chunk size that was not a multiple of the CU count cost ~8% from wave quantization.
- Applicability: any gather/scatter-heavy kernel on gfx1250 with multi-GB working sets.

### 2026-10-04 12:51 round-8 — ACCEPTED
- Context: FlyDSL gfx1250 kernels with a 4-wave (128-thread) workgroup.
- Signal: same per-wave work compiled to 999 VGPRs at 4 waves/WG vs ~500 at 8 waves/WG.
- Takeaway: with small workgroups the backend may use the full 1024-VGPR file (occupancy 1). Pass `value_attrs={"rocdl.waves_per_eu": 2}` at launch to cap at 512 and fit 2 WGs/CU; pair it with a smaller tile if the kernel would spill heavily (bn64+wpe2 was 2x slower, bn32+wpe2 15-25% faster). wpe 3/4 spill badly.
- Applicability: any gfx1250 FlyDSL kernel whose VGPR count jumps above 512 with a 4-wave workgroup.

### 2026-10-04 13:20 round-9 — ACCEPTED
- Context: CSR segmented-sum (dKV gather) where a few rows hold most entries (compressed pool).
- Signal: one-program-per-row gather 4.6x slower on cr128 than cr0 at similar bytes.
- Takeaway: split only heavy rows (> SEG entries) into fixed blocks of the sorted entry list, write per-block partials, add them in block order in a fix-up kernel (deterministic). Keep light rows on a 1-program-per-row path in the same launch — a pure fixed-block design was ~2x slower on short uniform rows (longer dependent-load chain + serial row loop).
- Applicability: any deterministic scatter-to-gather (inverted index) reduction with skewed row lengths.

### 2026-10-04 14:15 round-11 — ACCEPTED
- Context: WMMA-bound FlyDSL kernels (attention fwd/bwd) on gfx1250.
- Signal: free compiler knob listed in the FlyDSL arch-gfx1250 notes.
- Takeaway: `flyc.compile[{"llvm_options": {"amdgpu-expert-scheduling-mode": True}}](launch, ...)` gave +3..7% on every shape (largest on compute-heavy ones) with identical VGPRs and bitwise-same-quality results; verify `hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2` in `21_final_isa.s`. `amdgpu-sched-strategy=coexec` was neutral. Try it first on any new gfx1250 WMMA kernel.
- Applicability: any gfx1250 (gfx12+) compute-bound kernel.

### 2026-10-04 15:30 round-13 — REJECTED
- Context: `rocdl.global_load_async_to_lds_b128` + `s_wait_asynccnt` in a large (500-VGPR) FlyDSL kernel compiled with `amdgpu-expert-scheduling-mode`.
- Signal: GPU memory access fault at address nil; same kernel correct without expert mode; a tiny async-copy kernel works in both modes.
- Takeaway: async global->LDS copies and expert scheduling mode do not mix in the current compiler (the address VGPRs are likely clobbered before the async read; SCHED_MODE 2 leaves those hazards to software). Without expert mode the async gather removed 64 staging VGPRs and the spills, but expert mode alone was worth more (+5%).
- Applicability: any gfx1250 kernel combining async/TDM LDS copies with expert scheduling — test both modes.

### 2026-10-04 16:30 round-14 — REJECTED (re-profile reference numbers)
- Context: roofline numbers for gfx1250 measured with FlyDSL microbenchmarks.
- Takeaway: measure the WMMA and LDS peaks yourself with timing microbenchmarks; `ds_load_tr16_b128` delivers roughly three quarters of `ds_load_b128`'s LDS bandwidth. rocprofv3 WMMA counters read 0 on this stack, so use timing probes. The hipBLASLt bf16 GEMM was untuned on this stack, not a usable reference. `amdgpu-sched-strategy` max-ilp / max-memory-clause / iterative-* were all worse than the default (iterative-minreg produced wrong results).
- Applicability: any gfx1250 kernel roofline estimate.

### 2026-10-04 14:45 round-15 — REJECTED
- Context: attention forward on gfx1250 with every wave re-reading the shared K/V LDS tile (2 waves/SIMD, ~500 VGPRs).
- Signal: halving per-wave K/V LDS reads by pairing waves (2 head groups per wave, d split, S partial exchange) made it 10-25% slower, 6-21% even with the extra barrier removed.
- Takeaway: on paper LDS bandwidth looks like the floor, but these kernels are issue-bound; adding per-wave VALU/TRANS work (duplicated softmax) costs more than the saved LDS reads.
- Applicability: any LDS-sharing restructure of WMMA attention kernels on gfx1250.

### 2026-10-04 15:40 round-16 — REJECTED (comparison with a hand-written assembly kernel + TDM gather)
- Context: sparse-MLA forward on gfx1250 vs aiter's hand-written assembly (OPUS) bf16 sparse-prefill kernel.
- Signal: OPUS was ~1.7x faster than our kernel at the time. It runs 4 waves x 16 heads per WG at 1 wave/SIMD near the VGPR and LDS limits, 64-key tiles, TDM gather (`tensor_load_to_lds` with SGPR row indices from a scalar buffer load). Porting only the TDM gather into our 2-waves/SIMD kernel was 4-17% slower; OPUS-like 1-wave layouts in our compiler-scheduled code were 1.5x slower (stall on nearly every LDS fragment load).
- Takeaway: the OPUS win is a hand-pipelined single-wave kernel (deep LDS->register fragment prefetch), not the loader. TDM pitfalls: pred-off gathers don't count toward tensorcnt and all-OOB gathers complete early, so a count-based ring must drain with `s_wait_tensorcnt 0` at the tail. TDM gather pads LDS rows via the descriptor (interval 256 dw, amount 4 dw -> 1040 B pitch); `-1` indices are OOB and land as zeros.
- Applicability: any gfx1250 gather-attention kernel considering TDM or a 1-wave/SIMD layout.

### 2026-10-04 20:50 round-17 — ACCEPTED (single-wave forward learned from OPUS asm)
- Context: FlyDSL WMMA attention forward at one wave per SIMD (`waves_per_eu=1`) with a TDM gather ring.
- Signal: the compiler put two LDS loads + `s_wait_dscnt` right before every WMMA (no lookahead) and regrouped interleaved chains when prefetch was deep; debug runs without LDS loads were 32% faster.
- Takeaway: (1) prefetch LDS operands in the source (a Python queue of fragments N steps ahead) and pin the order with `rocdl.sched_barrier(0)` every few steps — prefetch 8 / pin 4 was best; (2) software-pipeline QK(t+1) against PV(t) in one block; (3) spend the 1024-VGPR file on more heads per wave (32 = two 16-head groups) so each LDS fragment feeds 2 WMMAs — the single biggest win (pro cr4 fwd 1.33 -> 0.93 ms, 950 VGPRs, no spills); (4) gfx1250 has no LDS bank-conflict counters; padding sweeps moved < 2%; (5) 16-bit TDM gather indices (dword0 `0x80000001`, 16 rows/op, two indices per dword) halve the op count against the 3-ops/wave queue; deeper rings lost (TDM LDS writes compete with operand reads); (6) expert scheduling mode miscompiled one VALU/WMMA interleaving (non-deterministic wrong results, correct with expert off) — re-verify every new interleaving several times.
- Applicability: any gfx1250 1-wave/SIMD WMMA kernel reading operands from LDS.

### 2026-10-04 21:15 round-18 — REJECTED
- Context: one-wave/SIMD WMMA attention forward on gfx1250 (expert scheduling mode).
- Signal: FA3-style intra-wave pipelining (softmax + O rescale of tile t spliced between the independent QK(t+1) WMMAs) was verified in the ISA but made the forward 1-2% slower.
- Takeaway: a wave's own VALU/TRANS work does not hide under its WMMAs; softmax cost here is issue cost. Reduce instruction count (lazy rescale, fewer selects) rather than reorder.
- Applicability: any single-wave WMMA kernel considering intra-wave softmax/GEMM overlap on gfx1250.

### 2026-10-04 21:30 OPUS asm study (no round) — where its softmax cycles go
- Context: aiter's hand-written gfx1250 sparse-prefill forward (OPUS), steady-state loop.
- Signal: its softmax runs as a contiguous block with no WMMAs inside (no intra-wave softmax/WMMA overlap, consistent with round 18).
- Takeaway (candidates, not yet tried): (1) thresholded lazy rescale (FA4-style): `growth = scale*tile_max - m_run`; if `growth <= 8.0` on every lane (`v_cmp_ge` + `s_cmp_eq vcc, exec` + `s_cbranch_vccnz`) keep the stale max and skip alpha + the 128 `v_pk_mul` O rescale — safe since exp2(s - m) <= 2^8; (2) lean exp path: one `v_fma_f32 x, s_scale, -m` per score alternated with `v_exp_f32` (VALU/TRANS co-issue), vs our pk_mul + cndmask + sub + back-to-back exps + v_nop; (3) split barrier: `s_barrier_signal`, the last PV WMMAs from already-loaded registers, then `s_barrier_wait`. OPUS also sets SCHED_MODE 2 (same as our expert mode).
- Applicability: next steps for the gfx1250 sparse-MLA forward; (1)-(3) apply to any single-wave WMMA attention forward.

### 2026-10-11 18:20 round-19 — REJECTED
- Context: one-wave/SIMD attention forward with 32 heads per wave (two 16-head groups).
- Signal: OPUS-style thresholded lazy rescale (ballot + wave-uniform `scf.if`, threshold 8 log2) was neutral; never rescaling at all was only 1.3% faster; forcing the branch every tile cost 2%.
- Takeaway: measure the bound first — once a wave owns more heads, per-tile softmax/rescale work is amortized over more WMMAs and OPUS's softmax tricks shrink to ~1%. FlyDSL recipe for wave-uniform branches with results: `llvm.call_intrinsic(i32, "llvm.amdgcn.ballot.i32", [i1])` then `scf.IfOp(cond, types, has_else=True)` with `scf.YieldOp` in both regions.
- Applicability: any gfx1250 attention kernel considering lazy rescale.

### 2026-10-11 18:16 round-20 — ACCEPTED (backward LDS prefetch)
- Context: fused per-token attention backward on gfx1250, 8 waves / WG (2 waves/SIMD, 512-VGPR cap).
- Signal: ISA had ~1 `s_wait_dscnt` per WMMA; source-level prefetch 8 + `sched_barrier(0)` every 4 steps gave -4..-5% bwd kernel time on H64 (363 VGPRs) but spilled on H128 (452 VGPRs -> 512 + 9..41 spills, up to 2.5x slower).
- Takeaway: the round-17 prefetch recipe needs ~64+ free VGPRs; gate it on the register budget per config. Pins without prefetch only constrain the scheduler (-11%), and prefetch without pins is undone by the backend (no gain) — they only work together. Split a big step list into `(load, mma)` steps where shared operands (A fragment for several B, B for several A) are loaded by the first step of their group and kept in a Python dict.
- Applicability: any gfx1250 FlyDSL WMMA kernel; check `.vgpr_count`/`.vgpr_spill_count` per config before enabling deep prefetch.

### 2026-10-11 18:22 round-21 — ACCEPTED (operand reuse must be explicit under sched_barrier)
- Context: FlyDSL WMMA loop where the same LDS fragment is read for several WMMAs (no LDS store in between), compiled with `sched_barrier(0)` pins.
- Signal: `ds_load_tr16` count rose 66 -> 98 after adding pins; unpinned builds had the same source but 66.
- Takeaway: the backend CSEs repeated LDS loads only within a scheduling region; `sched_barrier` splits regions, so reuse across pins must be written in the source (load once, keep in a Python variable). Count `ds_load_*` in the ISA before/after adding pins. Making reuse explicit where the compiler already merged it only adds live ranges (H128: +5%).
- Applicability: any pinned / hand-pipelined FlyDSL kernel.

### 2026-10-11 18:36 round-22 — ACCEPTED (backward on the single-wave, 32-heads-per-wave layout)
- Context: fused per-token sparse attention backward (S, dP, dS, dQ^T, per-slot dKV C^T), LDS ~317 KB on H128.
- Signal: 8 waves (2/SIMD) capped VGPRs at 512 with a 256-VGPR dQ^T accumulator — no room for LDS prefetch; ~1 LDS fragment per WMMA.
- Takeaway: the forward's recipe transfers: 4 waves x 32 heads (`waves_per_eu=1`, 914 VGPRs) so K (S/dP) and K^T (dQ^T) fragments feed two head groups, TDM gather for kv tiles (a register loader would need 64-128 staging VGPRs at 4 waves), prefetch 8 with pins. With a single kv buffer, issue the next gather after the last phase that reads K (barrier) and hide it under a phase that does not (C^T). -10..-15% kernel time on every shape, +9.3% step.
- Applicability: any per-row/per-token fused attention backward on gfx1250 whose accumulator footprint blocks prefetch at 2 waves/SIMD.

### 2026-10-11 19:27 round-23 — ACCEPTED (persistent forward)
- Context: per-token sparse attention forward, 1 wave/SIMD; pro 1 WG/CU, flash 2 WGs/CU.
- Signal: dropping the q load sped cr0/cr128 by 15-23%, i.e. the per-token prologue was exposed.
- Takeaway: persistent WGs (grid = CUs x WGs/CU) carrying the next token's q fragments and next index octets through an outer `range(..., init=...)`, with tile slots/gathers on a global tile index, hide the prologue at unchanged occupancy. Pays only where nothing else shares the CU (1 WG/CU: -8..-10%); with 2 WGs/CU the sibling already hid it. The outer loop costs ~9% at one token per WG, so always A/B against the non-loop kernel, not the loop at NPROG=T.
- Applicability: short-top-k / short-sequence attention kernels at 1 WG per CU.

### 2026-10-11 19:36 round-24 — ACCEPTED (branch-free skip of a speculative prefetch)
- Context: persistent loop that prefetches the next item's operands into registers.
- Takeaway: to cancel a speculative buffer_load on the last iteration without an `scf.if`, build its descriptor with `num_records = live ? size : 0`; OOB-checked loads return zeros with no memory traffic. Before blaming "loop overhead", diff the ISA of the loop at NPROG=T against the persistent build — identical code means the gap is wasted speculative work.
- Applicability: any FlyDSL persistent kernel with cross-iteration register prefetch.

### 2026-10-11 19:56 round-25 — REJECTED (persistent backward + L2 prefetch)
- Context: sparse-MLA backward, 1 WG/CU, LDS and VGPRs full, short top-k (4-6 tiles).
- Signal: linear fit over top-k lengths implied a large per-token fixed cost.
- Takeaway: don't infer fixed cost from a linear fit across compression ratios — per-tile cost changes with the kv pool's cache locality. Count bytes first: the kernel was at ~70% of copy bandwidth, so persistence was flat and `global_prefetch` of the next token (~400 KB/CU) made it 5-13% slower (extra traffic / L2 thrash).
- Applicability: any "hide the prologue" idea on a kernel that streams per-item inputs once.

### 2026-10-11 20:04 round-26 — ACCEPTED (compile-time split by tiles per token)
- Context: persistent forward (r23-24) applied to all compression ratios.
- Takeaway: gate structural changes that target the prologue on tiles-per-token at build time (here NT <= 8) and keep the straight-line kernel for long top-k; wrap the per-token body in a builder-level function so both paths share it. Check every shape against the pre-change kernel, not the geomean — the r23 geomean gain hid a 2-4% flash cr4 regression.
- Applicability: any kernel specialized per top-k / sequence length where a change helps short and hurts long rows.

### 2026-10-11 20:23 round-27 — REJECTED (custom CSR inverse of the top-k map)
- Context: backward dKV needs the inverse of the [T, topk] map (torch stable sort, 78-87 us at T=4096).
- Takeaway: a per-block counting sort with a [blocks x kv rows] table wins 2x at T=4096 but the table grows as T x num_kv (68 MB at T=16K) and loses there; LDS-atomic placement is deterministic only if a kv row occurs once per top-k row. In FlyDSL, `gpu.barrier()` drains global stores (`s_wait_storecnt 0`) — use `s_wait_dscnt 0` + `s_barrier_signal/wait` when only LDS ordering is needed.
- Applicability: any FlyDSL loop that interleaves global stores with workgroup barriers; any CSR/inverse-index build.

### 2026-10-11 20:36 round-28 — ACCEPTED (gather skips empty CSR rows)
- Context: dKV gather over a CSR inverse built per token chunk; heavy rows split into fixed blocks.
- Signal: stage times summed well below the full backward only when the intermediate was chunked (cr128 at 16K); per-chunk gather time fell chunk by chunk.
- Takeaway: a per-block loop `for r in range(row(first), row(last) + 1)` is O(row span), not O(entries): chunking leaves long runs of empty rows (between window and pool rows), walked serially. Jump to the next non-empty row (r + 1, else binary search of inv_ptr). Check chunked vs unchunked stage sums when a long-sequence shape regresses.
- Applicability: any CSR-based gather / segmented reduction over chunked inputs.

### 2026-10-11 21:05 round-29 — ACCEPTED (banded top-k: accumulate dKV across tokens on chip)
- Context: sparse-MLA backward where the top-k is a sliding window (cr0); per-token dKV went to a bf16 [T, topk, 512] intermediate reduced by sort + gather.
- Takeaway: for a banded top-k, give a workgroup a block of consecutive tokens (16) and fewer heads (64) so the window-wide fp32 dKV tiles (9 x 16 rows x 128 d per wave = 576 VGPRs) fit; process all top-k slots in one tile so dQ is written per d tile (no dQ accumulators); gather the block's kv window once (token i reads rows i..i+127) and read dS / P rows shifted by i per lane (out-of-range lanes point at a zero LDS row). Emit one fp32 partial per block; a closed-form per-row reduction replaces the CSR inverse. -40% / -53% backward, and more accurate than the bf16 intermediate.
- Measurement: SNR vs the old path / Triton fell (51.8 vs 67.7 dB) only because those two share the same bf16 rounding; check a changed reduction order against an fp64 reference before calling it a regression.
- Applicability: any attention backward whose key set is a band (SWA, local attention) or otherwise shared by neighbouring queries.

### 2026-10-11 21:36 round-30 — ACCEPTED (shared key rows from global; soffset against hoisted offsets)
- Context: extending the token-block SWA backward with cr128's pool rows (shared by all tokens) at ~980 VGPRs and a full LDS.
- Takeaway: key rows every workgroup shares can feed WMMA fragments straight from global memory (L2-resident; transpose a tiny K^T copy for the transposed operand) when LDS is full; the dead P tile of the window phase holds the pool dS / P. Near the VGPR limit, LICM hoists one VGPR per distinct `lane_off + uniform + const` offset and spills them in the prologue (all scratch stores before the loop): pass the wave-uniform part as the buffer soffset so the constant folds into the instruction offset (78 -> 46 spills), then trade prefetch depth / held fragments for the rest.
- Gate token-block kernels on workgroup count: each walks its tokens serially, so below ~CUs/2 workgroups the per-token kernel wins.
- Applicability: kernels with large loop-carried accumulators near the VGPR cap; attention backward with a shared (global) key subset.

### 2026-10-11 21:54 round-31 — ACCEPTED (ablate the forward's memory streams before restructuring)
- Context: short-top-k (128-192) sparse attention forward, 1 wave/SIMD, q / o streamed once per token.
- Signal: ablations with zero-size buffer descriptors (instructions still issued, no traffic): o store off -> 2x faster, q load off -> -11..-15%, kv gather off -> -16%. The o write, not the gather, dominates; the window-in-LDS forward (per-lane row offsets, closed-form mask) recovered the gather share (-11..-13%) only.
- Takeaway: ablate each memory stream with `num_records = 0` before choosing what to restructure. Re-using one output buffer across timing iterations skews store-heavy kernels by 6-10% (time with fresh allocations, as training does). Per-lane-address fragment loads (`ds_load_b128`, `ds_load_tr16`) make a sliding window a free row shift.
- Applicability: any attention forward with short rows; any store-heavy kernel benchmark.

### 2026-10-11 22:24 round-32 — ACCEPTED (unrolled tiles vs a loop at the VGPR cap)
- Context: 1-wave/SIMD attention forward at ~1000 VGPRs (q 256 + O 512), extra kv tiles of a different kind (shared pool) after the window tiles.
- Signal: unrolling the extra tiles spilled 38..123 VGPRs and the count swung wildly with unrelated edits (prefetch 8/6/4, pins, tile order); one `scf.for` over all tiles (base / mask by wave-uniform select) gave 0 spills on every shape. Conversely, with no extra tiles the 1-iteration loop cost 3% vs unrolled.
- Takeaway: at the register cap, put heterogeneous tiles into one loop with uniform selects rather than unrolling; keep unrolling only where it already fits. Put shared (token-independent) LDS regions first so their fragment offsets fit the 16-bit ds immediate. To fit two head groups' worth of LDS at 64 heads, use 4 waves as two token lanes sharing the buffers instead of two 2-wave workgroups. The round-30 soffset-for-constant-offset trick costs ~5% on stores when nothing spills — apply it only against an observed spill.
- Applicability: any hand-pipelined FlyDSL WMMA kernel near 1024 VGPRs.

### 2026-10-18 18:16 round-33 — ACCEPTED (transposed-fragment stores: coalesce through LDS)
- Context: transposed attention (O^T accumulators: lane = head, 8 contiguous d per fragment) writing o [T, H, 512] bf16 straight from the fragments.
- Signal: each `buffer_store` covered 16 heads x 32 B, 1 KB apart. A diagnostic with identical bytes but addresses remapped to 512 contiguous bytes per instruction ran 35-43% faster, within ~12% of dropping the store entirely.
- Takeaway: test the address pattern, not just the byte count — remap store offsets in a throwaway copy (output scrambled) to bound the gain before building. The fix is a wave-private LDS staging tile (fragments in, row halves out; LDS ops of one wave stay in order, so no barrier). Adding epilogue code near the VGPR cap can tip an unrolled loop into spills (flash cr0 32-row tiles: 168 spills, 2x slower); re-check spills per config after epilogue changes.
- Applicability: any WMMA kernel whose accumulators are transposed relative to the output row layout (attention fwd o, bwd dq / dkv).

### 2026-10-18 18:48 round-34 — ACCEPTED (benchmark the index layout training produces)
- Context: sparse attention fed by a learned top-k indexer (DSV4 CSA, cr4): per-token top-k over a causally growing compressed pool.
- Signal: the campaign's random no-mask indices were 99.8% valid; the real indexer (`torch.topk` sorted, masked picks -> -1) leaves them trailing and ~45% of slots empty at T=4096 (pro). The kernels computed every slot: main bwd kernel only -7% on real indices. A per-token loop bound (last valid slot, from one torch reduction) cut it 35% (pro cr4c bwd 3.47 -> 2.50 ms); the reduction costs ~1% on fully valid indices.
- Takeaway: build benchmark indices the way the producer builds them (causal mask, sort order, -1 placement) before optimizing a sparse kernel; a work-proportional loop bound needs no assumption beyond "slots past the last valid one are empty", so it stays correct for any layout. Per-token fixed work and launch order (longest tokens last) limit the gain below the slot ratio.
- Applicability: any top-k / sparse attention or gather kernel driven by indexer output.

### 2026-10-18 19:29 round-35 — ACCEPTED (the bwd intermediate store: ablate first, partial-row staging)
- Context: sparse-MLA backward main kernel (1 wave/SIMD, LDS nearly full with a single kv buffer) writing the C^T intermediate [T, topk, 512] bf16 from transposed fragments (lane = slot, 8 contiguous d).
- Signal: per-stream ablation: no-store -26..-43%, no-gather -5%. Same bytes remapped to 512 B / 128 B / 64 B contiguous per instruction: -19 / -15 / -14% (pro 16K), -33 / -27 / -24% (flash 16K). A 2.3 KB per-wave stage (4 d tiles x 16 slots, 3 tiles held in registers) realized the 128 B number: cr4 bwd -12..-20%, step +7.6%.
- Takeaway: before adding buffering to hide a load, ablate every memory stream — here the "obvious" kv-gather target was hidden and the store was not. Full-row staging is not needed: 128-byte (cache-line) row pieces get ~80% of the coalescing gain at a quarter of the LDS, which is what fits next to a single kv buffer. Ablation harness pitfall: closures defined in a loop over variants late-bind the loop variables — bind them as defaults or every variant times the last one.
- Applicability: any WMMA kernel storing transposed accumulators with little LDS headroom (bwd dkv / intermediate, fwd o at large head counts).

### 2026-10-18 20:10 round-36 — ACCEPTED (data-dependent trip counts near the VGPR cap)
- Context: 1-wave/SIMD attention forward at 900-950 VGPRs (O^T 512 + q 256), tile loop whose length should follow a per-token count read from the indices.
- Signal: replacing the static `scf.for` bound with a dynamic one (even a trivially dynamic one) took flash from 901 VGPRs / 0 spills to 1024 / 35 spills and 2x slower; clamping the bound >= 2 made it 103. Keeping the static trip count and skipping the body with a wave-uniform `scf.if` that threads the loop state gave 910 VGPRs / 0 spills and the full skipping gain.
- Takeaway: near the register cap, keep loop trip counts static and branch over unneeded iterations (wave-uniform condition, state passed through the else arm) rather than making the bound dynamic — the loop bypass edge of a dynamic bound costs registers. A per-token "last valid tile" is cheap in-kernel: lane c checks tile c, then `ballot` + `llvm.ctlz.i32` (wave32) gives a wave-uniform count with no barrier and no extra torch launch. FlyDSL: a loop body must have a single `yield` (compute the state in branches, yield once); `scf.IfOp` blocks come with a placeholder yield to erase.
- Applicability: any FlyDSL loop with a data-dependent length in a register-heavy kernel (sparse / variable-length attention, ragged GEMM).

### 2026-10-18 21:55 round-39 — ACCEPTED (profiling gfx1250 kernels: stamp the phases, then ablate)
- Context: 1-workgroup-per-CU sparse attention forward (4 waves at ~950 VGPRs), one token per workgroup, ~36% of WMMA peak at pro cr4 16K.
- Signal: rocprofv3 on this ROCm offers only SQ busy / wave / WMMA counters, and the WMMA ones misreport on a known-count microbenchmark. Stochastic PC sampling put 47% of samples at s_barrier_wait, but removing the barriers saved only 2.6%: the barrier is where a wave books idle time. Per-wave `llvm.readcyclecounter` stamps at phase boundaries, written into the output, showed the real split: the O epilogue was 20% of the workgroup lifetime (32 KB per wave stored as 32 B pieces of 16 rows per instruction). Staging O through 16.6 KB of LDS per wave for 512 B contiguous stores: fwd -14..-20%, step +2.1%.
- Takeaway: on gfx1250, validate counters on a microbenchmark with a known count before using them; read PC-sampling barrier/wait time as "where idle time lands", and confirm by removing the wait. Per-wave cycle stamps per phase (prologue / loop / epilogue) are cheap and decisive when there is one workgroup per CU, since nothing hides a prologue or an epilogue. The shader cycle counter is per SIMD: compare durations, not start times, across waves. Ablate every memory stream including the final output store: the forward's earlier component ablation skipped it and missed the largest cost.
- Applicability: any low-occupancy gfx1250 kernel (1 wave / SIMD) with a per-workgroup output, and any FlyDSL kernel storing transposed WMMA accumulators straight to global memory.

### 2026-10-18 22:30 round-40 — ACCEPTED (lazy rescale; judge accuracy against fp32, not the oracle kernel)
- Context: 1-wave/SIMD sparse attention forward on gfx1250 with O^T accumulators of 512 VGPRs per wave (d=512, 32 heads).
- Signal: a lagged running max (move only past 2^8, wave-uniform ballot, O rescale under `scf.if`) cut the cr4 forward 3%. The SNR vs the Triton oracle fell from 57-83 to 52-62 dB, but vs an fp32 reference it moved only 53.8 -> 53.4 dB against a 55.6 dB bf16-output cap: the oracle had agreed because it rounded P relative to the same max. Screened out the same session: 2 waves/SIMD (16 heads/wave) 1.5x slower — fragment reuse across head groups dominates latency hiding; WMMA `reuseA` hints did nothing (bits verified in ISA).
- Takeaway: when a change alters rounding order (lazy rescale, different max, split-K), measure it against an fp32 reference and the bf16-rounded-reference ceiling; a drop in agreement with another bf16 kernel is not an accuracy loss. Layout choice for big-d attention on gfx1250: maximize WMMAs per LDS fragment first; a second wave per SIMD is worth only a few percent.
- Applicability: any online-softmax kernel; any gfx1250 attention design choosing between occupancy and per-wave reuse.

### 2026-10-18 23:40 round-41 — ACCEPTED (stage the epilogue through dead LDS; whole-workgroup contiguous store)
- Context: gfx1250 sparse-MLA backward main kernel, 1 wave/SIMD, LDS full during the loop; dQ accumulators in WMMA layout (16 B per lane at a 1152 B row stride).
- Signal: per-wave `readcyclecounter` stamps put the dQ store at 8-12% of a workgroup even though it is only 32 KB per wave; staging dQ in the Q/dO area (dead after a barrier) and storing the token's H x 1152 B as one contiguous run cut the main kernel 4-6% and the step 2.2%, beating a direct contiguous remap of the register layout.
- Takeaway: a full LDS during the loop is no reason to skip epilogue staging: after the final barrier the operand areas are free. Stage per workgroup (not per wave) when the output tile is contiguous in global memory; then the global offset is just (tid + i*BLOCK)*16 and any wave split works.
- Applicability: any gfx1250 kernel whose epilogue writes WMMA accumulators to row-major memory.

### 2026-10-18 22:20 diagnosis — measure the HBM roofline with your own kernel, not torch copy
- Context: MI455X (gfx1250), deciding whether a kernel is bandwidth-bound.
- Signal: a FlyDSL buffer_load/store stream kernel (256 threads, 4 x 16 B per thread) reaches ~2.4x the copy bandwidth torch `copy_` shows (reads slightly above writes). Using the torch number as the ceiling had labelled the window backward "near the byte floor" when it ran at ~30% of the real one; per-token phase stamps then showed the dQ pass (LDS K^T reads + stores) at 60% of the time.
- Takeaway: establish the bandwidth roofline with a tuned in-DSL stream kernel before calling anything memory-bound; 16 x 16 B stores in flight per thread already halve write bandwidth.
- Applicability: any gfx1250 roofline claim.

### 2026-10-18 22:50 round-42 — ACCEPTED (stage stores in an LDS tile that is dead during the pass; check VGPR headroom per variant)
- Context: gfx1250 SWA backward (1 wave/SIMD, 1024 VGPRs with 576 of dKV accumulators), dQ written per WMMA fragment (16 B per lane, 1152 B row stride).
- Signal: per-token stamps showed the dQ pass at 60% of a token, ~8K of 14K cycles in the stores. Holding 8 bf16 tiles and staging them through the P tile (dead after C^T) as 256 B rows cut the cr0 kernel 21-23% (bwd -17%, step +3.4%). The same staging in the pool variant (already at 1024 VGPRs + 11 spills) went to 53 spills and ran 30-50% slower at any width; giving each wave all 64 heads in the dQ pass (more K^T reuse) also spilled the pool variant and barely moved cr0.
- Takeaway: a staged store costs VGPRs to hold tiles until the flush; enable it per kernel variant from the spill count, not globally. Phase stamps + "remove one piece" ablations attribute a slow pass faster than counters (here: store 8K, LDS reads 4.7K, WMMA 0.9K cycles).
- Applicability: any gfx1250 kernel near the VGPR cap with a per-fragment epilogue store.

### 2026-10-23 round-43 — ACCEPTED (L2 prefetch for the next workgroup: use SCOPE_DEV, issue at the end)
- Context: gfx1250 per-token backward, one workgroup per CU, cold-miss prologue (Q / dO / O / indices / lse, 200-400 KB per token).
- Signal: `llvm.amdgcn.global.prefetch` (global_prefetch_b8, 1 line per lane) over token tok+#CUs's inputs cut the next prologue 2-3K cycles. With cache policy 0 (SCOPE_WGP) it also fills this WGP's cache and cost +9K epilogue cycles (net loss); with SCOPE_DEV (policy bits [4:3] = 2, plus TH=1 non-speculative) it fills only L2 and won 2.6-2.9% on the flash kernel. Issuing it at workgroup start, or spread over the last tiles, was worse than right after the last tile.
- Takeaway: for cross-workgroup L2 warming, set scope SE/DEV (these scopes skip the WGP cache) and issue after the compute loop. `s_prefetch_data` is not an L2 prefetch for vector data: it fills the scalar constant cache, at most 32 x 128 B, and the length is taken mod 32 lines (8192 B silently becomes 128 B).
- Applicability: any gfx1250 kernel with one workgroup per CU per work item and a load-heavy prologue.

### 2026-10-23 round-44 — ACCEPTED (attention backward: delta = rowsum(P * dP) when a row's keys finish in one pass)
- Context: gfx1250 sparse-MLA window backward (128 slots per token, all in one S / dP pass), per token Q, dO, O in and dQ out.
- Signal: O was read only for delta = rowsum(O * dO). Computing delta as rowsum(P * dP) (exact: O = sum P V; a sink adds mass but no value row) with a DPP row_xmask butterfly over the 16 slot lanes plus one LDS exchange between slot-half waves cut the kernel 16% (bwd -14..-15%, step +2.85%). The SNR vs Triton fell 3 dB, but vs an fp32 reference dq improved slightly and dsink went from 55 to 127 dB: the Triton reference's delta carries the bf16 rounding of O.
- Takeaway: when every key of a query row is visited in one pass, form delta from P * dP and skip reading O. Judge numerics against fp32, not against another bf16 implementation that shares the error you removed. `rocdl.update_dpp(T.f32, x, x, 0x160 + m, 0xF, 0xF, True)` gives the xor-m lane within a 16-lane row (there is no permlane16 op in FlyDSL's rocdl dialect).
- Applicability: windowed / short-row attention backward; not tiled top-k kernels where dS of the first tile is needed before the last tile's dP.

### 2026-10-23 round-45 — ROLLED BACK (re-profile after a traffic-changing round before acting on an old ablation)
- Context: gfx1250 window backward dQ pass; a "remove the K^T LDS reads" ablation taken before round 44 promised -8%.
- Signal: round 44 removed the next token's O loads issued just before the dQ pass, and the pass fell from 7.8K to 4.6K cycles on its own. The all-heads-per-wave re-split (half the K^T reads, no spills) then made the pass 5.1K and the step -0.3%.
- Takeaway: ablation savings are not additive across rounds; after any round that changes memory traffic, re-stamp the phases before choosing the next lever.
- Applicability: any multi-round campaign using phase stamps / ablations.

### 2026-10-23 round-46 — ROLLED BACK (fragment reads one-per-step expose LDS latency; deeper prefetch is VGPR-capped)
- Context: gfx1250 window backward S/dP at 1 wave/SIMD near the 1024-VGPR cap (576 dKV accumulators).
- Signal: ablating the K fragment reads cut S/dP 5.3K -> 3.5K cycles, while the same number of Q/dO fragment reads cost only 0.5K: Q/dO are read 4 at a time and held over 4 steps, K one per step 4 steps ahead. Prefetch 8 recovered ~0.4K (-3% kernel in screening, < 1% in the bench); prefetch 16 spilled 68 VGPRs and ran +70%.
- Takeaway: count *how* fragments are read, not just how many: grouped reads amortize LDS latency, per-step reads expose it. When VGPRs cap the prefetch depth, restructure (share fragments across more WMMAs, or interleave an independent phase) instead of deepening the pipeline. readcyclecounter stamps do not fence VALU: sub-phase stamps inside a block of arithmetic are unreliable.
- Applicability: any WMMA loop at 1 wave/SIMD with per-step operand reads from LDS.

### 2026-10-23 round-48 — BASELINE (FlyDSL 0.3.4.1 / LLVM 24 costs ~5% through expert-mode dependency waits)
- Signal: same source, same VGPRs/attributes/IR; step 822.96 -> 782.50. `s_wait_alu depctr_va_vdst/vm_vsrc` counts jump (cr4 fwd ~0 -> 212, cr4 bwd 391 -> 707).
- Takeaway: after a FlyDSL/LLVM bump, re-baseline and diff the final ISA's instruction histogram first; under `amdgpu-expert-scheduling-mode` the compiler, not the hardware, enforces VALU->VMEM/DS distances, so a new LLVM can add stalls without touching registers. API ports for 0.3.x: `Vec.from_elements`, cache policy / `aux` as optional attributes, `fx.rocdl.sched_barrier`.
- Applicability: any gfx1250 FlyDSL kernel compiled with expert scheduling.

### 2026-10-23 round-49 — ROLLED BACK (source order cannot dodge a hazard wait inserted after scheduling)
- Signal: FlyDSL 0.3.4.1's LLVM puts `s_wait_alu depctr_va_vdst(0)` before a ds_load that overwrites the A operand of a just-issued WMMA (cr4 fwd loop: 103 waits; no RAW/WAW). Moving the load in the source changed nothing: the machine scheduler hoists loads to just after the last reader, and the wait is added afterwards. Pinning with sched_barrier cost 2-12% fwd.
- Takeaway: classify `s_wait_alu` waits by tracking VALU defs since the last wait (a small ISA script) before acting; waits with no RAW/WAW are compiler hazard rules, and only the compiler (or a register rotation the allocator cannot undo) can remove them.
- Applicability: gfx1250 WMMA loops fed by LDS fragment loads under expert scheduling with FlyDSL 0.3.4.x.

### 2026-10-23 round-50 — ACCEPTED (write-once intermediates: store with TH_NT)
- Signal: a large bf16 intermediate written by the main kernel and read back only after the whole launch (gather pass) shares L2 with randomly gathered kv rows.
- Takeaway: pass `aux=1` (gfx12 TH_NT) to `rocdl.RawPtrBufferStoreOp` for such streams; pro cr4 bwd -2%, flash flat; `aux=3` (WB) does nothing. Ablate the store first (drop the buffer_store in a patched copy) to size the ceiling: here stores were 21-29% of the kernel and the hint recovered ~3% of it.
- Applicability: gfx1250 FlyDSL kernels emitting write-once outputs larger than L2 while also gathering reused data.

### 2026-10-23 round-51 — ACCEPTED (overlap the per-chunk reduction with the next chunk's main kernel)
- Signal: sweeping the chunk cap showed a fixed ~0.1 ms per chunk (one token-round of a 1-WG-per-CU kernel draining before the serialized sort + gather).
- Takeaway: put the index-only inverse and the gather of chunk c on a non-blocking side stream (wait_stream / per-buffer events, two intermediates) so they run in chunk c+1's tail. Pass the torch stream to FlyDSL explicitly: `fx.Stream(None)` launches on the legacy null stream. Flash cr4 bwd -4..-7%, pro -1..-2%, for +1 intermediate of memory; halving the cap to keep memory flat loses more to extra chunks than overlap returns.
- Applicability: any chunked produce-then-reduce pipeline where the producer pins every CU.

### 2026-10-23 round-52 — ROLLED BACK (faster side-stream kernel, no end-to-end gain) + flyc.compile runs the launch
- Signal: a FlyDSL CSR gather beat Triton by 9% standalone (64-thread workgroups best: most workgroups are empty / short rows behind a chain of dependent index loads; 256 threads, deeper unroll and row pipelining were all slower), yet the cr4 backward did not move in an interleaved A/B: the gather runs on a side stream under a kernel that holds every CU.
- Takeaway: once work is overlapped, A/B the end-to-end op in one process before investing in the overlapped kernel; and when the bench headline moves on shapes the change cannot touch, treat it as noise. Separately: `flyc.compile(launch, *args)` executes the launch once — a `run_` that compiles then calls `fn` runs twice; fine for overwrite kernels, wrong for accumulating ones (return after compiling).
- Applicability: any multi-stream pipeline; any accumulating FlyDSL kernel with the compile-then-call wrapper pattern.

### 2026-10-23 round-53 — ACCEPTED (gpu.barrier() in a store-heavy loop: use an LDS-only barrier)
- Signal: a store ablation costing 20-30% of a kernel whose write bandwidth is low; ISA shows `s_wait_storecnt_dscnt 0x0` right before the loop's `s_barrier_signal`.
- Takeaway: `gpu.barrier()` waits for all outstanding global stores. Where the barrier only orders LDS (and TDM, already waited with s_wait_tensorcnt), emit `fx.rocdl.s_wait_dscnt(0); fx.rocdl.s_barrier_signal(-1); fx.rocdl.s_barrier_wait(-1)`. Sparse-MLA cr4 bwd w1: -5% kernel, -3..-5% bwd. Grep every FlyDSL loop that stores to global memory for this.
- Applicability: any FlyDSL gfx1250 kernel with per-iteration global stores and workgroup barriers.

## TDM stores (round 54)
- `tensor_store_from_lds` reads LDS linearly (no de-padding). To get a padded (bank-conflict-free) stage, make tile_d0 = cols + pad while tensor_dim0 = cols: the overhang is dropped as out of bounds, the LDS read still advances by the full tile width. Bit-exact.
- 3-D TDM descriptor: group1 s4 = tile_dim1 | tile_dim2<<16, s6 = tensor_dim1_stride[15:0]<<16 (s7 high bits), group2 s0 = tensor_dim2. Works, but stores with 32 B innermost rows are very slow; keep rows >= 256 B.
- TDM loads and stores share one in-order tensorcnt: any wait on a store issued after a prefetch gather also waits for the gather. Double-buffered store stages under a one-tile-ahead gather therefore stall on the gather.
- For the w1 C^T intermediate, the round-53 per-lane NT buffer_store from a padded stage beat TDM stores (flash +3%, pro has no LDS for a double stage).

## Cycle probes inside a FlyDSL kernel (round 55)
- `fx.Int64(llvm.call_intrinsic(i64, "llvm.readcyclecounter", [], [], []))` (from `flydsl._mlir.dialects.llvm`) reads a cycle counter; store per-phase deltas from one lane with an OOB-offset `buf_store` select for other tiles / lanes. Cheap enough to keep the phase split realistic.
- w1 (1 wave / SIMD) findings: real tiles run 2.3-2.5x the time of the same WMMAs with no LDS loads, because LDS fragment reads do not overlap WMMAs. Back-to-back dependent WMMAs (same accumulator) cost nothing extra — reordering them is useless.
- Hoisting tile-invariant WMMA operands out of the tile loop only paid 1-2% (C^T phase is not LDS-bandwidth-bound; it is latency / issue-bound at 1 wave per SIMD).

## Ablating a store also deletes its producers (round 56)
- Removing a result store (e.g. the dQ fragment `buf_store`) lets the compiler drop every WMMA and load that only fed it. The window-bwd "no dq store -20%" ablations (rounds 36-42) measured "no dQ at all". Ablate stores by issuing them to an out-of-range offset (traffic gone, issue kept) or by folding the results into one running sum stored once (stores gone, compute kept).
- cr128 window bwd: rowsum(P * dP) over window + pool slots replaces the O * dO delta (as cr0, round 44) — no O prefetch (64 VGPRs), no delta phase, -10..-11% bwd. Order matters at 1024 VGPRs: computing the small pool S / dP first (32 VGPRs of P / dP held) beat window-first (128 held) by 18%.

## Side streams under a kernel that holds every CU (round 59)
- A side-stream kernel only runs where the main kernel leaves CU slots (1 WG/CU with full LDS / VGPRs leaves none mid-chunk), so "overlapped" work costs about its standalone time as main-kernel slowdown plus tail. Size the side work standalone and cut it like exposed work; many small torch elementwise kernels are the worst case (each waits for slots: 6 key-building ops took 340 us beside w1 vs ~20 us alone).
- Several CSR inverses over slices of one index map come from one stable sort with composite keys `slice * num_rows + row` (int16 while it fits): each slice's row pointers are a host-side view of one searchsorted, no host sync; give the consumer row / entry base offsets.
- Ablation check: count WMMAs in the ablated ISA. Reusing one loaded fragment for several chains made identical chains the compiler merged (208 of 320 WMMAs left, -21% instead of the real -7%).

## Fuse torch elementwise chains on a starved side stream (round 60)
- Beside a kernel that holds every CU, each small torch launch waits for a CU slot, so a chain of six elementwise ops cost 340 us for ~20 us of work. A single Triton kernel did it in 25 us (pro cr4 4K step -0.9%).
- Next on the same stream: torch.sort's iota (`fill_reverse_indices`), `searchsorted` and the int64 -> int32 index cast each cost ~90-110 us there; a direct histogram + scan CSR would remove them.

## Know when the side chain leaves the critical path (round 62)
- After rounds 59-61 the sort chain beside w1 chunk 0 ends at ~490 us, well inside chunk 0. Cutting searchsorted 95 -> 15 us changed nothing (+0.2%). Before trimming a side-stream kernel, check in the timeline that its end gates something (the next chunk's gather or the tail).

## Persistence at ~950 VGPRs: the carried state costs more than the hidden prologue (rounds 63-64)
- Phase stamps of the 1-token-per-WG cr4 forward: ~9.4K of 63K cycles/token wait for the first kv gathers and q; reordering the prologue only moves the wait (TDM wait -> barrier).
- A persistent ragged version (ring index advanced by each token's own tile count; next token's tiles 0/1 issued after the loop's last barrier, gathers complete in issue order so a stray gather into a free buffer is harmless) hid the prologue, but carrying q (256 VGPRs) through the token loop sent the build to 1024 VGPRs + spills and the tile loop to +23% cycles (s_set_vgpr_msb 983 vs 658): check the per-tile cycles of the same code at grid = T to separate codegen from runtime. Reloading q per token recovered most of the loop but lost the gain to the q latency.
- Tools: patched-source ablations + per-wave stamps (incl. inter-token gaps), ISA stats per builder kwargs, per-basic-block counts and mnemonic diffs (see `tools/` in this repo).

## L2 prefetch transfers to the forward; interleave A/B timings (round 65)
- One-token-per-WG forward at 1 WG/CU: `global_prefetch` (SCOPE_DEV) of token tok - NPROG's q rows + indices after the last tile cut the prologue 9.7K -> 5.9K cycles, fwd -3.1% (pro cr4 4K). Skip it at 2 WGs/CU (flash): the sibling hides the prologue and the prefetch only adds traffic (+4%).
- Timing two kernels back to back in one process: the first one timed read 1.5-4% slow (A/A check), small shapes ±15%. Interleave (ABBA...) and take medians before trusting a few-percent screen.

## The 1-wave/SIMD forward loop is not LDS-read-bound (after round 65)
- Halving K or V^T fragment reads with the WMMA count kept (reuse a fragment for the next step's different B operand / accumulator) changed nothing; removing the in-loop TDM gathers saved 7% though nothing waited on them (TDM LDS writes + random rows). Don't restructure for fewer LDS reads (e.g. d-split PV across waves); look at gather traffic and issue/hazard stalls.
- Ablation pitfall: a gather where every slot reads one row turns into an L2 hotspot (2x slower); use distinct L2-hot rows to separate "memory source" from "TDM/LDS write" cost.

## Round 66 — ACCEPTED: batch LDS operand loads to dodge LLVM 24's WMMA-source reload wait
- Signal: FlyDSL 0.3.4.1 (LLVM 24, expert scheduling) puts `va_vdst(0)` before every ds_load that overwrites a register a WMMA read. A per-step operand stream (queue of 8) hit it ~110 times per 256-WMMA tile; the dense gfx1250 flash / MLA forwards, which load each phase's operands in bulk (`L32 | W32`), get 4-5 per loop on the same compiler.
- Fix: load batch b+1 (8 fragments) into other registers while batch b's WMMAs run, sched_barrier only at batch edges -> one reload switch per batch. Forward -10% (pro cr4 4K), bit-exact, waits 114 -> 69.
- Takeaway: when a compiler rule costs per register-reuse transition, compare the ISA op sequence (`tools/isa_seq.py`) with a kernel that is unaffected; the fix is structural (fewer, larger reuse transitions), not source reordering. Size the batch by VGPRs: 2 x BATCH fragments live; past the budget it spills and runs 2x slower.

## Round 67 — ACCEPTED: the same operand batching in the backward main kernel
- Batch 4 steps (backward steps load 1-5 fragments each, so its batches are smaller than the forward's 8): bwd -6% at pro cr4 4K, bit-exact, and 48 fewer VGPRs than the prefetch-8 stream. Batch 8 spills.
- Takeaway: pick the batch by fragments per batch, not steps; screen a range interleaved (3..6 were within noise) and prefer the one with the lowest VGPR count.

## Rounds 68-69 — ACCEPTED: batching in the cr0 / cr128 window kernels; not with global-memory operands
- Window forward: batch 6, -5..-7%; cr0 window backward: batch 4, -4..-5%; all bit-exact, no spills.
- cr128 backward (pool K / K^T fragments read from global memory, not LDS) got 18-36% slower with batching: grouping long-latency VMEM loads into batches exposes their latency, which the per-step stream hid. Gate batching to LDS-fed loops; screen each kernel separately.

## Round 70 — ACCEPTED: at 1024 VGPRs, stage outputs through LDS without holding them, and keep addresses out of VGPRs
- Diagnose with a loop mnemonic diff vs a sibling kernel: the cr128 bwd had 3.6x the `s_set_vgpr_msb` of cr0 (VGPR bank switching above v255) and 2.7x the `va_vdst` waits for +24% WMMAs.
- Direct 16 B-per-lane dQ fragment stores cost ~25% of the window backward vs a staged row copy-out. Holding QGW tiles in VGPRs until a row is whole spills at the VGPR limit; store each tile to the LDS stage as it completes instead.
- Spills of loop-invariant addresses are worse than they look: the scratch reloads wait on loadcnt, which also drains in-flight buffer_loads (here the pool K^T prefetch). 36 spills = +50%. Put per-store constants in instruction immediates and wave-uniform parts in soffset, one lane-varying VGPR per side (4 spills, -3.5%).

## Round 71 — ACCEPTED: fold small host-side torch epilogues into an existing reduce launch
- After round 70 the cr128 backward still spent ~0.03 ms in three tiny torch launches (partial sum over 512 workgroups, fp32 -> bf16 slice copy, rope zero fill). Extra workgroups in the existing `swa_dkv_reduce` grid (kernel-body `if` on the block index) replace them: -3.5% pro / -4.7% flash cr128 bwd.
- A reduction over many partials per output row needs several workgroups per row (column slices) and many loads in flight per thread (unrolled 16); a serial per-thread loop is latency-bound.
- Splitting heads differently inside a workgroup does not change dKV accumulator VGPRs: they cover window rows x 512 columns summed over heads, so any wave partition gives rows x 512 / 4 per lane.
