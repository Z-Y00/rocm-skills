### 2026-11-14 13:20 round-2 — ACCEPTED
- Context: gfx1250 FlyDSL bf16 variable-K wgrad (non-persistent path, accumulate=1), OUT 4096 x 4096, 32-64 experts, 512-2048 rows / expert.
- Signal: group_m 4-8 plus num_xcd 8 / xcd_band 256-512 cut the fc2 wgrad by 24-51% (bit-identical); OUT 8192 x 4096 (fc1) did not move. Small bands (16, 32) are slower than the default.
- Takeaway: on gfx1250 the tile walk order of a grouped GEMM with few hundred tiles per expert is a first-order knob; try XCD band >= 256 with group_m 4-8 before deeper changes.
- Applicability: grouped GEMMs whose per-expert tile count is a few hundred; also time candidates in a paired A/B in one process: run-to-run drift on MI455X was +-5-9%, and a sweep that clones multi-GB outputs between timings inflated the default's time by ~30%.
### 2026-11-14 13:35 round-3 — ACCEPTED
- Context: gfx1250 FlyDSL bf16 NT / NN grouped GEMM, Muse Spark shapes (N, K 4096-8192), 32 / 64 experts, 512-2048 rows / expert.
- Signal: the MiniMax-M3 persistent multicast keys transfer (fwd fc1 +5-11%, dgrad +3-9% at >= 1024 rows, paired). At 64 experts the NT set is refused ("nt_seam bit 2 needs ... tt_dec or vec_dec"): vec_dec is dropped above 32 experts, and tt_dec in its place launches and keeps the gain. At 512 rows / expert the persistent sets give nothing on dgrad.
- Takeaway: when transplanting a keyed NT set to > 32 experts, swap vec_dec for tt_dec. Validate with a paired A/B: single-op sweeps showed +20-37% for the same sets.
- Applicability: NT / NN persistent sets on gfx1250; not the variable-K path.
### 2026-11-14 13:52 round-4 — ACCEPTED
- Context: gfx1250 FlyDSL bf16 variable-K wgrad, accumulate=1 into fp32 main_grad, OUT 8192 x 4096, 512-2048 rows / expert.
- Signal: the default 256 x 256 tile runs one workgroup per CU, so each tile's main_grad read-modify-write epilogue idles the CU's compute; a 128 x 128 x 64 tile (~144 KB LDS) runs two per CU: +37% at 512 rows / expert, +11-19% at 1024-2048.
- Takeaway: for epilogue-heavy GEMMs (fp32 RMW output), trade tile size for two workgroups per CU before attempting persistent-epilogue overlap.
- Measurement: time candidates in step mode (all ops of the layer in training order, A / B alternating). One op called 20x back-to-back misreported the untuned wgrad by up to 30% vs its time inside the sequence.
### 2026-11-17 rounds 5-6 — ACCEPTED (revert + drop)
- Context: the round-4 keys and the fwd fc1 keys at 512 / 1024 rows, re-measured on a production node (the earlier rounds ran on a pre-production node).
- Signal: several keys that won on the first node lost on the second; reverting them and dropping the fwd fc1 entries at 512 / 1024 rows was a net gain.
- Takeaway: config keys tuned on one node or driver stack are hypotheses on the next; re-validate every key on the node the numbers will be reported from before stacking new work on it.
- Applicability: any keyed config table on young hardware / software stacks.
### 2026-11-17 round-7 — ROLLED BACK
- Context: persistent variable-K wgrad with the fp32 main_grad accumulate (old value through LDS), plus a C-prefetch knob.
- Signal: bit-identical and correct, but no gain over the non-persistent accumulate path with its tuned tile.
- Takeaway: once the epilogue-heavy wgrad runs two workgroups per CU, a persistent accumulate path has little left to hide; do not expect persistence to beat occupancy here.
### 2026-11-17 rounds 8-9 — ACCEPTED
- Context: fwd fc2 and dgrad fc1 / fc2 at 512-1024 rows; accumulating wgrad fc1 / fc2.
- Signal: LDS co-scheduling (`lds_cosched 3`) with `issue_first 1` won on the forward / dgrad; `lds_cosched 2` on the accumulating wgrad; all bit-identical against the keys turned off.
- Takeaway: LDS-read / WMMA co-scheduling depth is shape-family specific; sweep it per op (fwd, dgrad, wgrad) at every production row count instead of copying one op's value.
- Measurement: step mode (every op of the layer in training order, A/B alternating) and a bit-identity check against the keys forced off.
### 2026-11-21 round-10 — ACCEPTED
- Context: config tables keyed on exact `(N, K, avg_m)` with `avg_m = rows // experts`.
- Signal: real routing gives row counts between the tuned keys, so exact lookup silently fell back to defaults for most production batches; a log-scale nearest-key lookup within a factor of sqrt(2) of a keyed row count (scoped to one model's `(N, K)` pairs) recovered the tuned gain at the midpoints too.
- Takeaway: key tables on a runtime row count need a tolerant lookup, or most real calls miss; test midpoint row counts explicitly, and keep other models' keys exact unless they are tuned for it.
- Applicability: any per-shape config table indexed by a data-dependent dimension (rows per expert, tokens).
