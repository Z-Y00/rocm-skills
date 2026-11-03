# gfx1250 (MI455X) — Overview

gfx1250 is the MI455X compute target. For kernel writers it is closer to a wave32 RDNA-style
machine with a very large register file and LDS than to CDNA3/CDNA4: **wave32, WMMA (not MFMA),
a Tensor Data Mover (TDM), split barriers and a new set of wait counters**. Kernels ported from
gfx942/gfx950 need restructuring, not a recompile.

Read [`kernel-implementation-notes.md`](kernel-implementation-notes.md) next for per-subsystem
guidance, measured behavior and FlyDSL recipes. The findings come from a 71-round sparse-MLA
campaign ([AMD-AGI/Primus-Turbo#560](https://github.com/AMD-AGI/Primus-Turbo/pull/560)); the
round-by-round evidence is in
[`historical_experience/gfx1250/sparse_mla_fwd_bwd/flydsl/`](../../../../../historical_experience/gfx1250/sparse_mla_fwd_bwd/flydsl/).

Absolute peak numbers are deliberately not listed here. Measure your own roofline with in-DSL
microbenchmarks (see the notes, §9); `torch` copies and vendor GEMMs under-report this part.

## Execution Model in Brief

| Dimension | gfx1250 | Why it matters |
|---|---|---|
| Wave size | 32 | Lane maps, reductions (DPP row ops within 16 lanes), ballots (`ballot.i32`, `ctlz.i32`) are wave32 |
| Matrix op | WMMA, e.g. `v_wmma_f32_16x16x32_bf16`: A, B = v16bf16, C/D = v8f32 per lane | 16x16 tiles; lane `l16 = lane % 16` owns a row/column, `lane // 16` a K half; no AccVGPR pool |
| Waves per SIMD | `waves_per_eu` 1 → up to 1024 VGPRs per wave; 2 → 512 | The big win is usually 1 wave/SIMD with more work per wave, not more waves |
| VGPRs above v255 | addressed through `s_set_vgpr_msb` bank switches | Near 1024 VGPRs, the switch count (not only spills) grows; diff it against a sibling kernel |
| LDS per workgroup | up to 320 KiB | One workgroup per CU with a 4-wave WG is a common layout for big-d attention |
| Global → LDS | TDM `tensor_load_to_lds` (2-D tiles and row gathers, descriptor padding), `s_wait_tensorcnt` | Gathers rows by index without staging VGPRs; loads and stores share one in-order counter |
| LDS → VGPR transpose | `ds_load_tr16_b128` | Transposed WMMA operands (K^T, Q^T, dS^T) straight from LDS |
| Barriers | split `s_barrier_signal` / `s_barrier_wait` | Work can sit between signal and wait |
| Wait counters | `s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_dscnt`, `s_wait_xcnt`, `s_wait_tensorcnt`, `s_wait_alu depctr_*` | Finer-grained than `vmcnt/lgkmcnt`; a barrier that waits on `storecnt` stalls store-heavy loops |
| Scheduling mode | `amdgpu-expert-scheduling-mode` (SCHED_MODE 2): the compiler enforces VALU→DS/VMEM/WMMA distances | +3..7% on WMMA-bound kernels, but compiler versions change which waits it inserts |
| Cache hints | `global_prefetch_b8` with scope (WGP / SE / DEV), TH_NT store hint | L2-only prefetch for the next workgroup; non-temporal write-once outputs |

## Differences From CDNA (gfx942 / gfx950) That Change Kernel Structure

1. **Wave32 + WMMA.** Re-derive every lane→(row, col) map, reduction mask and tile shape. MFMA
   scheduling knobs (`sched_mfma` ratios, `s_setprio` tricks) do not transfer.
2. **Register budget.** One wave per SIMD gets 1024 VGPRs. Spending it on more heads per wave
   (each LDS fragment feeding 2+ WMMAs) beat adding waves in every attention kernel we tried.
3. **TDM replaces register-staged global loads.** Row gathers by index land directly in LDS;
   register loaders cost 64-128 staging VGPRs at 4 waves.
4. **The compiler owns hazard distances.** Under expert scheduling mode, a compiler upgrade can
   cost 5-20% through added hazard waits with identical source, registers and IR (see the notes, §3).
5. **Profiling is thinner.** rocprofv3 counters for WMMA misreported on this stack and there are
   no LDS bank-conflict counters; per-wave cycle stamps and ablations are the reliable tools.

## Porting Checklist (gfx950 → gfx1250)

- Rewrite matrix ops for WMMA 16x16x32 and wave32 lane maps; re-check every DPP / ballot mask.
- Choose the wave layout first: 4-wave workgroups at `waves_per_eu=1`, heads (or N) per wave
  sized to use ~900-1000 VGPRs without spills.
- Move global→LDS staging to TDM (2-D tiles or gathers); add `s_wait_tensorcnt` drains at the tail.
- Replace manual VGPR transposes with `ds_load_tr16_b128`.
- Compile with expert scheduling mode, then diff the final ISA's loop histogram for
  `s_wait_alu depctr_*`, `s_set_vgpr_msb` and `scratch_*`.
- Re-derive the roofline with a tuned stream kernel and WMMA / LDS microbenchmarks.
