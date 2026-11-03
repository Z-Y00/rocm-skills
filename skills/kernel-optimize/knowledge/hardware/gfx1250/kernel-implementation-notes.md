# gfx1250 (MI455X) — Kernel Implementation Notes

Per-subsystem guidance for gfx1250 kernels, measured on WMMA attention kernels written in FlyDSL
(0.2.x and 0.3.4.1 / LLVM 24). Read [`overview.md`](overview.md) first. Instruction names are as
they appear in the final ISA (`22_final_isa.s` in a FlyDSL IR dump). Each item states the signal
that found it, so it can be re-checked on a new compiler.

---

## 1. Wave Layout and the Register File

- **One wave per SIMD, more work per wave.** A 4-wave workgroup at `waves_per_eu=1` gives each
  wave 1024 VGPRs. Use them for more heads (or N columns) per wave so each LDS operand fragment
  feeds two or more WMMAs. Rebuilding the forward as 4 waves x 32 heads (two 16-head groups) with
  a TDM gather ring gave +22% forward TFLOPS, and the same layout cut the backward main kernel
  10-15%. A second wave per SIMD (16 heads per wave) screened 1.5x slower.
- **Small workgroups may take the whole file.** With 4-wave workgroups the backend can allocate
  all 1024 VGPRs. Pass `value_attrs={"rocdl.waves_per_eu": 2}` to cap at 512 for 2 WGs/CU, with a
  smaller tile if the kernel would spill.
- **When the LDS pins 1 WG/CU, keep the waves busy.** Split per-head-group work across two waves
  (slot sub-tiles, dQ d-range) instead of dropping to fewer waves when the head count is small.
- **Above v255, watch `s_set_vgpr_msb`.** Bank switches grow with pressure before spills appear:
  a kernel at the 1024 limit had 3.6x the switches of its sibling for 24% more WMMAs.
- **Spills of loop-invariant addresses are expensive.** LICM hoists one VGPR per distinct
  `lane_offset + uniform + const` address; at the limit they spill before the loop and reload
  from scratch inside it. Those reloads wait on `loadcnt`, which also drains in-flight
  buffer_loads (36 spills cost +50%). Pass the wave-uniform part as the buffer `soffset` and let
  constants fold into instruction immediates, keeping one lane-varying VGPR per address stream.
- **Keep trip counts static near the cap.** A dynamic `scf.for` bound took a kernel from 901 VGPRs
  and 0 spills to 1024 and 35 spills (2x slower). Keep the static count and skip unneeded
  iterations with a wave-uniform `scf.if` that threads the loop state.
- **Heterogeneous tiles in one loop.** Unrolling extra tiles of a different kind spilled 38-123
  VGPRs, and the count swung with unrelated edits; one `scf.for` with uniform selects for base and
  mask spilled nothing.
- **Size every batch, stage or prefetch by VGPRs.** In every screen, one step past the best size
  spilled and ran ~2x slower. Check `.vgpr_count` / `.vgpr_spill_count` per configuration.

## 2. WMMA Operand Streams From LDS

- **Load operands in batches, one batch ahead.** Load batch b+1 into separate registers while
  batch b's WMMAs run, with `sched_barrier(0)` only at batch edges. This dodges the LLVM 24
  hazard wait in §3: forward -10%, backward -6%, bit-exact. Batch 8 (forward) and 4 (backward,
  whose steps load more fragments each) were best; the next size up spilled.
- **Do not batch long-latency operands.** The same batching made a loop with WMMA operands from
  global memory 18-36% slower: grouping VMEM loads exposes their latency, which a per-step stream
  hid. Gate batching to LDS-fed loops.
- **Before LLVM 24: source-level prefetch plus pins.** A queue of fragments N steps ahead with
  `sched_barrier(0)` every few steps (prefetch 8 / pin 4). Pins without prefetch only constrain
  the scheduler (-11%), and prefetch without pins is undone by the backend.
- **Reuse across pins must be explicit.** The backend CSEs repeated LDS loads only within a
  scheduling region; `sched_barrier` splits regions. Load once and keep the fragment in a Python
  variable (count `ds_load_*` in the ISA before and after adding pins).
- **Grouped reads amortize latency; per-step reads expose it.** Q/dO fragments read 4 at a time
  and held for 4 steps cost 0.5K cycles; the same number of per-step K reads cost 5.3K cycles.
- **This loop shape is issue-bound, not LDS-bandwidth-bound.** Halving K or V^T fragment reads
  with the WMMA count unchanged changed nothing; pairing waves to share fragments (more VALU per
  wave) was 10-25% slower. Real tiles ran 2.3-2.5x the time of the same WMMAs without LDS loads.
- **Back-to-back dependent WMMAs (same accumulator) cost nothing extra.** Reordering them is useless.

## 3. Compiler and Scheduling (LLVM 24)

- **Turn on expert scheduling.**
  `flyc.compile[{"llvm_options": {"amdgpu-expert-scheduling-mode": True}}](...)`: +3..7% on every
  WMMA-bound shape at identical VGPRs. Verify `hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2` in the ISA.
  Other `amdgpu-sched-strategy` values lost (max-ilp, max-memory-clause) or crashed (iterative-*).
- **The LLVM 24 reload wait.** Under expert mode, LLVM 24 inserts `s_wait_alu depctr_va_vdst(0)`
  before any load that overwrites a register a WMMA has read, draining all outstanding WMMAs.
  Moving the load in the source does nothing: the machine scheduler hoists it to just after the
  last reader and the wait is added afterwards. The fix is structural: fewer, larger reuse
  transitions (batched loads, §2). Dense kernels that load each phase in bulk were unaffected.
- **After any compiler bump, rebaseline and diff the ISA.** Same source, registers and IR lost ~5%
  on the 0.2.x → 0.3.4.1 bump; `s_wait_alu depctr_va_vdst/vm_vsrc` counts jumped 2-200x.
- **Expert mode leaves hazards to software, and the compiler can get them wrong.**
  `global_load_async_to_lds` copies faulted under expert mode in a 500-VGPR kernel while a tiny
  async-copy kernel worked. Test new async/TDM patterns in both modes and bit-compare results.
- **Intra-wave softmax/GEMM overlap does not pay.** FA3-style splicing of softmax between
  independent WMMAs was verified in the ISA and made the forward 1-2% slower; a wave's own
  VALU/TRANS work does not hide under its WMMAs. Cut instruction count instead.

## 4. Tensor Data Mover (TDM)

- **Row gathers by index.** TDM gathers rows straight into LDS (16-bit indices where the row
  count allows) and can pad LDS rows through the descriptor, avoiding 64-128 staging VGPRs of a
  register loader. The descriptor/API details are in
  [`backend/flydsl/optimization-directions.md`](../../backend/flydsl/optimization-directions.md) §13.
- **Count-based rings need a tail drain.** Predicated-off gathers do not count toward
  `tensorcnt` and all-OOB gathers complete early: drain with `s_wait_tensorcnt 0` at the tail.
- **Loads and stores share one in-order `tensorcnt`.** Waiting on a store also waits for every
  earlier gather, so double-buffered TDM store stages under a one-tile-ahead gather stall.
- **TDM stores read LDS linearly.** For a padded stage, set `tile_d0 = cols + pad` with
  `tensor_dim0 = cols`: the overhang is dropped as OOB. Stores with 32 B innermost rows are very
  slow; keep rows ≥ 256 B. A per-lane non-temporal `buffer_store` from a padded stage beat TDM
  stores where LDS had no room for a double stage.
- **Deeper gather rings lost.** TDM LDS writes compete with operand reads; in-loop gathers cost
  ~7% even when nothing waited on them (TDM LDS writes plus random rows).

## 5. LDS

- **Up to 320 KiB per workgroup.** Lay out shared (token-independent) regions first so fragment
  offsets fit the 16-bit DS immediate.
- **No bank-conflict counters;** padding sweeps moved < 2%. Use per-lane-address fragment loads
  (`ds_load_b128`, `ds_load_tr16_b128`): a sliding window becomes a free per-lane row shift.
- **LDS ops of one wave stay in order,** so a wave-private stage needs no barrier between its
  writes and reads.
- **Dead LDS is staging space.** After the last reader's barrier, operand tiles (Q/dO, P, dS) can
  stage epilogue stores. A full LDS during the loop is no reason to skip epilogue staging.

## 6. Global Memory and Stores

- **Store pattern matters as much as bytes.** Transposed WMMA accumulators stored straight to
  row-major memory (16-32 B per lane, rows ~1 KB apart) cost 26-43% of a kernel. Stage through
  LDS and store contiguous runs: 512 B rows recovered the most; 128 B (cache-line) pieces got ~80%
  of that gain with a much smaller stage.
- **Do not hold tiles in VGPRs at the cap.** Write each finished tile to the LDS stage as it
  completes and copy rows out every few tiles; holding tiles until a row is whole spilled.
- **Buffer descriptors as switches.** `num_records = 0` turns a load into zeros with no traffic:
  use it to cancel speculative prefetches without `scf.if`, and to ablate memory streams.
- **Write-once outputs larger than L2:** `aux=1` (TH_NT) on `RawPtrBufferStoreOp` (-2% where the
  stream shared L2 with gathered rows; `aux=3` did nothing).
- **L2 prefetch for the next workgroup:** `llvm.amdgcn.global.prefetch` with scope DEV (cache
  policy bits [4:3] = 2, plus non-speculative TH) fills only L2; scope WGP also fills this WGP's
  cache and cost more than it saved. Issue it right after the compute loop. `s_prefetch_data`
  fills the scalar cache only (length taken mod 32 lines), not an L2 prefetch for vector data.
- **Randomly gathered buffers: chunk the working set.** The same gather ran 2.7x faster when its
  buffer spanned under ~2 GB; 2.3 GB chunks got no speedup. Sweep the size to find the knee.
- **Chunk per-workgroup work in multiples of the CU count** when LDS pins 1 WG/CU (wave
  quantization cost ~8%).

## 7. Barriers and Waits

- **`gpu.barrier()` drains global stores** (`s_wait_storecnt_dscnt 0` before the signal). Where
  the barrier only orders LDS, emit `s_wait_dscnt(0); s_barrier_signal(-1); s_barrier_wait(-1)`:
  -5% on a store-heavy loop.
- **PC sampling puts idle time at the barrier.** 47% of samples at `s_barrier_wait`, but removing
  the barriers saved 2.6%: the barrier is where a wave books its wait. Confirm by removing it.

## 8. Cross-Lane

- Row-local xor butterfly within 16 lanes: `rocdl.update_dpp(T.f32, x, x, 0x160 + m, 0xF, 0xF, True)`
  (DPP `row_xmask:m`); FlyDSL's rocdl dialect has no permlane16 op.
- Wave-uniform counts: lane c tests tile c, then `llvm.amdgcn.ballot.i32` + `llvm.ctlz.i32` gives
  the last valid tile with no barrier.

## 9. Measuring

- **Bandwidth roofline: write your own stream kernel.** A FlyDSL buffer_load/store kernel (256
  threads, 4 x 16 B per thread) reached ~2.4x the copy bandwidth `torch.copy_` showed. The torch
  number had labelled a kernel "near the byte floor" at ~30% of the real one.
- **Counters: validate before trusting.** rocprofv3 WMMA counters read 0 or misreported on a
  known-count microbenchmark. Time microbenchmarks instead.
- **Per-wave cycle stamps.** `llvm.readcyclecounter` at phase boundaries, written to an output
  from one lane (others store to an OOB offset), split prologue / loop / epilogue cheaply. The
  counter is per SIMD: compare durations, not start times. Stamps do not fence VALU, so sub-phase
  stamps inside arithmetic are unreliable.
- **Ablate by deleting work, timing only.** Patched copies that drop a stream or a phase bound
  what it can be worth. Pitfalls: removing a result store lets the compiler drop every WMMA that
  fed it (store to an OOB offset instead); a gather where every slot reads one row becomes an L2
  hotspot; reused fragments can make chains identical and merged (count WMMAs in the ablated ISA);
  closures built in a loop late-bind loop variables.
- **Interleave A/B in one process.** The first kernel timed read 1.5-4% slow in an A/A check and
  small shapes swing ±15%: alternate order and use min or median. Reusing one output buffer
  across timing iterations skews store-heavy kernels by 6-10%; time with fresh allocations.
- **ISA diff tools:** [`tools/isa_loop_hist.py`](../../../../../tools/isa_loop_hist.py) (mnemonic
  histogram diff of the hottest loop) and [`tools/isa_seq.py`](../../../../../tools/isa_seq.py)
  (run-length op sequence).

## 10. Pitfall Checklist

- Importing CDNA wave64 lane maps, MFMA shapes or MFMA scheduling ratios.
- Trusting `torch` copy bandwidth or vendor GEMM numbers as the roofline.
- Assuming a compiler upgrade is free under expert scheduling mode.
- Batching operand loads that come from global memory.
- Holding staged output tiles in VGPRs near 1024 VGPRs.
- Dynamic loop bounds in register-heavy loops.
- `gpu.barrier()` inside a loop that stores to global memory.
- Ablating a store by deleting it (the compiler deletes its producers too).
- Using `s_prefetch_data` as a vector-data L2 prefetch.
- Judging a rounding-order change against another bf16 kernel instead of an fp32 reference.
