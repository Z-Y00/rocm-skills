# Lessons learned — gfx1250 FlyDSL sparse-MLA fwd/bwd campaign (rounds 1-71)

Primus-Turbo campaign on MI455X (gfx1250), FlyDSL 0.3.4.1 (LLVM 24).
71 rounds: 50 accepted, 10 rejected, 3 rolled back. Step TFLOPS geomean (6 DSV4 shapes at 4K) went
from 317 to 912. Shipped as [AMD-AGI/Primus-Turbo#560](https://github.com/AMD-AGI/Primus-Turbo/pull/560). Per-round details are in `tips.md`; this file is the campaign-level view.

## Where the speed came from

1. **Shape-specialized kernels beat tuning the general one.** The largest single jumps came from
   noticing structure in the inputs, not from tuning instructions. Examples: cr0 is a fixed
   128-row window, and cr128 is that window plus a small shared pool. For those, token-block
   kernels load the shared kv rows to LDS once per block and accumulate dKV on chip, with no
   intermediate, sort or gather (rounds 29-32). Before building a faster general kernel, ask
   which inputs a cheaper special case covers.
2. **Matching the hardware model came before micro-tuning.** Rebuilding the forward and the
   backward as one wave per SIMD, 32 heads per wave, TDM gathers and an LDS ring (rounds 17 and
   22) was the step change. Tuning the 8-wave design had plateaued.
3. **Stores matter as much as loads.** Staging outputs through LDS so they leave as contiguous
   rows paid off repeatedly: forward O, the backward C^T intermediate, dQ in every kernel (rounds
   33, 35, 38, 39, 41, 42, 70). Direct 16-byte-per-lane fragment stores cost about 25% of the
   window backward.
4. **Skip work the indices say is dead.** Walking only up to each token's last valid top-k slot
   (rounds 34 and 36) and the lazy O rescale (round 40) were cheap and large for causal cr4.
5. **Remove whole passes, not just cycles.** Delta as rowsum(P * dP) stopped the backward from
   reading O (rounds 44 and 56). A fused dsink kernel replaced 6 torch kernels (round 58). The
   pool-partial sum moved into an existing reduce launch (round 71). Small host-side torch
   epilogues add up to 3-5% of a sub-millisecond kernel.

## Compiler lessons (LLVM 24 / gfx1250)

- **Upgrading the compiler can silently cost 10-20%.** Under expert scheduling, LLVM 24 inserts
  `s_wait_alu depctr_va_vdst(0)` before reloading a register a WMMA has read, which drains every
  outstanding WMMA. Kernels that reloaded operands step by step paid it on every step. The dense
  flash and MLA kernels were immune because they load each phase in bulk. Copying that shape
  into batched loads, one batch ahead, recovered 5-10% in every LDS-fed loop (rounds 66-69).
  Lesson: when a compiler upgrade regresses one kernel but not its siblings, diff their loop
  structure.
- **Batching does not suit long-latency operands.** The same batching made the cr128 backward
  18-36% slower, because its pool fragments come from global memory and the batch grouping
  exposes their latency. Screen each kernel separately.
- **Above 256 VGPRs, register-bank switches are a real cost.** `s_set_vgpr_msb` counts tripled
  in the cr128 backward against cr0. At the 1024-VGPR limit, extra live values become bank
  switches and waits before they show up as spills.
- **A few spills can cost 50%.** Spilled loop-invariant addresses are reloaded from scratch, and
  those reloads wait on loadcnt, which also drains in-flight buffer_loads. Fold constants into
  instruction immediates and wave-uniform parts into soffset, keeping one lane-varying VGPR per
  address stream.
- **Size batches and stages by VGPRs, not intuition.** In every batch screen, one size beyond the
  best one spilled: forward batch 12, w1 backward batch 8, cr128 dQ stage QGW 2.

## Diagnosis lessons

- **Diff instruction histograms against a sibling kernel.** Comparing the hottest loop's
  mnemonic counts against cr0, or against the dense flash kernel, found both of the late big
  issues (the va_vdst waits and the msb switches) in minutes. Raw ISA reading had not.
- **Ablate by deleting work, timing only.** Variants that remove a piece of work (results wrong,
  timing valid) bound what each component can be worth before anything is built. That stopped
  several dead ends: pool K staged in LDS was free, the barriers were free, and the real cost of
  cr128 was the dQ stores, not the pool math.
- **Check the arithmetic before proposing a restructure.** The idea of splitting heads to cut
  dKV accumulators was wrong: those accumulators are rows x 512 / 4 SIMDs no matter how heads
  are split. Writing out the register budget first would have caught it before it reached the
  user.
- **A per-shape performance model makes the priorities objective.** The ratio of modeled to
  measured time per shape showed which kernels to work on (pro cr4, then pro cr128 backward) and
  stopped effort on shapes already ahead of the model.

## Measurement lessons

- **Interleave variants and use min or median.** Small shapes swing ±15% from run to run, and
  machine drift of ±1.5% per day (round 10) is larger than many real wins. Comparing A and B in
  one process, alternating order, was the only reliable screen.
- **Check bit-exactness against the previous round's snapshot.** Most optimizations here change
  no math. A bitwise comparison against `rounds/round-N/kernel_snapshot` catches races and
  indexing bugs that SNR gates miss, and confirms when a code path is untouched (identical ISA).
  When summation order does change, report the relative difference.
- **Benchmark the full set, report the PR numbers.** The campaign bench (2 runs, 8 shapes) gates
  each round. The PR's `bench_sparse_mla_turbo.py` run (Timer mean, 18 shapes) is the stable
  number for comparisons and reviewers.
- **Rebaseline after moving nodes.** Two nodes of the same model differed by about 2%. Without
  the rebaseline, round 66 would have looked 2% better than it was.

## What didn't work (and why)

- Paired waves sharing K/V fragments, FA3-style softmax under QK, the aiter OPUS TDM loader, and
  thresholded lazy rescale (rounds 15, 16, 18, 19): wins on other GPUs or kernels, neutral or
  slower here. Ideas don't transfer by default; measure on this hardware.
- Async global-to-LDS copies (round 13) faulted under expert scheduling mode.
- Persistent workgroups for ragged or short work (rounds 25 and 64) lost to the straight-line
  kernels: token-to-token handoff overhead outweighed the overlap.
- Replacing torch sort or searchsorted with custom kernels (rounds 27 and 62): faster at one
  size, slower at another, or noise. Library primitives were good enough there.

## Process lessons

- **One variable per round, with a snapshot, kept the history usable.** Every accepted change
  can be bisected, and rejected ideas are recorded with their numbers, so nobody re-tries them.
- **Gate changes to the shapes they help.** Several changes helped one band and hurt another
  (batching cr0 vs cr128, the cr128 copy-out addressing). Keeping the other path's code, and
  its ISA, identical made those rounds safe to accept.
- **Shared nodes:** use only the assigned GPU, check others' processes before starting, and put
  the node back as found (container stopped, Docker daemon back to its original state).
- **Credentials and publishing:** never echo remote URLs that embed tokens (mask them), push
  over SSH only, confirm before pushing, keep the PR a single squashed commit, and back up a
  diverged remote branch before force-pushing.

## Open gaps (end of round 71)

- cr128 at 8K and 16K (pool > 32 rows) falls back to the general backward, the largest gap to the
  performance model. Extending the window kernel past 32 pool slots needs a VGPR home for the
  pool C^T, which is the same wall as below.
- The pro cr128 4K backward is at 1024 VGPRs: window dKV 576 + pool C^T 128. The remaining lever
  is splitting dKV columns across two workgroups, which roughly doubles the S / dP work.
- pro cr4 is still somewhat behind the model, most at 16K, in both forward and backward.
