# Common Pitfalls in RL Kernel Development

Read this before shipping a kernel that an RL stack runs on both sides: a rollout engine (vLLM /
SGLang, paged KV cache, HIP graphs) and a trainer (packed varlen sequences, autograd). The numerics
of bitwise train/infer match are in [`batch-invariance.md`](batch-invariance.md); this page collects
what went wrong when a batch-invariant attention kernel (Primus-Turbo RL attention, Triton and
FlyDSL backends, MI355X) met a real integration. Each item: the symptom, the cause, the fix.

## Gradients

**A gradient computed from the saved low-precision output.** The sink gradient
`dsink_h = -sum_t exp(sink_h - lse_th) * (dO . O)_th` was formed in torch from the bf16 forward
output: 1.5e-3 relative error against fp32 (vs 3e-7 achievable). Any gradient term that contracts
`dO` with `O` inherits `O`'s rounding. The backward's dQ pass already holds the exact quantity in
fp32: `dO . O = sum_j P_j * dP_j` over the row's keys. Store that per row and reduce from it.

**Atomics hidden in inherited code paths.** The upstream (aiter-derived) backward had the exact
in-kernel `dsink`, but accumulated it with `tl.atomic_add`, so it was disabled and the imprecise
torch formula used instead. Keep the exact per-row value, store it with a plain store into a
`[tokens, heads]` buffer, and reduce over tokens in one fixed-shape reduction: exact and
deterministic. Grep a forked kernel for `atomic_` before trusting its determinism claims.

**Tolerances that hide the bug.** A 5e-2 allclose on all gradients passed with the 1.5e-3 `dsink`
error. Give small or cancellation-prone terms (sinks, biases, scales) their own check: relative
norm error against an fp32 reference, at a bound the exact method meets (1e-4 here).

## Serving engines: HIP graphs and capture-time bounds

A graph is captured once with `max_seqlen_k = max_model_len` and replayed at every real length.
Anything sized from `max_seqlen_k` on the host is then paid at the bound, not the real length.

**Work that follows the bound.** Split-KV decode with a merge pass looping over
`cdiv(max_seqlen_k, segment)` segments, and split programs past a sequence's end still writing
empty partials: FlyDSL decode at 4k context ran up to 3.55x slower in a graph captured at 128k.
Loop over the sequence's own segment count read on the device, and let programs past its keys exit
before any store. Both are bit-exact when an empty segment merges as an exact no-op.

**Schedule heuristics fed by the bound.** An automatic "segments per split program" chosen from the
program count at `max_seqlen_k` picked 4 segments per program at 128k, collapsing a 4k sequence to
one program per (sequence, KV head) (32 programs on 256 CUs): still 2.95x after the work fix. Size
the grid from the batch and the CU count, and compute each program's run of segments on the device
from the live length, keeping one partial slot per segment so the merge order (and the bits) stay
fixed. The workspace stays sized by the bound; that costs memory only.

**Do not copy inference kernels' fix.** aiter's unified attention fixes `NUM_SEGMENTS` per launch
and computes `tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS * TILE)` on the device: graph friendly,
but segment boundaries move with the length, so the reduction order changes every decode step and
rollout no longer matches training. Keep fixed absolute segments; move only the assignment of
segments to programs.

**Compile-time constants derived from a length.** A Triton `tl.constexpr NUM_SEGMENTS =
cdiv(max_seqlen_k, 1024)` (used only as a buffer stride) recompiled split decode every 1024 keys of
context in eager mode. Pass length-derived values as runtime arguments, with
`@triton.jit(do_not_specialize=[...])`: Triton otherwise still specializes integer arguments on
`== 1` and divisibility by 16.

**A JIT launch cache keyed on runtime values.** The FlyDSL launcher cached compiled launchers in a
dict keyed on its scalar arguments (batch, max lengths, strides, split programs), all runtime
`Int32`s. Every eager decode step with a new `max_seqlen_k` missed it and called `flyc.compile`
again: FlyDSL's own cache made that ~0.2 ms rather than a real compile, but it was paid per launch
(twice per step with split + merge), 0.23-0.42 ms per step against a 40-50 us kernel. Key a launch
cache only on what changes the generated code (here everything compile-time is in the module's
traits, so the key is just the stream argument). A fresh value that compiles in ~0.2 ms is the tell
that the IR does not depend on it. Guard with a test that counts compiles across steps of new
lengths and batch (monkeypatch the compile entry point) and requires zero after the first.

**Early-exit programs are cheap; restructuring for them may not pay.** Triton split decode with one
program per fixed segment and a grid sized by `max_seqlen_k` cost at most 1.16x under a 128k graph
at 4k context: programs past a sequence's keys exit after a few loads. Moving it to FlyDSL's scheme
(fixed programs, a runtime run of segments per program) made eager decode up to 13 us slower
(batch 4 x 4k 27.8 -> 41 us; 31 us with the first segment peeled out of the loop; `num_stages=1` on
the outer loop changed nothing) and was no faster under the graph. The cost under a graph comes from
work and schedule choices that follow the bound (above), not from idle programs. Measure the
captured case against a capture at the real length before restructuring a grid.

**What a captured graph freezes, and the one condition for safe replay.** Host code runs once, at
capture: mode (fused / split), grid, program counts, workspace sizes and launch configuration are
all fixed from the capture-time arguments; replays only change device memory (`q`, `seq_lens`,
`cu_seqlens_q`, block table, KV cache). A replay at a growing length is therefore correct as long
as the kernels read the live length on the device and every real length stays at or below the
capture-time `max_seqlen_k`. Past it the result is wrong, not slow: segments beyond the grid are
never computed, and a device-computed run can index past a workspace sized by the bound (an
out-of-bounds write). A host check of `seq_lens.max()` syncs and cannot run inside a capture; clamp
on the device or document the contract. The caller must also keep the same buffers (the graph holds
their pointers) and a fixed batch shape (engines capture per padded batch size).

**The grid never changes the bits.** Only the reduction order does: tile size, absolute segment
boundaries, backend. Each segment starts from an empty state and walks its tiles in key order; the
segments' fp32 partials merge in segment order; sinks and normalization come last. Fused mode does
that in one program, split mode in any assignment of segments to programs with one slot per segment.
So the grid, program count, run grouping, mode and workspace may all follow `max_seqlen_k` while a
rollout still matches the training forward bitwise. Prove it with a graph captured at a large bound
and replayed at shorter lengths, bitwise against eager calls at the true length.

**Caches keyed on a tensor's version counter.** A mask cached the key starts it derived from
`seq_lens`, keyed on the tensor's identity and `seq_lens._version`. vLLM builds its attention
metadata under `torch.inference_mode()`, and inference tensors have no version counter: reading
`_version` raises "Inference tensors do not track version counter", so every masked paged call
failed (unmasked calls never touched the cache; the trainer runs under `no_grad`, which is fine).
The cache was also unsafe under HIP graphs: its check runs only on the host, so starts cached
before capture leave the cumsum out of the graph, and replays read stale starts after the engine
updates `seq_lens` in place. Compute such small derived tensors on every call (they are captured
into the graph), or inside the kernel's own metadata launch. Run the masked paths under
`torch.inference_mode()` in tests.

## Inputs and outputs as the integrations pass them

**Requiring contiguous inputs.** A FlyDSL wrapper asserted contiguous K / V and copied Q, while the
trainer passes Q / K / V as slices of a fused QKV projection (token stride `(Hq + 2 Hkv) * D`): one
extra copy per layer. The kernel already took a token stride. Accept any view whose heads are packed
per token (head stride `D`, unit last stride, rows 16-byte aligned) by passing its strides; give the
output its own stride if it used to share the input's; copy only layouts the kernel cannot address
(and say so in the docs) rather than raising. Test fused-QKV views bitwise against contiguous copies,
forward and backward (gradients land in the fused tensor).

**No `out=`.** vLLM passes a preallocated output buffer, padded rows included (`[rows, H * D]`,
`rows >= num_tokens`). Write the first `num_tokens` rows in place and leave the rest untouched. A
`torch.library.custom_op` cannot return a tensor aliasing an input: add a sibling op with
`mutates_args=("out",)` returning nothing. Under `torch.compile` the mutation is auto-functionalized;
Inductor usually re-inplaces it, otherwise it inserts a copy.

**Position conventions.** Engines track positions per sequence (0 = the sequence's first key); a
kernel API in global key indices forces a per-step conversion on the caller. Offer the sequence-
relative form at mask construction and convert once per step on the device (token -> sequence with
`searchsorted` on `cu_seqlens_q`, no host sync, graph safe), cached on the mask so every layer reuses
it. Kernels that already subtract each sequence's base need no change.

## Tuning heuristics on real shapes

**Find the factor before writing the rule.** Triton split decode beat fused by 1.1-1.7x at batch
64-256, context 4k-32k for gpt-oss-120b full attention (64 query / 8 KV heads, D 64) while Llama-3
8B (32 / 8, D 128) was best fused. Head size and query heads per KV head changed together, so add a
shape that separates them: Qwen3-32B (64 / 8, D 128) also gained up to 1.46x, and a D 64 shape with 4
query heads per KV head gained less. The rule became "split when 8+ query heads per KV head or
D <= 64", not "split at D 64". The old rule (split only when the fused grid has under 2 programs per
CU) never fired at these batches.

**Measurement order.** In a sweep timing several configurations back to back per shape, the first
configuration of each backend ran 2-7% slower (FlyDSL `MODE=auto` looked slower than `MODE=fused`
although both chose the same kernel). Repeat the first configuration at the end of the row, or
randomize the order, before calling a heuristic wrong.

**What a tuned inference kernel does with `max_seqlen_k`.** aiter's unified attention never passes
it to a kernel: the host uses it only to cap the split count (`min(cdiv(max_seqlen_k, tile), share)`,
share from the CU count and the 2D program count), to choose 2D vs split, and as a key of its tuned
config table; the GPU sizes each segment from the live `seq_len`. The batch-driven split count is the
part to copy; the length-derived segment size is the part that breaks train/infer bit equality.

## Coverage that looks complete but is not

An integrator's end-to-end zero-logprob test passed while covering none of these paths:

| Their test | Path it never exercises |
|---|---|
| Head dim 32 | Production builds (D 64 / 128 compile different code; FlyDSL takes only 64 / 128) |
| Sequences under 128 tokens | Fixed segment boundaries (512 / 1024 keys), long-context split decode |
| 3-5 sequences | The large-batch fused decode path |
| One sequence per forward, `no_grad` | Multi-document packing, the backward |
| `vis_lo` always 0 | Masks with a nonzero lower bound (spans, windows folded into intervals) |

On the kernel side: run rollout-vs-training tests at real model head shapes (e.g. Llama-3 8B
32/8/128, gpt-oss 64/8/64 with window 128 and sinks), at the default segment size as well as a
shrunk one, with chunked prefill crossing segment boundaries, alone and next to another long
sequence, in every split mode. Keep edge shapes (one KV head, a non-power-of-two GQA group, no GQA,
D 256) in a labelled list for the accuracy and mode-bitwise tests, not as stand-ins for models.
Capture a decode graph once at a length bound and replay it for steps of other lengths, bitwise
against eager.

## Toolchain

Bit equality across launch schedules is a property of the generated code, not only of the source:
the same kernel passed every schedule-invariance test on Triton 3.9 and failed many on Triton 3.7
and 3.8 (launch invariance at `BLOCK_M=32, num_warps=1`; paged vs contiguous and split vs fused,
mostly fp16), bf16 and fp16 alike.

**Contraction choice left to the compiler.** With `enable_fp_fusion` on (Triton's default), an
expression with two products and an add, here the split-KV segment merge `L * ca + l * cb` and
`A * ca + a * cb`, may become `fma(L, ca, l * cb)` or `fma(l, cb, L * ca)`: the compiler picks, and
the pick changed with the launch shape on 3.7 / 3.8 (different code layout) while 3.9 happened to be
consistent. Fix by construction: compile the kernels with `enable_fp_fusion=False` and spell every
fused multiply-add you want as `tl.fma`. The cost was within run-to-run noise (decode and training
forward, MI355X). Expressions with one product (`L * alpha + rowsum`) contract one way only, but
spell them too, so nothing depends on the flag.

**How it was found.** Compile the failing pair of launch shapes under the other release and compare
the ISA: both used the same MFMA instruction (16x16x32) for every shape, so the dot products were not
it; the shapes differed only in cross-lane code. Rows whose keys lay in one segment matched and rows
merging several segments did not, which pointed at the merge arithmetic. Shrink the segment size in
the repro so the multi-segment path runs at test sizes.

**Testing other releases from one suite.** One process cannot load two Triton releases. Install each
release in a list (e.g. 3.7, 3.8, 3.9) once with `pip install --no-deps --target <cache>/<version>`
and run the bitwise tests in a child process with that directory first on `PYTHONPATH` (and its own
`TRITON_CACHE_DIR`); check the child really loads that release; skip the running release (the suite
covers it) and skip when the install fails (no network). Split the child runs into groups that fit
the CI per-test timeout. Record the Triton version in debug dumps (see `batch-invariance.md`,
"Toolchain").
