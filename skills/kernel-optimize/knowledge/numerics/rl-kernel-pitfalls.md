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
the same kernel passed every schedule-invariance test on Triton 3.9 and failed 8 of 226 on an
integrator's Triton 3.7 (`BLOCK_M=128, num_warps=2`; fp16 split decode). Test on the Triton version
the integration ships, restrict schedule options to verified ones, and record the version in debug
dumps (see `batch-invariance.md`, "Toolchain").
