# Triton on ROCm — Pitfalls

Short, measured notes. Versions: Triton 3.4 (local interpreter), Triton 3.8 (rocm/primus:v26.8, MI355X).

- **Global constants in `@triton.jit` code** must be `NAME = tl.constexpr(value)`. Triton 3.8 rejects
  the annotated form `NAME: tl.constexpr = value` at module scope with
  `NameError: Cannot access global variable ... from within @jit'ed function` (older versions accepted
  it). Annotated locals inside a kernel are fine.
- **`TRITON_INTERPRET=1` miscomputes bf16.** On the CPU interpreter (numpy has no bf16), bf16
  attention returned values around 1e8 while the same kernel in fp16 was correct. Use the
  interpreter only with fp16/fp32 for logic checks; run bf16 on the GPU.
- **AMD launch options** (`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`) are not accepted by the
  interpreter; drop them when `TRITON_INTERPRET=1`.
- **int64 offsets** (`.to(tl.int64)` on indices) keep the AMD backend from emitting buffer loads.
  Use 32-bit offsets when the tensor has < 2^31 elements and select the 64-bit path with a constexpr.
- **pytest in a root container on an NFS home** leaves root-owned `.pytest_cache`; run with
  `-p no:cacheprovider`.
