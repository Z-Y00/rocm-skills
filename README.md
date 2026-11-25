# rocm-skills

Agent skills and kernel-optimization knowledge for AMD GPUs (ROCm), packaged so a coding agent
(Cursor, Claude Code, or similar) can drive kernel performance campaigns on its own.

**Install by symlink (the default for this repo):** clone it and run `./install.sh`. Every skill
and rule is linked into `~/.cursor/` (add `--claude` for `~/.claude/skills`, or `--project DIR`
for one repository), so `git pull` updates them in place. Do not copy the files; see
[Installing](#installing).

The skills and rules come from [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)
(`.cursor/skills`, `.cursor/rules`). On top of them, this repo adds what we learned optimizing
the DeepSeek-V4 sparse-MLA forward and backward kernels in FlyDSL on **MI455X (gfx1250)**
([AMD-AGI/Primus-Turbo#560](https://github.com/AMD-AGI/Primus-Turbo/pull/560)). Over 71 rounds
the training-step throughput went from 317 to 912 TFLOPS.

## Layout

| Path | What it is |
|---|---|
| `rules/iteration_rules.mdc` | Hard rules for a campaign: one variable per round, correctness first, full validation, accept/rollback lineage, gains must transfer to a real training step |
| `skills/kernel-optimize/` | The optimization loop, plus a knowledge base by op (`gemm`, `attention`), hardware (`gfx942`, `gfx950`, `gfx1250`), backend (`flydsl`, `triton`) and numerics (batch invariance) |
| `skills/primus-turbo-develop/` | Primus-Turbo project skill: architecture, accuracy and performance verification, profiling, handoff to `kernel-optimize`. It is an example of the project-skill contract that `kernel-optimize` expects |
| `skills/rccl-sdma/` | Running, verifying and measuring RCCL copy-engine (SDMA) collectives on MI355X: how RCCL picks the copy-engine path, Primus switches for Megatron, TorchTitan and Flux, multi-node settings, proving the copy engines ran, measurement method, debugging hangs, and building RCCL from a PR into the training image |
| `skills/primus-projection/` | Primus performance / memory projection: simulate vs benchmark mode, out-of-tree GEMM backends and hardware profiles (no unreleased specs in Primus), runtime patches for architectures the projector does not model, hardware calibration including the FlyDSL matrix-core clock, and the measured-vs-projected roofline ratio for a training proxy |
| `gpu-lock` (from [lhzhang333/gpulock](https://github.com/lhzhang333/gpulock)) | Fair per-GPU queueing on shared nodes (`gpulock run / status / hold / exclusive / history`). Not vendored: `install.sh` clones it to `~/.local/share/gpulock` (SSH, HTTPS fallback; private, needs read access), links `~/.local/bin/gpulock` and links the clone as the `gpu-lock` skill; gpulock updates its clone by itself |
| `skills/shared-gpu-node/` | Practices for agents running GPU work on a shared remote node over ssh: background runs under gpulock, container mounts, compile checks before queueing, flaky-link copies, snap docker paths, noise between sessions |
| `historical_experience/gfx1250/grouped_gemm_bf16/flydsl/` | `tips.md` from tuning the FlyDSL bf16 grouped GEMM for Muse Spark expert shapes (tile walk order, two workgroups per CU for fp32-accumulate wgrads, per-op LDS co-scheduling, tolerant row-count key lookup) |
| `historical_experience/gfx1250/sparse_mla_fwd_bwd/flydsl/` | `tips.md` (per-round signal / takeaway entries) and `lessons_learned.md` (campaign retrospective) |
| `historical_experience/gfx950/paged_attention/triton/` | `tips.md` from the OneTensor batch-invariant paged attention work on MI355X |
| `tools/` | ISA analysis helpers for FlyDSL IR dumps: `isa_loop_hist.py` (hot-loop mnemonic histogram and diff), `isa_seq.py` (run-length op sequence) |

New in this repo:

- [`skills/kernel-optimize/knowledge/hardware/gfx1250/`](skills/kernel-optimize/knowledge/hardware/gfx1250/overview.md):
  the gfx1250 execution model, the differences from CDNA, a porting checklist, and
  implementation notes (register file, WMMA operand streams, the LLVM 24 scheduling behavior,
  TDM, LDS, stores, barriers, measuring). Only relative guidance: measure your own roofline.
- [`skills/kernel-optimize/knowledge/ops/attention/math-optimizations.md`](skills/kernel-optimize/knowledge/ops/attention/math-optimizations.md):
  algebraic rewrites for attention forward and backward, including delta from P∘dP, lazy
  rescale, exp2 folding, the MLA shared-latent dKV, closed-form band masks, on-chip dKV
  accumulation, deterministic reductions, the sink gradient and valid-slot loop bounds.
- The gfx1250 sparse-MLA historical experience and the `tools/` scripts.
- [`skills/kernel-optimize/knowledge/numerics/batch-invariance.md`](skills/kernel-optimize/knowledge/numerics/batch-invariance.md):
  how to keep a kernel bitwise stable across batch shapes and between inference and training
  (aligned split-K / split-KV, split vs fused modes, exact online-softmax no-ops, masked loads, and
  the measured gfx950 rule that launch layouts with warps on both tile axes change bits).
- [`skills/kernel-optimize/knowledge/numerics/rl-kernel-pitfalls.md`](skills/kernel-optimize/knowledge/numerics/rl-kernel-pitfalls.md):
  common pitfalls when an RL rollout engine and a trainer share a kernel: gradients from a rounded
  output, hidden atomics, work and recompiles driven by a HIP graph's length bound, contiguous-only
  inputs, `out=` buffers, position conventions, and end-to-end tests that miss the real paths.
- [`skills/kernel-optimize/knowledge/backend/triton/`](skills/kernel-optimize/knowledge/backend/triton/overview.md) (overview, optimization directions, pitfalls)
  and the gfx950 paged-attention tips (`historical_experience/gfx950/paged_attention/triton/`).

## Installing

Install as **symlinks** into this checkout, so `git pull` updates every installed copy and edits
made through an installed path land in the repo:

```bash
git clone https://github.com/Z-Y00/rocm-skills.git && cd rocm-skills
./install.sh                  # ~/.cursor/skills/<skill> and ~/.cursor/rules/*.mdc -> this repo
./install.sh --claude         # also ~/.claude/skills/<skill>
./install.sh --project DIR    # DIR/.cursor/skills, DIR/.cursor/rules, DIR/agent/historical_experience
./install.sh --no-gpulock     # skip the gpu-lock skill (gpulock clone)
```

An existing entry that is not a symlink is moved to `<target>/.backup-<timestamp>/`, never
deleted. In `--project` mode a real `agent/historical_experience` directory (e.g. Primus-Turbo's
own) is kept; merge its tips into `historical_experience/` here by hand. A project can also add
this repo as a git submodule and point the same symlinks at it.

`kernel-optimize` links to `../../rules/iteration_rules.mdc` from its `SKILL.md`; with symlinks
that resolves to this repo's `rules/`. For a project other than Primus-Turbo, write a project
skill that provides the prerequisites listed in `skills/kernel-optimize/SKILL.md` (kernel path,
focused test and benchmark commands, scoring, execution mode), using `primus-turbo-develop` as
the template.

## License

MIT; see [`LICENSE`](LICENSE). The skills and rules are from Primus-Turbo (Copyright Advanced
Micro Devices, Inc.).
