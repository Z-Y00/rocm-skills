# rocm-skills

Agent skills and kernel-optimization knowledge for AMD GPUs (ROCm), packaged so a coding agent
(Cursor, Claude Code, or similar) can drive kernel performance campaigns on its own.

The skills and rules come from [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)
(`.cursor/skills`, `.cursor/rules`). On top of them, this repo adds what we learned optimizing
the DeepSeek-V4 sparse-MLA forward and backward kernels in FlyDSL on **MI455X (gfx1250)**
([AMD-AGI/Primus-Turbo#560](https://github.com/AMD-AGI/Primus-Turbo/pull/560)). Over 71 rounds
the training-step throughput went from 317 to 912 TFLOPS.

## Layout

| Path | What it is |
|---|---|
| `rules/iteration_rules.mdc` | Hard rules for a campaign: one variable per round, correctness first, full validation, accept/rollback lineage, gains must transfer to a real training step |
| `skills/kernel-optimize/` | The optimization loop, plus a knowledge base by op (`gemm`, `attention`), hardware (`gfx942`, `gfx950`, `gfx1250`) and backend (`flydsl`) |
| `skills/primus-turbo-develop/` | Primus-Turbo project skill: architecture, accuracy and performance verification, profiling, handoff to `kernel-optimize`. It is an example of the project-skill contract that `kernel-optimize` expects |
| `historical_experience/gfx1250/sparse_mla_fwd_bwd/flydsl/` | `tips.md` (per-round signal / takeaway entries) and `lessons_learned.md` (campaign retrospective) |
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

## Installing Into a Project

The skills expect this layout in the target repository:

```bash
mkdir -p .cursor/skills .cursor/rules agent
cp -r rocm-skills/skills/* .cursor/skills/            # or .claude/skills/
cp rocm-skills/rules/*.mdc .cursor/rules/
cp -r rocm-skills/historical_experience agent/        # read by kernel-optimize (READ_HISTORICAL_TIPS)
```

`kernel-optimize` links to `../../rules/iteration_rules.mdc` from its `SKILL.md`, which resolves
both in this repo and under `.cursor/`. For a project other than Primus-Turbo, write a project
skill that provides the prerequisites listed in `skills/kernel-optimize/SKILL.md` (kernel path,
focused test and benchmark commands, scoring, execution mode), using `primus-turbo-develop` as
the template.

## License

MIT; see [`LICENSE`](LICENSE). The skills and rules are from Primus-Turbo (Copyright Advanced
Micro Devices, Inc.).
