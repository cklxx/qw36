# Contributing to qw36

qw36 is a pure-C inference engine for Qwen 3.5/3.6 with three GPU
backends. It's a small, opinionated codebase — please read this once
before sending a PR.

## TL;DR

1. `make all && make -C cpu test` should be green on your machine.
2. Run `tests/quant_fastest_smoke.sh` (and `tests/precision_cpu_vs_metal.sh`
   on macOS) before pushing.
3. Every perf-affecting commit needs bench numbers in the commit body.
4. One file = one topic; no half-finished migrations.
5. The full ground rules live in [`AGENTS.md`](AGENTS.md) (§0 SOLID,
   editing, backend isolation, bench discipline, delegation). Read it.

## Repository shape

```
qw36/
├── common/             engine, vtable, CPU reference
├── cpu/                CPU backend stub + Makefile → ./qw36_cpu
├── metal/              Apple Metal backend → ./qw36_metal
├── cuda/               CUDA backend → ./qw36_cuda (off-host build)
├── amd/                AMD HIP backend → ./qw36_amd (off-host build)
├── docs/               design notes, post-mortems, codex briefs
├── tests/              smoke + correctness + bench scripts
└── tools/              dump_tensor, install.sh, gen_goldens
```

CPU is the reference: every GPU backend must produce numerically
equivalent output. CPU lives in `common/`; backends call into the
backend vtable defined in `common/qw36_gpu.h` (ABI is **frozen**).

## Building

```bash
make cpu                   # always works
make metal                 # macOS Apple Silicon
make cuda                  # needs nvcc; will compile-check on this host but
                           # requires a GPU host to run
make amd                   # needs hipcc; same caveat
make all                   # soft-builds whichever toolchains are present
```

If you don't have a GPU toolchain installed locally, `make all` will
quietly skip those backends. Use `--doctor` (once shipped) or
`./qw36_cpu --info` to confirm what backend ran.

## Test workflow

| script | what it proves | when to run |
|--------|----------------|-------------|
| `tests/quant_fastest_smoke.sh` | `--fast` on 0.8B says "Hello! How can I help you today?" | every commit |
| `tests/precision_cpu_vs_metal.sh` | step-0 fp32 bitwise CPU == Metal | every Metal change |
| `tests/kernel_golden.sh` | per-kernel goldens vs CPU reference | every kernel change |
| `tests/compare_mlx.sh [scope]` | side-by-side decode tok/s vs MLX | every perf-affecting change |
| `tests/kvcache_smoke.sh` | KV cache tiers round-trip | every kvcache change |

Run them ALL before opening a PR that touches kernels:

```bash
tests/quant_fastest_smoke.sh
tests/precision_cpu_vs_metal.sh
tests/kernel_golden.sh
```

## Perf discipline (read this twice)

Every commit that *could* affect tok/s carries data in the message.

- Quote machine load: `uptime | awk -F'load averages:' '{print $2}'`.
- Use a 3- or 5-run median. Single numbers are noise.
- **Wallclock tok/s is ground truth**, not `gpu_ms`. Persistent
  encoder is on by default and `gpu_ms` accumulators overcount.
- Test the env combo a user would actually run (`--fast`), not the
  combo that makes your change look best.
- Negative results are commits too. Land them with a
  `docs/<thing>_failed.md` entry so we don't try the same thing twice.

See `docs/performance_methodology.md` (when it lands) for the full
ritual. Until then, the failed-experiment files in `docs/` (`q4k_qmv_quad_failed.md`,
`fp16_state_root_cause.md`) are the templates.

## Editing rules

- **Preserve** as the default. Don't rewrite a kernel because you
  don't like the variable names; don't delete an opt-in env flag
  because it's "no longer needed" — opt-in research artifacts are
  intentional. Diff history is documentation.
- **No half-finished migrations.** If your PR converts the matmul
  vtable to a new signature, the same PR must convert all three
  backend stubs even if one is a no-op.
- **Backend isolation.** Common code (`common/*.c`) does NOT
  `#ifdef __APPLE__`. Backend-specific behaviour goes through the
  vtable.
- **No new dependencies.** mmap, pthread, and the Metal/CUDA/HIP
  frameworks are the floor. No JSON parsers, no logging libraries,
  no C++ in `common/`.

## Codex / Claude division of labour

This repo is built by two collaborators:

- **Claude** owns direction, integration, and final commits — the
  small diffs, the docs, the bench wrangling.
- **Codex** owns deep execution — large kernel rewrites,
  refactor passes, multi-file ABI changes.

Briefs live under `docs/briefs/`. A new brief should be small enough
that codex (or any contributor) can land it without a synchronous
clarification round. If the brief is bigger than ~200 lines of
prose, you've written a design doc, not a brief — file it under
`docs/` instead.

## Commit style

Commitizen-ish prefixes; mandatory scope tag.

```
metal: Q6K_SCALE16_MLX default-on (3-4% short, 3% sustained)

The MLX-style Q6K qmv variant landed opt-in in commit 7550375 with …
[bench numbers, env, REPEAT count, machine load]
```

Scopes: `metal`, `cuda`, `amd`, `common`, `tokenizer`, `gguf`,
`docs`, `tests`, `tools`, `chore`.

Push to `main` directly (small project, single author). No feature
branches. If something needs >1 commit, land them in sequence with
verification in between.

## Reviews

PRs to `main` are squash-merged or fast-forward. The PR description
should reproduce the headline number / behaviour change verbatim
from the commit body so reviewers don't have to dig.

If you're proposing a vtable / ABI change, file a design doc first
(`docs/<feature>_design.md`) and link it from the PR. The frozen
contract in `common/qw36_gpu.h` is the spine of the project; we
don't change it casually.

## What we don't accept

- PRs that add dependencies without prior discussion.
- PRs that disable existing tests "because they were flaky" — fix
  the flakiness or document it as a known limitation.
- PRs that change perf numbers without bench data in the message.
- PRs that delete failed-experiment docs because "it's been a while."

When in doubt, open a draft PR with the commit message and let the
maintainers tell you which of the rules above might bend.
