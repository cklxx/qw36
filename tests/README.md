# qw36 — tests/

What each script proves, when to run it, and which CI job picks it up.
**Correctness is top priority** — most of these gate CI; perf gates are
opt-in by model availability.

## TL;DR

```bash
make check                         # inner loop: smoke + CLI + kvcache + goldens + greedy + long-coherence
make perf                          # full bench: precision + compare_mlx short
QW36_TEST_MODEL=<path> make check  # same, with model-dependent gates exercised
```

## Catalog

### Correctness — CI blocking

| script | what it proves | trigger |
|--------|----------------|---------|
| [`quant_fastest_smoke.sh`](quant_fastest_smoke.sh) | `--fast` on 0.8B says "Hello! How can I help you today?" — the canonical end-to-end smoke | every PR |
| [`precision_cpu_vs_metal.sh`](precision_cpu_vs_metal.sh) | step-0 fp32 forward CPU == Metal bitwise on the canonical 0.8B model | macos-metal CI when model present |
| [`correctness_greedy.sh`](correctness_greedy.sh) | per-KV-config token match on raw + chat-template `Hello -n 16` — 8 configs (Q8 default / fp16-KV / bf16-KV / fp32-KV / fused-fp16 / chat-Q8 / chat-fp16 / chat-bf16) | macos-metal CI when model present |
| [`correctness_long_coherence.sh`](correctness_long_coherence.sh) | n=128 essay produces ≥30% lexical diversity, ≤5-run repetition, ≥30 words. Catches degeneracy / token-repeat-collapse. 3 configs (Q8 / fp16 / fused). | macos-metal CI when model present |
| [`cli_smoke.sh`](cli_smoke.sh) | `--doctor` and `--print-config` exit codes; 6 paths covered | every PR (linux-sanitizers + via `make check`) |
| [`golden_kernels.sh`](golden_kernels.sh) | per-kernel fixtures: rmsnorm / silu / swiglu / matmul / rope / qgate / residual_add — rtol 1e-5 | every PR (macos-metal) |
| [`kernel_golden.sh`](kernel_golden.sh) | per-tensor Q8_0 dequant matches `gguf-py` reference (35B-specific) | manual / 35B bring-up |
| [`q8_0_golden.sh`](q8_0_golden.sh) | Q8_0 dequant on 35B's `token_embd.weight` row 9419 + `output.weight` rows match `gguf-py` exactly | manual / 35B-specific |
| [`f16_materialize_audit.sh`](f16_materialize_audit.sh) | `output.weight` row count + byte size after `lazy_materialize_f16` | manual |
| [`mlx_safetensors_smoke.sh`](mlx_safetensors_smoke.sh) | C safetensors reader parses the local MLX 35B-A3B shard and validates representative affine tensor shapes | manual / MLX-loader bring-up |

### KV cache — CI blocking

| script | what it proves | trigger |
|--------|----------------|---------|
| [`kvcache_smoke.sh`](kvcache_smoke.sh) | 2-tier `(ram_lru, disk)` cache: insert / lookup / promotion across processes | every PR (linux-sanitizers) |
| [`kvcache_e2e.sh`](kvcache_e2e.sh) | end-to-end: 2nd run hits cache, `qw36_state_snapshot` / `_hydrate` reproduces logits bit-identical to a fresh prefill | macos-metal CI when model present |

### Performance — gated by model + load

| script | what it proves | trigger |
|--------|----------------|---------|
| [`perf_gate.sh`](perf_gate.sh) | wallclock tok/s within ±10% of `perf_baseline.json` on `(fast, n=64)` and `(fast, n=256)`; retests once on regress | macos-metal CI `perf-gate` job when `QW36_TEST_MODEL` set |
| [`compare_mlx.sh`](compare_mlx.sh) | side-by-side qw36 vs `mlx_lm.generate` median tok/s, `short` / `long` / `full` scopes | manual / `make perf` |
| [`quant_kernel_bench.sh`](quant_kernel_bench.sh) | single-layer quantized matmul microbench (no harness, manual output parse) | manual |
| [`metal_35b_moe_perf.sh`](metal_35b_moe_perf.sh) | 35B-A3B Metal-only MoE wallclock/perf helper | manual |

### Informational — not CI blocking

| script | what it proves | trigger |
|--------|----------------|---------|
| [`e2e_qwen35_smoke.sh`](e2e_qwen35_smoke.sh) | end-to-end `<think>` + ≥3-letter words sniff — currently "informational only, regressed" per `AGENTS.md` notes | manual |

## What each CI job actually runs

`.github/workflows/ci.yml` mapping (commit `688aca1`):

```
linux-cpu                 → cpu build + --help / --info smoke
macos-metal               → cpu + metal builds, golden_kernels.sh,
                             precision_cpu_vs_metal.sh,
                             correctness_greedy.sh,
                             correctness_long_coherence.sh
cuda-build-check          → cuda compile only (no GPU on runner)
linux-sanitizers          → asan + ubsan cpu builds, cli_smoke,
                             kvcache_smoke, sanitizer bounded run
lint                      → clang-format --dry-run on common/*.{c,h}
                             (PR-blocking on drift)
perf-gate                 → macos-latest, only when QW36_TEST_MODEL set
```

## Conventions

- **Skip cleanly when model is missing.** Every model-dependent test
  must short-circuit with a `[name] skip: model not found` message
  and `exit 0` if `QW36_TEST_MODEL` doesn't resolve. CI invokes them
  unconditionally; the script decides if it runs.
- **Stable on noisy hosts.** Perf tests retest under quiet conditions
  before failing; correctness tests use deterministic seeds + greedy
  decoding. See [`docs/performance_methodology.md`](../docs/performance_methodology.md)
  for the ritual.
- **Per-config explicit.** When testing per-KV-dtype or per-kernel
  variants, list each combination by label so failures are actionable.
- **Frozen baselines.** [`tests/perf_baseline.json`](perf_baseline.json) and the
  inline `EXPECTED` strings in `correctness_greedy.sh` are committed;
  updates need a commit body explaining what genuine win produced
  the new baseline.
- **Don't introduce flaky tests.** A test that fails intermittently
  is worse than no test. Pin RNG seeds, use exact-match where
  possible, document expected variance where not.

## Adding a new test

1. Write `tests/<feature>_<scope>.sh`. Start with the boilerplate
   (`set -euo pipefail`, model-existence guard, build guard).
2. If model-dependent, gate behind `QW36_TEST_MODEL`. If model-free,
   document why.
3. Add to `make check` (Makefile) if it's PR-grade.
4. Add to `.github/workflows/ci.yml` if it should be a hard CI gate.
5. Document the new entry in the table above.
6. Commit message body shows local pass + how to reproduce.

## See also

- [`docs/performance_methodology.md`](../docs/performance_methodology.md) — bench discipline.
- [`docs/architecture.md`](../docs/architecture.md) — engine internals.
- [`AGENTS.md`](../AGENTS.md) §0 — correctness-first project contract.
- [`ROADMAP.md`](../ROADMAP.md) — what's coming next.
