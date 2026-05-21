# qw36 — Roadmap

**Snapshot, 2026-05-21.** `--fast` profile delivers **204 tok/s short / 176
sustained / 92 tok/s @ n=2048** on Qwen3.5-0.8B-Q4_K_M on Apple Silicon
— **84% of MLX short, 31% of MLX at n=1024 essay**. The fp16 path is
ceiling-bound; the quant path has two unfinished kernel-level levers
(flash-attn decode, bf16 KV). 35B-A3B is functional on CPU; Metal
full-GPU is in flight (task #74, codex). KV prefix cache **skeleton**
landed (tier-composing, two tiers shipped: `ram_lru` + `disk`); engine
wiring is the next gate to long-context wallclock wins. CUDA/AMD
backends are compile-checked only.

The next 4–8 weeks fall into five themes. Each row has owner / effort /
dependency / definition-of-done. The bottom of the doc lists the **next
5 PRs** in priority order — that's the actionable bit.

## Themes

### Theme 1 — Correctness & coverage

> Every model we claim to support produces coherent text, and we have
> an automated way to prove it.

| id | task | owner | effort | dep | DoD |
|----|------|-------|--------|-----|-----|
| 1.1 | 35B-A3B Metal full-GPU close-out (#74 AC) | codex | 3–5 d | — | Metal output == CPU on greedy `Hello -n 32 --no-special`; `tests/precision_cpu_vs_metal.sh` extended to MoE; smoke in CI. |
| 1.2 | CUDA/AMD parity catch-up (#26) | claude orch + codex port | 5–8 d | remote GPU host | `make cuda && make amd` build; `tests/precision_cpu_vs_cuda.sh`; perf ladder column for one CUDA SKU. |
| 1.3 | Golden vector harness (#14, #27) | claude | 2–3 d | — | One `tests/golden_*.sh` per kernel category; CI macos-metal runs them; `tools/gen_goldens.c` regenerates deterministically. |
| 1.4 | Kernel goldens under CUDA/AMD | codex on GPU host | 1–2 d | 1.2, 1.3 | All three backends green in CI matrix. |
| 1.5 | Model-coverage matrix doc | claude | 1 d | 1.1 | `docs/model_support_matrix.md` green/yellow/red per (model, backend); linked from README. |

### Theme 2 — Performance

> Long-context (n≥2048) wallclock parity with MLX, or a documented
> reason why not.

| id | task | owner | effort | dep | DoD |
|----|------|-------|--------|-----|-----|
| 2.1 | **KV cache engine wiring (#75 AD)** — skeleton landed; `qw36_state_snapshot` / `_hydrate` + cache consult on `qw36_prefill`. | claude | 3–4 d | — | `tests/kvcache_e2e.sh`: second run on the same prompt skips prefill (TTFT drops to ~0); hit/miss counters in `--info`; disk tier round-trips a process restart. |
| 2.2 | **Flash-attn single-pass decode (#71 Z)** — fuse score / online softmax / V combine. | codex kernel + claude integration | 5–7 d | 1.3 goldens | n=2048 essay throughput 92 → 130+ tok/s; precision smoke unchanged; `QW36_METAL_FLASH_ATTN=1` becomes default after a 3-run wallclock gate. |
| 2.3 | bf16 KV cache (#73 AB) | codex | 3–4 d | 1.3 goldens | `QW36_METAL_BF16_KV=1` opt-in passes smoke; n=2048 tok/s matches or beats fp16 KV with no logit drift. |
| 2.4 | Auto-KV-transposed under `--fast` (#70 Y) | claude | ½ d | — | `QW36_METAL_KV_TRANSPOSED=auto|0|1`, default `auto`, flips at `seq_capacity > 512`. |
| 2.5 | Dispatch-count reduction (next J after gate_up) | codex | 2–3 d | — | −10% dispatches at decode steady state; +3–5% wallclock or "no win" doc. |
| 2.6 | Speculative decoding **design doc only** | claude | ½ d | 2.1 | `docs/speculative_decode_plan.md`. No code. |

### Theme 3 — Product

> A new user can install qw36, run an unknown model, and get a clear
> error or a working result. No silent failures.

| id | task | owner | effort | dep | DoD |
|----|------|-------|--------|-----|-----|
| 3.1 | `--doctor` mode | claude | 1–2 d | — | SDK/model/GPU/env conflict checks; OK/WARN/FAIL per line; non-zero exit on FAIL. |
| 3.2 | Model auto-fetch (`--fetch <name>`) | claude | 2 d | — | Curated registry (0.8B-Q4_K_M, 35B-A3B-Q4_K_XL, +2 more) with size+sha256; cached under `~/.cache/qw36/models/`. |
| 3.3 | CLI knob consolidation + `--print-config` | claude | 1–2 d | — | Effective profile + every flag's value + provenance; README env table trimmed to ~6 user-facing + research appendix. |
| 3.4 | Error-message audit | claude | 1 d | 3.1 | Every `fprintf(stderr, …)` and `qw36__die` follows what-failed / why / next-step format. |
| 3.5 | Installer polish (`--uninstall`, sha pin, source-build hint) | claude | ½ d | — | README quick-start works end-to-end on vanilla M-series Mac and vanilla Linux. |

### Theme 4 — Engineering quality

> A contributor PR can't break the perf ladder or smuggle in a UB
> regression without CI catching it.

| id | task | owner | effort | dep | DoD |
|----|------|-------|--------|-----|-----|
| 4.1 | CI matrix expansion (macOS `--fast` smoke, ASAN/UBSAN cpu, clang-tidy) | claude | 2 d | 4.5 | 5+ CI jobs; < 10 min per PR. |
| 4.2 | Sanitizer build targets | claude | 1 d | 4.1 | `make -C cpu asan ubsan` build; CI green; baseline committed. |
| 4.3 | Perf-regression gate | claude | 2–3 d | 4.1, 4.5 | `tests/perf_gate.sh` + `tests/perf_baseline.json`; CI fails on > 5% regression (3-run median, retest on noise). |
| 4.4 | clang-tidy (`bugprone-*` + `cert-*`) | claude | 1 d | — | `.clang-tidy` + baseline `tools/clang_tidy_baseline.txt`. |
| 4.5 | clang-format | claude | ½ d | — | `.clang-format` committed; CI runs `clang-format --dry-run -Werror`. |

### Theme 5 — Documentation & community

> A competent C engineer can clone, read for an afternoon, and submit
> a useful PR.

| id | task | owner | effort | dep | DoD |
|----|------|-------|--------|-----|-----|
| 5.1 | Architecture deep-dive | claude | 2 d | 2.1 | `docs/architecture.md`: vtable, lazy weight materialization, per-token forward, fusion flags, persistent encoder, KV layout. |
| 5.2 | Model-support matrix | claude | (in 1.5) | 1.5 | (in 1.5) |
| 5.3 | Troubleshooting / FAQ | claude | 1–2 d | 3.4 | `docs/troubleshooting.md`; linked from `--doctor` output. |
| 5.4 | CONTRIBUTING.md | claude | 1 d | 4.1–4.5 | One walked-through external contributor PR using only the doc. |
| 5.5 | Performance methodology doc | claude | 1 d | — | `docs/performance_methodology.md`: median-of-N, wallclock not gpu_ms, load discipline. |

## Next 5 concrete PRs in priority order

These are the ones to land in this exact sequence.

### PR #1 — `common: qw36_state_snapshot / _hydrate + KV prefix cache wired into qw36_prefill` (task 2.1)

Skeleton has been in tree for two commits with no consumer. Wiring it
converts the project from "fast decoder" into "fast server-able
decoder". Single biggest product-level uplift in the window.

Verification: `tests/kvcache_e2e.sh` — second invocation on the same
prompt skips prefill; `tests/precision_cpu_vs_metal.sh` extended to
confirm hit produces byte-equal logits to a miss + prefill.

### PR #2 — `metal: qw36_attn_decode_flash_f16kv — single-pass online-softmax streaming kernel` (task 2.2, #71 Z)

Long-context is the widest remaining gap to MLX. The transposed-K
work (b4bb6f6) cleared the layout obstacle. This is the highest-ROI
kernel commit left.

Verification: precision goldens from PR #3 (or in-flight); n=2048
wallclock 3-run median.

### PR #3 — `tests: kernel goldens for rmsnorm / rope / qgate / decode-attn / dn-step / moe-route` (task 1.3, #14)

Must land before PR #2 merges. Flash-attn is precisely the case where
"smoke passes but logits drift over 24 layers" can sneak in — goldens
are what makes PR #2 reviewable. Developed in parallel with PR #1.

Verification: one C tool generates goldens, per-kernel shell harness
consumes them, all green on current main before PR #2 starts.

### PR #4 — `35B-A3B: Metal MoE full-GPU close-out` (task 1.1, #74 AC)

Codex is already on it. Slotted here because it's a parallel track —
if codex closes sooner, it moves up; if codex stalls, claude drops in
with the small-diff fallback per AGENTS delegation rule.

Verification: `tests/precision_cpu_vs_metal.sh` extended to optionally
take a 35B model path; 8-token greedy diff CPU vs Metal.

### PR #5 — `common: QW36_PROFILE consolidation + qw36_metal --print-config + --doctor` (tasks 3.1 + 3.3)

Once the perf/correctness churn from PRs #1-#4 settles, this is the
lowest-risk, highest-credibility-per-hour commit on the list. Converts
the env-flag jungle into a documented surface; gives `--doctor` as the
first-touch tool when external users start hitting issues.

Verification: `tests/cli_smoke.sh` exercises every profile +
`--print-config` against expected env state.

## Explicit non-goals for this window

- **Speculative decoding.** Multi-day port on top of three moving
  pieces. Park in `docs/speculative_decode_plan.md` as future work.
- **CUDA/AMD perf optimization** beyond functional parity (task 1.2).
  Needs dedicated hardware time we don't have.
- **Writing a kernel compiler** (the only honest path to MLX's
  244 tok/s). Explicit non-goal per `FINAL_STATUS.md` § "Realistic
  horizon."
- **MoE backend on CUDA/AMD.** Still NULL; only consumer is the 35B
  model which lives on Metal. Revisit when we have a CUDA box that
  can actually load 35B.

## What this roadmap was built on

Four parallel audits run 2026-05-21:

- **Docs audit** — found stale `README.md` perf (81 tok/s vs actual
  204), `AGENTS.md` perf claims out of date, missing CONTRIBUTING /
  architecture / troubleshooting / methodology docs, broken cross-refs
  in `docs/INDEX.md`.
- **Toolchain audit** — CI has only 3 jobs (linux-cpu, macos-metal,
  cuda-compile); missing AMD/HIP, sanitizers, lint, perf-gate. No
  `.clang-format`, no `make check`/`make perf`, no `--doctor`.
- **Codebase health** — 4 unused functions, ~7 undocumented env
  knobs, public-header doc gaps, magic constants (160-byte affine32
  block, 16-element sub-group). No blocker bugs; all hygiene.
- **Roadmap synthesis** — feeds the above into the table above.

Audit reports are in this session's history; their findings drove the
specific DoD wording in each row.
