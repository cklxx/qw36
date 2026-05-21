# qw36 documentation index

This directory holds design notes, research briefs, and post-mortems.
Top-level files in the repo root carry the project contract and
user-facing summaries; everything below is engineering memory for
future contributors.

## Top-level (read these first)

| file | what it is |
|------|------------|
| [`README.md`](../README.md) | user-facing intro — install, build, CLI flags, env knobs |
| [`ROADMAP.md`](../ROADMAP.md) | 4–8 week plan: themes, tasks, owner, definition-of-done, next 5 PRs |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | how to land a useful PR — build, test, perf discipline, commit style |
| [`AGENTS.md`](../AGENTS.md) / `CLAUDE.md` symlink | project contract — execution phases, editing rules, codex/claude delegation, perf benchmarking discipline |
| [`FINAL_STATUS.md`](../FINAL_STATUS.md) | running perf ladder + MLX comparison + the levers we tried |
| [`CHANGELOG.md`](../CHANGELOG.md) | release notes |
| [`DIVISION_OF_WORK.md`](../DIVISION_OF_WORK.md) | historical Codex / Claude split |

## Architecture & design (`docs/`)

| file | what it is |
|------|------------|
| [`architecture.md`](architecture.md) | engine deep-dive — vtable contract, lazy weight materialization, per-token forward, fusion flags, persistent encoder, KV layouts; the "read for an afternoon" doc |
| [`kvcache_design.md`](kvcache_design.md) | tier-composing KV prefix cache (vtable per medium; ram_lru + disk on Mac, future vram + redis) |
| [`moe_design.md`](moe_design.md) | MoE GPU rewrite plan — MLX-style SwitchGLU + gather_qmm operating on Q4/Q5/Q6 expert weights directly (in flight, codex) |
| [`kv_quant_plan.md`](kv_quant_plan.md) | KV cache quantization scoreboard (fp16 + bf16 shipped; Q8_0/Q4_0 planned) |
| [`state_snapshot_plan.md`](state_snapshot_plan.md) | qw36_state_snapshot / _hydrate design for the KV prefix cache follow-up (task #82) |
| [`speculative_decode_plan.md`](speculative_decode_plan.md) | speculative decoding sketch — explicit non-goal for the current window |
| [`q4k_kernel_design_v2.md`](q4k_kernel_design_v2.md) | Q4_K affine32 + qmv_fast kernel design, the lever that unlocked >170 tok/s |
| [`q4k_qmv_quad_design.md`](q4k_qmv_quad_design.md) | early qmv_quad-style Q4_K design notes (predecessor to v2) |
| [`q4k_kernel_research_task.md`](q4k_kernel_research_task.md) | research brief that triggered task R |
| [`quant_matmul_research_brief.md`](quant_matmul_research_brief.md) | how qw36 reasons about quant matmul performance |

## Reference (`docs/`)

| file | what it is |
|------|------------|
| [`env_knobs.md`](env_knobs.md) | every `QW36_*` env knob the engine reads, with file:line, default, and lifecycle (stable / internal / research) |
| [`model_support_matrix.md`](model_support_matrix.md) | green/yellow/red per (model, backend) cell with testing scope |
| [`backend_parity.md`](backend_parity.md) | Metal / CUDA / AMD vtable-slot coverage and porting order |
| [`performance_methodology.md`](performance_methodology.md) | how qw36 measures, why median-of-N + wallclock-not-gpu_ms, what counts as a real win |
| [`troubleshooting.md`](troubleshooting.md) | top failure modes (build, runtime, perf) with concrete checks |
| [`../tests/README.md`](../tests/README.md) | every test script catalogued — what it proves, when it runs, which CI job |

## Post-mortems / negative results (`docs/`)

Keep these — they document blind alleys so we don't retry them.

| file | what it is |
|------|------------|
| [`q4k_qmv_quad_failed.md`](q4k_qmv_quad_failed.md) | first Q4_K qmv_quad port was 29 tok/s vs 58 baseline; root cause: GGUF Q4_K layout has per-32 scales, qmv_quad assumed per-group |
| [`fp16_state_root_cause.md`](fp16_state_root_cause.md) | fp16 state buffer caused multi-layer logit drift; bf16 would have been right (per agent-infer) |
| [`moe_kernel_failed.md`](moe_kernel_failed.md) | MoE rewrite dead ends — 1-row-per-TG (24× slower) and Q8-only (wrong direction); the lessons that drove the SwitchGLU design |
| [`qwen36_35b_a3b_status.md`](qwen36_35b_a3b_status.md) | Qwen3.6-35B-A3B (MoE) functional-smoke bring-up notes: MoE/aliasing bug, GGUF V-head reorder, gibberish-bisection history |

## Codex task briefs (`docs/briefs/`)

Brief documents written by Claude to delegate large units of work to
codex. Kept after completion as the historical record of what was
asked / how the work was structured.

| file | what it is |
|------|------------|
| [`briefs/lm_head_quant.md`](briefs/lm_head_quant.md) | task U — lm_head Q6K_SCALE16 path |
| [`briefs/kcache_transpose.md`](briefs/kcache_transpose.md) | task W — transposed K/V cache layout for long-context attention |

## Conventions

- One file = one topic. Don't fold three unrelated post-mortems into a
  single .md just because they're chronologically adjacent.
- Lead with the **conclusion** — the file is being read by someone
  trying to decide if the lever is worth retrying.
- Quote machine load (`uptime` 1-min avg) and `REPEAT=N` on any perf
  number. See `memory/feedback_bench_under_load.md` for why.
- Negative results are first-class. Keep `*_failed.md` and
  `*_status.md` files even after the experiment is closed.
