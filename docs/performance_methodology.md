# Performance methodology

How qw36 measures, why we measure that way, and what counts as a real
win. Codifies the rules already lived in `AGENTS.md §0` and the
failed-experiment files; this is the user-facing version.

## What we measure

**Decode wallclock tok/s — single ground truth.**

We do NOT trust accumulated per-kernel `gpu_ms` as a fitness metric.
On Metal with the persistent compute encoder (default since commit
112e85f), GPU kernels overlap. The wallclock cost of a token is less
than the sum of its kernels' `gpu_ms` — `gpu_ms` over-counts. We use
PERF mode only to *attribute* time inside a token (find the dominant
kernel), never to *grade* a change.

A commit "wins" if and only if the wallclock tok/s improves.

## How we measure

The standard rig:

```bash
# under quiet load (1-min uptime < 3)
QW36_METAL_QUANT_GPU=1 QW36_METAL_FAST=1 \
    ./qw36_metal -m <gguf> -p "Hello" -n 64
```

For perf-affecting decisions: run **N ≥ 5** times, report the
**median**. Single runs are noise.

For long-context decisions (`n=512`, 1024, 2048): same N, run on a
prompt that does not hit EOS early. The repo's standard is `"Write a
detailed essay about computer science history in at least 2000 words."`
(see `tests/compare_mlx.sh`).

### Why median, not mean

Apple Silicon's unified-memory contention with WindowServer + other
Metal clients spikes the tail. A single bad run pulls the mean down
by ~10-15% but the median is stable. Median + N=5 has rejected several
"+5% wins" that were noise.

### Why wallclock, not `gpu_ms`

Concretely, on n=256 sustained:

```
  qw36_attn_decode_fused_f16kv_f32 — accumulated 275 ms (PERF mode)
  wallclock for the same workload — 5.7 ms / token × 268 tokens = 1530 ms
```

The kernels add up to 30% of wallclock. If you optimize the kernel
that's biggest in PERF, you're not necessarily optimizing the
wallclock. We've shipped a kernel that was -10% in PERF mode and a
wash in wallclock (commit 876e914, the x4 attention experiment). Use
PERF to pick the lever, wallclock to grade the win.

### What "under load" means

`uptime` 1-minute load average. The lever:

- Load < 3: standard. Five-rep median is publishable.
- Load 3-5: caveat. Quote it; cross-check with a quieter re-run before
  flipping a default.
- Load > 5: do not bench. Reschedule.

We've reversed at least one default decision (Q4K_AFFINE32_MLX, commit
8d45cca → aa293f7) after retesting under a quieter load. See
`memory/feedback_bench_under_load.md` (engineering memory, not public).

## What goes in the commit body

Every perf-affecting commit:

```
<scope>: <change> (<headline number>)

<what changed in plain words>

Bench (model, M-class GPU, load avg X.Y, N-rep median):
  n=64 short:      <before> tok/s
  n=64 short:      <after>  tok/s   (+/- %)
  n=256 sustained: <before>
  n=256 sustained: <after>

Smoke (tests/quant_fastest_smoke.sh): pass.
Correctness: <how you checked, e.g. logit diff vs CPU>
```

Missing bench numbers = the commit is not landable. Exception:
documentation-only commits, `tests/*` shell tweaks that don't affect
the engine, memory entries.

## Side-by-side: qw36 vs MLX

`tests/compare_mlx.sh` is the canonical apples-to-apples bench. Same
prompts, same n, both greedy + `--ignore-chat-template`. Median of
`REPEAT` runs (defaults to 3; use 5 for ±5% decisions). Output is a
markdown table per `tests/compare_mlx.sh [short|long|full]`.

If you change the kernel set, run the `short` scope (n=64, 256) before
opening the PR and paste the table into the PR description. The
`long` scope is for KV / attention work specifically.

## Failed experiments are first-class

If your change makes things slower or doesn't move the needle, that's
still a commit:

```
metal: x4-batched attention scoring variant (opt-in research, ~0% win)

[Bench data showing it was a wash]
[Why it didn't work — what was the hypothesis, what actually happened]
```

Add an entry under `docs/*_failed.md` if the experiment is
substantial. Future contributors should be able to find "has anyone
tried X?" in 30 seconds. Examples: `docs/q4k_qmv_quad_failed.md`,
`docs/fp16_state_root_cause.md`.

## Common traps

- **Cold cache.** The first run after a binary rebuild reads pages
  off disk; tok/s is artificially low. Run a warm-up `-n 8` and
  discard it.
- **Different prompts.** `"Hello"` hits EOS at 13 tokens and rewards
  short-context. `"Write a long essay…"` runs the full `-n` and tests
  attention scaling. Pick the prompt that matches the lever you're
  pulling.
- **Backend selection.** `./qw36_metal -m … -n 64` with no env runs
  the legacy fp32 path. Most levers only show up under `--fast`.
- **EOS bias.** A model that hits EOS quickly looks fast on tok/s
  but is doing less per-token work. Use a long-form prompt for
  attention benches.

## Open questions documented here on purpose

We do not have a public **perf-regression gate** in CI yet. Until
task 4.3 lands, the gate is "every PR shows bench data and a reviewer
eyeballs it." The gate is the next quality lever after kernel
correctness goldens (task 1.3) land.
