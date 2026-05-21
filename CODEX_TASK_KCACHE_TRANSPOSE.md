# Codex brief — transposed K cache layout (W task, long-context attention)

**Context.** qw36_metal hit 200 tok/s short-context with the triple-affine
+ lm_head Q6K_SCALE16 path. But long-context bench vs MLX
(`tests/compare_mlx.sh long`) shows the remaining gap:

| prompt          | n    | qw36  | MLX   | qw36/MLX |
|-----------------|-----:|------:|------:|---------:|
| long_essay      | 1024 | ~100  | ~290  | **31%**  |
| detailed_essay  | 1024 | ~100  | ~290  | **34%**  |

qw36 drops 200 → 100 tok/s from n=64 → n=1024 on essay-style prompts.
MLX stays ~290 flat. The drop is attention scaling — our fused decode
attention kernel reads K and V cache linearly each step, and at n=1024
× 24 layers × 8 heads that bandwidth becomes the bottleneck.

I tried reducing barriers (commit 876e914 x4 batched scoring) and it
moved the needle by 0-2% — confirming this is **K/V cache bandwidth
bound, not synchronization bound**. See
`memory/feedback_attention_bandwidth.md`.

## Your task: transpose K cache so adjacent-t reads coalesce

Current K cache layout (per layer):
```
k_cache[t * kv_len + kvh * head_dim + lane]
```
For a fixed lane (head_dim element), reading K[t] and K[t+1] is
`kv_len` (= n_kv * head_dim) floats apart in memory — 1024 floats for
this model. That stride wastes cache lines: each fp16 load pulls in 32
neighboring fp16 values that the lane never reads.

**Target layout:**
```
k_cache_t[kvh * head_dim * seq_capacity + lane * seq_capacity + t]
```
Now K[t] and K[t+1] for the same lane are adjacent in memory. half2 /
half4 vector loads at runtime become natural. Bandwidth utilization
should approach MLX's ceiling.

V cache benefits the same way and should be transposed in step.

## What to touch

1. **`common/qw36.c` allocation** — k_cache_dev / v_cache_dev sizing is
   the same total bytes; layout interpretation changes downstream.
2. **`metal/qw36_metal.metal`** — the four attention kernels need to read
   the cache with the new stride:
   - `qw36_attn_decode_fused_f32`
   - `qw36_attn_decode_fused_f16kv_f32`
   - `qw36_attn_decode_fused_f16kv_x4_f32` (opt-in research kernel,
     update for correctness even if it stays opt-in)
   - The KV write path inside each kernel (where t = seq_pos, the kernel
     writes the new K/V into the cache).
3. **`metal/qw36_metal.m`** — buffer dims passed to kernels do not change
   in size; only the access pattern in the kernels does. Some helper
   that touches the cache (e.g. dn_*) may need updates if it reads K/V —
   please grep `k_cache` / `v_cache` and audit.

**Gating.** Risky enough that I'd like an opt-in env first:
`QW36_METAL_KV_TRANSPOSED=1`. Keep the legacy layout dispatch path so we
can A/B easily. Default-on once smoke + the compare_mlx.sh long bench
shows ≥+20% at n=1024.

## Acceptance

1. `make -C metal` clean, no new warnings.
2. `tests/quant_fastest_smoke.sh` passes both ways
   (`QW36_METAL_KV_TRANSPOSED=0` and `=1`).
3. `tests/compare_mlx.sh long` shows the long-context bench improving
   (target: n=1024 > 150 tok/s, from current ~100). Numbers in commit
   body.
4. Output text on a 256-token essay generation is sensible (compare
   first 30 tokens against the no-transpose run — they should match
   greedy since temp=0).

## Don't break

- Vanilla / DeltaNet hybrid layer dispatch routing.
- The Q-gate (vanilla layers in Qwen3.5 multiply attn_out by
  sigmoid(q_gate); the kernel already has `q_has_gate` plumbing).
- f16 KV under quant_gpu (commit 6619ac8, recently fixed).

## Files for reference

- `memory/feedback_attention_bandwidth.md` — why barriers aren't the
  bottleneck (this confirms transposition is the right lever).
- `tests/compare_mlx.sh` — bench harness.
- `metal/qw36_metal.metal:972-1270` — current decode attention kernels.

## Done criteria

- `make -C metal` clean
- smoke green both env values
- compare_mlx.sh long shows ≥150 tok/s at n=1024 (best of 3) with
  KV_TRANSPOSED=1 vs ~100 today
- commit with the bench numbers in the body
