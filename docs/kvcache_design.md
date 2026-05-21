# KV prefix cache — L1 / L2 / L3 design

Goal: reuse the per-layer KV/conv/delta state across requests whose
prompts share a leading token sequence. Typical wins are large for
serving (system prompts, few-shot prefixes) and visible for the
interactive CLI when the user repeats a long prefix.

## Tier model

| tier | scope                                    | persistence            | medium             |
|------|------------------------------------------|------------------------|--------------------|
| L1   | the current `qw36_state` for an in-flight session | per-engine, hot | RAM (already exists) |
| L2   | LRU of recent **completed** sessions' prefixes    | per-engine            | RAM, opt-in        |
| L3   | shared on-disk store of stable prefixes           | per-machine, restart-survive | disk file       |

L1 is already implemented — it's `qw36_state` mid-forward. L2 and L3
are the new infrastructure this design doc covers.

We're not trying to reproduce vLLM's `PagedAttention` or SGLang's
`RadixAttention` here — those are GPU-resident block tables aimed at
multi-request batching. qw36 is single-stream; the win is **prefix
reuse across consecutive calls**, not concurrent batching.

## State shape per prefix

For a model with `L` layers, after writing `n` tokens of prefix:

| component         | dtype  | size                                                  | note |
|-------------------|--------|-------------------------------------------------------|------|
| `k_cache[l]`      | fp16   | `n * n_kv * head_dim * 2 bytes`                       | vanilla GQA layers only |
| `v_cache[l]`      | fp16   | `n * n_kv * head_dim * 2 bytes`                       | vanilla GQA layers only |
| `conv_state[l]`   | fp32   | `(conv_kernel - 1) * conv_dim * 4 bytes`              | DN layers only |
| `delta_state[l]`  | fp32   | `n_v_heads * key_dim * val_dim * 4 bytes`             | DN layers only |
| `seq_pos`         | u32    | 4 bytes                                               |   |

For Qwen3.5-0.8B-Q4_K_M (24 layers, ~half DN):

- vanilla K/V: 12 layers × n × 2KB ≈ **24n KB**
- DN conv: 12 × 3 × 6144 × 4 = 0.9 MB constant
- DN delta: 12 × 2 × 64 × 4096 × 4 = 25 MB constant

So per-prefix budget is dominated by DN delta (~26 MB constant) plus
K/V cache that scales 24 KB/token.

For 35B-A3B-Q4_K_XL (40 layers, 10 vanilla / 30 DN):

- vanilla K/V: 10 × n × 2 × 2 × 256 × 2 = **20n KB** (n_kv=2, head_dim=256)
- DN conv: 30 × 3 × 8192 × 4 = **2.9 MB**
- DN delta: 30 × 32 × 128 × 128 × 4 = **60 MB**

State per 1024-token prefix on 35B: ~80 MB. A reasonable L2 cap of
2-4 GB holds 25-50 prefixes; L3 on disk can be much larger.

## API (see `common/qw36_kvcache.h`)

```c
typedef struct qw36_kv_prefix_cache qw36_kv_prefix_cache;

qw36_kv_prefix_cache *qw36_kv_cache_create(const qw36_kv_cache_cfg *cfg);
void                  qw36_kv_cache_destroy(qw36_kv_prefix_cache *c);

/* Look up the longest matching token prefix and rehydrate its state.
 * Returns the matched prefix length (0 if no hit). The engine is
 * responsible for resuming from that position. */
size_t qw36_kv_cache_lookup(qw36_kv_prefix_cache *c,
                            const uint32_t *tokens, size_t n_tokens,
                            qw36_state *dst);

/* Snapshot the current state at `seq_pos` for a given token prefix. */
int qw36_kv_cache_insert(qw36_kv_prefix_cache *c,
                         const uint32_t *tokens, size_t n_tokens,
                         const qw36_state *src);

void qw36_kv_cache_stats(const qw36_kv_prefix_cache *c,
                         qw36_kv_cache_stats_t *out);
```

`qw36_kv_cache_cfg` knobs: `l2_capacity_bytes`, `l3_dir`,
`l3_capacity_bytes`. Setting either capacity to 0 disables that tier.

## Key choice

Prefix key = `fnv1a64(tokens[0..n-1])`. Cheap, no allocations, no
external dep. Collisions are surfaced by storing the full token vector
alongside the state and verifying it before rehydrate.

## Eviction policy

LRU within each tier. On L2 miss we check L3 (mmap that file, copy
into L2). On L2 evict we may write back to L3 if the prefix is large
enough to be worth re-loading later (threshold env-tunable).

## What this is NOT

- Not a batched-request scheduler. qw36 stays single-stream.
- Not a GPU-resident table. The state lives in host memory and gets
  re-uploaded to GPU buffers on rehydrate via the same path as engine
  init.
- Not yet PagedAttention. KV is contiguous per-layer; rehydrate is a
  bulk memcpy, not a page-table lookup.

## Out of scope (v0)

- Token-stream-prefix matching with branching (a la RadixAttention).
  v0 only supports exact prefix match.
- Cross-engine shared cache (would need a network layer or a daemon).

## Tracking

- Task #75 (AD): infrastructure with L1 conceptual / L2 in-RAM LRU /
  L3 disk skeleton. Engine wiring deferred until codex's 35B Metal
  work in `common/qw36.c` finishes to avoid conflicts.
