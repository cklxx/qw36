# KV prefix cache — tier-composing design

Goal: reuse the per-layer KV / conv / delta state across requests
whose prompts share a leading token sequence. The cache is generic
across backends; only the **storage tiers** differ per platform.

## Tier-as-strategy model

A `qw36_kv_prefix_cache` is just an **ordered list of tiers**, hottest
first. Each tier implements `qw36_kv_tier_ops`:

```c
typedef struct {
    int  (*get)(...);          /* hit / miss */
    int  (*put)(...);          /* insert; tier copies bytes */
    size_t (*used_bytes)(...);
    size_t (*capacity_bytes)(...);
    void (*clear)(...);
    void (*destroy)(...);
} qw36_kv_tier_ops;
```

The cache loop is platform-agnostic: lookup walks tiers top-down with
promotion, insert writes the hot tier with optional writethrough,
eviction is a tier-internal concern. **Adding a new storage medium is
one new vtable, zero changes to the cache scheduler.**

## Per-platform tier stacks

This is what we ship today; the abstraction supports more.

### Mac (Apple Silicon, unified memory) — 2 tiers

```
[ ram_lru ] → [ disk ]
```

Unified memory means there is no distinct GPU memory tier — buffers
the engine uploads to MTL go into the same RAM the cache LRU lives
in. So Mac is naturally 2-tier: in-process LRU + on-disk persistence.

### Discrete GPU (CUDA / HIP, future) — 3 tiers

```
[ vram_lru ] → [ ram_lru ] → [ disk ]
```

VRAM is a distinct memory pool; on hit we want the KV state to already
live there to skip the host→device upload. The same generic cache
loop applies. The VRAM tier knows how to allocate / free `cudaMalloc`
buffers; the cache doesn't care.

### Multi-machine / serving (future) — 4 tiers

```
[ vram_lru ] → [ ram_lru ] → [ disk ] → [ redis | rdma | s3 ]
```

A shared cold tier lets several inference servers share warm prefixes
(e.g. system prompts) without each paying the cost once. Same vtable
contract.

### CPU-only (Linux / minimal Mac) — 2 tiers

```
[ ram_lru ] → [ disk ]
```

Identical to Mac. Cache code path is shared.

## Cache loop (the part that's reused across all stacks)

Lookup:
1. Try longest prefix (`n_tokens`) → shorter (down to `min_prefix_tokens`).
2. At each length, walk tiers hot→cold.
3. On hit at tier `i`: promote payload into tiers `0..i-1` (best-effort).
4. Return matched prefix length + payload.

Insert:
1. Put into tier `0` (hottest).
2. If `writeback_on_evict` is set, mirror into colder tiers.

Eviction:
- Each tier owns its own policy (LRU, FIFO, ARC, …).
- Future cross-tier writeback on evict is a tier-internal call; the
  cache loop doesn't get involved.

## Why composition over inheritance

The earlier draft hard-coded L1/L2/L3 enums and rolled disk into the
cache struct. That made adding a 4th tier (VRAM) a cache-wide change.
The strategy split lets us:

- Reuse the cache scheduler across all backends.
- Unit-test each tier in isolation.
- Swap tiers per environment (single binary, multiple storage stacks).
- Add VRAM / network tiers without touching scheduler invariants.

## State shape per prefix (engine-side, not cache-side)

The cache stores **opaque payload bytes**. The engine encodes its
`qw36_state` into a blob via `qw36_state_snapshot` (next to
qw36_state, sees its internals) and decodes via `qw36_state_hydrate`.
That keeps the cache 100% backend-agnostic.

For 0.8B-Q4_K_M (24 layers, half DN):

| component                   | size                                          |
|-----------------------------|-----------------------------------------------|
| vanilla k_cache + v_cache   | 12 × n × n_kv × head_dim × 2 bytes (fp16 KV) |
| DN conv_state               | 12 × (kernel-1) × conv_dim × 4 bytes (const) |
| DN delta_state              | 12 × n_v × key_dim × val_dim × 4 bytes (const) |
| seq_pos                     | 4 bytes                                       |

For 35B-A3B-Q4_K_XL (40 layers, 10 vanilla / 30 DN), per 1024-token
prefix ≈ 80 MB. A 2 GB RAM LRU holds ~25 prefixes; disk effectively
unbounded.

## What this is NOT

- Not vLLM PagedAttention. KV remains contiguous per-layer; rehydrate
  is a bulk memcpy/upload.
- Not RadixAttention. v0 is exact-prefix match; radix-sharing is a
  future evolution and reuses the tier vtable.
- Not a batched-request scheduler. qw36 stays single-stream; the win
  is sequential prefix reuse.
- Not yet engine-wired. The cache stores opaque blobs; the
  `qw36_state_snapshot` / `_hydrate` helpers that bridge engine state
  ↔ blob are the next task on top of this skeleton.

## File format (disk tier)

Single shot per prefix hash, little-endian on this host. Cross-machine
sharing would add an endian flag; v0 doesn't claim it.

```
uint32  magic    = 0x57515646 ('KVQW')
uint32  version  = 1
uint32  n_tokens
uint32  reserved
uint64  payload_bytes
uint32  tokens[n_tokens]
uint8   payload[payload_bytes]
```

Filename: `qw36_kv_<hex16>.bin` where the hex is `fnv1a64(tokens)`.

## Tracking

- Task #75 (AD): tier abstraction + ram_lru + disk tiers + cache loop.
- Engine integration (snapshot/hydrate hooks in `qw36_prefill` /
  `qw36_forward`): next task after codex's 35B Metal commits land.
- Future tier ctors (CUDA VRAM, Redis, S3) plug into the same vtable
  — they're independent work units that don't touch the cache.
