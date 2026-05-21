# `qw36_state_snapshot` / `_hydrate` plan (task #82)

The follow-up to the KV prefix cache engine attach point (`0296183`).
Today the cache lookup runs and the post-prefill insert stores zero
bytes; this doc captures the design for the **state serialization**
that actually makes second-run prefill skip.

## The shape of qw36_state

Per `common/qw36.h:177` the live state for an N-token sequence is:

| field                       | who lives there                                  | bytes (per layer, per token) |
|-----------------------------|--------------------------------------------------|------------------------------|
| `k_cache_dev[L]`            | vanilla GQA layers; size = `n_kv * head_dim`     | `2 × n_kv × head_dim` (fp16) |
| `v_cache_dev[L]`            | same                                             | same                         |
| `conv_state_dev[L]`         | DN layers; size = `(kernel-1) × conv_dim`        | constant per layer (not per-token) |
| `delta_state_dev[L]`        | DN layers; size = `n_v_heads × key_dim × val_dim` | constant per layer       |
| `seq_pos`                   | scalar — how many tokens written                 | 4 bytes                      |

For 0.8B (24 layers, 12 vanilla + 12 DN, n_kv=2, head_dim=128):
- vanilla per-token KV: 12 × 2 × 2 × 128 = **6 KB / token**
- DN constants: 12 × ((4-1) × 6144 × 4 + 2 × 64 × 4096 × 4) = **24 MB**
- Total at 1024 tokens: 24 MB + 6 MB = **30 MB**

For 35B-A3B (40 layers, 10 vanilla + 30 DN, n_kv=2, head_dim=256):
- vanilla per-token KV: 10 × 2 × 2 × 256 = **10 KB / token**
- DN constants: 30 × ((4-1) × 8192 × 4 + 32 × 128 × 128 × 4) = **63 MB**
- Total at 1024 tokens: 63 MB + 10 MB = **73 MB**

These are the sizes a serialization needs to dump. Tractable for
L2 RAM-LRU (cache already in tree) at maybe 4-8 prefixes per GB.
Trivial on disk.

## Why this isn't yet shipped

Two real obstacles, not effort:

### 1. State buffers live on the device

Under `--fast` the K/V cache lives in GPU memory (`k_cache_dev[l]`),
not host memory. The host arrays (`k_cache[l]`) exist but are stale
on this path. Snapshot needs to **download** from device → host
before serializing; hydrate needs to **upload** host → device.

The backend vtable already has `download(buf, dst, bytes)` (Metal
implements it; CUDA/AMD likewise). So the kernel-side hooks exist;
what's missing is the engine-side glue that iterates per-layer
buffers and downloads them at the right moment.

### 2. Atomicity / ordering with persistent encoder

Under `--fast` the persistent compute encoder batches kernels.
`download()` triggers a `flush` if needed. We need to flush before
reading device buffers (Metal's `metal_download` already does this
— see `metal/qw36_metal.m:1109`), so the engine just needs to call
`download` and trust the backend to serialize ops.

For hydrate, the analogous concern: after uploading new KV cache
bytes, the next forward step must see them. `upload` is a fresh
allocation + memcpy on Metal, so the device buffer is brand-new and
visible. Need to swap the old `k_cache_dev[l]` for the new one (or
copy into existing).

## Proposed API

```c
/* Returns the size of the serialization blob that snapshot would
 * produce, given the current state. Cheap — does NOT actually
 * download from device. Useful for the cache to pre-allocate. */
size_t qw36_state_blob_size(const qw36_state *st, const qw36_engine *eng);

/* Snapshot: download all device buffers and serialize into `buf`.
 * `buf_cap` must be >= qw36_state_blob_size(...). Returns bytes
 * written on success, 0 on failure (small buf / backend rejects
 * download). */
size_t qw36_state_snapshot(const qw36_state *st, const qw36_engine *eng,
                           void *buf, size_t buf_cap);

/* Hydrate: deserialize and upload back to device buffers. Sets
 * st->seq_pos to the stored value. Returns 0 on success, -1 on
 * size/version/layer-count mismatch. */
int qw36_state_hydrate(qw36_state *st, const qw36_engine *eng,
                       const void *buf, size_t bytes);
```

## Blob format

```
uint32  magic    = 0x51564B53  ('SKVQ')
uint32 version  = 1
uint32 seq_pos
uint32 num_layers
uint32 layer_flags[num_layers]  // bit 0: vanilla GQA, bit 1: DeltaNet
uint32 hidden_size, n_kv, head_dim, kernel, n_v_heads, key_dim, val_dim
uint32 kv_dtype  // QW36_DTYPE_F32 / F16 / BF16 / Q8_0(future)
uint32 reserved
// Per layer (in layer order), only the kinds matching the layer flag:
//   if vanilla:
//     <seq_pos * n_kv * head_dim * kv_elem_bytes> bytes for k_cache
//     <same>                                            for v_cache
//   if DN:
//     <(kernel-1) * conv_dim * 4> bytes for conv_state (fp32)
//     <n_v_heads * key_dim * val_dim * 4> bytes for delta_state (fp32)
```

Big-endian / cross-machine: not yet. v0 is single-host, host
endian. If cross-machine shows up as a need, we add an endian flag
and convert on read.

## Implementation plan

This is the actual work, in order, with code agent estimates:

1. **Helper: `qw36_state_blob_size`** (1 hour). Compute the
   serialization size given layer types + seq_pos + kv_dtype.
   Doesn't touch device buffers.
2. **Helper: `qw36__download_layer_kv(state, eng, l, kbuf, vbuf)`**
   (2 hours). Iterates one vanilla layer's k/v cache, calls the
   backend's `download` to copy `seq_pos * n_kv * head_dim * elem`
   bytes from device into the host-side buffers.
3. **`qw36_state_snapshot` impl** (2 hours). Walks layers, writes
   the header, calls the helper above for vanilla layers and the
   DN download helpers for DN layers. Linear write into the blob.
4. **`qw36_state_hydrate` impl** (2 hours). Walks the blob, validates
   header against current engine config (catches model mismatch),
   uploads per-layer buffers via `backend->copy_from_host` or
   re-`alloc + upload`. Sets `st->seq_pos`.
5. **Wire into `qw36_prefill`** (1 hour). At entry: lookup → if hit
   and `hydrate` succeeds, set start = matched. At exit: snapshot
   into a temp buffer, insert into the cache, free temp buffer.
6. **`tests/kvcache_e2e.sh`** (2 hours). First-run produces N
   tokens, time it. Second-run with same prompt prefix; time should
   drop dramatically (TTFT goes from N×prefill to ~0); logits at
   token N+1 must be identical between the two runs.

Total: ~10 hours of code agent time, single session. The actual
risk is in step 4 — uploading hydrated K/V cache might race with
the persistent encoder if not flushed properly, similar to the
problem `metal_download` solved on the snapshot side.

## What can go wrong

- **Dtype mismatch.** The blob stores `kv_dtype` in the header.
  Hydrating a fp16 blob into a bf16 engine = corrupt KV. Validate
  on hydrate; refuse mismatched dtypes (return -1, count as miss).
- **Model mismatch.** Loading a 0.8B blob into a 35B engine would
  catastrophically fail. The header captures num_layers + hidden_size
  + head_dim; any mismatch returns -1.
- **Seq capacity overflow.** If matched_n > st->seq_capacity, we
  truncate the hydrate (or refuse). The cache shouldn't store
  prefixes longer than reasonable session sizes anyway.
- **Persistent encoder ordering.** Snapshot must `flush` before
  `download`. Metal backend already does this; CUDA/AMD ports must
  follow suit when those backends actually exercise this path.
- **DN constant-size state.** conv_state and delta_state don't grow
  with seq_pos — they're per-layer fixed buffers. The serialization
  stores them once per DN layer regardless of seq_pos.

## What this PR is NOT

- Not a multi-process cache. The disk tier survives process restart;
  the RAM tier doesn't. That's fine — disk handles persistence,
  RAM handles intra-session prefix reuse.
- Not radix prefix matching. v0 is exact-prefix only. Future radix
  support keeps the same blob format.
- Not draft-model state sharing. Speculative decoding (deferred,
  `docs/speculative_decode_plan.md`) has its own state contract.

## After this lands

`docs/architecture.md` § "KV prefix cache" gets the new section
describing snapshot/hydrate. `README.md` Known Limitations drops the
"engine wiring deferred" caveat. `ROADMAP.md` PR #1 marks complete
(currently shows skeleton + attach-point landed).

## Estimate

~10 hours of code agent time over one session. The cache itself
exists; the engine has the attach point. This is mechanical
serialization + four backend `download`/`upload` calls per layer.
The main work is testing — the `tests/kvcache_e2e.sh` smoke needs
to verify both the speed win AND that logits at token N+1 are
identical with vs without the cache hit.
