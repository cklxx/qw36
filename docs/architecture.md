# qw36 architecture

How qw36 actually works, end-to-end. Written for a competent C engineer
who's read the README, who wants to land a PR within an afternoon, and
who needs to know where the load-bearing seams are before touching
anything.

Companion docs you'll want open in adjacent tabs:

- [`AGENTS.md`](../AGENTS.md) — project contract (SOLID rules, perf
  discipline, codex/claude split)
- [`docs/env_knobs.md`](env_knobs.md) — every `QW36_*` env knob
- [`docs/performance_methodology.md`](performance_methodology.md) —
  what "wins" means around here
- [`docs/kvcache_design.md`](kvcache_design.md) — tier composition

## 30-second mental model

qw36 is **single-stream**: one prompt, one sequence, one binary. There
is no scheduler, no batcher, no API server. Inference is `embed →
N layers → output_norm → lm_head → sample`, where `N layers` alternate
between vanilla GQA attention + dense/MoE MLP and Gated DeltaNet
(SSM-style) blocks, depending on the model.

The hot path is **CPU + one GPU backend**. CPU implements every kernel
as a reference (slow but correct). Each GPU backend (Metal / CUDA /
HIP) implements the same kernels via a frozen vtable. The engine
orchestrates by calling the vtable; common/qw36.c is the only place
that ties it all together.

```
+--------------+                  +---------------------+
|   CLI args   |                  |     GGUF on disk    |
+------+-------+                  +----------+----------+
       |                                     |
       v                                     v
+--------------+      mmap (NoCopy)   +--------------+
| qw36_engine  |<--------------------+   GGUF loader |
|              |                     +--------------+
| weights:     |
|   lazy_w[]   |  ----------------+
|              |                  |
| state:       |     vtable      v
|   x, x_rms,  |    dispatch  +-----------+
|   k/v cache, +<-----------+  qw36_gpu  |
|   conv, dn   |             |  backend   |
+------+-------+             | (metal/    |
       |                      |  cuda/    |
       v                      |  hip)     |
   forward()                  +-----------+
       |
       v
  logits → sample → token
```

## File layout (and why)

```
common/
├── qw36.h            public C API: engine open/close, forward, prefill
├── qw36_internal.h   private: engine struct, forward_ctx, lazy_w
├── qw36_gpu.h        FROZEN vtable contract (every backend implements)
├── qw36.c            engine lifecycle + per-token forward orchestration
├── qw36_dequant.c    GGUF dtype → fp32/fp16; affine32 / scale16 repacks
├── qw36_ops.c        CPU rmsnorm/silu/residual/matmul reference kernels
├── qw36_attn_vanilla.c  vanilla GQA + Q-gate (Qwen3.5/3.6 attention)
├── qw36_attn_deltanet.c Gated DeltaNet decode (DN hybrid layers)
├── qw36_mlp.c        SwiGLU dense MLP
├── qw36_moe.c        Top-K MoE + shared expert
├── qw36_kvcache.c    tier-composing prefix cache (ram_lru + disk + …)
├── qw36_gguf.c       GGUF v3 mmap loader
├── qw36_tokenizer.c  BBPE + Qwen3 chat-template specials
├── qw36_policy.c     QW36_PROFILE → env flag resolution
└── qw36_cli.c        main(), --doctor, --print-config, --info

cpu/         cpu_stub.c: qw36_backend_create() → NULL (CPU reference)
metal/       qw36_metal.m + qw36_metal.metal (MSL kernels)
cuda/, amd/  parallel layouts for the two GPU vendors
tests/       smoke + correctness + bench scripts
tools/       gen_goldens.c, check_goldens.c, dump_tensor.c, install.sh
```

The split is intentional. Common code does NOT `#ifdef __APPLE__`. The
vtable absorbs backend differences. Adding a 4th backend (Vulkan?
WebGPU?) is one new directory + one vtable impl, zero changes in
common/.

## The vtable contract (`common/qw36_gpu.h`)

Frozen ABI. Every backend implements every slot, even when it's a
no-op. Changing the vtable is a 3-backend commit. The slots
(non-exhaustive):

```
upload / download         host ↔ device transfers (zero-copy on Metal)
alloc                     device buffer allocation
matmul                    Y = W·x (M=1 GEMV; fp32/fp16/quant in, fp32 out)
rmsnorm                   per-row normalize + scale
residual_add              y += x
embedding_lookup          token id → row of embed table
swiglu                    silu(g) * u
attention                 vanilla GQA forward (Q-gate aware)
deltanet                  DN forward (conv1d + gated_delta + tail)
moe_forward               router + top-K experts + shared expert
softmax / sample          (sampling is CPU-side; backend reaches into logits)
```

The vtable file declares the contract and documents each function's
shape; backend impls are in `metal/qw36_metal.m`, `cuda/qw36_cuda.cu`,
`amd/qw36_amd.cpp`. CPU is special: `cpu/qw36_cpu_stub.c` returns NULL
from `qw36_backend_create()` and the engine falls back to the
reference path defined in qw36.c. That's why the CPU binary works
without a GPU SDK on the host.

## Lazy weight materialization

GGUF weights stay mmap'd. The engine wraps each tensor in a `lazy_w`
descriptor:

```
struct qw36_lazy_w {
    void       *data;       /* mmap'd raw bytes */
    qw36_dtype  dtype;      /* Q4_K, Q5_K_AFFINE32, F16, … */
    uint32_t    rows, cols, n_extra;
    void       *gpu_buf;    /* uploaded copy or NoCopy view */
    int         ggml_type;
};
```

The "lazy" part: weights are **not** dequantized at load. Instead:

1. On engine open, the loader walks every tensor in the model and
   creates `lazy_w` descriptors pointing at the mmap'd bytes.
2. For Metal under `--fast`, the engine repacks K-quant blocks into
   `_AFFINE32` / `_SCALE16` variants (better GPU access pattern; see
   `docs/q4k_kernel_design_v2.md`) and uploads the repacked bytes.
3. For paths that need fp16/fp32 weights, `lazy_materialize_f16` /
   `_f32` converts on demand — **cached** in `lw->data` so the
   conversion happens once per tensor.

The materialize cache is the unlock for "fast model open under 5s on
a 0.8B GGUF": no upfront dequant pass.

## Per-token forward (vanilla layer)

`qw36_forward(eng, st, token)` in qw36.c is the orchestrator. The
high-level shape:

```c
embed(token) → x        (x is the residual stream, size = hidden)
for layer L in 0..num_hidden_layers-1:
    if layer L is vanilla:
        x_rms = rmsnorm(x, w_input_layernorm)
        q, k, v = matmul3(x_rms, W_qkv)         // fused QKV
        q = q_gate(q)                            // Qwen3.5/3.6 Q-gate
        apply_rope(q, k)
        write_kv_cache(k, v)
        attn_out = attention(q, k_cache[layer], v_cache[layer])
        x = x + W_o · attn_out
        x_rms = rmsnorm(x, w_post_attn_layernorm)
        x = x + (mlp or moe)(x_rms)
    if layer L is deltanet:
        x_rms = rmsnorm(x, w_input_layernorm)
        qkv = matmul(x_rms, W_attn_qkv)          // [2*key_dim + value_dim]
        qkv = conv1d_silu(qkv, w_conv1d, conv_state[layer])
        q, k, v = split(qkv, [key_dim, key_dim, value_dim])
        q, k = l2norm(q), l2norm(k)
        z = matmul(x_rms, W_z)
        a = matmul(x_rms, W_alpha)
        b = matmul(x_rms, W_beta)
        gated_delta_step(q, k, v, a, b, z, dn_state[layer])
        x = x + W_dn_out · result
        x_rms = rmsnorm(x, w_post_attn_layernorm)
        x = x + (mlp or moe)(x_rms)
x = rmsnorm(x, w_output_norm)
logits = matmul(x, W_lm_head)
```

Real code is in `common/qw36.c:qw36_forward` (~200 lines, monolithic
by design — the orchestrator stays in one place for inlining and
easier diff review).

## Fusion flags (the load-bearing detail)

Performance wins come from fusing adjacent ops so we don't materialize
intermediates to memory. The engine tracks per-step flags on a
`qw36_forward_ctx`:

| flag                       | what it means                                      |
|----------------------------|----------------------------------------------------|
| `post_attn_rmsnorm_done`   | the attn block already wrote x_rms_dev = rmsnorm(x+attn_out); MLP must skip rmsnorm |
| `input_rmsnorm_done`       | residual_add fused into next rmsnorm dispatch     |
| `qkv_fused`                | one matmul produced [q;k;v] concatenated         |
| `dn_qkvzab_fused`          | DN layer concat-fuses (qkv, z, alpha, beta)      |

Adding a new fusion is: (1) write the fused kernel, (2) add a flag,
(3) the next op checks the flag and skips its work. Mostly done in
codex's perf cycle; see commits `H` (residual+rmsnorm), `I` (QKV
concat), `K` (DN qkvzab), `N` (conv1d+gated_delta), `X` (gate_up
quant fusion).

## Persistent compute encoder (Metal)

Metal's CommandBuffer-per-op overhead is high. We open ONE compute
encoder at the start of a token, dispatch every kernel into it, and
commit only once per token. Commit at `O` (`metal_compute_encoder_for_op`).
Dispatching new kernels: call `metal_compute_encoder_for_op(ctx)`
rather than `[cmdBuf computeCommandEncoder]`. The encoder is opaque to
the engine — `qw36.c` just sees a sequence of `backend->matmul(...)`
calls and trusts the backend to batch.

PERF mode (`QW36_METAL_PERF=1`) intentionally disables the persistent
encoder so each kernel's `gpu_ms` is attributable. PERF mode wallclock
is ~1/4 of normal — never bench under PERF.

## KV cache (state, not weights)

Two layouts coexist:

- **Legacy** `[t][head][dim]`: contiguous in time. Writes are cheap;
  reads spread across heads. Good at short context.
- **Transposed** `[head][dim][t]`: contiguous in head. Reads coalesce
  along the lane. +24% at n=2048, -10% at n=64.

`QW36_METAL_KV_TRANSPOSED` is a tri-state since `27f64ec`:
`auto|0|1`, default `auto` → on when `seq_capacity > 512`. The flip
happens at attention dispatch; the underlying buffer is layout-agnostic.

DN layers have their own state instead:

- `conv_state[L]`: short ring buffer for the depthwise causal conv1d
  (size = (kernel-1) × conv_dim)
- `delta_state[L]`: per-head SSM recurrent state (size = n_v_heads ×
  key_dim × val_dim)

State buffers live in `qw36_state` (host arrays + device buffers).
See `common/qw36.h:177` for the struct.

## KV prefix cache (state across requests)

`common/qw36_kvcache.{h,c}` ships an ordered tier list — `ram_lru`,
`disk`, with hooks for future `vram_lru` / `redis` / `s3`. The
scheduler is generic (lookup walks tiers hot→cold with promotion;
insert writes the hot tier with optional writethrough); adding a new
medium is one new vtable, zero scheduler change.

Engine attach point (PR #1, commit `0296183`): the engine has a
`kv_cache` slot; `qw36_engine_attach_kv_cache(eng, cache)` plugs one
in. `qw36_prefill` consults it on entry (lookup) and inserts on exit
(zero-byte payload today; full `qw36_state_snapshot/_hydrate` is the
next PR). See `docs/kvcache_design.md`.

## State buffer dtypes

The residual stream defaults to fp32 (`x_dev` / `x_rms_dev` / `q_dev`).
Switching the state to fp16 has bitten us twice (see
`docs/fp16_state_root_cause.md`) — 24-layer fp16 accumulation diverges
the residual enough to flip token decisions. Opt-in via
`QW36_METAL_FP16_*` env flags; default off.

KV cache **does** default to fp16 under `--fast` (it's a bandwidth
win and the per-step accumulation is bounded). bf16 KV (`#73 AB`) is
the next experiment to widen the exponent range and let us go further
down the residual stream.

## Sampling

Backend-agnostic, CPU-side. After forward, `st->logits` is fp32 on
host; `qw36_sample` reads it. Temperature, top-p, top-k all live in
`qw36_sampler` (qw36.h). Greedy (`temp <= 0`) takes the argmax.
No structured-output / grammar / speculative-decode logic yet.

## Adding a new model architecture

Roughly the work-graph:

1. Confirm GGUF metadata uses `general.architecture = <name>`. qw36
   currently recognises `qwen35` and `qwen35moe`; new arches go through
   `qw36_engine_open` arch-detection.
2. If MLP shape differs, update `qw36_mlp.c` / `qw36_moe.c`.
3. If attention shape differs (Q-gate, RoPE flavour, head counts),
   update `qw36_attn_vanilla.c`.
4. If a new state-space variant (different conv kernel, different
   gating), update `qw36_attn_deltanet.c`.
5. Run the precision smoke (CPU vs reference): `tests/precision_cpu_vs_metal.sh`
   needs CPU-only equivalent.
6. Add a golden fixture (`tools/gen_goldens.c` extends per kernel).

The vtable doesn't change unless you add a new op category. Most new
arches don't.

## Adding a new backend

1. Create `<backend>/` directory mirroring `metal/` (Makefile,
   `qw36_<backend>.{m,cu,cpp}`).
2. Implement every `qw36_gpu.h` slot. Use the CPU reference as the
   truth source; precision smoke must pass.
3. Update top-level `Makefile`'s `make all` to detect the toolchain.
4. Add a CI job (.github/workflows/ci.yml) mirroring the existing
   cuda-build-check pattern.
5. Run `tests/golden_kernels.sh` against the new backend (once the
   cross-backend check_goldens lands).

CUDA and AMD currently compile-check only on this host; full kernel
parity is task 1.2 in [`ROADMAP.md`](../ROADMAP.md).

## Where the surprises live

- **Tied embeddings.** Some models alias `lm_head` to `embed_tokens`.
  Our Q6K_SCALE16 lm_head path needs to detect tied + un-alias so
  the GPU repack doesn't clobber the embedding lookup.
- **MTP / nextn layers.** Qwen3.6-35B-A3B's `block_count` includes
  one MTP head. The engine reads `nextn_predict_layers` and skips it.
- **GGUF V-head ordering.** llama.cpp stores DN value heads as
  V-within-K groups. The engine reorders them at load + inverse-
  reorders the `dn_out` projection on the way out. See
  `docs/qwen36_35b_a3b_status.md`.
- **fp16 state drift.** Bisected to 24-layer accumulation, not any
  single op. Don't try the same shortcut again without bisecting
  layer-by-layer. See `docs/fp16_state_root_cause.md`.
- **Q4K qmv_quad layout assumption.** MLX's pattern assumes one
  scale per group; GGUF Q4_K has per-32 scales. Direct port failed;
  affine32 repack is what actually wins. See
  `docs/q4k_qmv_quad_failed.md`.

## What this doc deliberately doesn't cover

- Sampling internals (qw36_sample is small; read the source).
- GGUF v3 spec (read llama.cpp's gguf-py for the canonical reference).
- MLX comparison methodology (read `docs/performance_methodology.md`).
- The CUDA / AMD kernel-by-kernel TODO list (read `ROADMAP.md` theme 1.2).

If you're touching qw36 for the first time and this doc didn't answer
your question, that's a doc bug — open an issue, or open a PR.
