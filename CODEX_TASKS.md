# Codex task brief

You own the three GPU backends. Claude owns host-side C (loader, tokenizer,
CPU reference, CLI). The frozen contract is in `common/qw36.h` and
`common/qw36_gpu.h` — do not change those without asking.

## Your files

```
amd/qw36_amd.cpp        # HIP backend
amd/Makefile

metal/qw36_metal.m      # Objective-C host
metal/qw36_metal.metal  # MSL kernels
metal/Makefile

cuda/qw36_cuda.cu       # CUDA backend
cuda/Makefile
```

Every function with a `/* TODO(codex) */` comment is yours. Do not touch any
file under `common/` except to read it.

## What "done" looks like (per backend)

The backend produces fp32-identical output to `common/qw36.c` on the golden
vectors under `tests/` (Claude will land those). For v0:

- **Numerics**: fp32 accumulator everywhere, weights in storage dtype.
- **Layout**: row-major. `W[rows, cols]`, `y = W·x` ⇒ `y[r] = Σ_c W[r,c]·x[c]`.
- **Build**: `make amd` / `make metal` / `make cuda` from the repo root.
- **Correctness**: each kernel matches `common/qw36.c::*_f32` within 1e-4
  on random fp32 inputs.

## Priority order

Same per backend:

1. `init` / `destroy`
2. `upload` / `download` / `alloc` / `free`
3. `rmsnorm`
4. `matmul` (naive tiled is fine for v0)
5. `embedding_lookup` (trivial)
6. `residual_add` (trivial)
7. `swiglu_mlp` (compose: matmul → silu·up → matmul)
8. `attention` — biggest. See spec below.
9. `moe_forward` — last; can stay NULL for v0.

## Numerics spec (matches CPU reference)

- **RMSNorm**: `y = x * rsqrt(mean(x²) + eps) * w`. eps from config (default 1e-6).
- **RoPE**: pairs `(x[2i], x[2i+1])` rotated by `pos · θ^{−2i/d_rot}`,
  where `θ = config.rope_theta` and `d_rot = head_dim · partial_rotary_factor`.
  Apply only to the first `d_rot` components of each head.
- **GQA**: `n_kv ≤ n_heads`. Query head `h` reads kv head `h mod n_kv`
  (group-replicate). q_norm and k_norm are per-head RMSNorm applied to each
  head's slice before RoPE.
- **Attention scores**: `q_h · k_h / sqrt(head_dim)`, causal mask
  `pos > seq_pos ⇒ -inf`.
- **Softmax**: numerically stable (subtract max, fp32 exp, fp32 sum).
- **SiLU**: `x · sigmoid(x)`. SwiGLU: `silu(gate(x)) * up(x)` → `down(...)`.

## Suggested order of attack

Start with **metal/** if you're on Apple Silicon — it'll let you smoke-test
end to end fastest. Then cuda/. AMD last (most likely to need a remote box
to actually verify).

For each backend, the inner loop is:

```
1. write the kernel
2. run tests/harness (Claude will land this) → compare to CPU reference
3. fix until bit-equal at fp32
4. commit one kernel per logical commit, with bench numbers in the message
```

## Boundaries with Claude

- **Do not edit** `common/*` or `Makefile` (root) or `tests/*`.
- **Do edit** anything under `amd/`, `metal/`, `cuda/`.
- If you need a helper in `common/`, write a one-line `/* HANDOFF(codex→claude): need X */` comment in the call site and ping Claude.

## How to communicate progress

Append a row to the table at the bottom of `DIVISION_OF_WORK.md` after each
landed kernel. Commit messages should look like:

```
metal: rmsnorm kernel, matches CPU reference within 1e-6

- threadgroup reduce over hidden, one tg per token
- fp32 accumulator, weights loaded as fp16 → fp32
```

That's it. Start with `metal/` `init` + `destroy` + `upload`/`download` —
get the round-trip working before any math kernel.
