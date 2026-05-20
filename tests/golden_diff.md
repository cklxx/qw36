# Golden intermediate-tensor diff workflow (task #31 closer)

## Goal

Pinpoint the layer + op where qw36's forward diverges numerically from a
known-good reference (agent-infer MLX or HF transformers). Once
identified, the fix is usually a one-line transpose / order / scale
correction.

## Step 0 — what we already know

| signal                                 | value          |
|----------------------------------------|----------------|
| qw36 vs llama.cpp top-1 (`Hello`, n=1) | `{$` vs `<th`  |
| qw36 CPU vs Metal step-0 logits        | bit-equal      |
| qw36 vs Python Q6_K embed[9419]        | byte-equal     |
| qw36 vs Python output_norm.weight[0..15]| byte-equal    |
| residual stream ‖x‖ trajectory         | bounded, well-behaved |
| `QW36_SKIP_DN=1` output                | different garbage (\n cluster) |

So: dequant, embedding lookup, weight loading, GQA mapping, mRoPE,
backend dispatch are all *not* the bug. The bug is in the arithmetic of
the forward itself, and it affects both vanilla GQA and DeltaNet layers
(since skipping DN still produces garbage, just differently flavored).

## Step 1 — emit qw36 intermediates

Set the env var and capture stderr:

```bash
QW36_DEBUG_LAYER=1 ./qw36_cpu \
    -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
    -p "Hello" -n 1 --debug-top 5 \
    2> qw36_layers.txt
```

Currently this dumps ‖x‖ per layer. Extend in `common/qw36.c` around
`debug_layer` to dump:
- input embed (full row, first 32 fp32)
- after input_layernorm in L0
- q/k/v projection output in L0 / L3 (first vanilla)
- post-RoPE q/k
- attention scores top-5 in L3
- post-attention output (pre-residual)
- final logits top-K

## Step 2 — get reference intermediates

### Option A: agent-infer (Rust + MLX)

`agent-infer/crates/mlx-sys/src/mlx_qwen35_model.cpp` already has a
`keep_intermediates` plumbing (line 1389 onward). Build with that
plumbing exposed and run the same prompt:

```bash
cd ../agent-infer
cargo build --release --features mlx-debug-intermediates \\
    -p mlx-sys -p infer
# Then a small Rust binary or test that calls qwen35_compiled_forward
# with current_keep_step_intermediates=true and dumps artifacts.intermediates[*]
```

The intermediates vector contains (in order, see `forward_impl` 1330):
`xn`, `attn_out`, `xn2` for each layer.

### Option B: HF transformers (Python)

```bash
pip install transformers==4.46 torch
python tools/hf_dump.py --model Qwen/Qwen3.5-0.8B \\
    --prompt 'Hello' --layer 3 --out hf_layer3.json
```

Use `model.model.layers[3].self_attn` and register a forward hook on
each sub-module (input_layernorm, q_proj, k_proj, v_proj, q_norm,
k_norm, o_proj). Save the input/output of each as JSON of fp32 lists.

### Option C: poison the original tensors

Force a *known* input through the engine — set st->x to a fixed
pattern (e.g. [1, 0, 0, ..., 0]) and dump the output of each layer.
A reference can be computed by hand for the first layer.

## Step 3 — diff

```bash
python tools/diff_layers.py qw36_layers.txt hf_layer3.json --rtol 5e-3
```

The first op that diverges by more than rtol is the bug site.

## Suspect-tree (in likelihood order)

1. **DN attn_qkv split order**: the fused projection produces a
   `qkv_dim = q_dim + k_dim + v_dim` vector. Our split assumes
   `[Q | K | V]` block layout. Some HF checkpoints store `[q_h0, k_h0,
   v_h0, q_h1, k_h1, v_h1, ...]` per-head interleaved. Cross-check by
   comparing first 16 elements of our `q_post_split` against HF
   `model.layers[0].self_attn.q_proj(x_normed)` (DeltaNet variant uses
   the fused qkv but the equivalent split should be byte-equal).
2. **conv1d state shift direction**: agent-infer's
   `conv1d_decode_batch_kernel` does `my_state[t] = old_state[t+1]` then
   `my_state[k-1] = x_new`. Our code matches but only when the loop is
   indexed `[oldest, ..., newest]`. If the GGUF stores the conv1d
   weight in `[k, channels]` with k=time but our reader interprets it
   as `[channels, k]`, weights are applied to wrong taps.
3. **ssm_norm.weight broadcast**: shape is `[val_dim=128]` and applied
   per value head. Cross-check that our rmsnorm_f32 doesn't accidentally
   read past the weight.
4. **Final norm + lm_head**: we observed `output_norm.weight[0] =
   0.22` while others are ~4. If our final RMSNorm somehow swaps
   indices 0..1, that single different-magnitude element changes
   logit direction non-trivially.

## Once #31 is closed

Per session-wrap criteria:
- `tests/e2e_qwen35_smoke.sh ./qw36_metal` must return ok (output
  contains `<think>` and ≥3-letter words).
- Re-run perf bench — should still be ~82 tok/s sustained at n=128.
- Then unblock #30 (push to 200 tok/s).
