# Correctness Investigation — task #31

## Problem statement

The forward path runs end-to-end on Metal (81 tok/s decode, fp16 weights)
and CPU (1.7 tok/s), but the generated tokens are degenerate. The model
prefers a small cluster of tokens regardless of prompt:

| prompt                       | qw36 top-1            | llama.cpp top-1 |
|------------------------------|-----------------------|-----------------|
| `Hello` (chat-templated)     | `{$`  (id 6785, 12.09)| `<th` (start of `<think>`) |
| `What is 2+2?`               | ` $`  (id 393)        | thinking / answer |
| `The capital of France is`   | `:$`  (id 21031)      | thinking / answer |
| `Once upon a time`           | ` مرور` (id 197211)   | thinking / answer |

llama-completion on the same `Qwen3.5-0.8B-Q4_K_M.gguf` produces fluent
output (`Hello! How can I help you today?`), so the model is correct
and the bug lives in our forward.

## What we've ruled out

- **Tokenizer**: Special tokens (`<|im_start|>`, `<|im_end|>`) are
  detected. Token count for `Hello` chat-wrapped is 9 in both impls.
- **Q4_K / Q5_K / Q6_K dequant**: All match ggml-quants.c byte-equal
  block layouts (see commits c479cad, 2df5298, ce9343e).
- **GQA mapping**: `kvh = h * n_kv / n_heads` (commit 6946fe1) matches
  the agent-infer / standard convention.
- **mRoPE sections**: Parsed from `qwen35.rope.dimension_sections` and
  only the first section uses seq_pos; others use axis-0 (commit 727bb10).
- **num_attention_heads override**: Derived from `attn_q.weight` shape
  (commit 089eaec) — model metadata is wrong (says 8, tensor is 16).
- **CPU↔Metal divergence**: Not the cause. Step 0 is bit-equal in
  fp32 mode (tests/precision_cpu_vs_metal.sh). Both backends share
  the same algorithmic bug.
- **DeltaNet specifically**: Setting `QW36_SKIP_DN=1` to zero out the
  18/24 DN layers produces *different* garbage (newline-heavy), so DN
  is contributing nonsense but isn't the sole bug — vanilla layers are
  also somehow off.

## Hypotheses to test next

1. **Weight access pattern is row-major-transposed.** matmul_lazy reads
   `W[r, c] = data[r * cols + c]` from rows of length `cols`. If GGUF
   stores `Wᵀ` for some tensors, our matmul computes the wrong product.
   *Test*: pick one weight (e.g. `blk.3.attn_q.weight`), dump first 16
   fp32 entries via our dequant, compare to llama.cpp's dump or to a
   safetensors copy.
2. **mRoPE pair convention**: agent-infer uses `(x[i], x[i+d/2])`
   (NEOX half-rotation). Our default is `(x[2i], x[2i+1])` (interleaved).
   `QW36_ROPE_NEOX=1` switches modes; we should bench-test both for
   semantic output, not just numeric stability.
3. **QK normalization order**: We do `L2(q) * 1/sqrt(d)`; agent-infer
   does `RMS(q) * 1/sqrt(d)²`. Mathematically equivalent for the
   resulting scaled dot product, but might differ on a specific
   weight load if eps is wrong.
4. **DeltaNet conv1d state ordering**: our convention is
   `state[0]` = oldest. agent-infer's `conv1d_decode_batch_kernel`
   shifts state buffer differently; cross-check.
5. **DeltaNet GQA mapping in our CPU code is `v % n_key`**, not
   `v * n_key / n_value` (file `common/qw36.c:700`). For 0.8B that's
   the same (n_v=n_k=16) but should be fixed for clarity.

## Useful tools we have

- `--debug-top K` prints per-step top-K logits.
- `--dump-tokens` prints the tokenizer's id sequence for a prompt.
- `QW36_DEBUG_LAYER=1` traces per-layer ‖x‖ through the forward.
- `QW36_SKIP_DN=1` zeroes the DN block contribution.
- `QW36_SKIP_CONV1D=1` bypasses depthwise conv1d in DN.
- `QW36_ROPE_NEOX=1` switches RoPE pair convention.

## Suggested next investigative steps

1. Write `tools/dump_tensor.c`: take a model + tensor name + first N
   elements, print dequantized fp32. Compare with llama.cpp's
   `gguf-dump`. If mismatched, dequant is wrong.
2. Run with `QW36_DEBUG_LAYER=1` and a known-coherent prompt; capture
   per-layer ‖x‖. Trajectory should be similar to a reference run
   (which we don't have yet — agent-infer's `keep_intermediates` path
   could produce one).
3. Build a tiny "model" with hand-crafted weights (identity layers,
   known embedding) and assert qw36_forward returns exactly the
   embedding.
