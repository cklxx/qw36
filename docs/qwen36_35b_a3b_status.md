# Qwen3.6-35B-A3B (MoE) status — gibberish output

Date: 2026-05-21. Model: `~/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
(22 GB, 753 tensors). Tracked as task #74 (AC).

Update, 2026-05-21:

- Added `tests/q8_0_golden.sh`: `token_embd.weight` row 9419 and
  `output.weight` rows 0 / 248319 match `gguf-py` Q8_0 dequantization
  exactly.
- Hardened `lazy_materialize_f16`: checked shape math and row-wise
  dequant-to-f16 conversion avoid a giant temporary fp32 buffer and preserve
  all 248320 output rows. `tests/f16_materialize_audit.sh` verified
  `output.weight rows=248320 cols=2048 f16_bytes=1017118720` and checked
  rows 0 / 124160 / 248319 against direct dequantization.
- `qwen35moe.nextn_predict_layers=1` is now honored at load time:
  `block_count 41 -> transformer layers 40`; blk.40 remains MTP/nextn data,
  not a normal forward layer.
- The previous `MAX_LAYERS=0` "Release means C row indexing is broken"
  hypothesis is rejected. Direct `gguf-py` computation of
  `token_embd[9419] -> output_norm -> output.weight` produces the exact same
  top-10 as qw36, headed by id 16870 (` Release`). That zero-layer result is
  not evidence of a qw36 dequant/materialize bug.

## Symptom

```
./qw36_metal --fast -m ~/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -p Hello -n 32 --no-special
→ "nö,and,and richer lock lock使它 rembourseteree集中e集中eenlzerzern..."
```

Gibberish from token 0 of decode. Same Hello prompt on
`Qwen3.5-0.8B-Q4_K_M.gguf` produces "Hello! How can I help you today?".

## Bisection results

| config                          | top-1 first token       | conclusion |
|---------------------------------|-------------------------|------------|
| metal `--fast`, all 41 layers   | `nö` / `oxygen` / etc.  | broken |
| metal `--fast`, `MAX_LAYERS=4`  | `ENDebru` / gibberish   | broken |
| cpu  `MAX_LAYERS=1`             | `oxygenning`            | broken (DN layer 0) |
| cpu  `MAX_LAYERS=8`             | `achuYRO`               | broken |
| cpu  `MAX_LAYERS=0`             | ` Release` (id 16870)   | matches `gguf-py`; not a row-indexing bug |
| cpu  `MAX_LAYERS=0`, **0.8B**   | `Hello` (id 9419) ✓     | works (tied embeddings) |
| cpu  `MAX_LAYERS=1`, **0.8B**   | ` bye`                  | works (semantically near) |
| metal `--profile reference`     | OOM (exit 137)          | n/a — 22 GB Q→fp16 won't fit on this host |

**Reading.** With `MAX_LAYERS=0` only the path `embed_lookup → output_norm →
lm_head matmul → argmax` runs. 0.8B has tied embeddings so lm_head sees the
embed of "Hello" and argmax returns "Hello". 35B-A3B has **untied** output
weight (both `token_embd.weight` and `output.weight` exist as separate
Q8_0 tensors). The earlier claim that the zero-layer untied result should be
a semantic neighbor was not evidence-backed; `gguf-py` produces ` Release`
for the same zero-layer computation.

That makes the embed/norm/lm_head trio (the path before any transformer
layer fires) suspect on 35B. The model loads:

- `general.architecture = qwen35moe`
- `block_count = 41` (40 transformer + 1 nextn / MTP)
- `attention.head_count = 16`, `head_count_kv = 2`, `head_dim = 256`
- `ssm.group_count = 16`, `ssm.state_size = 128`, `ssm.inner_size = 4096`,
  `ssm.conv_kernel = 4`
- `expert_count = 256`, `expert_used_count = 8`, `expert_feed_forward_length = 512`
- `full_attention_interval = 4` → vanilla layers at multiples of 4,
  rest are DeltaNet (matches qw36's `vanilla_q=11/41 ssm=30/41`).
- Embed + lm_head are Q8_0, shape `[2048, 248320]`.
- attn_qkv (DN) is Q8_0 `[2048, 8192]` = `2*key_dim + value_dim`.

## Top suspects (ranked)

1. **Q8_0 row indexing for huge matrices.**  `embed_lookup` reads row 9419
   of a 248320-row Q8_0 tensor. Offset = `9419 × (2048/32) × 34 = 20,495,744`
   bytes — fits in `size_t` fine, but the GGUF loader passes data via mmap;
   any pointer-arithmetic wraparound or signed-int promotion bug would only
   surface on huge tensors. `qw36__dequant_row` for Q8_0 looks correct in
   isolation (commit point: `common/qw36_dequant.c:676`) but hasn't been
   diff-tested against a Python reference for this specific tensor.

2. **Untied lm_head + Q8_0 dispatch.**  The lm_head path landed for tied
   embeddings in commit `743a158`; the untied case (separate
   `output.weight`) follows the legacy lazy_materialize_f16 path. For
   248320 × 2048 Q8_0 = ~497 MB tensor, the materialize-to-fp16 produces
   a 1 GB fp16 buffer that the MPS GEMV reads. Worth confirming the
   materialize loop walks all 248320 rows correctly (no row-count truncation).

3. **MTP / nextn_predict_layers = 1.**  block_count is 41 but transformer
   layers are 40; the 41st (blk.40) is a multi-token-prediction head.
   qw36 now subtracts `nextn_predict_layers` at load time, so default forward
   executes 40 transformer layers and skips blk.40.

4. **DN config defaults for 35B-A3B.**  `dn_value_head_dim` is derived
   from `inner_size / dn_value_head_dim` after `value_head_dim` is set to
   `state_size`. For 35B: `state_size=128, inner_size=4096` → 32 value
   heads of 128. That matches the model's value_dim = 4096. Likely OK.

## Recommended next steps

1. Continue the layer bisection from `MAX_LAYERS=1`; `MAX_LAYERS=0` is no
   longer a useful failure signal because it matches the direct `gguf-py`
   zero-layer computation.
2. Use the 40-layer count when comparing against MLX/HF references; blk.40 is
   MTP/nextn and should not be part of normal autoregressive forward.

## Reasonable interim posture

35B-A3B is documented as functional-smoke-only in AGENTS.md §Project shape
("仅用于功能冒烟，不用于 perf 主线"). The model loads, runs end-to-end, and
exits cleanly — qw36's plumbing for huge GGUFs works. The output text is
incorrect because the embed-to-lm_head untied/Q8_0/MTP path has at least
one unfixed bug. **Do not optimize 35B perf** until at least
MAX_LAYERS=0 yields a coherent top-1 on Hello.
