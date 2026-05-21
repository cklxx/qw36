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
- Fixed the real CPU correctness bug: MoE used the same `st->x_rms` buffer for
  input and output, then zeroed `y` before expert matmuls, so every MoE block
  saw an all-zero normalized input. `qw36__moe_forward_f32` now preserves an
  aliasing input copy.
- Aligned Qwen3.5 GGUF DeltaNet layout with `../agent-infer`: V-head blocks in
  qkv/z/a/b activations, conv1d V rows, `dt_bias`, and `ssm_a` are converted
  from GGUF V-within-K order to HF K-grouped order; `ssm_a` keeps the
  `log(abs(raw))` transform; `dn_out` receives the inverse V reorder.
- Bound `ffn_gate_inp_shexp.weight` and applied the Qwen3.5 shared-expert
  scalar gate `sigmoid(shared_expert_gate(x)) * shared_expert(x)`.

## Current Status

```
./qw36_cpu -m ~/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -p Hello -n 8 --no-special --debug-top 5
→ ",\n\nI am trying to use the"
```

This matches the local `llama-completion` CPU reference for the same raw
prompt (`",\n\nI am trying to use the"`). The default chat-template path is
also no longer multilingual garbage; first tokens are `<think>\n</think>\n`.

`QW36_MAX_LAYERS=0 --no-special` on 35B still returns id 16870 (` Release`),
and still matches the direct `gguf-py` zero-layer computation. That path only
tests `embed_lookup -> output_norm -> untied output.weight`, not transformer
quality. With the default chat template, `QW36_MAX_LAYERS=0` top-1 is
`' most'`, not `Release`, but the meaningful correctness check is the full
40-layer CPU result above.

## Model Notes

- `general.architecture = qwen35moe`
- `block_count = 41` (40 transformer + 1 nextn / MTP)
- `attention.head_count = 16`, `head_count_kv = 2`, `head_dim = 256`
- `ssm.group_count = 16`, `ssm.state_size = 128`, `ssm.inner_size = 4096`,
  `ssm.conv_kernel = 4`
- `expert_count = 256`, `expert_used_count = 8`,
  `expert_feed_forward_length = 512`,
  `expert_shared_feed_forward_length = 512`
- Full-attention layers are 3, 7, 11, ..., 39; the other 30 layers are
  DeltaNet (current survey: `vanilla_q=10/40 fused_qkv=30/40 ssm=30/40`).
- Embed + lm_head are Q8_0, shape `[2048, 248320]`.
- attn_qkv (DN) is Q8_0 `[2048, 8192]` = `2*key_dim + value_dim`.

## Root Causes

1. **MoE input/output aliasing.** CPU MoE was invoked as
   `qw36__moe_forward_f32(st->x_rms, st->x_rms, ...)`. The implementation
   zeroed `y` before expert matmuls, clearing `x` as well. `QW36_DEBUG_LAYER=1`
   showed `||y||=0.000000` on every MoE block before the fix.
2. **Missing Qwen3.5 shared-expert scalar gate.** The GGUF tensor
   `ffn_gate_inp_shexp.weight` was not bound; shared experts were added
   unscaled instead of `sigmoid(linear(x)) * shared_expert(x)`.
3. **Qwen3.5 GGUF V-head layout.** llama.cpp stores value heads as
   V-within-K groups. Runtime math expects K-grouped value heads. The fix
   mirrors `../agent-infer`'s `reverse_v_reorder*` loaders and forwards the
   inverse reorder into raw `ssm_out.weight`.

The negative DN probe env paths were deleted rather than kept as opt-in
alternatives: `ssm_a` is always `log(abs(raw))`, Q/K head mapping is always
`v / values_per_key`, and `attn_qkv` is always `[Q all | K all | V all]`.

## Verification

- `make -C cpu`
- `tests/q8_0_golden.sh`
- `tests/f16_materialize_audit.sh`
- `tests/quant_fastest_smoke.sh` (0.8B Metal fastest path unchanged)
- `./qw36_cpu ... -p Hello -n 8 --no-special --debug-top 5`
  starts with `,\n\nI am trying to use the`
- `QW36_MAX_LAYERS=0 ./qw36_cpu ... -p Hello -n 1 --debug-top 5`
  top-1 is `' most'` (chat template, not `Release`)
