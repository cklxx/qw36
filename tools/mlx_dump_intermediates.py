#!/usr/bin/env python3
"""Dump intermediate tensors from an MLX Qwen3.5 model.

Used as a golden reference for qw36 — once you have this dump, run qw36
with QW36_DEBUG_LAYER=1 / --debug-top and diff against these values to
locate the first divergent op.

Usage:
    python3 tools/mlx_dump_intermediates.py \
        --model /Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit \
        --prompt "Hello" \
        --layer 3 \
        --out /tmp/qw35_golden_L3.json

The captured layer 3 (first full_attention layer in this checkpoint)
intermediates: pre-norm x, post input_layernorm, q_proj raw output,
queries (post split), gate (post split), q_norm output, k_proj output,
v_proj output, post-RoPE q, post-RoPE k, attn output (pre-gate),
sigmoid(gate)*attn output, o_proj output.
"""
import argparse, json, sys
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def to_list(a, take=32):
    """Realise an mx.array and return first N flat values as Python floats."""
    arr = np.asarray(a.astype(mx.float32))
    flat = arr.flatten()[:take]
    return [float(v) for v in flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--layer", type=int, default=3,
                    help="0-based layer index (3 = first vanilla)")
    ap.add_argument("--take", type=int, default=32,
                    help="how many values per tensor to dump")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    print(f"loading {args.model}...", file=sys.stderr)
    model, tok = load(args.model)
    text_model = model.language_model.model

    layers = text_model.layers
    print(f"got {len(layers)} layers; targeting L{args.layer}", file=sys.stderr)
    L = layers[args.layer]
    if hasattr(L, "self_attn"):
        attn = L.self_attn
    else:
        raise SystemExit(f"layer {args.layer} is a linear_attn layer; pick a full-attention layer (try 3,7,11,15,19,23)")

    captured = {}

    def cap(name, val):
        try:
            mx.eval(val)
        except Exception:
            pass
        captured[name] = {
            "shape": list(val.shape),
            "first_n": to_list(val, args.take),
        }

    # Tokenize: use the chat template like our CLI / mlx-lm default.
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt)
    print(f"prompt token count: {len(ids)}", file=sys.stderr)

    x_in = mx.array([ids])           # [1, L_tok]
    embed = text_model.embed_tokens(x_in)
    cap("embed", embed)

    # Walk layers 0..target_layer-1 normally; intercept target.
    cur = embed
    cache = None
    # we need per-layer mask; for prompt of length 1 (decoding) it's None, for
    # prefill it's causal — but mlx_lm's scaled_dot_product_attention handles
    # this. For simplicity, just run the model.layers up to target then
    # manually invoke the attention parts of target.
    for i in range(args.layer):
        cur = layers[i](cur)
    cap("residual_in_to_L%d" % args.layer, cur)

    # Manual Q-gate path, mirroring qwen3_next.Qwen3NextAttention.__call__:
    x_normed = L.input_layernorm(cur)
    cap("input_layernorm_out", x_normed)

    B, Lq, D = x_normed.shape
    nh = attn.num_attention_heads
    nkv = attn.num_key_value_heads
    hd = attn.head_dim

    q_raw = attn.q_proj(x_normed)
    cap("q_proj_raw", q_raw)
    queries, gate = mx.split(q_raw.reshape(B, Lq, nh, -1), 2, axis=-1)
    cap("q_split_pre_norm", queries)
    cap("gate_split", gate)

    gate_flat = gate.reshape(B, Lq, -1)
    keys = attn.k_proj(x_normed)
    values = attn.v_proj(x_normed)
    cap("k_proj_raw", keys)
    cap("v_proj_raw", values)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)   # [B, nh, Lq, hd]
    keys = attn.k_norm(keys.reshape(B, Lq, nkv, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, Lq, nkv, -1).transpose(0, 2, 1, 3)
    cap("q_post_norm", queries)
    cap("k_post_norm", keys)

    queries = attn.rope(queries)
    keys = attn.rope(keys)
    cap("q_post_rope", queries)
    cap("k_post_rope", keys)

    # scaled dot-product attention without cache (prefill-like).
    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attn.scale
    )
    cap("attn_out_pre_gate", output)
    out2 = output.transpose(0, 2, 1, 3).reshape(B, Lq, -1)
    cap("attn_out_flat_pre_gate", out2)

    gated = out2 * mx.sigmoid(gate_flat)
    cap("attn_out_gated", gated)
    o = attn.o_proj(gated)
    cap("o_proj_out", o)

    out_json = {
        "model": args.model,
        "prompt": args.prompt,
        "layer": args.layer,
        "n_heads": nh,
        "n_kv": nkv,
        "head_dim": hd,
        "hidden_size": D,
        "n_prompt_tokens": int(len(ids)),
        "captured": captured,
    }

    text = json.dumps(out_json, indent=2)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
