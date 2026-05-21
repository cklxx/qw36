#!/usr/bin/env bash
# tests/correctness_greedy.sh — per-KV-config greedy token-match gate.
#
# Captures the deterministic-on-this-host invariant: across all KV cache
# dtypes (Q8 default, fp16, bf16, fp32) and both attention kernels
# (flash, legacy fused), `Hello -n 16` produces identical first-16-token
# output. Any config-introduced divergence is a correctness regression.
#
# Run before merging anything that touches the attention dispatch, the
# KV cache layout, the fused decode kernels, or the residual stream.
# `make check` runs this when QW36_TEST_MODEL is set.
#
# Frozen baselines below are captured on commit 511d31e against the
# canonical 0.8B-Q4_K_M model.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${QW36_TEST_MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "[correctness-greedy] skip: model not found at $MODEL"
    exit 0
fi

if [ ! -x ./qw36_metal ]; then
    echo "[correctness-greedy] no ./qw36_metal — building"
    make -s metal 1>/dev/null
fi

# Each baseline tuple: label, env-setup, substring-expected
#
# The substring is a known-distinctive segment of the generation that
# uniquely identifies the canonical output. Using substring instead of
# exact match dodges newline-escaping pain and is robust to slight
# trailing-token jitter (the canonical Hello completion has been
# end-of-sequence in some runs).
RAW_EXPECTED='I am writing a Python script that uses the `scipy.optimize`'
CHAT_EXPECTED='Hello! How can I help you today?'

run_case() {
    local label="$1"; local env_setup="$2"; local prompt_flags="$3"; local expected="$4"
    local out
    out=$(env $env_setup ./qw36_metal --fast -m "$MODEL" -p "Hello" -n 16 $prompt_flags 2>&1)
    if [[ "$out" == *"$expected"* ]]; then
        printf '  OK   %s\n' "$label"
        return 0
    else
        printf '  FAIL %s\n' "$label"
        printf '       expected substring: %s\n' "$expected"
        printf '       output tail:\n'
        echo "$out" | tail -6 | sed 's/^/         /'
        return 1
    fi
}

fail=0
echo "[correctness-greedy] Hello -n 16 --no-special (raw)"
run_case "Q8-default"   ""                                                        "--no-special" "$RAW_EXPECTED" || fail=1
run_case "fp16-KV"      "QW36_METAL_Q8_KV=0"                                      "--no-special" "$RAW_EXPECTED" || fail=1
run_case "bf16-KV"      "QW36_METAL_Q8_KV=0 QW36_METAL_BF16_KV=1"                 "--no-special" "$RAW_EXPECTED" || fail=1
run_case "fp32-KV"      "QW36_METAL_QUANT_GPU=1 QW36_METAL_Q8_KV=0 QW36_METAL_FP16_KV=0"  "--no-special" "$RAW_EXPECTED" || fail=1
run_case "fused-fp16"   "QW36_METAL_Q8_KV=0 QW36_METAL_FLASH_ATTN=0"              "--no-special" "$RAW_EXPECTED" || fail=1

echo "[correctness-greedy] Hello -n 16 (chat-template)"
run_case "chat-Q8"      ""                       "" "$CHAT_EXPECTED" || fail=1
run_case "chat-fp16"    "QW36_METAL_Q8_KV=0"     "" "$CHAT_EXPECTED" || fail=1
run_case "chat-bf16"    "QW36_METAL_Q8_KV=0 QW36_METAL_BF16_KV=1" "" "$CHAT_EXPECTED" || fail=1

if [ "$fail" = "0" ]; then
    echo "[correctness-greedy] all 8 configs match baseline"
    exit 0
else
    echo "[correctness-greedy] FAIL: at least one config diverged."
    echo "                     Sanity-check the change before updating baseline."
    exit 1
fi
