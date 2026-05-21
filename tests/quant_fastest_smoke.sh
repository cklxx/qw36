#!/usr/bin/env bash
# Smoke-test the fastest opt-in Metal quant path. This is not a bit-equal
# CPU comparison; it is an e2e guard that catches the known degenerate-output
# failures for Q-gate / Q6K_SCALE16 / quantized lm_head combinations.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "skip: model not found at $MODEL"
    exit 0
fi

if [ ! -x ./qw36_metal ]; then make -s metal; fi

hello_out=$(./qw36_metal --fast -m "$MODEL" -p "Hello" -n 16 2>&1)
if [[ "$hello_out" != *"Hello! How can I help you today?"* ]]; then
    echo "FAIL: fastest quant path did not produce the expected Hello continuation"
    echo "$hello_out"
    exit 1
fi

poem_out=$(./qw36_metal --fast -m "$MODEL" -p "写一首关于秋天的古诗" -n 24 2>&1)
if [[ "$poem_out" != *"《"* ]]; then
    echo "FAIL: fastest quant path did not produce a titled Chinese poem"
    echo "$poem_out"
    exit 1
fi

echo "ok: fastest quant path smoke passed"
