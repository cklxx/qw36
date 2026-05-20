#!/usr/bin/env bash
# e2e_qwen35_smoke.sh — assert qw36 produces coherent Qwen3.5 output.
#
# Qwen3.5 with the proper chat template should emit a <think>...</think>
# block before its answer. If our tokenizer or forward path silently
# regresses (e.g. lost special-token detection, RoPE direction, or a
# dequantizer), the model output collapses to repeated punctuation. This
# script does a cheap end-to-end run and greps for the thinking marker
# plus at least one common English word.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
BIN="${1:-./qw36_metal}"
PROMPT='What is the capital of France?'
N=24

if [ ! -f "$MODEL" ]; then
    echo "skip: model not found at $MODEL"
    exit 0
fi

if [ ! -x "$BIN" ]; then
    case "$BIN" in
        *metal*) make -s metal ;;
        *cpu*)   make -s cpu   ;;
        *)       make -s cpu   ;;
    esac
fi

OUT=$(timeout 60 "$BIN" -m "$MODEL" -p "$PROMPT" -n "$N" 2>&1 || true)
echo "$OUT" | tail -5

# Three checks:
#  1. <think> appears (chat template recognised)
#  2. At least one ASCII-word of >= 3 letters (coherent English)
#  3. Output is not just punctuation
echo "$OUT" | grep -q '<think>' || { echo "FAIL: no <think> in output"; exit 1; }
echo "$OUT" | grep -qE '[A-Za-z]{3,}' || { echo "FAIL: no English word"; exit 1; }
n_punc=$(echo "$OUT" | tr -d -c '\$:;,()\[\]"' | wc -c | tr -d ' ')
n_alpha=$(echo "$OUT" | tr -d -c 'A-Za-z' | wc -c | tr -d ' ')
if [ "$n_alpha" -lt "$n_punc" ]; then
    echo "FAIL: output is mostly punctuation (alpha=$n_alpha punc=$n_punc)"; exit 1
fi
echo "ok: $BIN generates coherent Qwen3 output (alpha=$n_alpha punc=$n_punc)"
