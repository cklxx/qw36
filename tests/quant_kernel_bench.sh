#!/usr/bin/env bash
# Side-by-side bench of QUANT_GPU paths. Pre/post optimization.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL"; exit 0
fi

if [ ! -x ./qw36_metal ]; then make -s metal; fi

run_summary() {
  "$@" 2>&1 | grep -E "generated|prefill|affine32 repacked"
}

echo "=== baseline (fp16 path, no quant) ==="
for i in 1 2 3; do
  run_summary env QW36_METAL_FP16_WEIGHTS=1 ./qw36_metal -m "$MODEL" -p "Hello" -n 128
done

echo
echo "=== QUANT_GPU current (default kernel) ==="
for i in 1 2 3; do
  run_summary env QW36_METAL_QUANT_GPU=1 ./qw36_metal -m "$MODEL" -p "Hello" -n 128
done

echo
echo "=== QUANT_GPU + Q4K_AFFINE32 opt-in ==="
for i in 1 2 3; do
  run_summary env QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_AFFINE32=1 ./qw36_metal -m "$MODEL" -p "Hello" -n 128
done

echo
echo "=== QUANT_GPU + QK_REPACK opt-in (Q4_K + Q5_K + Q6_K) ==="
for i in 1 2 3; do
  run_summary env QW36_METAL_QUANT_GPU=1 QW36_METAL_QK_REPACK=1 ./qw36_metal -m "$MODEL" -p "Hello" -n 128
done

echo
echo "=== QUANT_GPU + QK_REPACK + Q6K_SCALE16 experiment ==="
for i in 1 2 3; do
  run_summary env QW36_METAL_QUANT_GPU=1 QW36_METAL_QK_REPACK=1 QW36_METAL_Q6K_SCALE16=1 ./qw36_metal -m "$MODEL" -p "Hello" -n 128
done

echo
echo "=== QUANT_GPU + Q4K_QUAD opt-in (legacy experimental kernel) ==="
for i in 1 2 3; do
  run_summary env QW36_METAL_QUANT_GPU=1 QW36_METAL_Q4K_QUAD=1 ./qw36_metal -m "$MODEL" -p "Hello" -n 128 ||
    echo "legacy Q4K_QUAD run failed"
done

echo
echo "=== correctness check (Hello continuation, affine32) ==="
QW36_METAL_QUANT_GPU=1 QW36_METAL_QK_REPACK=1 ./qw36_metal -m "$MODEL" -p "Hello" -n 16 2>&1 | tail -3
