#!/usr/bin/env bash
# Side-by-side decode tok/s comparison: qw36 (Q4_K_M GGUF) vs MLX (4-bit).
#
# Same prompts, same n, same temperature (greedy). Both backends run on
# the same M-class Apple GPU. Output is a markdown table.
#
# Prereqs:
#   - ./qw36_metal built (make -C metal)
#   - mlx_lm.generate on PATH (pip install mlx-lm)
#   - Qwen3.5-0.8B-Q4_K_M.gguf (qw36 path)
#   - Qwen3.5-0.8B-MLX-4bit/  (MLX 4-bit weights dir)
#
# Usage:
#   tests/compare_mlx.sh                     # quick run (n=64, 256)
#   tests/compare_mlx.sh long                # adds n=512, 1024
#   tests/compare_mlx.sh full                # adds n=2048
#   QW36_MODEL=...  MLX_MODEL=...  tests/compare_mlx.sh
set -euo pipefail
cd "$(dirname "$0")/.."

QW36_MODEL="${QW36_MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
MLX_MODEL="${MLX_MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit}"
REPEAT="${REPEAT:-3}"
SCOPE="${1:-short}"

# Fixed prompt dataset — same on both sides.  Picked so the model does NOT
# immediately emit EOS (the Hello prompt without chat template stays in
# completion mode rather than chat mode, so generation runs the full --max-tokens).
declare -a PROMPTS=(
  "Hello"
  "Write a long essay"
  "Write a detailed essay about computer science history in at least 2000 words."
)
declare -a PROMPT_LABELS=("hello" "long_essay" "detailed_essay")

# Context lengths to bench.  Short and medium always; long/full opt-in.
declare -a NS_SHORT=(64 256)
declare -a NS_LONG=(64 256 512 1024)
declare -a NS_FULL=(64 256 512 1024 2048)

case "$SCOPE" in
  short) NS=("${NS_SHORT[@]}") ;;
  long)  NS=("${NS_LONG[@]}") ;;
  full)  NS=("${NS_FULL[@]}") ;;
  *) echo "scope must be 'short' | 'long' | 'full'"; exit 2 ;;
esac

if [ ! -f "$QW36_MODEL" ]; then echo "skip: qw36 model missing at $QW36_MODEL"; exit 0; fi
if [ ! -d "$MLX_MODEL" ];  then echo "skip: MLX model missing at $MLX_MODEL"; exit 0; fi
if [ ! -x ./qw36_metal ]; then make -s -C metal; fi
if ! command -v mlx_lm.generate >/dev/null 2>&1; then
    echo "skip: mlx_lm.generate not on PATH"; exit 0
fi

# Capture each backend's decode tok/s for a given prompt + N.  Returns the
# median of $REPEAT runs.
qw36_decode_tps() {
    local prompt="$1" n="$2"
    local samples=()
    for _ in $(seq "$REPEAT"); do
        local out tps
        out=$(./qw36_metal --fast -m "$QW36_MODEL" -p "$prompt" -n "$n" 2>&1) || true
        tps=$(echo "$out" | grep -oE "generated [0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s" \
              | tail -1 | grep -oE "[0-9.]+ tok/s" | grep -oE "[0-9.]+" | head -1)
        [ -n "$tps" ] && samples+=("$tps")
    done
    median "${samples[@]}"
}

mlx_decode_tps() {
    local prompt="$1" n="$2"
    local samples=()
    for _ in $(seq "$REPEAT"); do
        local out tps
        out=$(mlx_lm.generate \
              --model "$MLX_MODEL" \
              --prompt "$prompt" \
              --max-tokens "$n" --temp 0 --ignore-chat-template 2>&1) || true
        tps=$(echo "$out" | grep -oE "Generation: [0-9]+ tokens, [0-9.]+ tokens-per-sec" \
              | tail -1 | grep -oE "[0-9.]+ tokens-per-sec" | grep -oE "[0-9.]+" | head -1)
        [ -n "$tps" ] && samples+=("$tps")
    done
    median "${samples[@]}"
}

median() {
    if [ "$#" -eq 0 ]; then echo "n/a"; return; fi
    printf '%s\n' "$@" | sort -n | awk '
        { v[NR]=$1 }
        END {
            if (NR == 0) { print "n/a"; exit }
            mid = int((NR + 1) / 2)
            printf "%.1f\n", v[mid]
        }'
}

# Output a markdown table.  Columns: prompt | n | qw36 tok/s | MLX tok/s | qw36/MLX %
echo ""
echo "load_avg=$(uptime | awk -F'load averages:' '{print $2}' | xargs)"
echo "repeat=$REPEAT  scope=$SCOPE"
echo ""
echo "| prompt | n | qw36 (Q4_K_M) | MLX (4-bit) | qw36/MLX |"
echo "|--------|--:|--------------:|------------:|---------:|"

for pi in "${!PROMPTS[@]}"; do
    label="${PROMPT_LABELS[$pi]}"
    prompt="${PROMPTS[$pi]}"
    for n in "${NS[@]}"; do
        q=$(qw36_decode_tps "$prompt" "$n")
        m=$(mlx_decode_tps "$prompt" "$n")
        ratio="n/a"
        if [[ "$q" != "n/a" && "$m" != "n/a" ]]; then
            ratio=$(awk -v a="$q" -v b="$m" 'BEGIN { if (b > 0) printf "%.0f%%", 100*a/b; else print "n/a" }')
        fi
        echo "| $label | $n | $q | $m | $ratio |"
    done
done

echo ""
echo "Methodology"
echo "- qw36:     ./qw36_metal --fast"
echo "- MLX:      mlx_lm.generate --temp 0 --ignore-chat-template"
echo "- Both:     greedy decode, decode tok/s (excludes prefill)"
echo "- median of $REPEAT runs reported"
