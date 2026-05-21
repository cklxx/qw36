#!/usr/bin/env bash
# tests/perf_gate.sh — block CI on > X% perf regression against baseline.
#
# v0 deliberately small: 3-run median per cell on the cells the
# project cares about, compared to tests/perf_baseline.json. Uses
# wallclock tok/s (the source-of-truth metric per
# docs/performance_methodology.md) — never per-kernel gpu_ms.
#
# Knobs:
#   PERF_GATE_MODEL     — path to GGUF (default: agent-infer's 0.8B)
#   PERF_GATE_REPEAT    — runs per cell (default 3; CI uses 5)
#   PERF_GATE_TOLERANCE_PCT — fail threshold (default 5)
#   PERF_GATE_CELLS     — pipe-separated subset of profile:n  (default: short cells)
#                           e.g. PERF_GATE_CELLS="fast:64|fast:256"
#   PERF_GATE_RETRY_ON_REGRESS — retest once if a cell regresses
#                                (defeats one-off noise; default 1)
#
# Exits 1 on regression > tolerance after retest. Exit 0 on clean
# pass or skip (model missing). Prints a one-line table per cell.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${PERF_GATE_MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
REPEAT="${PERF_GATE_REPEAT:-3}"
TOLERANCE="${PERF_GATE_TOLERANCE_PCT:-5}"
CELLS="${PERF_GATE_CELLS:-fast:64|fast:256}"
RETRY="${PERF_GATE_RETRY_ON_REGRESS:-1}"
BASELINE="${PERF_GATE_BASELINE:-tests/perf_baseline.json}"

if [ ! -f "$MODEL" ]; then
    echo "[perf-gate] model not found at $MODEL — skipping (CI will mark green)."
    exit 0
fi

if [ ! -x ./qw36_metal ]; then
    echo "[perf-gate] no ./qw36_metal binary — building"
    make -s metal 1>/dev/null
fi

# Median-of-N helper, BSD-portable (no sort -n -k…).
median() {
    printf '%s\n' "$@" | awk '
        { v[NR]=$1 }
        END {
            n = NR
            for (i = 2; i <= n; i++) {
                k = v[i]; j = i - 1
                while (j >= 1 && v[j] > k) { v[j+1] = v[j]; j-- }
                v[j+1] = k
            }
            if (n % 2) print v[(n+1)/2]
            else print (v[n/2] + v[n/2+1]) / 2
        }'
}

run_cell() {
    local profile="$1"
    local n="$2"
    local prompt
    case "$prompt_kind" in
        Hello) prompt="Hello" ;;
        essay) prompt="Write a detailed essay about computer science history in at least 2000 words." ;;
        *)     prompt="Hello" ;;
    esac
    local env_setup
    case "$profile" in
        fast)  env_setup="QW36_PROFILE=fast" ;;
        quant) env_setup="QW36_METAL_QUANT_GPU=1" ;;
        fp16)  env_setup="QW36_METAL_FP16_WEIGHTS=1" ;;
        *) echo "unknown profile $profile" >&2; return 1 ;;
    esac
    local results=()
    for _ in $(seq 1 "$REPEAT"); do
        local out
        out=$(env $env_setup ./qw36_metal -m "$MODEL" -p "$prompt" -n "$n" \
              2>&1 | grep -E "generated .* in [0-9.]+s \([0-9.]+ tok/s\)" | tail -1)
        local tps
        tps=$(echo "$out" | sed -E 's/.*\(([0-9.]+) tok\/s\).*/\1/')
        results+=("$tps")
    done
    median "${results[@]}"
}

baseline_value() {
    local profile="$1"; local n="$2"
    python3 - "$BASELINE" "$profile" "$n" <<'PY'
import json, sys
p, n = sys.argv[2], int(sys.argv[3])
with open(sys.argv[1]) as f:
    b = json.load(f)
for c in b["cells"]:
    if c["profile"] == p and c["n"] == n:
        print(c["tok_per_s"])
        sys.exit(0)
sys.exit(1)
PY
}

prompt_kind=Hello

echo "[perf-gate] baseline=$BASELINE  repeat=$REPEAT  tolerance=±${TOLERANCE}%"
printf '  %-8s %-6s %-7s %-9s %-7s\n' "profile" "n" "median" "baseline" "delta"
echo "  -------- ------ ------- --------- -------"

regress=0
IFS='|'
for cell in $CELLS; do
    profile="${cell%%:*}"
    n="${cell##*:}"
    if ! base=$(baseline_value "$profile" "$n" 2>/dev/null); then
        printf '  %-8s %-6s  (no baseline) — skipping\n' "$profile" "$n"
        continue
    fi
    med=$(run_cell "$profile" "$n")
    delta=$(python3 -c "
b = float($base); m = float($med)
pct = 100.0 * (m - b) / b
sign = '+' if pct >= 0 else ''
print(f'{sign}{pct:.1f}%')
")
    fail=$(python3 -c "
b = float($base); m = float($med); tol = float($TOLERANCE)
pct = 100.0 * (m - b) / b
print('fail' if pct < -tol else 'ok')
")
    printf '  %-8s %-6s %-7s %-9s %-7s  %s\n' "$profile" "$n" "$med" "$base" "$delta" "$fail"
    if [ "$fail" = "fail" ]; then regress=1; fi
done
unset IFS

if [ "$regress" = "1" ] && [ "$RETRY" = "1" ]; then
    echo "[perf-gate] regression detected; retesting once under quiet conditions"
    sleep 2
    regress=0
    IFS='|'
    for cell in $CELLS; do
        profile="${cell%%:*}"
        n="${cell##*:}"
        if ! base=$(baseline_value "$profile" "$n" 2>/dev/null); then continue; fi
        med=$(run_cell "$profile" "$n")
        fail=$(python3 -c "
b = float($base); m = float($med); tol = float($TOLERANCE)
print('fail' if 100.0 * (m - b) / b < -tol else 'ok')
")
        if [ "$fail" = "fail" ]; then
            regress=1
            printf '  RETRY FAIL  %-8s %-6s  median=%s baseline=%s\n' "$profile" "$n" "$med" "$base"
        else
            printf '  RETRY OK    %-8s %-6s  median=%s baseline=%s\n' "$profile" "$n" "$med" "$base"
        fi
    done
    unset IFS
fi

if [ "$regress" = "1" ]; then
    echo "[perf-gate] FAIL: at least one cell regressed > ${TOLERANCE}% on both runs."
    echo "          See docs/performance_methodology.md for the discipline."
    exit 1
fi
echo "[perf-gate] OK: all cells within ±${TOLERANCE}% of baseline."
