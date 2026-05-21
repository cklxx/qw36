#!/usr/bin/env bash
# tests/correctness_long_coherence.sh — gate against long-generation
# degeneracy.
#
# Failure modes this catches:
#   1. Token-repetition collapse  (model emits the same token 6+ times in a row)
#   2. Lexical diversity collapse  (>70% of tokens are duplicates)
#   3. Single-character emission   (model can only emit punctuation / spaces)
#
# All three are real failure modes we've hit during this project:
#   - 35B-A3B before codex's MoE/DN alignment fix produced repetitive
#     gibberish.
#   - fp16-state drift produced single-char streams in early iterations.
#   - Bad Q-gate could produce locked-in repeat patterns.
#
# Per the AGENTS §0 "推理的正确性是第一位" directive: CI MUST block on
# this for the canonical 0.8B reference model.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${QW36_TEST_MODEL:-/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf}"
N_TOKENS="${LONG_COHERENCE_N:-128}"

if [ ! -f "$MODEL" ]; then
    echo "[long-coherence] skip: model not found at $MODEL"
    exit 0
fi

if [ ! -x ./qw36_metal ]; then
    echo "[long-coherence] no ./qw36_metal — building"
    make -s metal 1>/dev/null
fi

PROMPT="Write a detailed essay about computer science history in at least 2000 words."

extract_gen() {
    perl -e '
        my $in = 0; my $buf = "";
        while (<STDIN>) {
            if (/qw36: prefill \d+ tokens/) { $in = 1; next; }
            if (/qw36: generated \d+ tokens/) {
                if (s/^(.*)qw36: generated .*/$1/) { $buf .= $_; }
                last;
            }
            $buf .= $_ if $in;
        }
        print $buf;
    '
}

check_config() {
    local label="$1"; local env_setup="$2"
    local out
    out=$(env $env_setup ./qw36_metal --fast -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" 2>&1 | extract_gen)
    local n_chars
    n_chars=$(printf '%s' "$out" | wc -c | tr -d ' ')
    if [ "$n_chars" -lt 200 ]; then
        printf '  FAIL %-18s : output too short (%s chars)\n' "$label" "$n_chars"
        printf '       output: %s\n' "$out"
        return 1
    fi

    local n_words n_unique ratio_pct
    n_words=$(printf '%s' "$out" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '[\n*]' | grep -c '^.\{1,\}$' || true)
    n_unique=$(printf '%s' "$out" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '[\n*]' | grep '^.\{1,\}$' | sort -u | wc -l | tr -d ' ')
    if [ "$n_words" -lt 30 ]; then
        printf '  FAIL %-18s : too few words (%s)\n' "$label" "$n_words"
        printf '       output: %s\n' "$out"
        return 1
    fi
    ratio_pct=$(( n_unique * 100 / n_words ))
    if [ "$ratio_pct" -lt 30 ]; then
        printf '  FAIL %-18s : lexical diversity %s%% (%s/%s unique) < 30%% floor\n' \
               "$label" "$ratio_pct" "$n_unique" "$n_words"
        printf '       sample: %s ...\n' "${out:0:200}"
        return 1
    fi

    local max_run
    max_run=$(printf '%s' "$out" | tr -s '[:space:]' '\n' | awk '
        BEGIN { max=0; cur=1; prev="" }
        { if ($0 == prev) cur++; else cur=1
          if (cur > max) max = cur; prev = $0 }
        END { print max }
    ')
    if [ "$max_run" -ge 6 ]; then
        printf '  FAIL %-18s : token repeated %s times in a row\n' "$label" "$max_run"
        printf '       output: %s\n' "$out"
        return 1
    fi

    printf '  OK   %-18s : %s words, %s unique (%s%%), max-run %s\n' \
           "$label" "$n_words" "$n_unique" "$ratio_pct" "$max_run"
    return 0
}

fail=0
echo "[long-coherence] essay prompt, n=$N_TOKENS, --fast"
check_config "Q8-default"   ""                                        || fail=1
check_config "fp16-KV"      "QW36_METAL_Q8_KV=0"                      || fail=1
check_config "fused-fp16"   "QW36_METAL_Q8_KV=0 QW36_METAL_FLASH_ATTN=0" || fail=1

if [ "$fail" = "0" ]; then
    echo "[long-coherence] all configs produce coherent long output"
    exit 0
else
    echo "[long-coherence] FAIL: at least one config went degenerate."
    echo "                 This is a correctness regression — DO NOT merge."
    exit 1
fi
