#!/usr/bin/env bash
# tools/download_model.sh — fetch a Qwen 3.5 / 3.6 GGUF checkpoint.
#
# Usage:
#   tools/download_model.sh                            # default: 0.8b-q4_k_m
#   tools/download_model.sh 0.8b-q4_k_m
#   tools/download_model.sh 35b-a3b-q4_k_xl
#   tools/download_model.sh list                       # show registered variants
#
# Honors:
#   DEST_DIR  — where to write the .gguf (default: ./models/)
#   FORCE     — re-download even if sha256 matches
#
# The registry below pairs each variant with a HuggingFace repo + file
# + sha256 (when available). We fall back to `huggingface-cli download`
# if installed, then plain `curl`, then `wget`. After the file is on
# disk we verify the sha256 if one was registered.
#
# Zero-deps on purpose: a contributor on a fresh checkout can `make all
# && tools/download_model.sh` and get inference running without
# installing Python or Git LFS.

set -euo pipefail

# ---------------------------------------------------------------- registry
# Format: NAME|REPO|FILE|EXPECTED_SHA256|SIZE_HUMAN
# SHA256 may be the literal string "skip" if upstream hasn't published
# a digest; we still download but warn.
read -r -d '' REGISTRY <<'EOF' || true
0.8b-q4_k_m|Qwen/Qwen3.5-0.8B-GGUF|Qwen3.5-0.8B-Q4_K_M.gguf|skip|~500 MB
0.8b-q5_k_m|Qwen/Qwen3.5-0.8B-GGUF|Qwen3.5-0.8B-Q5_K_M.gguf|skip|~610 MB
0.8b-q8_0|Qwen/Qwen3.5-0.8B-GGUF|Qwen3.5-0.8B-Q8_0.gguf|skip|~870 MB
35b-a3b-q4_k_xl|unsloth/Qwen3.6-35B-A3B-Instruct-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf|skip|~22 GB
EOF

list_variants() {
    echo "Registered variants (qw36 has CI smoke against 0.8b-q4_k_m):"
    echo "  name              repo                                                file"
    echo "  ----              ----                                                ----"
    while IFS='|' read -r name repo file sha size; do
        [ -z "$name" ] && continue
        printf "  %-18s %-50s %s  (%s)\n" "$name" "$repo" "$file" "$size"
    done <<< "$REGISTRY"
}

variant_lookup() {
    local want="$1"
    while IFS='|' read -r name repo file sha size; do
        [ -z "$name" ] && continue
        if [ "$name" = "$want" ]; then
            printf '%s\n%s\n%s\n%s\n' "$repo" "$file" "$sha" "$size"
            return 0
        fi
    done <<< "$REGISTRY"
    return 1
}

VARIANT="${1:-0.8b-q4_k_m}"
if [ "$VARIANT" = "list" ] || [ "$VARIANT" = "--list" ] || [ "$VARIANT" = "-l" ]; then
    list_variants
    exit 0
fi

if ! info=$(variant_lookup "$VARIANT"); then
    echo "qw36: unknown variant '$VARIANT'." >&2
    echo >&2
    list_variants
    exit 2
fi
REPO=$(echo "$info"  | sed -n '1p')
FILE=$(echo "$info"  | sed -n '2p')
SHA=$(echo "$info"   | sed -n '3p')
SIZE=$(echo "$info"  | sed -n '4p')

DEST_DIR="${DEST_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
mkdir -p "$DEST_DIR"
DEST_PATH="$DEST_DIR/$FILE"

echo "qw36: variant     = $VARIANT  ($SIZE)"
echo "qw36: huggingface = $REPO/$FILE"
echo "qw36: → $DEST_PATH"

# Already present + sha matches?
already_ok() {
    [ -f "$DEST_PATH" ] || return 1
    if [ -n "${FORCE:-}" ]; then return 1; fi
    if [ "$SHA" = "skip" ]; then
        # No registered sha → trust the existing file.
        return 0
    fi
    if command -v shasum >/dev/null 2>&1; then
        local got
        got=$(shasum -a 256 "$DEST_PATH" | awk '{print $1}')
        [ "$got" = "$SHA" ]
        return $?
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        local got
        got=$(sha256sum "$DEST_PATH" | awk '{print $1}')
        [ "$got" = "$SHA" ]
        return $?
    fi
    return 0  # no sha tool → fall back to "exists ⇒ ok"
}
if already_ok; then
    echo "qw36: already on disk, skipping download (set FORCE=1 to re-fetch)."
    echo "qw36: try ./qw36_metal --doctor -m \"$DEST_PATH\""
    exit 0
fi

# Prefer huggingface-cli if available; it handles auth + resume better.
if command -v huggingface-cli >/dev/null 2>&1; then
    echo "qw36: using huggingface-cli"
    huggingface-cli download "$REPO" "$FILE" \
        --local-dir "$DEST_DIR" \
        --local-dir-use-symlinks False
elif command -v curl >/dev/null 2>&1; then
    URL="https://huggingface.co/${REPO}/resolve/main/${FILE}?download=true"
    echo "qw36: using curl"
    curl -L --progress-bar --fail "$URL" -o "$DEST_PATH"
elif command -v wget >/dev/null 2>&1; then
    URL="https://huggingface.co/${REPO}/resolve/main/${FILE}?download=true"
    echo "qw36: using wget"
    wget --show-progress -O "$DEST_PATH" "$URL"
else
    echo "qw36: no huggingface-cli / curl / wget available; install one and re-run." >&2
    exit 3
fi

# Verify sha256 if registered.
if [ "$SHA" != "skip" ]; then
    if command -v shasum >/dev/null 2>&1; then
        got=$(shasum -a 256 "$DEST_PATH" | awk '{print $1}')
    elif command -v sha256sum >/dev/null 2>&1; then
        got=$(sha256sum "$DEST_PATH" | awk '{print $1}')
    else
        got=""
    fi
    if [ -n "$got" ] && [ "$got" != "$SHA" ]; then
        echo "qw36: sha256 mismatch!" >&2
        echo "  expected: $SHA" >&2
        echo "  got:      $got" >&2
        echo "  file:     $DEST_PATH" >&2
        echo "qw36: leaving file in place for inspection; delete + retry to re-download." >&2
        exit 4
    fi
fi

echo "qw36: download complete."
echo "qw36: smoke it with:"
echo "  ./qw36_metal --doctor -m \"$DEST_PATH\""
echo "  ./qw36_metal --fast -m \"$DEST_PATH\" -p \"Hello\" -n 16"
