#!/usr/bin/env bash
# tools/download_model.sh — fetch a Qwen 3.6 GGUF checkpoint.
#
# Usage: tools/download_model.sh [variant]   (default: 4b-instruct-q4_k_m)
#
# TODO(Claude): wire up to HF / a mirror, with sha256 verification.

set -euo pipefail

VARIANT="${1:-4b-instruct-q4_k_m}"
DEST_DIR="${DEST_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
mkdir -p "$DEST_DIR"

echo "qw36: download_model is a stub. Variant requested: $VARIANT"
echo "qw36: drop your Qwen 3.6 GGUF into $DEST_DIR/ manually for now."
exit 1
