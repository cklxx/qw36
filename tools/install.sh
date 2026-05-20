#!/usr/bin/env bash
# install.sh — install the latest qw36 release into ~/.local/bin.
#
# Usage:
#   tools/install.sh              # latest release
#   tools/install.sh v0.1.0       # pinned tag
#
# Detects platform (macos-arm64 / linux-x86_64), fetches the matching
# tarball from GitHub Releases, verifies its SHA-256 against the
# .sha256 sidecar published alongside the asset, and extracts the
# binaries to ~/.local/bin.

set -euo pipefail

REPO="cklxx/qw36"
TAG="${1:-latest}"
PREFIX="${QW36_PREFIX:-$HOME/.local}"
BINDIR="$PREFIX/bin"

uname_s=$(uname -s)
uname_m=$(uname -m)

case "$uname_s/$uname_m" in
    Darwin/arm64)        PLATFORM="macos-arm64" ;;
    Linux/x86_64)        PLATFORM="linux-x86_64" ;;
    Darwin/x86_64)       echo "error: macOS Intel is not a published target (Apple Silicon only)"; exit 1 ;;
    *)                   echo "error: unsupported platform $uname_s/$uname_m"; exit 1 ;;
esac

mkdir -p "$BINDIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if command -v gh >/dev/null 2>&1; then
    DOWNLOADER="gh"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
else
    echo "error: need either 'gh' or 'curl' to download the release"
    exit 1
fi

echo "==> repo=$REPO tag=$TAG platform=$PLATFORM"

# Resolve the actual tag if we asked for "latest", so the filename
# pattern matches.
if [ "$TAG" = "latest" ]; then
    if [ "$DOWNLOADER" = "gh" ]; then
        TAG=$(gh release view --repo "$REPO" --json tagName -q .tagName)
    else
        TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
            | grep -E '"tag_name"' | head -n1 | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')
    fi
    echo "==> resolved latest tag: $TAG"
fi

ASSET="qw36-${TAG}-${PLATFORM}.tar.gz"
SHA="${ASSET}.sha256"

cd "$TMP"

if [ "$DOWNLOADER" = "gh" ]; then
    gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --pattern "$SHA"
else
    URL_BASE="https://github.com/$REPO/releases/download/$TAG"
    curl -fsSL -o "$ASSET" "$URL_BASE/$ASSET"
    curl -fsSL -o "$SHA"   "$URL_BASE/$SHA"
fi

echo "==> verifying sha256"
# The sha256 sidecar was produced by `shasum -a 256 <file>` so its second
# field is the path relative to the release stage dir. Compare hashes only.
expected=$(awk '{print $1}' "$SHA")
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$ASSET" | awk '{print $1}')
else
    actual=$(shasum -a 256 "$ASSET" | awk '{print $1}')
fi
if [ "$expected" != "$actual" ]; then
    echo "error: sha256 mismatch"
    echo "  expected: $expected"
    echo "  actual:   $actual"
    exit 1
fi

echo "==> extracting"
tar -xzf "$ASSET"
STAGE="qw36-${TAG}-${PLATFORM}"

install_one() {
    local name="$1"
    if [ -f "$STAGE/$name" ]; then
        install -m 0755 "$STAGE/$name" "$BINDIR/$name"
        echo "    installed $BINDIR/$name"
    fi
}

install_one qw36_cpu
install_one qw36_metal
install_one qw36_cuda

# Metal needs default.metallib next to the binary.
if [ -f "$STAGE/default.metallib" ]; then
    install -m 0644 "$STAGE/default.metallib" "$BINDIR/default.metallib"
    echo "    installed $BINDIR/default.metallib"
fi

echo
echo "done. add $BINDIR to your PATH if it isn't already:"
echo "    export PATH=\"$BINDIR:\$PATH\""
