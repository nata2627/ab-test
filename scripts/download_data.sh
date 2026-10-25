#!/usr/bin/env bash
# Скачивает Criteo Uplift v2.1 (CC BY-NC-SA 4.0) и сверяет sha256.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
SHA256="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
OUT="data/raw/criteo-research-uplift-v2.1.csv.gz"

mkdir -p "$(dirname "$OUT")"
if [ ! -f "$OUT" ]; then
    curl -L --fail --retry 3 -o "$OUT.part" "$URL"
    mv "$OUT.part" "$OUT"
fi

actual=$(shasum -a 256 "$OUT" | cut -d' ' -f1)
if [ "$actual" != "$SHA256" ]; then
    echo "sha256 не совпал: $actual" >&2
    exit 1
fi
echo "ok: $OUT"
