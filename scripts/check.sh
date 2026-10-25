#!/usr/bin/env bash
# Единая проверка проекта: линтер, тесты, запрет слов и длинного тире в публичных текстах.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

"$PY" -m ruff check src tests scripts
"$PY" -m pytest -q

public=(README.md docs src tests scripts)
# Список запрещенных слов и длинное тире хранятся в base64, чтобы не попадать в сам код.
banned=$(printf '%s' 'Q2xhdWRlfEFudGhyb3BpY3zQsNGB0YHQuNGB0YLQtdC90YJ80JjQmHzigJQ=' | base64 -d)
if grep -rInE "$banned" "${public[@]}"; then
    echo "check: запрещенные слова или длинное тире в публичных файлах" >&2
    exit 1
fi

if [ -f scripts/verify_numbers.py ]; then
    "$PY" scripts/verify_numbers.py
fi

echo "check: ok"
