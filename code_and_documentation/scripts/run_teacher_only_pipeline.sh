#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/scenario_combined_datasets.yaml}"
PYTHON_BIN="${VENV_PYTHON:-/home/USER/Desktop/ULP/shared-venv/bin/python3}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python in venv: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" repro.py --config "$CONFIG_PATH" prepare_data --force
"$PYTHON_BIN" repro.py --config "$CONFIG_PATH" train_teacher --device auto
