#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/env_server.sh"

if [[ "${VENV_PYTHON}" == "python3" ]]; then
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[install] ERROR: active venv not detected. Set VENV_PYTHON or activate venv first." >&2
    exit 1
  fi
fi

if [[ ! -x "${VENV_PYTHON}" ]] && ! command -v "${VENV_PYTHON}" >/dev/null 2>&1; then
  echo "[install] ERROR: Python executable not found: ${VENV_PYTHON}" >&2
  exit 1
fi

echo "[install] Using python: ${VENV_PYTHON}"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "[install] Installing MetricGAN dependencies..."
"${VENV_PYTHON}" -m pip install -r "${METRICGAN_PROJECT}/requirements.txt"

echo "[install] Installing ultra-low-power-se dependencies..."
"${VENV_PYTHON}" -m pip install -r "${ULTRA_PROJECT}/requirements_server.txt"

echo "[install] Done."
