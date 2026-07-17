#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/env_server.sh"

echo "[bootstrap] Checking server runtime prerequisites..."
echo "[bootstrap] KINGSTON_MOUNT=${KINGSTON_MOUNT}"
echo "[bootstrap] STACK_ROOT=${STACK_ROOT}"

if [[ ! -d "${STACK_ROOT}" ]]; then
  echo "[bootstrap] ERROR: STACK_ROOT not found: ${STACK_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${METRICGAN_PROJECT}" ]]; then
  echo "[bootstrap] ERROR: metricgan project missing: ${METRICGAN_PROJECT}" >&2
  exit 1
fi

if [[ ! -d "${ULTRA_PROJECT}" ]]; then
  echo "[bootstrap] ERROR: ultra project missing: ${ULTRA_PROJECT}" >&2
  exit 1
fi

if [[ ! -f "${METRICGAN_PROJECT}/requirements.txt" ]]; then
  echo "[bootstrap] ERROR: missing requirements.txt in metricgan project" >&2
  exit 1
fi

if [[ ! -f "${ULTRA_PROJECT}/requirements_server.txt" ]]; then
  echo "[bootstrap] ERROR: missing requirements_server.txt in ultra project" >&2
  exit 1
fi

echo "[bootstrap] Base structure OK."
echo "[bootstrap] Next:"
echo "  1) bash ${THIS_DIR}/install_requirements_server.sh"
echo "  2) bash ${THIS_DIR}/check_env_gpu.sh"
