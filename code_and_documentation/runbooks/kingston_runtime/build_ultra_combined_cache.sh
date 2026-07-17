#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/env_server.sh"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST_ROOT}/train_combined_staged.csv}"
TEST_MANIFEST="${TEST_MANIFEST:-${MANIFEST_ROOT}/test_combined_staged.csv}"
OUTDIR="${ULTRA_CACHE_DIR}"
JOBS="${ULTRA_CACHE_JOBS:-$(nproc)}"
MEL_CEPS="${ULTRA_MEL_CEPS:-0}"

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "[ultra_cache] ERROR: train manifest missing: ${TRAIN_MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${TEST_MANIFEST}" ]]; then
  echo "[ultra_cache] ERROR: test manifest missing: ${TEST_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"

echo "[ultra_cache] Building train cache..."
(
  cd "${ULTRA_PROJECT}"
  "${VENV_PYTHON}" core/features/create_training_data.py \
    --csv "${TRAIN_MANIFEST}" \
    --outdir "${OUTDIR}" \
    --subset-name train \
    --mel-ceps "${MEL_CEPS}" \
    --jobs "${JOBS}"
)

echo "[ultra_cache] Building test cache..."
(
  cd "${ULTRA_PROJECT}"
  "${VENV_PYTHON}" core/features/create_training_data.py \
    --csv "${TEST_MANIFEST}" \
    --outdir "${OUTDIR}" \
    --subset-name test \
    --mel-ceps "${MEL_CEPS}" \
    --jobs "${JOBS}"
)

echo "[ultra_cache] Done."
