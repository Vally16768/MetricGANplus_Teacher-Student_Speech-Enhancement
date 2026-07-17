#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/env_server.sh"

TS="$(date +%Y%m%d_%H%M%S)"
EXPORT_DIR="${RESULTS_ROOT}/exports"
mkdir -p "${EXPORT_DIR}"
OUT="${EXPORT_DIR}/ulp_training_results_${TS}.tar.gz"

echo "[export] Creating archive: ${OUT}"
tar -czf "${OUT}" \
  -C "${STACK_ROOT}" \
  results \
  data/manifests

echo "[export] Done: ${OUT}"
