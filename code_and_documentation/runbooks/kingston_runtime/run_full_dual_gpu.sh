#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/env_server.sh"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RESULTS_ROOT}/dual_run_${TS}"
METRICGAN_RUN_ROOT="${RUN_ROOT}/metricgan_teacher"
ULTRA_RUN_ROOT="${RUN_ROOT}/ultra_teacher"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${METRICGAN_RUN_ROOT}" "${ULTRA_RUN_ROOT}" "${LOG_DIR}"

TRAIN_MANIFEST="${MANIFEST_ROOT}/train_combined_staged.csv"
VAL_RANK_MANIFEST="${MANIFEST_ROOT}/val_rank_combined_staged.csv"
VAL_SELECT_MANIFEST="${MANIFEST_ROOT}/val_select_combined_staged.csv"
TEST_MANIFEST="${MANIFEST_ROOT}/test_combined_staged.csv"

for f in "${TRAIN_MANIFEST}" "${VAL_RANK_MANIFEST}" "${VAL_SELECT_MANIFEST}" "${TEST_MANIFEST}"; do
  [[ -f "${f}" ]] || { echo "[full] ERROR: missing manifest ${f}" >&2; exit 1; }
done

ULTRA_TRAIN_MANIFEST="${ULTRA_CACHE_DIR}/train_manifest.csv"
ULTRA_TEST_MANIFEST="${ULTRA_CACHE_DIR}/test_manifest.csv"
ULTRA_TRAIN_STATS="${ULTRA_CACHE_DIR}/train_feature_stats.npz"
for f in "${ULTRA_TRAIN_MANIFEST}" "${ULTRA_TEST_MANIFEST}" "${ULTRA_TRAIN_STATS}"; do
  [[ -f "${f}" ]] || { echo "[full] ERROR: missing ultra cache file ${f}" >&2; exit 1; }
done

METRICGAN_CFG="${METRICGAN_RUN_ROOT}/metricgan_runtime.yaml"
"${VENV_PYTHON}" "${THIS_DIR}/generate_metricgan_runtime_config.py" \
  --base-config "${METRICGAN_RUNTIME_CONFIG}" \
  --output-config "${METRICGAN_CFG}" \
  --output-root "${METRICGAN_RUN_ROOT}/outputs" \
  --tracking-root "${METRICGAN_RUN_ROOT}/tracking" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-rank-manifest "${VAL_RANK_MANIFEST}" \
  --val-select-manifest "${VAL_SELECT_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --experiment-name "metricgan_kingston_full_${TS}"

METRICGAN_LOG="${LOG_DIR}/metricgan_teacher.log"
ULTRA_LOG="${LOG_DIR}/ultra_teacher.log"

echo "[full] Starting full MetricGAN pipeline on GPU ${METRICGAN_GPU}..."
(
  cd "${METRICGAN_PROJECT}"
  CUDA_VISIBLE_DEVICES="${METRICGAN_GPU}" "${VENV_PYTHON}" repro.py \
    --config "${METRICGAN_CFG}" run_all --device cuda:0
) > "${METRICGAN_LOG}" 2>&1 &
METRICGAN_PID=$!
echo "${METRICGAN_PID}" > "${METRICGAN_RUN_ROOT}/metricgan_teacher.pid"

echo "[full] Starting ultra teacher on GPU ${ULTRA_GPU}..."
(
  cd "${ULTRA_PROJECT}"
  CUDA_VISIBLE_DEVICES="${ULTRA_GPU}" "${VENV_PYTHON}" core/train.py \
    --train-manifest "${ULTRA_TRAIN_MANIFEST}" \
    --test-manifest "${ULTRA_TEST_MANIFEST}" \
    --train-stats "${ULTRA_TRAIN_STATS}" \
    --outdir "${ULTRA_RUN_ROOT}/train_out" \
    --batch-size "${ULTRA_BATCH_SIZE:-32}" \
    --epochs "${ULTRA_EPOCHS:-300}" \
    --lr "${ULTRA_LR:-1e-3}" \
    --stft-center true \
    --log-base ln \
    --log-eps 1e-6 \
    --peak-target 0.95
) > "${ULTRA_LOG}" 2>&1 &
ULTRA_PID=$!
echo "${ULTRA_PID}" > "${ULTRA_RUN_ROOT}/ultra_teacher.pid"

cat > "${RUN_ROOT}/pids.txt" <<EOF
metricgan_pid=${METRICGAN_PID}
ultra_pid=${ULTRA_PID}
EOF

echo "[full] Started."
echo "[full] MetricGAN PID: ${METRICGAN_PID} log: ${METRICGAN_LOG}"
echo "[full] Ultra PID: ${ULTRA_PID} log: ${ULTRA_LOG}"
echo "[full] Run root: ${RUN_ROOT}"
