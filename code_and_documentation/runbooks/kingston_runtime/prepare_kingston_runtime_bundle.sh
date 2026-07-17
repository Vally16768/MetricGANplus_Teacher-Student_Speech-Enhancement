#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICGAN_SRC="$(cd "${THIS_DIR}/../.." && pwd)"
ULP_ROOT="$(cd "${METRICGAN_SRC}/.." && pwd)"
ULTRA_SRC="${ULTRA_SRC:-${ULP_ROOT}/ultra-low-power-se}"

KINGSTON_MOUNT="${KINGSTON_MOUNT:-/media/${USER}/KINGSTON}"
STACK_ROOT="${STACK_ROOT:-${KINGSTON_MOUNT}/ulp-stack}"
RUNBOOKS_DST="${STACK_ROOT}/runbooks"
PROJECTS_DST="${STACK_ROOT}/projects"
DATA_DST="${STACK_ROOT}/data"
RESULTS_DST="${STACK_ROOT}/results"

STAGED_MANIFEST_DIR="${STAGED_MANIFEST_DIR:-${METRICGAN_SRC}/outputs/combined_datasets/combined/staged_manifests}"
STAGED_AUDIO_DST="${DATA_DST}/staged/ULP_STAGE_AUDIO"
MANIFEST_DST="${DATA_DST}/manifests/metricgan"
PER_DOMAIN_RUNTIME_DST="${MANIFEST_DST}/per_domain_runtime"
MANIFEST_SRC_ARCHIVE="${DATA_DST}/manifests/source_local_staged"
PER_DOMAIN_RUNTIME_SRC="${STAGED_MANIFEST_DIR}/per_domain_runtime"

if [[ -x "/home/USER/Desktop/ULP/shared-venv/bin/python3" ]]; then
  PYTHON_BIN="${VENV_PYTHON:-/home/USER/Desktop/ULP/shared-venv/bin/python3}"
else
  PYTHON_BIN="${VENV_PYTHON:-python3}"
fi
COPY_WORKERS="${COPY_WORKERS:-12}"

for cmd in rsync "${PYTHON_BIN}"; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[prepare_bundle] ERROR: missing command '${cmd}'" >&2
    exit 1
  fi
done

if [[ ! -d "${METRICGAN_SRC}" ]]; then
  echo "[prepare_bundle] ERROR: metricgan source missing: ${METRICGAN_SRC}" >&2
  exit 1
fi
if [[ ! -d "${ULTRA_SRC}" ]]; then
  echo "[prepare_bundle] ERROR: ultra source missing: ${ULTRA_SRC}" >&2
  exit 1
fi
if [[ ! -d "${KINGSTON_MOUNT}" ]]; then
  echo "[prepare_bundle] ERROR: KINGSTON mount missing: ${KINGSTON_MOUNT}" >&2
  exit 1
fi

TRAIN_MANIFEST="${STAGED_MANIFEST_DIR}/train_combined_staged.csv"
VAL_RANK_MANIFEST="${STAGED_MANIFEST_DIR}/val_rank_combined_staged.csv"
VAL_SELECT_MANIFEST="${STAGED_MANIFEST_DIR}/val_select_combined_staged.csv"
TEST_MANIFEST="${STAGED_MANIFEST_DIR}/test_combined_staged.csv"

for f in "${TRAIN_MANIFEST}" "${VAL_RANK_MANIFEST}" "${VAL_SELECT_MANIFEST}" "${TEST_MANIFEST}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[prepare_bundle] ERROR: missing staged manifest: ${f}" >&2
    exit 1
  fi
done

mkdir -p "${RUNBOOKS_DST}" "${PROJECTS_DST}" "${DATA_DST}" "${RESULTS_DST}" "${MANIFEST_DST}" "${MANIFEST_SRC_ARCHIVE}"

echo "[prepare_bundle] Copying runbooks..."
rsync -a \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  "${THIS_DIR}/" "${RUNBOOKS_DST}/"

METRICGAN_DST="${PROJECTS_DST}/MetricGANplus_Teacher-Student_Speech-Enhancement"
ULTRA_DST="${PROJECTS_DST}/ultra-low-power-se"

echo "[prepare_bundle] Copying metricgan project..."
rsync -a \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache/" \
  --exclude "outputs/" \
  --exclude "tracking/" \
  "${METRICGAN_SRC}/" "${METRICGAN_DST}/"

echo "[prepare_bundle] Copying ultra project..."
rsync -a \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache/" \
  --exclude "runs/" \
  "${ULTRA_SRC}/" "${ULTRA_DST}/"

echo "[prepare_bundle] Archiving source manifests..."
for f in "${TRAIN_MANIFEST}" "${VAL_RANK_MANIFEST}" "${VAL_SELECT_MANIFEST}" "${TEST_MANIFEST}"; do
  cp -a "${f}" "${MANIFEST_SRC_ARCHIVE}/$(basename "${f}")"
done

echo "[prepare_bundle] Sync staged audio (runtime-only from manifests)..."
"${PYTHON_BIN}" "${RUNBOOKS_DST}/sync_staged_audio_from_manifests.py" \
  --manifests "${TRAIN_MANIFEST}" "${VAL_RANK_MANIFEST}" "${VAL_SELECT_MANIFEST}" "${TEST_MANIFEST}" \
  --dest-stage-root "${STAGED_AUDIO_DST}" \
  --workers "${COPY_WORKERS}" \
  --summary-json "${RESULTS_DST}/bundle_sync_summary.json"

echo "[prepare_bundle] Rewriting manifests to KINGSTON paths..."
"${PYTHON_BIN}" "${RUNBOOKS_DST}/rewrite_manifests_to_kingston.py" \
  --manifests "${TRAIN_MANIFEST}" "${VAL_RANK_MANIFEST}" "${VAL_SELECT_MANIFEST}" "${TEST_MANIFEST}" \
  --output-dir "${MANIFEST_DST}" \
  --stage-root "${STAGED_AUDIO_DST}" \
  --strict-exists \
  --summary-json "${RESULTS_DST}/bundle_manifest_rewrite_summary.json"

if [[ -d "${PER_DOMAIN_RUNTIME_SRC}" ]]; then
  echo "[prepare_bundle] Copying per-domain runtime manifests..."
  mkdir -p "${PER_DOMAIN_RUNTIME_DST}"
  rsync -a "${PER_DOMAIN_RUNTIME_SRC}/" "${PER_DOMAIN_RUNTIME_DST}/"
fi

echo "[prepare_bundle] Copying MetricGAN teacher resume checkpoint (if available)..."
RESUME_STATE_SRC="$("${PYTHON_BIN}" - <<PY
from pathlib import Path
root = Path("${METRICGAN_SRC}/outputs/combined_datasets/checkpoints/teacher")
candidates = sorted(root.glob("**/latest_state.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0].as_posix() if candidates else "")
PY
)"

RESUME_MODEL_DST=""
RESUME_STATE_DST=""
if [[ -n "${RESUME_STATE_SRC}" ]] && [[ -f "${RESUME_STATE_SRC}" ]]; then
  RESUME_SRC_DIR="$(dirname "${RESUME_STATE_SRC}")"
  RESUME_RUN_NAME="$(basename "${RESUME_SRC_DIR}")"
  RESUME_DST_DIR="${RESULTS_DST}/metricgan/checkpoints/teacher_resume/${RESUME_RUN_NAME}"
  mkdir -p "${RESUME_DST_DIR}"
  for fname in latest_state.pt model.pt training_history.csv training_history.json training_history.png; do
    if [[ -f "${RESUME_SRC_DIR}/${fname}" ]]; then
      cp -a "${RESUME_SRC_DIR}/${fname}" "${RESUME_DST_DIR}/${fname}"
    fi
  done
  if [[ -f "${RESUME_DST_DIR}/model.pt" ]]; then
    RESUME_MODEL_DST="${RESUME_DST_DIR}/model.pt"
  fi
  if [[ -f "${RESUME_DST_DIR}/latest_state.pt" ]]; then
    RESUME_STATE_DST="${RESUME_DST_DIR}/latest_state.pt"
  fi
fi

echo "[prepare_bundle] Generating MetricGAN runtime config..."
BASE_CFG="${METRICGAN_DST}/configs/scenario_combined_datasets.yaml"
RUNTIME_CFG="${METRICGAN_DST}/configs/scenario_combined_datasets_kingston_runtime.yaml"
VOICEBANK_TRAIN_FIT_ARG=()
VOICEBANK_VAL_RANK_ARG=()
VOICEBANK_VAL_SELECT_ARG=()
DNS5_TRAIN_FIT_ARG=()
DNS5_VAL_RANK_ARG=()
DNS5_VAL_SELECT_ARG=()
if [[ -f "${PER_DOMAIN_RUNTIME_DST}/voicebank_train_fit.csv" ]]; then
  VOICEBANK_TRAIN_FIT_ARG=(--voicebank-train-fit-manifest "${PER_DOMAIN_RUNTIME_DST}/voicebank_train_fit.csv")
fi
if [[ -f "${PER_DOMAIN_RUNTIME_DST}/voicebank_val_rank.csv" ]]; then
  VOICEBANK_VAL_RANK_ARG=(--voicebank-val-rank-manifest "${PER_DOMAIN_RUNTIME_DST}/voicebank_val_rank.csv")
fi
if [[ -f "${PER_DOMAIN_RUNTIME_DST}/voicebank_val_select.csv" ]]; then
  VOICEBANK_VAL_SELECT_ARG=(--voicebank-val-select-manifest "${PER_DOMAIN_RUNTIME_DST}/voicebank_val_select.csv")
fi
if [[ -f "${PER_DOMAIN_RUNTIME_DST}/dns5_train_fit.csv" ]]; then
  DNS5_TRAIN_FIT_ARG=(--dns5-train-fit-manifest "${PER_DOMAIN_RUNTIME_DST}/dns5_train_fit.csv")
fi
if [[ -f "${PER_DOMAIN_RUNTIME_DST}/dns5_val_rank.csv" ]]; then
  DNS5_VAL_RANK_ARG=(--dns5-val-rank-manifest "${PER_DOMAIN_RUNTIME_DST}/dns5_val_rank.csv")
fi
if [[ -f "${PER_DOMAIN_RUNTIME_DST}/dns5_val_select.csv" ]]; then
  DNS5_VAL_SELECT_ARG=(--dns5-val-select-manifest "${PER_DOMAIN_RUNTIME_DST}/dns5_val_select.csv")
fi
"${PYTHON_BIN}" "${RUNBOOKS_DST}/generate_metricgan_runtime_config.py" \
  --base-config "${BASE_CFG}" \
  --output-config "${RUNTIME_CFG}" \
  --output-root "${RESULTS_DST}/metricgan/outputs" \
  --tracking-root "${RESULTS_DST}/metricgan/tracking" \
  --train-manifest "${MANIFEST_DST}/train_combined_staged.csv" \
  --val-rank-manifest "${MANIFEST_DST}/val_rank_combined_staged.csv" \
  --val-select-manifest "${MANIFEST_DST}/val_select_combined_staged.csv" \
  --test-manifest "${MANIFEST_DST}/test_combined_staged.csv" \
  "${VOICEBANK_TRAIN_FIT_ARG[@]}" \
  "${VOICEBANK_VAL_RANK_ARG[@]}" \
  "${VOICEBANK_VAL_SELECT_ARG[@]}" \
  "${DNS5_TRAIN_FIT_ARG[@]}" \
  "${DNS5_VAL_RANK_ARG[@]}" \
  "${DNS5_VAL_SELECT_ARG[@]}" \
  --teacher-resume-model "${RESUME_MODEL_DST}" \
  --teacher-resume-state "${RESUME_STATE_DST}" \
  --experiment-name "metricgan_combined_datasets_kingston"

echo "[prepare_bundle] Writing bundle summary..."
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
summary = {
    "stack_root": Path("${STACK_ROOT}").resolve().as_posix(),
    "metricgan_project": Path("${METRICGAN_DST}").resolve().as_posix(),
    "ultra_project": Path("${ULTRA_DST}").resolve().as_posix(),
    "manifest_dir": Path("${MANIFEST_DST}").resolve().as_posix(),
    "staged_audio_root": Path("${STAGED_AUDIO_DST}").resolve().as_posix(),
    "metricgan_runtime_config": Path("${RUNTIME_CFG}").resolve().as_posix(),
    "resume_model": "${RESUME_MODEL_DST}",
    "resume_state": "${RESUME_STATE_DST}",
}
out = Path("${RESULTS_DST}/bundle_summary.json").resolve()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(out.as_posix())
PY

echo "[prepare_bundle] Done."
echo "[prepare_bundle] STACK_ROOT=${STACK_ROOT}"
