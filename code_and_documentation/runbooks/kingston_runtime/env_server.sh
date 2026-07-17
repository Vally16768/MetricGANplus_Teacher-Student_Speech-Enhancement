#!/usr/bin/env bash
set -euo pipefail

export KINGSTON_MOUNT="${KINGSTON_MOUNT:-/media/${USER}/KINGSTON}"
export STACK_ROOT="${STACK_ROOT:-${KINGSTON_MOUNT}/ulp-stack}"

export METRICGAN_PROJECT="${METRICGAN_PROJECT:-${STACK_ROOT}/projects/MetricGANplus_Teacher-Student_Speech-Enhancement}"
export ULTRA_PROJECT="${ULTRA_PROJECT:-${STACK_ROOT}/projects/ultra-low-power-se}"

export DATA_ROOT="${DATA_ROOT:-${STACK_ROOT}/data}"
export STAGED_AUDIO_ROOT="${STAGED_AUDIO_ROOT:-${DATA_ROOT}/staged/ULP_STAGE_AUDIO}"
export MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests/metricgan}"

export RESULTS_ROOT="${RESULTS_ROOT:-${STACK_ROOT}/results}"
export RUNBOOKS_ROOT="${RUNBOOKS_ROOT:-${STACK_ROOT}/runbooks}"

export METRICGAN_GPU="${METRICGAN_GPU:-0}"
export ULTRA_GPU="${ULTRA_GPU:-1}"

export METRICGAN_RUNTIME_CONFIG="${METRICGAN_RUNTIME_CONFIG:-${METRICGAN_PROJECT}/configs/scenario_combined_datasets_kingston_runtime.yaml}"
export ULTRA_CACHE_DIR="${ULTRA_CACHE_DIR:-${ULTRA_PROJECT}/runs/feat_cache_combined}"

if [[ -z "${VENV_PYTHON:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python3" ]]; then
    export VENV_PYTHON="${VIRTUAL_ENV}/bin/python3"
  else
    export VENV_PYTHON="python3"
  fi
fi
