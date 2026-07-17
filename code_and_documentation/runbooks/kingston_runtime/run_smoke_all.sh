#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/env_server.sh"

TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_ROOT="${RESULTS_ROOT}/smoke/run_${TS}"
SMOKE_MANIFESTS="${SMOKE_ROOT}/manifests"
SMOKE_CFG_DIR="${SMOKE_ROOT}/configs"
mkdir -p "${SMOKE_MANIFESTS}" "${SMOKE_CFG_DIR}"

TRAIN_SRC="${MANIFEST_ROOT}/train_combined_staged.csv"
VAL_RANK_SRC="${MANIFEST_ROOT}/val_rank_combined_staged.csv"
VAL_SELECT_SRC="${MANIFEST_ROOT}/val_select_combined_staged.csv"
TEST_SRC="${MANIFEST_ROOT}/test_combined_staged.csv"
PER_DOMAIN_SRC="${MANIFEST_ROOT}/per_domain_runtime"

for f in "${TRAIN_SRC}" "${VAL_RANK_SRC}" "${VAL_SELECT_SRC}" "${TEST_SRC}"; do
  [[ -f "${f}" ]] || { echo "[smoke] ERROR: missing manifest ${f}" >&2; exit 1; }
done

"${VENV_PYTHON}" - <<PY
import csv
from pathlib import Path

def subset(src: Path, dst: Path, keep: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", newline="", encoding="utf-8") as in_f, dst.open("w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=["noisy", "clean"])
        writer.writeheader()
        for i, row in enumerate(reader):
            if i >= keep:
                break
            writer.writerow({"noisy": row["noisy"], "clean": row["clean"]})

subset(Path("${TRAIN_SRC}"), Path("${SMOKE_MANIFESTS}/train_smoke.csv"), 128)
subset(Path("${VAL_RANK_SRC}"), Path("${SMOKE_MANIFESTS}/val_rank_smoke.csv"), 24)
subset(Path("${VAL_SELECT_SRC}"), Path("${SMOKE_MANIFESTS}/val_select_smoke.csv"), 24)
subset(Path("${TEST_SRC}"), Path("${SMOKE_MANIFESTS}/test_smoke.csv"), 24)

voicebank_test_expected = 824
voicebank_full_test = Path("${SMOKE_MANIFESTS}/voicebank_test_full.csv")
dns5_full_test = Path("${SMOKE_MANIFESTS}/dns5_test_full.csv")
with Path("${TEST_SRC}").open("r", newline="", encoding="utf-8") as in_f:
    reader = csv.DictReader(in_f)
    rows = list(reader)
if len(rows) >= voicebank_test_expected:
    with voicebank_full_test.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["noisy", "clean"])
        writer.writeheader()
        writer.writerows(rows[:voicebank_test_expected])
    with dns5_full_test.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["noisy", "clean"])
        writer.writeheader()
        writer.writerows(rows[voicebank_test_expected:])
    subset(voicebank_full_test, Path("${SMOKE_MANIFESTS}/voicebank_test_smoke.csv"), 12)
    subset(dns5_full_test, Path("${SMOKE_MANIFESTS}/dns5_test_smoke.csv"), 12)

if Path("${PER_DOMAIN_SRC}").is_dir():
    subset(Path("${PER_DOMAIN_SRC}/voicebank_train_fit.csv"), Path("${SMOKE_MANIFESTS}/voicebank_train_fit_smoke.csv"), 32)
    subset(Path("${PER_DOMAIN_SRC}/voicebank_val_rank.csv"), Path("${SMOKE_MANIFESTS}/voicebank_val_rank_smoke.csv"), 12)
    subset(Path("${PER_DOMAIN_SRC}/voicebank_val_select.csv"), Path("${SMOKE_MANIFESTS}/voicebank_val_select_smoke.csv"), 12)
    subset(Path("${PER_DOMAIN_SRC}/dns5_train_fit.csv"), Path("${SMOKE_MANIFESTS}/dns5_train_fit_smoke.csv"), 96)
    subset(Path("${PER_DOMAIN_SRC}/dns5_val_rank.csv"), Path("${SMOKE_MANIFESTS}/dns5_val_rank_smoke.csv"), 12)
    subset(Path("${PER_DOMAIN_SRC}/dns5_val_select.csv"), Path("${SMOKE_MANIFESTS}/dns5_val_select_smoke.csv"), 12)
PY

METRICGAN_SMOKE_CFG="${SMOKE_CFG_DIR}/metricgan_smoke.yaml"
VOICEBANK_TRAIN_FIT_ARG=()
VOICEBANK_VAL_RANK_ARG=()
VOICEBANK_VAL_SELECT_ARG=()
VOICEBANK_TEST_ARG=()
DNS5_TRAIN_FIT_ARG=()
DNS5_VAL_RANK_ARG=()
DNS5_VAL_SELECT_ARG=()
DNS5_TEST_ARG=()
if [[ -f "${SMOKE_MANIFESTS}/voicebank_train_fit_smoke.csv" ]]; then
  VOICEBANK_TRAIN_FIT_ARG=(--voicebank-train-fit-manifest "${SMOKE_MANIFESTS}/voicebank_train_fit_smoke.csv")
fi
if [[ -f "${SMOKE_MANIFESTS}/voicebank_val_rank_smoke.csv" ]]; then
  VOICEBANK_VAL_RANK_ARG=(--voicebank-val-rank-manifest "${SMOKE_MANIFESTS}/voicebank_val_rank_smoke.csv")
fi
if [[ -f "${SMOKE_MANIFESTS}/voicebank_val_select_smoke.csv" ]]; then
  VOICEBANK_VAL_SELECT_ARG=(--voicebank-val-select-manifest "${SMOKE_MANIFESTS}/voicebank_val_select_smoke.csv")
fi
if [[ -f "${SMOKE_MANIFESTS}/voicebank_test_smoke.csv" ]]; then
  VOICEBANK_TEST_ARG=(--voicebank-test-manifest "${SMOKE_MANIFESTS}/voicebank_test_smoke.csv")
fi
if [[ -f "${SMOKE_MANIFESTS}/dns5_train_fit_smoke.csv" ]]; then
  DNS5_TRAIN_FIT_ARG=(--dns5-train-fit-manifest "${SMOKE_MANIFESTS}/dns5_train_fit_smoke.csv")
fi
if [[ -f "${SMOKE_MANIFESTS}/dns5_val_rank_smoke.csv" ]]; then
  DNS5_VAL_RANK_ARG=(--dns5-val-rank-manifest "${SMOKE_MANIFESTS}/dns5_val_rank_smoke.csv")
fi
if [[ -f "${SMOKE_MANIFESTS}/dns5_val_select_smoke.csv" ]]; then
  DNS5_VAL_SELECT_ARG=(--dns5-val-select-manifest "${SMOKE_MANIFESTS}/dns5_val_select_smoke.csv")
fi
if [[ -f "${SMOKE_MANIFESTS}/dns5_test_smoke.csv" ]]; then
  DNS5_TEST_ARG=(--dns5-test-manifest "${SMOKE_MANIFESTS}/dns5_test_smoke.csv")
fi
"${VENV_PYTHON}" "${THIS_DIR}/generate_metricgan_runtime_config.py" \
  --base-config "${METRICGAN_PROJECT}/configs/scenario_combined_datasets_kingston_runtime.yaml" \
  --output-config "${METRICGAN_SMOKE_CFG}" \
  --output-root "${SMOKE_ROOT}/metricgan/outputs" \
  --tracking-root "${SMOKE_ROOT}/metricgan/tracking" \
  --train-manifest "${SMOKE_MANIFESTS}/train_smoke.csv" \
  --val-rank-manifest "${SMOKE_MANIFESTS}/val_rank_smoke.csv" \
  --val-select-manifest "${SMOKE_MANIFESTS}/val_select_smoke.csv" \
  --test-manifest "${SMOKE_MANIFESTS}/test_smoke.csv" \
  "${VOICEBANK_TRAIN_FIT_ARG[@]}" \
  "${VOICEBANK_VAL_RANK_ARG[@]}" \
  "${VOICEBANK_VAL_SELECT_ARG[@]}" \
  "${VOICEBANK_TEST_ARG[@]}" \
  "${DNS5_TRAIN_FIT_ARG[@]}" \
  "${DNS5_VAL_RANK_ARG[@]}" \
  "${DNS5_VAL_SELECT_ARG[@]}" \
  "${DNS5_TEST_ARG[@]}" \
  --experiment-name "metricgan_kingston_smoke"

"${VENV_PYTHON}" - <<PY
from pathlib import Path
import yaml
p = Path("${METRICGAN_SMOKE_CFG}")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["teacher_training"]["epochs"] = 1
cfg["teacher_training"]["min_epochs"] = 1
cfg["teacher_training"]["early_stop_patience"] = 1
cfg["teacher_training"]["phase_a"]["epochs"] = 1
cfg["teacher_training"]["phase_a"]["min_epochs"] = 1
cfg["teacher_training"]["phase_a"]["early_stop_patience"] = 1
cfg["teacher_training"]["phase_b"]["epochs"] = 1
cfg["teacher_training"]["phase_b"]["min_epochs"] = 1
cfg["teacher_training"]["phase_b"]["early_stop_patience"] = 1
cfg["teacher_training"]["phase_c"]["epochs"] = 1
cfg["teacher_training"]["phase_c"]["min_epochs"] = 1
cfg["teacher_training"]["phase_c"]["early_stop_patience"] = 1
cfg["teacher_training"]["pesq_proxy"]["epochs"] = 1
cfg["stage1"]["phase_s1"]["epochs"] = 1
cfg["stage1"]["phase_s1"]["min_epochs"] = 1
cfg["stage1"]["phase_s1"]["early_stop_patience"] = 1
cfg["stage1"]["phase_s2"]["epochs"] = 1
cfg["stage1"]["phase_s2"]["min_epochs"] = 1
cfg["stage1"]["phase_s2"]["early_stop_patience"] = 1
cfg["stage1"]["target_pesq_floor"] = -999.0
cfg["training"]["eval_every"] = 1
cfg["training"]["rank_eval_every"] = 1
cfg["training"]["select_eval_every"] = 1
cfg["training"]["checkpoint_every_minutes"] = 60
cfg["training"]["checkpoint_keep_last"] = 1
cfg["training"]["autotune_loader"]["enabled"] = False
cfg["training"]["num_workers"] = 2
cfg["teacher_cache"]["num_workers"] = 2
cfg["teacher_cache"]["batch_size"] = 4
p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

mkdir -p "${SMOKE_ROOT}/logs"

echo "[smoke] MetricGAN full pipeline smoke..."
(
  cd "${METRICGAN_PROJECT}"
  CUDA_VISIBLE_DEVICES="${METRICGAN_GPU}" "${VENV_PYTHON}" repro.py \
    --config "${METRICGAN_SMOKE_CFG}" run_all --device cuda:0 \
    > "${SMOKE_ROOT}/logs/metricgan_smoke_run_all.log" 2>&1
)

echo "[smoke] ultra cache mini..."
ULTRA_SMOKE_CACHE="${SMOKE_ROOT}/ultra/cache"
mkdir -p "${ULTRA_SMOKE_CACHE}"
(
  cd "${ULTRA_PROJECT}"
  "${VENV_PYTHON}" core/features/create_training_data.py \
    --csv "${SMOKE_MANIFESTS}/train_smoke.csv" \
    --outdir "${ULTRA_SMOKE_CACHE}" \
    --subset-name train \
    --jobs 4
  "${VENV_PYTHON}" core/features/create_training_data.py \
    --csv "${SMOKE_MANIFESTS}/test_smoke.csv" \
    --outdir "${ULTRA_SMOKE_CACHE}" \
    --subset-name test \
    --jobs 4
)

echo "[smoke] ultra train smoke..."
(
  cd "${ULTRA_PROJECT}"
  CUDA_VISIBLE_DEVICES="${ULTRA_GPU}" "${VENV_PYTHON}" core/train.py \
    --train-manifest "${ULTRA_SMOKE_CACHE}/train_manifest.csv" \
    --test-manifest "${ULTRA_SMOKE_CACHE}/test_manifest.csv" \
    --train-stats "${ULTRA_SMOKE_CACHE}/train_feature_stats.npz" \
    --outdir "${SMOKE_ROOT}/ultra/train_smoke" \
    --batch-size 8 \
    --epochs 1 \
    --lr 1e-3 \
    --stft-center true \
    --log-base ln \
    --log-eps 1e-6 \
    --peak-target 0.95 \
    > "${SMOKE_ROOT}/logs/ultra_smoke_train.log" 2>&1
)

echo "[smoke] Done: ${SMOKE_ROOT}"
