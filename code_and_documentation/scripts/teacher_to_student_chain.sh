#!/usr/bin/env bash
set -euo pipefail

CFG="configs/scenario_combined_datasets.yaml"
PID_FILE="/mnt/ldm/DNS-Challenge/logs/metricgan_teacher_train.pid"
PY="/home/USER/Desktop/ULP/shared-venv/bin/python3"

echo "[$(date '+%F %T')] chain started"

while true; do
  if [[ ! -f "$PID_FILE" ]]; then
    sleep 30
    continue
  fi

  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
    sleep 60
    continue
  fi
  break
done

echo "[$(date '+%F %T')] teacher finished, start cache+student"

"$PY" repro.py --config "$CFG" build_teacher_cache --force
"$PY" repro.py --config "$CFG" train_stage1 --device auto
"$PY" repro.py --config "$CFG" train_qat --device auto

echo "[$(date '+%F %T')] chain finished"
