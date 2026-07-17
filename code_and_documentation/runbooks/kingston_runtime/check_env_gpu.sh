#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/env_server.sh"

echo "[check_env] STACK_ROOT=${STACK_ROOT}"
echo "[check_env] VENV_PYTHON=${VENV_PYTHON}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[check_env] nvidia-smi:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "[check_env] WARNING: nvidia-smi not found."
fi

"${VENV_PYTHON}" - <<'PY'
import json
import platform

payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
}

try:
    import torch
    payload["torch_version"] = torch.__version__
    payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    payload["torch_cuda_device_count"] = int(torch.cuda.device_count())
except Exception as exc:
    payload["torch_error"] = str(exc)

try:
    import tensorflow as tf
    payload["tf_version"] = tf.__version__
    payload["tf_gpu_count"] = len(tf.config.list_physical_devices("GPU"))
except Exception as exc:
    payload["tf_error"] = str(exc)

for module in ("pesq", "pystoi", "librosa", "soundfile", "onnxruntime", "yaml"):
    try:
        __import__(module)
        payload[f"import_{module}"] = True
    except Exception as exc:
        payload[f"import_{module}"] = False
        payload[f"import_{module}_error"] = str(exc)

print(json.dumps(payload, indent=2, sort_keys=True))
PY

"${VENV_PYTHON}" "${THIS_DIR}/verify_pesq_consistency.py" \
  --metricgan-project "${METRICGAN_PROJECT}" \
  --ultra-project "${ULTRA_PROJECT}"

echo "[check_env] OK"
