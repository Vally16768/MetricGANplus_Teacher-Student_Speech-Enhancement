from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def require_cuda_device(device: str | None) -> str:
    requested = (device or "").strip().lower()
    if requested in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if not requested.startswith("cuda"):
        raise ValueError(f"Unsupported device {device!r}. Use cpu, cuda or cuda:N.")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available on this machine.")
    if requested == "cuda":
        return requested
    try:
        index = int(requested.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid CUDA device {device!r}. Use cuda or cuda:N.") from exc
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise ValueError(f"Invalid CUDA device index {index}; available device count is {device_count}.")
    return requested


def require_training_cuda(device: str | None) -> str:
    """Resolve a training device and reject every CPU fallback."""
    resolved = require_cuda_device(device)
    if not resolved.startswith("cuda"):
        raise RuntimeError(
            "Training is GPU-only in this project. Use --device cuda or cuda:N."
        )
    return resolved


def require_shared_venv(default_path: str | Path) -> Path:
    """Require the configured shared virtual environment without storing a personal path."""
    configured = os.environ.get("METRICGAN_SHARED_VENV", "").strip()
    expected = Path(configured) if configured else Path(default_path)
    expected = expected.expanduser().resolve(strict=False)
    active = Path(sys.prefix).resolve(strict=False)
    if active != expected:
        raise RuntimeError(
            "Training must run from the shared project virtual environment. "
            f"Active prefix={active}; expected prefix={expected}. "
            "Set METRICGAN_SHARED_VENV if the shared environment is not a sibling "
            "of the repository."
        )
    return expected
