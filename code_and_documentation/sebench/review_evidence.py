"""Reviewer-requested current-protocol baselines and uncertainty evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from sebench.audio import resample_mono_audio
from sebench.checkpoints import load_model_from_checkpoint
from sebench.t3_training import sha256_file
from sebench.training import evaluate_manifest


class NoisyPassthrough(nn.Module):
    """Identity enhancement baseline."""

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform

    def denoise_single(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform


class BandwidthLimitedTeacher(nn.Module):
    """Run the WB teacher using only an 8-kHz-bandlimited input."""

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher

    def forward(self, waveform_8k: torch.Tensor) -> torch.Tensor:
        original_length = int(waveform_8k.shape[-1])
        waveform_16k = resample_mono_audio(waveform_8k, 8_000, 16_000)
        enhanced_16k = self.teacher(waveform_16k)
        enhanced_8k = resample_mono_audio(enhanced_16k, 16_000, 8_000)
        return enhanced_8k[..., :original_length]

    def denoise_single(self, waveform_8k: torch.Tensor) -> torch.Tensor:
        if waveform_8k.ndim != 2:
            raise ValueError("Expected waveform shaped (batch, length).")
        original_length = int(waveform_8k.shape[-1])
        waveform_16k = resample_mono_audio(waveform_8k, 8_000, 16_000)
        if hasattr(self.teacher, "denoise_single"):
            enhanced_16k = self.teacher.denoise_single(waveform_16k)
        else:
            enhanced_16k = self.teacher(waveform_16k.unsqueeze(1)).squeeze(1)
        enhanced_8k = resample_mono_audio(enhanced_16k, 16_000, 8_000)
        return enhanced_8k[..., :original_length]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_current_protocol_baselines(
    *,
    teacher_checkpoint: str | Path,
    test_manifest: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    max_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Reviewer baseline evaluation is CUDA-only.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    teacher, _ = load_model_from_checkpoint(teacher_checkpoint, device=device)
    systems = {
        "NOISY-WB": (NoisyPassthrough().to(device), "wb", 16_000),
        "NOISY-NB": (NoisyPassthrough().to(device), "nb", 8_000),
        "MATCHED-INPUT-TEACHER-NB": (
            BandwidthLimitedTeacher(teacher).to(device),
            "nb",
            8_000,
        ),
    }
    results: dict[str, Any] = {}
    for name, (model, bandwidth, sample_rate) in systems.items():
        results[name] = evaluate_manifest(
            model,
            str(test_manifest),
            device,
            sample_rate=sample_rate,
            bandwidth=bandwidth,
            compute_dnsmos=False,
            compute_composite=False,
            max_files=max_files,
            batch_size=1,
            cache_audio=True,
            progress_callback=progress_callback,
            sample_metrics_out=root / "sample_metrics" / f"{name}.csv",
        )
    summary = {
        "schema_version": 1,
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "test_manifest_sha256": sha256_file(test_manifest),
        "max_files": max_files,
        "test_is_selection_input": False,
        "systems": results,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int = 20260730,
    draws: int = 10_000,
) -> dict[str, float]:
    """Paired percentile bootstrap for mean(left - right)."""
    lhs = np.asarray(left, dtype=np.float64).reshape(-1)
    rhs = np.asarray(right, dtype=np.float64).reshape(-1)
    if lhs.shape != rhs.shape or lhs.size < 2:
        raise ValueError("Paired bootstrap inputs must have the same support.")
    generator = np.random.default_rng(int(seed))
    deltas = lhs - rhs
    means = np.empty(int(draws), dtype=np.float64)
    for start in range(0, int(draws), 1_000):
        count = min(1_000, int(draws) - start)
        indices = generator.integers(
            0,
            deltas.size,
            size=(count, deltas.size),
        )
        means[start : start + count] = deltas[indices].mean(axis=1)
    return {
        "count": int(deltas.size),
        "draws": int(draws),
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }
