"""Bounded, deterministic mask-logit calibration for T4-A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_training import clone_state_dict, sha256_file
from sebench.training import evaluate_manifest


T4_LOGIT_BIAS_GRID = (
    -0.10,
    -0.08,
    -0.06,
    -0.04,
    -0.02,
    -0.01,
    0.0,
    0.01,
    0.02,
    0.04,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_uniform_mask_logit_bias(
    model: torch.nn.Module,
    delta: float,
) -> None:
    """Fold a bounded inference logit shift into the ordinary checkpoint."""
    if abs(float(delta)) > 0.10:
        raise ValueError("T4 uniform mask-logit bias is bounded to +/-0.10.")
    target = model.base_model if hasattr(model, "base_model") else model
    generator = getattr(target, "mask_generator", None)
    linear = getattr(generator, "linear2", None)
    if not isinstance(linear, torch.nn.Linear) or linear.bias is None:
        raise TypeError("T4 requires the official MetricGAN mask linear2 bias.")
    with torch.no_grad():
        linear.bias.add_(float(delta))


def _triplet(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }


def run_t4_logit_bias_scan(
    *,
    teacher_checkpoint: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T4 teacher calibration is CUDA-only.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    base_state = clone_state_dict(model)
    candidates: list[dict[str, Any]] = []
    for index, delta in enumerate(T4_LOGIT_BIAS_GRID, start=1):
        model.load_state_dict(base_state)
        apply_uniform_mask_logit_bias(model, delta)
        metrics = evaluate_manifest(
            model,
            str(val_rank_manifest),
            device,
            sample_rate=16_000,
            bandwidth="wb",
            compute_dnsmos=False,
            compute_composite=False,
            batch_size=1,
            progress_callback=progress_callback,
        )
        triplet = _triplet(metrics)
        checks = {
            "stoi_guardrail": triplet["stoi_mean"]
            >= float(baseline_rank_metrics["stoi_mean"]) - 0.002,
            "sisdr_guardrail": triplet["sisdr_mean"]
            >= float(baseline_rank_metrics["sisdr_mean"]) - 0.25,
        }
        candidates.append(
            {
                "delta": float(delta),
                "val_rank_metrics": triplet,
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
        if progress_callback:
            progress_callback(
                f"T4 logit candidate {index}/{len(T4_LOGIT_BIAS_GRID)} "
                f"delta={delta:+.3f} pesq={triplet['pesq_mean']:.6f}"
            )
    zero = next(item for item in candidates if item["delta"] == 0.0)
    for key in ("pesq_mean", "stoi_mean", "sisdr_mean"):
        if (
            abs(
                float(zero["val_rank_metrics"][key])
                - float(baseline_rank_metrics[key])
            )
            > 1e-7
        ):
            raise RuntimeError(f"T4 zero-delta baseline mismatch for {key}.")
    selected = max(
        (item for item in candidates if item["eligible"]),
        key=lambda item: float(item["val_rank_metrics"]["pesq_mean"]),
    )
    selected_delta = float(selected["delta"])
    model.load_state_dict(base_state)
    apply_uniform_mask_logit_bias(model, selected_delta)
    checkpoint_path = root / "T4-A-SELECTED.pt"
    save_checkpoint_package(
        checkpoint_path,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T4-A-UNIFORM-LOGIT-BIAS",
            "uniform_mask_logit_bias": selected_delta,
            "selection_split": "val_rank",
            "test_read": False,
        },
    )
    if selected_delta == 0.0:
        select_metrics = _triplet(baseline_select_metrics)
    else:
        select_metrics = _triplet(
            evaluate_manifest(
                model,
                str(val_select_manifest),
                device,
                sample_rate=16_000,
                bandwidth="wb",
                compute_dnsmos=False,
                compute_composite=False,
                batch_size=1,
                progress_callback=progress_callback,
            )
        )
    deltas = {
        key: float(select_metrics[key]) - float(baseline_select_metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }
    gate_checks = {
        "nonzero_candidate": selected_delta != 0.0,
        "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
        "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
        "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T4-A-UNIFORM-LOGIT-BIAS",
        "grid": list(T4_LOGIT_BIAS_GRID),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "test_read": False,
        "candidates": candidates,
        "selected_delta": selected_delta,
        "selected_val_rank_metrics": selected["val_rank_metrics"],
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": deltas,
        "gate": {
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
        "selected_checkpoint": checkpoint_path.as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    _atomic_json(root / "summary.json", summary)
    return summary
