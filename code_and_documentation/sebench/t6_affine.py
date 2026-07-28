"""True-PESQ affine-logit calibration for T6."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_training import clone_state_dict, sha256_file
from sebench.t5_zeroth_order import (
    apply_frequency_logit_curve,
    frequency_curve_from_knots,
    prepare_t5_support_manifests,
)
from sebench.training import evaluate_manifest


T6_SCALES = (0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40)
T6_SUPPORT_START = 192


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _triplet(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }


def _checks(metrics: dict[str, float], baseline: dict[str, Any]) -> dict[str, bool]:
    return {
        "finite_metrics": all(math.isfinite(value) for value in metrics.values()),
        "stoi_guardrail": metrics["stoi_mean"]
        >= float(baseline["stoi_mean"]) - 0.002,
        "sisdr_guardrail": metrics["sisdr_mean"]
        >= float(baseline["sisdr_mean"]) - 0.25,
    }


def apply_affine_logit_calibration(
    model: torch.nn.Module,
    *,
    scale: float,
    coefficients: Iterable[float],
) -> torch.Tensor:
    """Fold `scale * original_logit + curve` into the final linear layer."""
    value = float(scale)
    if not math.isfinite(value) or not 0.5 <= value <= 1.5:
        raise ValueError("T6 logit scale must be finite and in [0.5, 1.5].")
    target = model.base_model if hasattr(model, "base_model") else model
    linear = getattr(getattr(target, "mask_generator", None), "linear2", None)
    if not isinstance(linear, torch.nn.Linear) or linear.bias is None:
        raise TypeError("T6 requires the official MetricGAN mask linear2 layer.")
    with torch.no_grad():
        linear.weight.mul_(value)
        linear.bias.mul_(value)
    return apply_frequency_logit_curve(model, coefficients)


def run_t6_affine_search(
    *,
    teacher_checkpoint: str | Path,
    identities_path: str | Path,
    t5_summary_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    scales: Iterable[float] = T6_SCALES,
    fit_count: int = 96,
    calibration_count: int = 96,
    support_start: int = T6_SUPPORT_START,
    top_fit: int = 5,
    top_calibration: int = 3,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T6 true-PESQ search is CUDA-only.")
    normalized_scales = tuple(float(item) for item in scales)
    if not normalized_scales:
        raise ValueError("T6 requires at least one logit scale.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    support = prepare_t5_support_manifests(
        identities_path,
        root / "support",
        fit_count=fit_count,
        calibration_count=calibration_count,
        start_index=support_start,
    )
    t5 = json.loads(Path(t5_summary_path).read_text(encoding="utf-8"))
    sweep_by_id = {int(row["sweep"]): row for row in t5["sweeps"]}
    curves = {
        "T5-SWEEP-1": list(sweep_by_id[1]["coefficients"]),
        "T5-SWEEP-3": list(sweep_by_id[3]["coefficients"]),
    }
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    base_state = clone_state_dict(model)

    def evaluate(
        scale: float | None,
        coefficients: list[float] | None,
        manifest: str,
    ) -> dict[str, float]:
        model.load_state_dict(base_state)
        if scale is not None and coefficients is not None:
            apply_affine_logit_calibration(
                model, scale=scale, coefficients=coefficients
            )
        return _triplet(
            evaluate_manifest(
                model,
                manifest,
                device,
                sample_rate=16_000,
                bandwidth="wb",
                compute_dnsmos=False,
                compute_composite=False,
                max_files=max_eval_files,
                batch_size=1,
                progress_callback=progress_callback,
            )
        )

    fit_manifest = support["fit"]["manifest"]
    calibration_manifest = support["calibration"]["manifest"]
    baseline_fit = evaluate(None, None, fit_manifest)
    baseline_calibration = evaluate(None, None, calibration_manifest)
    if max_eval_files is not None:
        baseline_rank_metrics = evaluate(None, None, str(val_rank_manifest))
        baseline_select_metrics = evaluate(None, None, str(val_select_manifest))
    fit_candidates: list[dict[str, Any]] = []
    for curve_name, coefficients in curves.items():
        for scale in normalized_scales:
            metrics = evaluate(scale, coefficients, fit_manifest)
            checks = _checks(metrics, baseline_fit)
            fit_candidates.append(
                {
                    "curve": curve_name,
                    "coefficients": coefficients,
                    "scale": scale,
                    "fit_metrics": metrics,
                    "fit_checks": checks,
                    "fit_eligible": all(checks.values()),
                }
            )
            if progress_callback:
                progress_callback(
                    f"T6 fit {curve_name} scale={scale:.2f} "
                    f"pesq={metrics['pesq_mean']:.6f}"
                )
    fit_shortlist = sorted(
        (row for row in fit_candidates if row["fit_eligible"]),
        key=lambda row: float(row["fit_metrics"]["pesq_mean"]),
        reverse=True,
    )[: int(top_fit)]
    calibration_candidates: list[dict[str, Any]] = []
    for row in fit_shortlist:
        metrics = evaluate(
            float(row["scale"]),
            list(row["coefficients"]),
            calibration_manifest,
        )
        checks = _checks(metrics, baseline_calibration)
        calibration_candidates.append(
            {
                **row,
                "calibration_metrics": metrics,
                "calibration_checks": checks,
                "calibration_eligible": all(checks.values()),
            }
        )
    calibration_shortlist = sorted(
        (row for row in calibration_candidates if row["calibration_eligible"]),
        key=lambda row: float(row["calibration_metrics"]["pesq_mean"]),
        reverse=True,
    )[: int(top_calibration)]
    rank_candidates: list[dict[str, Any]] = [
        {
            "name": "T0",
            "scale": None,
            "coefficients": None,
            "val_rank_metrics": _triplet(baseline_rank_metrics),
            "checks": {"finite_metrics": True, "stoi_guardrail": True, "sisdr_guardrail": True},
            "eligible": True,
        }
    ]
    for index, row in enumerate(calibration_shortlist, start=1):
        metrics = evaluate(
            float(row["scale"]),
            list(row["coefficients"]),
            str(val_rank_manifest),
        )
        checks = _checks(metrics, baseline_rank_metrics)
        rank_candidates.append(
            {
                "name": f"T6-CANDIDATE-{index}",
                "curve": row["curve"],
                "scale": row["scale"],
                "coefficients": row["coefficients"],
                "val_rank_metrics": metrics,
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
        if progress_callback:
            progress_callback(
                f"T6 rank candidate={index} pesq={metrics['pesq_mean']:.6f}"
            )
    selected = max(
        (row for row in rank_candidates if row["eligible"]),
        key=lambda row: float(row["val_rank_metrics"]["pesq_mean"]),
    )
    model.load_state_dict(base_state)
    if selected["scale"] is not None:
        curve = apply_affine_logit_calibration(
            model,
            scale=float(selected["scale"]),
            coefficients=list(selected["coefficients"]),
        )
    else:
        curve = torch.zeros(257)
    selected_path = root / "T6-SELECTED.pt"
    save_checkpoint_package(
        selected_path,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T6-TRUE-PESQ-AFFINE-LOGIT",
            "selected_name": selected["name"],
            "scale": selected["scale"],
            "coefficients": selected["coefficients"],
            "curve_min": float(curve.min()),
            "curve_max": float(curve.max()),
            "test_read": False,
        },
    )
    if selected["scale"] is None:
        select_metrics = _triplet(baseline_select_metrics)
    else:
        select_metrics = evaluate(
            float(selected["scale"]),
            list(selected["coefficients"]),
            str(val_select_manifest),
        )
    deltas = {
        key: float(select_metrics[key]) - float(baseline_select_metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }
    gate_checks = {
        "nonzero_candidate": selected["scale"] is not None,
        "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
        "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
        "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
        "production_support": (
            max_eval_files is None
            and fit_count == 96
            and calibration_count == 96
            and support_start == T6_SUPPORT_START
            and normalized_scales == T6_SCALES
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T6-TRUE-PESQ-AFFINE-LOGIT",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "t5_summary_sha256": sha256_file(t5_summary_path),
        "support": support,
        "baseline_fit_metrics": baseline_fit,
        "baseline_calibration_metrics": baseline_calibration,
        "fit_candidates": fit_candidates,
        "calibration_candidates": calibration_candidates,
        "rank_candidates": rank_candidates,
        "selected_name": selected["name"],
        "selected_scale": selected["scale"],
        "selected_coefficients": selected["coefficients"],
        "selected_val_rank_metrics": selected["val_rank_metrics"],
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": deltas,
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "selected_checkpoint": selected_path.as_posix(),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "test_read": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
