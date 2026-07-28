"""True-PESQ confidence-conditioned mask-logit search for T7."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_training import clone_state_dict, sha256_file
from sebench.t5_zeroth_order import prepare_t5_support_manifests
from sebench.training import evaluate_manifest


T7_LOWS = (-0.20, -0.30, -0.40)
T7_HIGHS = (0.00, 0.05)
T7_THRESHOLDS = (-6.0, -4.0, -2.0, 0.0)
T7_TEMPERATURE = 1.5
T7_SUPPORT_START = 384
T7_FIT_SUPPORT = 96
T7_CALIBRATION_SUPPORT = 96


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


def configure_confidence_calibration(
    model: torch.nn.Module,
    *,
    enabled: bool,
    low: float = 0.0,
    high: float = 0.0,
    threshold: float = -4.0,
    temperature: float = T7_TEMPERATURE,
) -> None:
    """Configure the official teacher's deployable T7 transform."""
    target = model.base_model if hasattr(model, "base_model") else model
    configure = getattr(target, "configure_confidence_calibration", None)
    if configure is None:
        raise TypeError("T7 requires a confidence-calibratable MetricGAN teacher.")
    values = (low, high, threshold, temperature)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("T7 calibration values must be finite.")
    if float(temperature) <= 0.0:
        raise ValueError("T7 temperature must be positive.")
    configure(
        enabled=bool(enabled),
        low=float(low),
        high=float(high),
        threshold=float(threshold),
        temperature=float(temperature),
    )


def confidence_candidate_grid(
    *,
    lows: Iterable[float] = T7_LOWS,
    highs: Iterable[float] = T7_HIGHS,
    thresholds: Iterable[float] = T7_THRESHOLDS,
    temperature: float = T7_TEMPERATURE,
) -> list[dict[str, float]]:
    candidates = [
        {
            "low": float(low),
            "high": float(high),
            "threshold": float(threshold),
            "temperature": float(temperature),
        }
        for low, high, threshold in product(lows, highs, thresholds)
    ]
    if not candidates:
        raise ValueError("T7 requires at least one candidate.")
    for candidate in candidates:
        if candidate["low"] > candidate["high"]:
            raise ValueError("T7 requires low <= high.")
    return candidates


def run_t7_confidence_search(
    *,
    teacher_checkpoint: str | Path,
    identities_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    lows: Iterable[float] = T7_LOWS,
    highs: Iterable[float] = T7_HIGHS,
    thresholds: Iterable[float] = T7_THRESHOLDS,
    temperature: float = T7_TEMPERATURE,
    fit_count: int = T7_FIT_SUPPORT,
    calibration_count: int = T7_CALIBRATION_SUPPORT,
    support_start: int = T7_SUPPORT_START,
    top_fit: int = 8,
    top_calibration: int = 4,
    top_rank: int = 2,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T7 true-PESQ search is CUDA-only.")
    candidates = confidence_candidate_grid(
        lows=lows,
        highs=highs,
        thresholds=thresholds,
        temperature=temperature,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    support = prepare_t5_support_manifests(
        identities_path,
        root / "support",
        fit_count=fit_count,
        calibration_count=calibration_count,
        start_index=support_start,
    )
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    base_state = clone_state_dict(model)

    def evaluate(
        candidate: dict[str, float] | None,
        manifest: str,
    ) -> dict[str, float]:
        model.load_state_dict(base_state)
        if candidate is None:
            configure_confidence_calibration(model, enabled=False)
        else:
            configure_confidence_calibration(model, enabled=True, **candidate)
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

    fit_manifest = str(support["fit"]["manifest"])
    calibration_manifest = str(support["calibration"]["manifest"])
    baseline_fit = evaluate(None, fit_manifest)
    baseline_calibration = evaluate(None, calibration_manifest)
    if max_eval_files is not None:
        baseline_rank_metrics = evaluate(None, str(val_rank_manifest))
        baseline_select_metrics = evaluate(None, str(val_select_manifest))

    fit_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        metrics = evaluate(candidate, fit_manifest)
        checks = _checks(metrics, baseline_fit)
        fit_candidates.append(
            {
                "candidate": candidate,
                "fit_metrics": metrics,
                "fit_checks": checks,
                "fit_eligible": all(checks.values()),
            }
        )
        if progress_callback:
            progress_callback(
                f"T7 fit candidate={index}/{len(candidates)} "
                f"low={candidate['low']:+.2f} high={candidate['high']:+.2f} "
                f"threshold={candidate['threshold']:+.1f} "
                f"pesq={metrics['pesq_mean']:.6f}"
            )
    fit_shortlist = sorted(
        (row for row in fit_candidates if row["fit_eligible"]),
        key=lambda row: float(row["fit_metrics"]["pesq_mean"]),
        reverse=True,
    )[: int(top_fit)]

    calibration_candidates: list[dict[str, Any]] = []
    for row in fit_shortlist:
        metrics = evaluate(row["candidate"], calibration_manifest)
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

    rank_screen: list[dict[str, Any]] = []
    for index, row in enumerate(calibration_shortlist, start=1):
        metrics = evaluate(row["candidate"], str(val_rank_manifest))
        checks = _checks(metrics, baseline_rank_metrics)
        rank_screen.append(
            {
                **row,
                "name": f"T7-CANDIDATE-{index}",
                "val_rank_metrics": metrics,
                "rank_checks": checks,
                "rank_eligible": all(checks.values()),
            }
        )
        if progress_callback:
            progress_callback(
                f"T7 rank candidate={index} pesq={metrics['pesq_mean']:.6f}"
            )
    rank_shortlist = sorted(
        (row for row in rank_screen if row["rank_eligible"]),
        key=lambda row: float(row["val_rank_metrics"]["pesq_mean"]),
        reverse=True,
    )[: int(top_rank)]
    rank_candidates: list[dict[str, Any]] = [
        {
            "name": "T0",
            "candidate": None,
            "val_rank_metrics": _triplet(baseline_rank_metrics),
            "rank_checks": {
                "finite_metrics": True,
                "stoi_guardrail": True,
                "sisdr_guardrail": True,
            },
            "rank_eligible": True,
        },
        *rank_shortlist,
    ]
    selected = max(
        rank_candidates,
        key=lambda row: float(row["val_rank_metrics"]["pesq_mean"]),
    )
    selected_candidate = selected["candidate"]
    model.load_state_dict(base_state)
    if selected_candidate is None:
        configure_confidence_calibration(model, enabled=False)
    else:
        configure_confidence_calibration(
            model, enabled=True, **selected_candidate
        )
    selected_path = root / "T7-SELECTED.pt"
    save_checkpoint_package(
        selected_path,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T7-CONFIDENCE-CONDITIONED-LOGIT",
            "selected_name": selected["name"],
            "candidate": selected_candidate,
            "selection_split": "val_rank",
            "test_read": False,
        },
    )
    reloaded, reloaded_package = load_model_from_checkpoint(
        selected_path, device=device
    )
    reloaded_target = (
        reloaded.base_model if hasattr(reloaded, "base_model") else reloaded
    )
    roundtrip = {
        "enabled": bool(
            getattr(reloaded_target, "confidence_calibration_enabled", False)
        ),
        "model_config": dict(reloaded_package.get("model_config", {})),
    }

    if selected_candidate is None:
        select_metrics = _triplet(baseline_select_metrics)
    else:
        select_metrics = evaluate(selected_candidate, str(val_select_manifest))
    deltas = {
        key: float(select_metrics[key]) - float(baseline_select_metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }
    gate_checks = {
        "nonzero_candidate": selected_candidate is not None,
        "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
        "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
        "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
        "checkpoint_roundtrip": (
            selected_candidate is None or roundtrip["enabled"]
        ),
        "production_support": (
            max_eval_files is None
            and int(fit_count) == T7_FIT_SUPPORT
            and int(calibration_count) == T7_CALIBRATION_SUPPORT
            and int(support_start) == T7_SUPPORT_START
            and candidates == confidence_candidate_grid()
            and int(top_fit) == 8
            and int(top_calibration) == 4
            and int(top_rank) == 2
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T7-CONFIDENCE-CONDITIONED-LOGIT",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "support": support,
        "candidate_grid": candidates,
        "baseline_fit_metrics": baseline_fit,
        "baseline_calibration_metrics": baseline_calibration,
        "fit_candidates": fit_candidates,
        "calibration_candidates": calibration_candidates,
        "rank_candidates": rank_candidates,
        "selected_name": selected["name"],
        "selected_candidate": selected_candidate,
        "selected_val_rank_metrics": selected["val_rank_metrics"],
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": deltas,
        "checkpoint_roundtrip": roundtrip,
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "selected_checkpoint": selected_path.as_posix(),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "test_read": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
