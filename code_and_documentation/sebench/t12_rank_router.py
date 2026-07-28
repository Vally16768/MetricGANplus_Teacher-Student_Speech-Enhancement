"""Rank-selected risk policy for the frozen T9 multi-action router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_training import sha256_file
from sebench.t9_multi_router import (
    T9_ACTION_LOWS,
    _aggregate,
    _predict,
    collect_multi_action_records,
    configure_multi_action_router,
)
from sebench.t11_penalty_router import penalize_ridges
from sebench.training import evaluate_manifest


T12_PENALTIES = (
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.045,
    0.050,
)
T12_THRESHOLDS = (
    0.0000,
    0.0025,
    0.0050,
    0.0075,
    0.0100,
    0.0125,
    0.0150,
    0.0175,
    0.0200,
)


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


def rank_policy_grid(
    records: list[dict[str, Any]],
    raw_predictions: np.ndarray,
    *,
    lows: Iterable[float],
    penalties: Iterable[float],
    thresholds: Iterable[float],
    baseline_metrics: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate predeclared policies from one exact val_rank action pass."""
    action_lows = tuple(float(value) for value in lows)
    if not records or raw_predictions.shape != (len(records), len(action_lows)):
        raise ValueError("T12 policy-grid inputs are inconsistent.")
    baseline = (
        _triplet(baseline_metrics)
        if baseline_metrics is not None
        else _triplet(_aggregate(records, np.full(len(records), -1, dtype=np.int64)))
    )
    candidates: list[dict[str, Any]] = []
    for penalty in tuple(float(value) for value in penalties):
        adjusted = raw_predictions - penalty * np.square(
            np.asarray(action_lows, dtype=np.float64)
        ).reshape(1, -1)
        best_actions = np.argmax(adjusted, axis=1)
        best_scores = adjusted[np.arange(adjusted.shape[0]), best_actions]
        for threshold in tuple(float(value) for value in thresholds):
            decisions = np.where(best_scores >= threshold, best_actions, -1)
            metrics = _aggregate(records, decisions)
            deltas = {
                key: float(metrics[key]) - float(baseline[key])
                for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
            }
            checks = {
                "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
                "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
                "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
                "nonzero_router": int(metrics["base_count"]) < len(records),
            }
            candidates.append(
                {
                    "penalty": penalty,
                    "threshold": threshold,
                    "metrics": metrics,
                    "deltas": deltas,
                    "checks": checks,
                    "eligible": all(checks.values()),
                }
            )
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible if eligible else candidates
    selected = max(
        pool,
        key=lambda row: (
            row["deltas"]["pesq_mean"],
            row["deltas"]["sisdr_mean"],
            row["deltas"]["stoi_mean"],
            row["penalty"],
            row["threshold"],
        ),
    )
    return candidates, selected


def run_t12_rank_search(
    *,
    teacher_checkpoint: str | Path,
    t9_checkpoint: str | Path,
    t9_summary_path: str | Path,
    t11_summary_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    penalties: Iterable[float] = T12_PENALTIES,
    thresholds: Iterable[float] = T12_THRESHOLDS,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T12 rank-selected router search is CUDA-only.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    t9 = json.loads(Path(t9_summary_path).read_text(encoding="utf-8"))
    t11 = json.loads(Path(t11_summary_path).read_text(encoding="utf-8"))
    if (
        t11.get("status") != "failed"
        or t11.get("val_select_deltas") is None
        or float(t11["val_select_deltas"]["pesq_mean"]) >= 0.01
        or not bool(t11["gate"]["checks"]["stoi_drop_at_most_0_002"])
        or not bool(t11["gate"]["checks"]["sisdr_drop_at_most_0_25"])
        or bool(t11.get("test_read"))
    ):
        raise ValueError("T12 requires the auxiliary-safe below-PESQ T11 result.")
    lows = tuple(float(value) for value in t9["action_lows"])
    if lows != T9_ACTION_LOWS:
        raise ValueError("T12 requires the frozen T9 action set.")
    ridges = list(t9["ridges"])
    model, package = load_model_from_checkpoint(t9_checkpoint, device=device)
    records = collect_multi_action_records(
        model,
        val_rank_manifest,
        device=device,
        lows=lows,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    raw_predictions = np.stack(
        [
            _predict(
                ridge,
                np.asarray(
                    [row["actions"][index]["features"] for row in records],
                    dtype=np.float64,
                ),
            )
            for index, ridge in enumerate(ridges)
        ],
        axis=1,
    )
    observed_baseline = _triplet(
        _aggregate(records, np.full(len(records), -1, dtype=np.int64))
    )
    production = (
        max_eval_files is None
        and tuple(float(value) for value in penalties) == T12_PENALTIES
        and tuple(float(value) for value in thresholds) == T12_THRESHOLDS
    )
    baseline_reconciled = (
        not production
        or all(
            abs(observed_baseline[key] - float(baseline_rank_metrics[key])) <= 1e-5
            for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
        )
    )
    candidates, selected = rank_policy_grid(
        records,
        raw_predictions,
        lows=lows,
        penalties=penalties,
        thresholds=thresholds,
        baseline_metrics=(
            baseline_rank_metrics if production else observed_baseline
        ),
    )
    adjusted_ridges = penalize_ridges(
        ridges,
        lows,
        float(selected["penalty"]),
    )
    configure_multi_action_router(
        model,
        ridges=adjusted_ridges,
        threshold=float(selected["threshold"]),
        lows=lows,
    )
    checkpoint = root / "T12-RANK-SELECTED-ROUTED.pt"
    save_checkpoint_package(
        checkpoint,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T12-RANK-SELECTED-RISK-POLICY",
            "strength_penalty": selected["penalty"],
            "decision_threshold": selected["threshold"],
            "selection_split": "val_rank",
            "test_read": False,
        },
    )
    reloaded, reloaded_package = load_model_from_checkpoint(checkpoint, device=device)
    target = reloaded.base_model if hasattr(reloaded, "base_model") else reloaded
    roundtrip = bool(
        getattr(target, "multi_router_enabled", False)
        and tuple(getattr(target, "multi_router_lows", ())) == lows
        and float(getattr(target, "multi_router_threshold"))
        == float(selected["threshold"])
    )
    prevalidation = bool(selected["eligible"]) and baseline_reconciled and roundtrip

    select_metrics = select_deltas = None
    select_evaluated = bool(prevalidation and production)
    if select_evaluated:
        select_metrics = _triplet(
            evaluate_manifest(
                reloaded,
                str(val_select_manifest),
                device,
                sample_rate=16_000,
                bandwidth="wb",
                compute_dnsmos=False,
                compute_composite=False,
                max_files=None,
                batch_size=1,
                progress_callback=progress_callback,
            )
        )
        select_deltas = {
            key: select_metrics[key] - float(baseline_select_metrics[key])
            for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
        }
    gate_checks = {
        "prevalidation_passed": prevalidation,
        "rank_pesq_gain_at_least_0_01": bool(
            selected["checks"]["pesq_gain_at_least_0_01"]
        ),
        "rank_stoi_guardrail": bool(
            selected["checks"]["stoi_drop_at_most_0_002"]
        ),
        "rank_sisdr_guardrail": bool(
            selected["checks"]["sisdr_drop_at_most_0_25"]
        ),
        "pesq_gain_at_least_0_01": (
            select_deltas is not None and select_deltas["pesq_mean"] >= 0.01
        ),
        "stoi_drop_at_most_0_002": (
            select_deltas is not None and select_deltas["stoi_mean"] >= -0.002
        ),
        "sisdr_drop_at_most_0_25": (
            select_deltas is not None and select_deltas["sisdr_mean"] >= -0.25
        ),
        "checkpoint_roundtrip": roundtrip,
        "production_grid": production,
        "rank_baseline_reconciled": baseline_reconciled,
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T12-RANK-SELECTED-RISK-POLICY",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "source_t9_checkpoint_sha256": sha256_file(t9_checkpoint),
        "source_t9_summary_sha256": sha256_file(t9_summary_path),
        "source_t11_summary_sha256": sha256_file(t11_summary_path),
        "selection_split": "val_rank",
        "rank_count": len(records),
        "observed_rank_baseline": observed_baseline,
        "rank_baseline_reconciled": baseline_reconciled,
        "policy_candidates": candidates,
        "selected_policy": selected,
        "prevalidation": {
            "passed": prevalidation,
            "checkpoint_roundtrip": roundtrip,
        },
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": select_deltas,
        "checkpoint_roundtrip": {
            "passed": roundtrip,
            "model_config": reloaded_package.get("model_config", {}),
        },
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "selected_checkpoint": checkpoint.as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "val_select_read": select_evaluated,
        "test_read": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
