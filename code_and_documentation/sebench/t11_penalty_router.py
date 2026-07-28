"""Risk-penalized action utility search for T11."""

from __future__ import annotations

import csv
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
from sebench.training import evaluate_manifest


T11_CALIBRATION_START = 128
T11_CALIBRATION_SUPPORT = 72
T11_PENALTIES = (0.005, 0.01, 0.02, 0.03, 0.04)
T11_THRESHOLDS = (0.005, 0.01, 0.015, 0.02, 0.025)


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


def prepare_t11_calibration_manifest(
    identities_path: str | Path,
    t9_summary: dict[str, Any],
    t10_summary: dict[str, Any],
    output_dir: str | Path,
    *,
    calibration_count: int = T11_CALIBRATION_SUPPORT,
    calibration_start: int = T11_CALIBRATION_START,
) -> dict[str, Any]:
    payload = json.loads(Path(identities_path).read_text(encoding="utf-8"))
    audit_pool = [
        row for row in payload["records"] if row.get("partition") == "audit"
    ]
    start = int(calibration_start)
    rows = audit_pool[start : start + int(calibration_count)]
    if len(rows) != int(calibration_count):
        raise ValueError("T11 support lacks the required remaining audit identities.")
    tokens = {str(row["token"]) for row in rows}
    clean_tokens = {str(row["clean_token"]) for row in rows}
    excluded_tokens = {
        str(value)
        for split in ("fit", "calibration")
        for value in t9_summary["support"][split]["tokens"]
    } | {str(value) for value in t10_summary["support"]["tokens"]}
    excluded_clean = {
        str(value)
        for split in ("fit", "calibration")
        for value in t9_summary["support"][split]["clean_tokens"]
    } | {str(value) for value in t10_summary["support"]["clean_tokens"]}
    if tokens & excluded_tokens or clean_tokens & excluded_clean:
        raise ValueError("T11 support overlaps T9/T10 pair or clean identities.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "calibration.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("noisy", "clean"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"noisy": row["noisy"], "clean": row["clean"]})
    temporary.replace(path)
    result = {
        "schema_version": 1,
        "source_identities_sha256": sha256_file(identities_path),
        "source_partition": "audit",
        "start": start,
        "count": len(rows),
        "manifest": path.as_posix(),
        "manifest_sha256": sha256_file(path),
        "tokens": sorted(tokens),
        "clean_tokens": sorted(clean_tokens),
        "predecessor_pair_overlap": 0,
        "predecessor_clean_overlap": 0,
    }
    _atomic_json(root / "support.json", result)
    return result


def penalize_ridges(
    ridges: list[dict[str, Any]],
    lows: Iterable[float],
    penalty: float,
) -> list[dict[str, Any]]:
    action_lows = tuple(float(value) for value in lows)
    if len(ridges) != len(action_lows) or float(penalty) < 0.0:
        raise ValueError("T11 penalty inputs are invalid.")
    adjusted: list[dict[str, Any]] = []
    for ridge, low in zip(ridges, action_lows, strict=True):
        row = dict(ridge)
        row["bias"] = float(ridge["bias"]) - float(penalty) * abs(low) ** 2
        adjusted.append(row)
    return adjusted


def run_t11_penalty_search(
    *,
    teacher_checkpoint: str | Path,
    t9_checkpoint: str | Path,
    t9_summary_path: str | Path,
    t10_summary_path: str | Path,
    identities_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    calibration_count: int = T11_CALIBRATION_SUPPORT,
    calibration_start: int = T11_CALIBRATION_START,
    penalties: Iterable[float] = T11_PENALTIES,
    thresholds: Iterable[float] = T11_THRESHOLDS,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T11 penalty router search is CUDA-only.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    t9 = json.loads(Path(t9_summary_path).read_text(encoding="utf-8"))
    t10 = json.loads(Path(t10_summary_path).read_text(encoding="utf-8"))
    if (
        t10.get("status") != "failed"
        or t10["val_select_deltas"] is None
        or float(t10["val_select_deltas"]["pesq_mean"]) >= 0.01
        or bool(t10.get("test_read"))
    ):
        raise ValueError("T11 requires the auxiliary-safe below-PESQ T10 result.")
    lows = tuple(float(value) for value in t9["action_lows"])
    if lows != T9_ACTION_LOWS:
        raise ValueError("T11 requires the frozen T9 action set.")
    ridges = list(t9["ridges"])
    support = prepare_t11_calibration_manifest(
        identities_path,
        t9,
        t10,
        root / "support",
        calibration_count=calibration_count,
        calibration_start=calibration_start,
    )
    model, package = load_model_from_checkpoint(t9_checkpoint, device=device)
    records = collect_multi_action_records(
        model,
        support["manifest"],
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
    baseline = _aggregate(
        records, np.full(len(records), -1, dtype=np.int64)
    )
    candidates: list[dict[str, Any]] = []
    for penalty in tuple(float(value) for value in penalties):
        adjusted_scores = raw_predictions - penalty * np.square(
            np.asarray(lows, dtype=np.float64)
        ).reshape(1, -1)
        best_actions = np.argmax(adjusted_scores, axis=1)
        best_scores = adjusted_scores[
            np.arange(adjusted_scores.shape[0]), best_actions
        ]
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
                "nonzero_router": metrics["base_count"] < len(records),
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
    selected = (
        max(eligible, key=lambda row: row["metrics"]["pesq_mean"])
        if eligible
        else max(candidates, key=lambda row: row["metrics"]["pesq_mean"])
    )
    adjusted_ridges = penalize_ridges(ridges, lows, float(selected["penalty"]))
    configure_multi_action_router(
        model,
        ridges=adjusted_ridges,
        threshold=float(selected["threshold"]),
        lows=lows,
    )
    checkpoint = root / "T11-PENALIZED-ROUTED.pt"
    save_checkpoint_package(
        checkpoint,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T11-RISK-PENALIZED-MULTI-ACTION",
            "strength_penalty": selected["penalty"],
            "decision_threshold": selected["threshold"],
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
    prevalidation = bool(selected["eligible"]) and roundtrip

    rank_metrics = rank_deltas = select_metrics = select_deltas = None
    rank_checks: dict[str, bool] = {"prevalidation_passed": prevalidation}
    if prevalidation:
        rank_metrics = _triplet(
            evaluate_manifest(
                reloaded,
                str(val_rank_manifest),
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
        rank_deltas = {
            key: rank_metrics[key] - float(baseline_rank_metrics[key])
            for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
        }
        rank_checks.update(
            {
                "beats_t0_pesq": rank_deltas["pesq_mean"] > 0.0,
                "stoi_guardrail": rank_deltas["stoi_mean"] >= -0.002,
                "sisdr_guardrail": rank_deltas["sisdr_mean"] >= -0.25,
            }
        )
        if all(rank_checks.values()):
            select_metrics = _triplet(
                evaluate_manifest(
                    reloaded,
                    str(val_select_manifest),
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
            select_deltas = {
                key: select_metrics[key] - float(baseline_select_metrics[key])
                for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
            }
    gate_checks = {
        "prevalidation_passed": prevalidation,
        "rank_passed": bool(rank_checks) and all(rank_checks.values()),
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
        "production_support": (
            max_eval_files is None
            and int(calibration_count) == T11_CALIBRATION_SUPPORT
            and int(calibration_start) == T11_CALIBRATION_START
            and tuple(float(value) for value in penalties) == T11_PENALTIES
            and tuple(float(value) for value in thresholds) == T11_THRESHOLDS
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T11-RISK-PENALIZED-MULTI-ACTION",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "source_t9_checkpoint_sha256": sha256_file(t9_checkpoint),
        "source_t9_summary_sha256": sha256_file(t9_summary_path),
        "source_t10_summary_sha256": sha256_file(t10_summary_path),
        "support": support,
        "calibration_count": len(records),
        "baseline_calibration_metrics": baseline,
        "policy_candidates": candidates,
        "selected_policy": selected,
        "prevalidation": {
            "passed": prevalidation,
            "checkpoint_roundtrip": roundtrip,
        },
        "val_rank_metrics": rank_metrics,
        "val_rank_deltas": rank_deltas,
        "val_rank_checks": rank_checks,
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": select_deltas,
        "checkpoint_roundtrip": {
            "passed": roundtrip,
            "model_config": reloaded_package.get("model_config", {}),
        },
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "selected_checkpoint": checkpoint.as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "test_read": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
