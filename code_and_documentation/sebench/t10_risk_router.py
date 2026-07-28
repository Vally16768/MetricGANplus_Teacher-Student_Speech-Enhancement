"""Fresh-support conservative margin calibration for the frozen T9 router."""

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


T10_CALIBRATION_SUPPORT = 128
T10_THRESHOLDS = (0.020, 0.0225, 0.025, 0.0275, 0.030, 0.035, 0.040)


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


def prepare_t10_calibration_manifest(
    identities_path: str | Path,
    t9_summary: dict[str, Any],
    output_dir: str | Path,
    *,
    calibration_count: int = T10_CALIBRATION_SUPPORT,
) -> dict[str, Any]:
    """Freeze fresh T3-audit calibration identities disjoint from T9."""
    payload = json.loads(Path(identities_path).read_text(encoding="utf-8"))
    audit = [
        row for row in payload["records"] if row.get("partition") == "audit"
    ][: int(calibration_count)]
    if len(audit) != int(calibration_count):
        raise ValueError("T10 support lacks the required audit identities.")
    tokens = {str(row["token"]) for row in audit}
    clean_tokens = {str(row["clean_token"]) for row in audit}
    t9_support = dict(t9_summary["support"])
    excluded_tokens = {
        str(value)
        for split in ("fit", "calibration")
        for value in t9_support[split]["tokens"]
    }
    excluded_clean = {
        str(value)
        for split in ("fit", "calibration")
        for value in t9_support[split]["clean_tokens"]
    }
    if tokens & excluded_tokens or clean_tokens & excluded_clean:
        raise ValueError("T10 calibration overlaps T9 pair/clean identities.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "calibration.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("noisy", "clean"))
        writer.writeheader()
        for row in audit:
            writer.writerow({"noisy": row["noisy"], "clean": row["clean"]})
    temporary.replace(path)
    summary = {
        "schema_version": 1,
        "source_identities_sha256": sha256_file(identities_path),
        "source_partition": "audit",
        "count": len(audit),
        "manifest": path.as_posix(),
        "manifest_sha256": sha256_file(path),
        "tokens": sorted(tokens),
        "clean_tokens": sorted(clean_tokens),
        "t9_pair_overlap": 0,
        "t9_clean_overlap": 0,
    }
    _atomic_json(root / "support.json", summary)
    return summary


def run_t10_conservative_search(
    *,
    teacher_checkpoint: str | Path,
    t9_checkpoint: str | Path,
    t9_summary_path: str | Path,
    identities_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    calibration_count: int = T10_CALIBRATION_SUPPORT,
    thresholds: Iterable[float] = T10_THRESHOLDS,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T10 conservative router search is CUDA-only.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    t9_summary = json.loads(Path(t9_summary_path).read_text(encoding="utf-8"))
    if (
        t9_summary.get("status") != "failed"
        or bool(t9_summary["prevalidation"]["passed"])
        or float(t9_summary["oracle_calibration_pesq_gain"]) < 0.025
        or bool(t9_summary.get("test_read"))
    ):
        raise ValueError("T10 requires the terminal auxiliary-risk T9 result.")
    action_lows = tuple(float(value) for value in t9_summary["action_lows"])
    if action_lows != T9_ACTION_LOWS:
        raise ValueError("T10 requires the frozen four T9 actions.")
    ridges = list(t9_summary["ridges"])
    support = prepare_t10_calibration_manifest(
        identities_path,
        t9_summary,
        root / "support",
        calibration_count=calibration_count,
    )
    model, package = load_model_from_checkpoint(t9_checkpoint, device=device)
    records = collect_multi_action_records(
        model,
        support["manifest"],
        device=device,
        lows=action_lows,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    columns = []
    for action_index, ridge in enumerate(ridges):
        features = np.asarray(
            [row["actions"][action_index]["features"] for row in records],
            dtype=np.float64,
        )
        columns.append(_predict(ridge, features))
    predictions = np.stack(columns, axis=1)
    best_actions = np.argmax(predictions, axis=1)
    best_predictions = predictions[
        np.arange(predictions.shape[0]), best_actions
    ]
    baseline = _aggregate(
        records,
        np.full(len(records), -1, dtype=np.int64),
    )
    rows: list[dict[str, Any]] = []
    for threshold in tuple(float(value) for value in thresholds):
        decisions = np.where(best_predictions >= threshold, best_actions, -1)
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
        rows.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "deltas": deltas,
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    selected = (
        max(eligible, key=lambda row: row["metrics"]["pesq_mean"])
        if eligible
        else max(rows, key=lambda row: row["metrics"]["pesq_mean"])
    )
    configure_multi_action_router(
        model,
        ridges=ridges,
        threshold=float(selected["threshold"]),
        lows=action_lows,
    )
    checkpoint = root / "T10-CONSERVATIVE-ROUTED.pt"
    save_checkpoint_package(
        checkpoint,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T10-FRESH-CALIBRATION-CONSERVATIVE-MARGIN",
            "source_t9_checkpoint_sha256": sha256_file(t9_checkpoint),
            "decision_threshold": selected["threshold"],
            "test_read": False,
        },
    )
    reloaded, reloaded_package = load_model_from_checkpoint(checkpoint, device=device)
    target = reloaded.base_model if hasattr(reloaded, "base_model") else reloaded
    roundtrip = bool(
        getattr(target, "multi_router_enabled", False)
        and tuple(getattr(target, "multi_router_lows", ())) == action_lows
        and float(getattr(target, "multi_router_threshold"))
        == float(selected["threshold"])
    )
    prevalidation_checks = {
        "eligible_margin": bool(selected["eligible"]),
        "checkpoint_roundtrip": roundtrip,
    }
    prevalidation_passed = all(prevalidation_checks.values())

    rank_metrics: dict[str, float] | None = None
    rank_deltas: dict[str, float] | None = None
    rank_checks: dict[str, bool] = {"prevalidation_passed": prevalidation_passed}
    select_metrics: dict[str, float] | None = None
    select_deltas: dict[str, float] | None = None
    if prevalidation_passed:
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
        "prevalidation_passed": prevalidation_passed,
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
            and int(calibration_count) == T10_CALIBRATION_SUPPORT
            and tuple(float(value) for value in thresholds) == T10_THRESHOLDS
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T10-FRESH-CALIBRATION-CONSERVATIVE-MARGIN",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "source_t9_checkpoint_sha256": sha256_file(t9_checkpoint),
        "source_t9_summary_sha256": sha256_file(t9_summary_path),
        "support": support,
        "calibration_count": len(records),
        "baseline_calibration_metrics": baseline,
        "threshold_candidates": rows,
        "selected_threshold": selected,
        "prevalidation": {
            "checks": prevalidation_checks,
            "passed": prevalidation_passed,
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
