"""Train-only multi-objective action router for T13."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_training import sha256_file
from sebench.t8_router import fit_ridge_router
from sebench.t9_multi_router import (
    T9_ACTION_LOWS,
    _aggregate,
    _predict,
    collect_multi_action_records,
    configure_multi_action_router,
)
from sebench.training import evaluate_manifest


T13_SUPPORT_COUNT = 584
T13_STOI_WEIGHTS = (0.0, 1.0, 2.0, 4.0)
T13_SISDR_WEIGHTS = (0.0, 0.01, 0.02, 0.04)
T13_STRENGTH_PENALTIES = (0.0, 0.01, 0.02)
T13_THRESHOLDS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015)


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


def merge_train_support(
    manifests: Iterable[str | Path],
    output_dir: str | Path,
    *,
    max_rows_per_manifest: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, Any]] = []
    for raw_path in manifests:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        if max_rows_per_manifest is not None:
            source_rows = source_rows[: int(max_rows_per_manifest)]
        added = 0
        for row in source_rows:
            pair = (str(row["noisy"]), str(row["clean"]))
            if pair in seen:
                continue
            seen.add(pair)
            rows.append({"noisy": pair[0], "clean": pair[1]})
            added += 1
        sources.append(
            {
                "sha256": sha256_file(path),
                "available_rows": len(source_rows),
                "added_rows": added,
            }
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "fit.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("noisy", "clean"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    summary = {
        "schema_version": 1,
        "manifest": path.as_posix(),
        "manifest_sha256": sha256_file(path),
        "count": len(rows),
        "duplicate_pairs_removed": sum(
            int(source["available_rows"]) for source in sources
        )
        - len(rows),
        "sources": sources,
    }
    _atomic_json(root / "support.json", summary)
    return summary


def fit_action_metric_ridges(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    fitted: dict[str, list[dict[str, Any]]] = {
        "pesq": [],
        "stoi": [],
        "sisdr": [],
    }
    for action_index in range(len(T9_ACTION_LOWS)):
        features = np.asarray(
            [row["actions"][action_index]["features"] for row in records],
            dtype=np.float64,
        )
        for metric in fitted:
            labels = np.asarray(
                [
                    float(row["actions"][action_index][metric])
                    - float(row["base"][metric])
                    for row in records
                ],
                dtype=np.float64,
            )
            fitted[metric].append(fit_ridge_router(features, labels))
    return fitted


def fold_multiobjective_ridges(
    metric_ridges: dict[str, list[dict[str, Any]]],
    *,
    stoi_weight: float,
    sisdr_weight: float,
    strength_penalty: float,
    lows: Iterable[float] = T9_ACTION_LOWS,
) -> list[dict[str, Any]]:
    action_lows = tuple(float(value) for value in lows)
    if any(
        len(metric_ridges[metric]) != len(action_lows)
        for metric in ("pesq", "stoi", "sisdr")
    ):
        raise ValueError("T13 metric-ridge/action count mismatch.")
    folded: list[dict[str, Any]] = []
    for index, low in enumerate(action_lows):
        pesq = metric_ridges["pesq"][index]
        stoi = metric_ridges["stoi"][index]
        sisdr = metric_ridges["sisdr"][index]
        mean = np.asarray(pesq["feature_mean"], dtype=np.float64)
        scale = np.asarray(pesq["feature_scale"], dtype=np.float64)
        if not (
            np.allclose(mean, stoi["feature_mean"], atol=1e-12, rtol=0.0)
            and np.allclose(mean, sisdr["feature_mean"], atol=1e-12, rtol=0.0)
            and np.allclose(scale, stoi["feature_scale"], atol=1e-12, rtol=0.0)
            and np.allclose(scale, sisdr["feature_scale"], atol=1e-12, rtol=0.0)
        ):
            raise ValueError("T13 ridge normalizers do not match.")
        weights = (
            np.asarray(pesq["weights"], dtype=np.float64)
            + float(stoi_weight) * np.asarray(stoi["weights"], dtype=np.float64)
            + float(sisdr_weight) * np.asarray(sisdr["weights"], dtype=np.float64)
        )
        bias = (
            float(pesq["bias"])
            + float(stoi_weight) * float(stoi["bias"])
            + float(sisdr_weight) * float(sisdr["bias"])
            - float(strength_penalty) * abs(low) ** 2
        )
        folded.append(
            {
                "feature_mean": mean.tolist(),
                "feature_scale": scale.tolist(),
                "weights": weights.tolist(),
                "bias": bias,
            }
        )
    return folded


def run_t13_multiobjective_search(
    *,
    teacher_checkpoint: str | Path,
    t9_checkpoint: str | Path,
    t9_summary_path: str | Path,
    t10_summary_path: str | Path,
    t11_summary_path: str | Path,
    t12_summary_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    stoi_weights: Iterable[float] = T13_STOI_WEIGHTS,
    sisdr_weights: Iterable[float] = T13_SISDR_WEIGHTS,
    strength_penalties: Iterable[float] = T13_STRENGTH_PENALTIES,
    thresholds: Iterable[float] = T13_THRESHOLDS,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T13 multi-objective router search is CUDA-only.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    t9 = json.loads(Path(t9_summary_path).read_text(encoding="utf-8"))
    t10 = json.loads(Path(t10_summary_path).read_text(encoding="utf-8"))
    t11 = json.loads(Path(t11_summary_path).read_text(encoding="utf-8"))
    t12 = json.loads(Path(t12_summary_path).read_text(encoding="utf-8"))
    if (
        t12.get("status") != "failed"
        or t12.get("val_select_deltas") is None
        or float(t12["val_select_deltas"]["pesq_mean"]) >= 0.01
        or not bool(t12["gate"]["checks"]["stoi_drop_at_most_0_002"])
        or not bool(t12["gate"]["checks"]["sisdr_drop_at_most_0_25"])
        or bool(t12.get("test_read"))
    ):
        raise ValueError("T13 requires the auxiliary-safe below-PESQ T12 result.")
    lows = tuple(float(value) for value in t9["action_lows"])
    if lows != T9_ACTION_LOWS:
        raise ValueError("T13 requires the frozen T9 action set.")
    support_manifests = (
        t9["support"]["fit"]["manifest"],
        t9["support"]["calibration"]["manifest"],
        t10["support"]["manifest"],
        t11["support"]["manifest"],
    )
    support = merge_train_support(
        support_manifests,
        root / "support",
        max_rows_per_manifest=10 if max_eval_files is not None else None,
    )
    model, package = load_model_from_checkpoint(t9_checkpoint, device=device)
    fit_records = collect_multi_action_records(
        model,
        support["manifest"],
        device=device,
        lows=lows,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    metric_ridges = fit_action_metric_ridges(fit_records)
    rank_records = collect_multi_action_records(
        model,
        val_rank_manifest,
        device=device,
        lows=lows,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    prediction_by_metric = {
        metric: np.stack(
            [
                _predict(
                    ridge,
                    np.asarray(
                        [
                            row["actions"][index]["features"]
                            for row in rank_records
                        ],
                        dtype=np.float64,
                    ),
                )
                for index, ridge in enumerate(ridges)
            ],
            axis=1,
        )
        for metric, ridges in metric_ridges.items()
    }
    observed_baseline = _triplet(
        _aggregate(
            rank_records,
            np.full(len(rank_records), -1, dtype=np.int64),
        )
    )
    production = (
        max_eval_files is None
        and support["count"] == T13_SUPPORT_COUNT
        and tuple(float(value) for value in stoi_weights) == T13_STOI_WEIGHTS
        and tuple(float(value) for value in sisdr_weights) == T13_SISDR_WEIGHTS
        and tuple(float(value) for value in strength_penalties)
        == T13_STRENGTH_PENALTIES
        and tuple(float(value) for value in thresholds) == T13_THRESHOLDS
    )
    baseline_reconciled = (
        not production
        or all(
            abs(observed_baseline[key] - float(baseline_rank_metrics[key])) <= 1e-5
            for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
        )
    )
    baseline = baseline_rank_metrics if production else observed_baseline
    candidates: list[dict[str, Any]] = []
    low_penalties = np.square(np.asarray(lows)).reshape(1, -1)
    for stoi_weight in tuple(float(value) for value in stoi_weights):
        for sisdr_weight in tuple(float(value) for value in sisdr_weights):
            for strength_penalty in tuple(
                float(value) for value in strength_penalties
            ):
                utility = (
                    prediction_by_metric["pesq"]
                    + stoi_weight * prediction_by_metric["stoi"]
                    + sisdr_weight * prediction_by_metric["sisdr"]
                    - strength_penalty * low_penalties
                )
                actions = np.argmax(utility, axis=1)
                scores = utility[np.arange(utility.shape[0]), actions]
                for threshold in tuple(float(value) for value in thresholds):
                    decisions = np.where(scores >= threshold, actions, -1)
                    metrics = _aggregate(rank_records, decisions)
                    deltas = {
                        key: float(metrics[key]) - float(baseline[key])
                        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
                    }
                    checks = {
                        "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
                        "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
                        "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
                        "nonzero_router": int(metrics["base_count"])
                        < len(rank_records),
                    }
                    candidates.append(
                        {
                            "stoi_weight": stoi_weight,
                            "sisdr_weight": sisdr_weight,
                            "strength_penalty": strength_penalty,
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
        ),
    )
    folded = fold_multiobjective_ridges(
        metric_ridges,
        stoi_weight=float(selected["stoi_weight"]),
        sisdr_weight=float(selected["sisdr_weight"]),
        strength_penalty=float(selected["strength_penalty"]),
        lows=lows,
    )
    configure_multi_action_router(
        model,
        ridges=folded,
        threshold=float(selected["threshold"]),
        lows=lows,
    )
    checkpoint = root / "T13-MULTIOBJECTIVE-ROUTED.pt"
    save_checkpoint_package(
        checkpoint,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T13-TRAIN-ONLY-MULTIOBJECTIVE-ROUTER",
            "selected_policy": {
                key: selected[key]
                for key in (
                    "stoi_weight",
                    "sisdr_weight",
                    "strength_penalty",
                    "threshold",
                )
            },
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
    if prevalidation and production:
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
        "production_contract": production,
        "rank_baseline_reconciled": baseline_reconciled,
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T13-TRAIN-ONLY-MULTIOBJECTIVE-ROUTER",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "source_t9_checkpoint_sha256": sha256_file(t9_checkpoint),
        "source_t9_summary_sha256": sha256_file(t9_summary_path),
        "source_t10_summary_sha256": sha256_file(t10_summary_path),
        "source_t11_summary_sha256": sha256_file(t11_summary_path),
        "source_t12_summary_sha256": sha256_file(t12_summary_path),
        "support": support,
        "fit_count": len(fit_records),
        "metric_ridges": metric_ridges,
        "rank_count": len(rank_records),
        "observed_rank_baseline": observed_baseline,
        "rank_baseline_reconciled": baseline_reconciled,
        "policy_candidates": candidates,
        "selected_policy": selected,
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": select_deltas,
        "checkpoint_roundtrip": {
            "passed": roundtrip,
            "model_config": reloaded_package.get("model_config", {}),
        },
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "selected_checkpoint": checkpoint.as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "val_select_read": select_metrics is not None,
        "test_read": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
