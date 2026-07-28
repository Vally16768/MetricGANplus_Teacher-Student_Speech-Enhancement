"""Quadratic train-only multi-objective router for T14."""

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
from sebench.t13_multiobjective_router import (
    T13_SISDR_WEIGHTS,
    T13_STOI_WEIGHTS,
    T13_STRENGTH_PENALTIES,
    T13_SUPPORT_COUNT,
    T13_THRESHOLDS,
    fold_multiobjective_ridges,
    merge_train_support,
)
from sebench.training import evaluate_manifest


T14_RIDGE_LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


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


def quadratic_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 16:
        raise ValueError("T14 quadratic transform expects [N,16].")
    products = [
        x[:, left] * x[:, right]
        for left in range(16)
        for right in range(left, 16)
    ]
    return np.concatenate((x, np.stack(products, axis=1)), axis=1)


def fit_quadratic_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    lambdas: Iterable[float] = T14_RIDGE_LAMBDAS,
    folds: int = 5,
) -> dict[str, Any]:
    x = quadratic_features(features)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] < folds:
        raise ValueError("T14 ridge support is invalid.")

    def solve(train_x: np.ndarray, train_y: np.ndarray, penalty: float):
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        normalized = (train_x - mean) / scale
        bias = float(train_y.mean())
        weights = np.linalg.solve(
            normalized.T @ normalized
            + float(penalty) * np.eye(normalized.shape[1]),
            normalized.T @ (train_y - bias),
        )
        return mean, scale, weights, bias

    fold_ids = np.arange(x.shape[0]) % int(folds)
    cv: list[dict[str, float]] = []
    for penalty in tuple(float(value) for value in lambdas):
        errors = []
        for fold in range(int(folds)):
            train = fold_ids != fold
            mean, scale, weights, bias = solve(x[train], y[train], penalty)
            predictions = ((x[~train] - mean) / scale) @ weights + bias
            errors.append(float(np.mean((predictions - y[~train]) ** 2)))
        cv.append({"lambda": penalty, "cv_mse": float(np.mean(errors))})
    selected = min(cv, key=lambda row: row["cv_mse"])
    mean, scale, weights, bias = solve(x, y, float(selected["lambda"]))
    predictions = ((x - mean) / scale) @ weights + bias
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "selected_lambda": float(selected["lambda"]),
        "cv": cv,
        "fit_mse": float(np.mean((predictions - y) ** 2)),
        "fit_pearson": float(np.corrcoef(predictions, y)[0, 1]),
    }


def fit_quadratic_metric_ridges(
    records: list[dict[str, Any]],
    *,
    lows: Iterable[float] = T9_ACTION_LOWS,
) -> dict[str, list[dict[str, Any]]]:
    action_lows = tuple(float(value) for value in lows)
    fitted = {"pesq": [], "stoi": [], "sisdr": []}
    for action_index in range(len(action_lows)):
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
                ]
            )
            fitted[metric].append(fit_quadratic_ridge(features, labels))
    return fitted


def run_t14_quadratic_search(
    *,
    teacher_checkpoint: str | Path,
    t9_checkpoint: str | Path,
    t9_summary_path: str | Path,
    t10_summary_path: str | Path,
    t11_summary_path: str | Path,
    t13_summary_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    metric_ridge_fitter: Callable[
        [list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]]
        | tuple[dict[str, list[dict[str, Any]]], dict[str, Any]],
    ] = fit_quadratic_metric_ridges,
    strategy: str = "T14-QUADRATIC-MULTIOBJECTIVE",
    checkpoint_filename: str = "T14-QUADRATIC-ROUTED.pt",
    prerequisite_name: str = "T13",
    action_lows: Iterable[float] = T9_ACTION_LOWS,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T14 quadratic router search is CUDA-only.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lows = tuple(float(value) for value in action_lows)
    if not lows:
        raise ValueError("Quadratic router action set cannot be empty.")
    t9 = json.loads(Path(t9_summary_path).read_text(encoding="utf-8"))
    t10 = json.loads(Path(t10_summary_path).read_text(encoding="utf-8"))
    t11 = json.loads(Path(t11_summary_path).read_text(encoding="utf-8"))
    t13 = json.loads(Path(t13_summary_path).read_text(encoding="utf-8"))
    if (
        t13.get("status") != "failed"
        or t13.get("val_select_deltas") is None
        or float(t13["val_select_deltas"]["pesq_mean"]) >= 0.01
        or not bool(t13["gate"]["checks"]["stoi_drop_at_most_0_002"])
        or not bool(t13["gate"]["checks"]["sisdr_drop_at_most_0_25"])
        or bool(t13.get("test_read"))
    ):
        raise ValueError(
            f"T14 search requires the auxiliary-safe below-PESQ "
            f"{prerequisite_name} result."
        )
    support = merge_train_support(
        (
            t9["support"]["fit"]["manifest"],
            t9["support"]["calibration"]["manifest"],
            t10["support"]["manifest"],
            t11["support"]["manifest"],
        ),
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
    fitted = metric_ridge_fitter(fit_records)
    if isinstance(fitted, tuple):
        metric_ridges, fit_diagnostics = fitted
    else:
        metric_ridges, fit_diagnostics = fitted, {}
    rank_records = collect_multi_action_records(
        model,
        val_rank_manifest,
        device=device,
        lows=lows,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    predictions = {
        metric: np.stack(
            [
                _predict(
                    ridge,
                    quadratic_features(
                        np.asarray(
                            [
                                row["actions"][index]["features"]
                                for row in rank_records
                            ]
                        )
                    ),
                )
                for index, ridge in enumerate(ridges)
            ],
            axis=1,
        )
        for metric, ridges in metric_ridges.items()
    }
    observed_baseline = _triplet(
        _aggregate(rank_records, np.full(len(rank_records), -1, dtype=np.int64))
    )
    production = max_eval_files is None and support["count"] == T13_SUPPORT_COUNT
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
    for stoi_weight in T13_STOI_WEIGHTS:
        for sisdr_weight in T13_SISDR_WEIGHTS:
            for strength_penalty in T13_STRENGTH_PENALTIES:
                utility = (
                    predictions["pesq"]
                    + stoi_weight * predictions["stoi"]
                    + sisdr_weight * predictions["sisdr"]
                    - strength_penalty * low_penalties
                )
                actions = np.argmax(utility, axis=1)
                scores = utility[np.arange(utility.shape[0]), actions]
                for threshold in T13_THRESHOLDS:
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
    selected = max(
        eligible if eligible else candidates,
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
        feature_transform="quadratic",
    )
    checkpoint = root / checkpoint_filename
    save_checkpoint_package(
        checkpoint,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={"strategy": strategy, "test_read": False},
    )
    reloaded, reloaded_package = load_model_from_checkpoint(checkpoint, device=device)
    target = reloaded.base_model if hasattr(reloaded, "base_model") else reloaded
    roundtrip = bool(
        getattr(target, "multi_router_feature_transform", None) == "quadratic"
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
                batch_size=1,
                progress_callback=progress_callback,
            )
        )
        select_deltas = {
            key: select_metrics[key] - float(baseline_select_metrics[key])
            for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
        }
    checks = {
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
        "status": "passed" if all(checks.values()) else "failed",
        "strategy": strategy,
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "source_t9_checkpoint_sha256": sha256_file(t9_checkpoint),
        "source_t13_summary_sha256": sha256_file(t13_summary_path),
        "support": support,
        "fit_count": len(fit_records),
        "action_lows": list(lows),
        "metric_ridges": metric_ridges,
        "fit_diagnostics": fit_diagnostics,
        "rank_count": len(rank_records),
        "policy_candidates": candidates,
        "selected_policy": selected,
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": select_deltas,
        "checkpoint_roundtrip": {
            "passed": roundtrip,
            "model_config": reloaded_package.get("model_config", {}),
        },
        "gate": {"checks": checks, "passed": all(checks.values())},
        "selected_checkpoint": checkpoint.as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "val_select_read": select_metrics is not None,
        "test_read": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
