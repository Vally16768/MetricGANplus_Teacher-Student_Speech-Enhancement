"""Out-of-fold calibrated quadratic router for T15.

T15 changes only the train-support estimator used by T14.  The action set,
quadratic feature map, rank policy family, validation gate, and test embargo
remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from sebench.t9_multi_router import T9_ACTION_LOWS
from sebench.t14_quadratic_router import (
    T14_RIDGE_LAMBDAS,
    fit_quadratic_ridge,
    quadratic_features,
    run_t14_quadratic_search,
)


T15_OUTER_FOLDS = 5
T15_INNER_FOLDS = 4
T15_SLOPE_BOUNDS = (0.0, 1.5)


def _solve_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    penalty: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("T15 ridge support is invalid.")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (x - mean) / scale
    bias = float(y.mean())
    weights = np.linalg.solve(
        normalized.T @ normalized
        + float(penalty) * np.eye(normalized.shape[1]),
        normalized.T @ (y - bias),
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "weights": weights,
        "bias": bias,
    }


def _ridge_predict(ridge: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return (
        (x - np.asarray(ridge["feature_mean"], dtype=np.float64))
        / np.asarray(ridge["feature_scale"], dtype=np.float64)
    ) @ np.asarray(ridge["weights"], dtype=np.float64) + float(ridge["bias"])


def _select_lambda(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    lambdas: Iterable[float],
    folds: int,
) -> tuple[float, list[dict[str, float]]]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.shape[0] < folds:
        raise ValueError("T15 inner-CV support is smaller than the fold count.")
    fold_ids = np.arange(x.shape[0], dtype=np.int64) % int(folds)
    rows: list[dict[str, float]] = []
    for penalty in tuple(float(value) for value in lambdas):
        errors = []
        for fold in range(int(folds)):
            train = fold_ids != fold
            ridge = _solve_ridge(x[train], y[train], penalty)
            predictions = _ridge_predict(ridge, x[~train])
            errors.append(float(np.mean((predictions - y[~train]) ** 2)))
        rows.append({"lambda": penalty, "cv_mse": float(np.mean(errors))})
    selected = min(rows, key=lambda row: (row["cv_mse"], row["lambda"]))
    return float(selected["lambda"]), rows


def fit_affine_calibration(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    slope_bounds: tuple[float, float] = T15_SLOPE_BOUNDS,
) -> dict[str, float]:
    """Fit y ~= slope * prediction + intercept with a bounded slope."""
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if predicted.shape != target.shape or predicted.size < 2:
        raise ValueError("T15 affine calibration inputs are invalid.")
    centered = predicted - float(predicted.mean())
    variance = float(centered @ centered)
    raw_slope = (
        float(centered @ (target - float(target.mean()))) / variance
        if variance > 1e-12
        else 0.0
    )
    slope = float(np.clip(raw_slope, slope_bounds[0], slope_bounds[1]))
    intercept = float(target.mean() - slope * predicted.mean())
    calibrated = slope * predicted + intercept
    return {
        "raw_slope": raw_slope,
        "slope": slope,
        "intercept": intercept,
        "oof_mse_before": float(np.mean((predicted - target) ** 2)),
        "oof_mse_after": float(np.mean((calibrated - target) ** 2)),
        "oof_pearson_before": _safe_pearson(predicted, target),
        "oof_pearson_after": _safe_pearson(calibrated, target),
    }


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def apply_affine_to_ridge(
    ridge: dict[str, Any],
    calibration: dict[str, float],
) -> dict[str, Any]:
    """Fold an affine output calibration into a serialized ridge."""
    slope = float(calibration["slope"])
    result = dict(ridge)
    result["weights"] = (
        slope * np.asarray(ridge["weights"], dtype=np.float64)
    ).tolist()
    result["bias"] = (
        slope * float(ridge["bias"]) + float(calibration["intercept"])
    )
    result["oof_calibration"] = dict(calibration)
    return result


def fit_oof_calibrated_quadratic_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    lambdas: Iterable[float] = T14_RIDGE_LAMBDAS,
    outer_folds: int = T15_OUTER_FOLDS,
    inner_folds: int = T15_INNER_FOLDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Nested-CV OOF calibration followed by a full-support T14 ridge."""
    raw = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    x = quadratic_features(raw)
    if x.shape[0] != y.shape[0] or x.shape[0] < outer_folds:
        raise ValueError("T15 OOF support is invalid.")
    fold_ids = np.arange(x.shape[0], dtype=np.int64) % int(outer_folds)
    oof = np.empty(y.shape, dtype=np.float64)
    outer_rows: list[dict[str, Any]] = []
    for fold in range(int(outer_folds)):
        train = fold_ids != fold
        selected_lambda, inner_cv = _select_lambda(
            x[train],
            y[train],
            lambdas=lambdas,
            folds=min(int(inner_folds), int(train.sum())),
        )
        ridge = _solve_ridge(x[train], y[train], selected_lambda)
        oof[~train] = _ridge_predict(ridge, x[~train])
        outer_rows.append(
            {
                "fold": fold,
                "train_count": int(train.sum()),
                "validation_count": int((~train).sum()),
                "selected_lambda": selected_lambda,
                "inner_cv": inner_cv,
            }
        )
    calibration = fit_affine_calibration(oof, y)
    final = fit_quadratic_ridge(raw, y, lambdas=lambdas, folds=outer_folds)
    calibrated = apply_affine_to_ridge(final, calibration)
    diagnostics = {
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "fold_assignment": "row_index_modulo_fold_count",
        "slope_bounds": list(T15_SLOPE_BOUNDS),
        "calibration": calibration,
        "outer_cv": outer_rows,
        "full_support_selected_lambda": float(final["selected_lambda"]),
    }
    return calibrated, diagnostics


def fit_oof_calibrated_metric_ridges(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    fitted: dict[str, list[dict[str, Any]]] = {
        "pesq": [],
        "stoi": [],
        "sisdr": [],
    }
    diagnostics: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in fitted
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
            ridge, detail = fit_oof_calibrated_quadratic_ridge(features, labels)
            fitted[metric].append(ridge)
            diagnostics[metric].append(
                {"action_low": T9_ACTION_LOWS[action_index], **detail}
            )
    return fitted, {
        "method": "nested-oof-affine-calibration",
        "metrics": diagnostics,
    }


def run_t15_oof_calibrated_search(
    *,
    teacher_checkpoint: str | Path,
    t9_checkpoint: str | Path,
    t9_summary_path: str | Path,
    t10_summary_path: str | Path,
    t11_summary_path: str | Path,
    t14_summary_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run T15 through the frozen T14 selection/evaluation protocol."""
    return run_t14_quadratic_search(
        teacher_checkpoint=teacher_checkpoint,
        t9_checkpoint=t9_checkpoint,
        t9_summary_path=t9_summary_path,
        t10_summary_path=t10_summary_path,
        t11_summary_path=t11_summary_path,
        t13_summary_path=t14_summary_path,
        val_rank_manifest=val_rank_manifest,
        val_select_manifest=val_select_manifest,
        baseline_rank_metrics=baseline_rank_metrics,
        baseline_select_metrics=baseline_select_metrics,
        output_dir=output_dir,
        device=device,
        max_eval_files=max_eval_files,
        progress_callback=progress_callback,
        metric_ridge_fitter=fit_oof_calibrated_metric_ridges,
        strategy="T15-OOF-CALIBRATED-QUADRATIC-MULTIOBJECTIVE",
        checkpoint_filename="T15-OOF-CALIBRATED-ROUTED.pt",
        prerequisite_name="T14",
    )
