"""Train-only adaptive T0/T7 utterance router for T8."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from metrics.pesq import pesq_score
from metrics.sisdr import sisdr
from metrics.stoi import stoi_score
from sebench.audio import tensor_to_numpy_mono
from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_training import sha256_file
from sebench.t5_zeroth_order import prepare_t5_support_manifests
from sebench.t7_confidence import configure_confidence_calibration
from sebench.training import _load_eval_audio_rows, evaluate_manifest


T8_SUPPORT_START = 576
T8_FIT_SUPPORT = 256
T8_CALIBRATION_SUPPORT = 128
T8_RIDGE_LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
T8_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02)
T8_CANDIDATE = {
    "low": -0.30,
    "high": 0.0,
    "threshold": 0.0,
    "temperature": 1.5,
}


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


def _aggregate_records(
    records: list[dict[str, Any]],
    decisions: np.ndarray,
) -> dict[str, float]:
    use_candidate = np.asarray(decisions, dtype=bool)
    if use_candidate.shape != (len(records),):
        raise ValueError("T8 decision count does not match records.")
    result: dict[str, float] = {}
    for metric in ("pesq", "stoi", "sisdr"):
        values = [
            float(row[f"candidate_{metric}"] if choose else row[f"base_{metric}"])
            for row, choose in zip(records, use_candidate, strict=True)
        ]
        result[f"{metric}_mean"] = float(np.mean(values))
    result["candidate_fraction"] = float(use_candidate.mean())
    result["candidate_count"] = int(use_candidate.sum())
    return result


def fit_ridge_router(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    lambdas: Iterable[float] = T8_RIDGE_LAMBDAS,
    folds: int = 5,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != 16:
        raise ValueError("T8 ridge expects [N,16] features and N labels.")
    if x.shape[0] < folds or folds < 2:
        raise ValueError("T8 ridge support is too small for cross-validation.")
    lambda_values = tuple(float(value) for value in lambdas)
    if not lambda_values or any(value <= 0.0 for value in lambda_values):
        raise ValueError("T8 ridge lambdas must be positive.")

    def solve(train_x: np.ndarray, train_y: np.ndarray, penalty: float):
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        normalized = (train_x - mean) / scale
        target_mean = float(train_y.mean())
        centered = train_y - target_mean
        gram = normalized.T @ normalized
        weights = np.linalg.solve(
            gram + penalty * np.eye(normalized.shape[1]),
            normalized.T @ centered,
        )
        return mean, scale, weights, target_mean

    rows: list[dict[str, float]] = []
    fold_ids = np.arange(x.shape[0]) % int(folds)
    for penalty in lambda_values:
        errors: list[float] = []
        for fold in range(int(folds)):
            train = fold_ids != fold
            held = ~train
            mean, scale, weights, bias = solve(x[train], y[train], penalty)
            predictions = ((x[held] - mean) / scale) @ weights + bias
            errors.append(float(np.mean((predictions - y[held]) ** 2)))
        rows.append({"lambda": penalty, "cv_mse": float(np.mean(errors))})
    selected = min(rows, key=lambda row: row["cv_mse"])
    mean, scale, weights, bias = solve(x, y, float(selected["lambda"]))
    predictions = ((x - mean) / scale) @ weights + bias
    correlation = float(np.corrcoef(predictions, y)[0, 1])
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "selected_lambda": float(selected["lambda"]),
        "cv": rows,
        "fit_mse": float(np.mean((predictions - y) ** 2)),
        "fit_pearson": correlation,
    }


def configure_adaptive_router(
    model: torch.nn.Module,
    *,
    ridge: dict[str, Any],
    threshold: float,
    enabled: bool = True,
) -> None:
    target = model.base_model if hasattr(model, "base_model") else model
    configure_confidence_calibration(
        model,
        enabled=False,
        **T8_CANDIDATE,
    )
    configure = getattr(target, "configure_adaptive_router", None)
    if configure is None:
        raise TypeError("T8 requires an adaptive-router MetricGAN teacher.")
    configure(
        enabled=enabled,
        feature_mean=ridge["feature_mean"],
        feature_scale=ridge["feature_scale"],
        weights=ridge["weights"],
        bias=float(ridge["bias"]),
        threshold=float(threshold),
    )


@torch.inference_mode()
def collect_router_records(
    model: torch.nn.Module,
    manifest_path: str | Path,
    *,
    device: str,
    max_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    rows = _load_eval_audio_rows(
        str(manifest_path),
        sample_rate=16_000,
        use_cache=True,
        max_files=max_files,
        progress_callback=progress_callback,
    )
    target = model.base_model if hasattr(model, "base_model") else model
    records: list[dict[str, Any]] = []
    model.eval()
    for index, (row, noisy, clean, sample_rate) in enumerate(rows, start=1):
        noisy_batch = noisy.unsqueeze(0).to(device)
        configure_confidence_calibration(model, enabled=False)
        target.configure_adaptive_router(
            enabled=False,
            feature_mean=(),
            feature_scale=(),
            weights=(),
            bias=0.0,
            threshold=0.0,
        )
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            base = model.denoise_single(noisy_batch).squeeze(0)
            spec = target._stft(noisy_batch)
            magnitude = spec.abs().clamp_min(1e-8)
            features_frequency = torch.log1p(magnitude)
            logits = target.mask_generator.forward_logits(
                features_frequency.transpose(1, 2)
            )
            base_mask = target.mask_generator.learnable_sigmoid(logits)
        configure_confidence_calibration(
            model,
            enabled=True,
            **T8_CANDIDATE,
        )
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            candidate = model.denoise_single(noisy_batch).squeeze(0)
            candidate_logits = target._confidence_candidate_logits(logits)
            candidate_mask = target.mask_generator.learnable_sigmoid(
                candidate_logits
            )
            router_features = target.confidence_router_features(
                noisy_batch,
                magnitude,
                logits,
                base_mask,
                candidate_logits,
                candidate_mask,
            )
        aligned = min(noisy.numel(), clean.numel(), base.numel(), candidate.numel())
        clean_np = tensor_to_numpy_mono(clean[:aligned])
        base_np = tensor_to_numpy_mono(base[:aligned].cpu())
        candidate_np = tensor_to_numpy_mono(candidate[:aligned].cpu())
        base_pesq = float(
            pesq_score(clean_np, base_np, sample_rate, bandwidth="wb")
        )
        candidate_pesq = float(
            pesq_score(clean_np, candidate_np, sample_rate, bandwidth="wb")
        )
        records.append(
            {
                "token": Path(row.noisy).stem,
                "features": router_features.squeeze(0).float().cpu().tolist(),
                "base_pesq": base_pesq,
                "candidate_pesq": candidate_pesq,
                "delta_pesq": candidate_pesq - base_pesq,
                "base_stoi": float(stoi_score(clean_np, base_np, sample_rate)),
                "candidate_stoi": float(
                    stoi_score(clean_np, candidate_np, sample_rate)
                ),
                "base_sisdr": float(sisdr(clean_np, base_np)),
                "candidate_sisdr": float(sisdr(clean_np, candidate_np)),
            }
        )
        if progress_callback and (index == 1 or index == len(rows) or index % 64 == 0):
            progress_callback(
                f"T8 labeled {index}/{len(rows)} from {Path(manifest_path).name}"
            )
    return records


def _predict(ridge: dict[str, Any], features: np.ndarray) -> np.ndarray:
    mean = np.asarray(ridge["feature_mean"], dtype=np.float64)
    scale = np.asarray(ridge["feature_scale"], dtype=np.float64)
    weights = np.asarray(ridge["weights"], dtype=np.float64)
    return ((features - mean) / scale) @ weights + float(ridge["bias"])


def run_t8_router_search(
    *,
    teacher_checkpoint: str | Path,
    identities_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    fit_count: int = T8_FIT_SUPPORT,
    calibration_count: int = T8_CALIBRATION_SUPPORT,
    support_start: int = T8_SUPPORT_START,
    lambdas: Iterable[float] = T8_RIDGE_LAMBDAS,
    thresholds: Iterable[float] = T8_THRESHOLDS,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T8 router search is CUDA-only.")
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
    fit_records = collect_router_records(
        model,
        support["fit"]["manifest"],
        device=device,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    calibration_records = collect_router_records(
        model,
        support["calibration"]["manifest"],
        device=device,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    x_fit = np.asarray([row["features"] for row in fit_records], dtype=np.float64)
    y_fit = np.asarray([row["delta_pesq"] for row in fit_records], dtype=np.float64)
    ridge = fit_ridge_router(x_fit, y_fit, lambdas=lambdas)
    x_calibration = np.asarray(
        [row["features"] for row in calibration_records],
        dtype=np.float64,
    )
    predictions = _predict(ridge, x_calibration)
    baseline_calibration = _aggregate_records(
        calibration_records,
        np.zeros(len(calibration_records), dtype=bool),
    )
    oracle_calibration = _aggregate_records(
        calibration_records,
        np.asarray(
            [row["delta_pesq"] > 0.0 for row in calibration_records],
            dtype=bool,
        ),
    )
    threshold_rows: list[dict[str, Any]] = []
    for threshold in tuple(float(value) for value in thresholds):
        metrics = _aggregate_records(
            calibration_records,
            predictions >= threshold,
        )
        deltas = {
            key: float(metrics[key]) - float(baseline_calibration[key])
            for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
        }
        checks = {
            "pesq_gain_at_least_0_005": deltas["pesq_mean"] >= 0.005,
            "stoi_drop_at_most_0_0015": deltas["stoi_mean"] >= -0.0015,
            "sisdr_drop_at_most_0_15": deltas["sisdr_mean"] >= -0.15,
        }
        threshold_rows.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "deltas": deltas,
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
    eligible_thresholds = [row for row in threshold_rows if row["eligible"]]
    selected_threshold = (
        max(eligible_thresholds, key=lambda row: row["metrics"]["pesq_mean"])
        if eligible_thresholds
        else max(threshold_rows, key=lambda row: row["metrics"]["pesq_mean"])
    )
    oracle_gain = (
        oracle_calibration["pesq_mean"] - baseline_calibration["pesq_mean"]
    )
    prevalidation_checks = {
        "oracle_gain_at_least_0_015": oracle_gain >= 0.015,
        "learned_threshold_eligible": bool(selected_threshold["eligible"]),
        "finite_ridge": all(
            math.isfinite(float(value))
            for value in (
                ridge["fit_mse"],
                ridge["fit_pearson"],
                ridge["bias"],
            )
        ),
    }
    prevalidation_passed = all(prevalidation_checks.values())

    configure_adaptive_router(
        model,
        ridge=ridge,
        threshold=float(selected_threshold["threshold"]),
        enabled=True,
    )
    checkpoint = root / "T8-ROUTED.pt"
    save_checkpoint_package(
        checkpoint,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T8-TRAIN-ONLY-ADAPTIVE-ROUTER",
            "candidate": T8_CANDIDATE,
            "ridge_lambda": ridge["selected_lambda"],
            "decision_threshold": selected_threshold["threshold"],
            "test_read": False,
        },
    )
    reloaded, reloaded_package = load_model_from_checkpoint(checkpoint, device=device)
    reloaded_target = (
        reloaded.base_model if hasattr(reloaded, "base_model") else reloaded
    )
    roundtrip = bool(
        getattr(reloaded_target, "adaptive_router_enabled", False)
        and len(getattr(reloaded_target, "adaptive_router_weights", ())) == 16
    )

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
            key: float(rank_metrics[key]) - float(baseline_rank_metrics[key])
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
                key: float(select_metrics[key]) - float(baseline_select_metrics[key])
                for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
            }
    gate_checks = {
        "prevalidation_passed": prevalidation_passed,
        "rank_passed": bool(rank_checks) and all(rank_checks.values()),
        "nonzero_router": selected_threshold["metrics"]["candidate_count"] > 0,
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
            and int(fit_count) == T8_FIT_SUPPORT
            and int(calibration_count) == T8_CALIBRATION_SUPPORT
            and int(support_start) == T8_SUPPORT_START
            and tuple(float(value) for value in lambdas) == T8_RIDGE_LAMBDAS
            and tuple(float(value) for value in thresholds) == T8_THRESHOLDS
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T8-TRAIN-ONLY-ADAPTIVE-ROUTER",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "support": support,
        "candidate": T8_CANDIDATE,
        "ridge": ridge,
        "fit_count": len(fit_records),
        "calibration_count": len(calibration_records),
        "baseline_calibration_metrics": baseline_calibration,
        "oracle_calibration_metrics": oracle_calibration,
        "oracle_calibration_pesq_gain": oracle_gain,
        "threshold_candidates": threshold_rows,
        "selected_threshold": selected_threshold,
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
