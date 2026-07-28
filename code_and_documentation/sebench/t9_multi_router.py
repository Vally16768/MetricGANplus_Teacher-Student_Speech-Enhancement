"""Train-only multi-action T0/confidence router for T9."""

from __future__ import annotations

import csv
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
from sebench.t8_router import fit_ridge_router
from sebench.training import _load_eval_audio_rows, evaluate_manifest


T9_FIT_START = 576
T9_FIT_SUPPORT = 256
T9_CALIBRATION_SUPPORT = 128
T9_ACTION_LOWS = (-0.20, -0.40, -0.60, -0.80)
T9_RIDGE_LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
T9_THRESHOLDS = (0.0, 0.005, 0.01, 0.015, 0.02)


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


def prepare_t9_support_manifests(
    identities_path: str | Path,
    output_dir: str | Path,
    *,
    fit_count: int = T9_FIT_SUPPORT,
    calibration_count: int = T9_CALIBRATION_SUPPORT,
    fit_start: int = T9_FIT_START,
) -> dict[str, Any]:
    """Freeze T9 fit from train and fresh calibration from its named partition."""
    payload = json.loads(Path(identities_path).read_text(encoding="utf-8"))
    records = list(payload["records"])
    train = [row for row in records if row.get("partition") == "train"]
    calibration_pool = [
        row for row in records if row.get("partition") == "calibration"
    ]
    start = int(fit_start)
    fit = train[start : start + int(fit_count)]
    calibration = calibration_pool[: int(calibration_count)]
    if len(fit) != int(fit_count) or len(calibration) != int(calibration_count):
        raise ValueError("T9 support does not contain the required identities.")

    fit_tokens = {str(row["token"]) for row in fit}
    calibration_tokens = {str(row["token"]) for row in calibration}
    fit_clean = {str(row["clean_token"]) for row in fit}
    calibration_clean = {str(row["clean_token"]) for row in calibration}
    if fit_tokens & calibration_tokens or fit_clean & calibration_clean:
        raise ValueError("T9 fit/calibration support is not pair/clean disjoint.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    def write_manifest(name: str, rows: list[dict[str, Any]]) -> Path:
        path = root / f"{name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("noisy", "clean"))
            writer.writeheader()
            for row in rows:
                writer.writerow({"noisy": row["noisy"], "clean": row["clean"]})
        temporary.replace(path)
        return path

    fit_path = write_manifest("fit", fit)
    calibration_path = write_manifest("calibration", calibration)
    summary = {
        "schema_version": 1,
        "source_identities_sha256": sha256_file(identities_path),
        "fit_source_partition": "train",
        "fit_start": start,
        "calibration_source_partition": "calibration",
        "fit": {
            "count": len(fit),
            "manifest": fit_path.as_posix(),
            "manifest_sha256": sha256_file(fit_path),
            "tokens": sorted(fit_tokens),
            "clean_tokens": sorted(fit_clean),
        },
        "calibration": {
            "count": len(calibration),
            "manifest": calibration_path.as_posix(),
            "manifest_sha256": sha256_file(calibration_path),
            "tokens": sorted(calibration_tokens),
            "clean_tokens": sorted(calibration_clean),
        },
        "pair_overlap": 0,
        "clean_overlap": 0,
    }
    _atomic_json(root / "support.json", summary)
    return summary


def configure_multi_action_router(
    model: torch.nn.Module,
    *,
    ridges: list[dict[str, Any]],
    threshold: float,
    lows: Iterable[float] = T9_ACTION_LOWS,
    enabled: bool = True,
    feature_transform: str = "identity",
) -> None:
    target = model.base_model if hasattr(model, "base_model") else model
    action_lows = tuple(float(value) for value in lows)
    if len(ridges) != len(action_lows):
        raise ValueError("T9 ridge/action count mismatch.")
    target.configure_confidence_calibration(
        enabled=False,
        low=0.0,
        high=0.0,
        threshold=0.0,
        temperature=1.5,
    )
    target.configure_adaptive_router(
        enabled=False,
        feature_mean=(),
        feature_scale=(),
        weights=(),
        bias=0.0,
        threshold=0.0,
    )
    target.configure_multi_action_router(
        enabled=enabled,
        lows=action_lows,
        feature_means=[row["feature_mean"] for row in ridges],
        feature_scales=[row["feature_scale"] for row in ridges],
        weights=[row["weights"] for row in ridges],
        biases=[float(row["bias"]) for row in ridges],
        threshold=float(threshold),
        feature_transform=feature_transform,
    )


def _waveform_for_mask(
    target: torch.nn.Module,
    *,
    spec: torch.Tensor,
    features_frequency: torch.Tensor,
    mask: torch.Tensor,
    length: int,
) -> torch.Tensor:
    masked = mask.transpose(1, 2) * features_frequency
    enhanced_magnitude = torch.expm1(masked).clamp_min(0.0)
    enhanced_spec = torch.polar(enhanced_magnitude, torch.angle(spec))
    return target._istft(enhanced_spec, length).squeeze(0).squeeze(0)


@torch.inference_mode()
def collect_multi_action_records(
    model: torch.nn.Module,
    manifest_path: str | Path,
    *,
    device: str,
    lows: Iterable[float] = T9_ACTION_LOWS,
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
    action_lows = tuple(float(value) for value in lows)
    target.configure_multi_action_router(
        enabled=False,
        lows=(),
        feature_means=(),
        feature_scales=(),
        weights=(),
        biases=(),
        threshold=0.0,
    )
    target.configure_adaptive_router(
        enabled=False,
        feature_mean=(),
        feature_scale=(),
        weights=(),
        bias=0.0,
        threshold=0.0,
    )
    target.configure_confidence_calibration(
        enabled=False,
        low=0.0,
        high=0.0,
        threshold=0.0,
        temperature=1.5,
    )
    model.eval()
    records: list[dict[str, Any]] = []
    for index, (row, noisy, clean, sample_rate) in enumerate(rows, start=1):
        noisy_batch = noisy.unsqueeze(0).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            spec = target._stft(noisy_batch)
            magnitude = spec.abs().clamp_min(1e-8)
            features_frequency = torch.log1p(magnitude)
            logits = target.mask_generator.forward_logits(
                features_frequency.transpose(1, 2)
            )
            base_mask = target.mask_generator.learnable_sigmoid(logits)
            base = _waveform_for_mask(
                target,
                spec=spec,
                features_frequency=features_frequency,
                mask=base_mask,
                length=noisy.numel(),
            )
            action_payloads: list[tuple[torch.Tensor, torch.Tensor]] = []
            for low in action_lows:
                candidate_logits = target._confidence_candidate_logits_for(
                    logits,
                    low=low,
                    high=0.0,
                    threshold=0.0,
                    temperature=1.5,
                )
                candidate_mask = target.mask_generator.learnable_sigmoid(
                    candidate_logits
                )
                candidate = _waveform_for_mask(
                    target,
                    spec=spec,
                    features_frequency=features_frequency,
                    mask=candidate_mask,
                    length=noisy.numel(),
                )
                router_features = target.confidence_router_features(
                    noisy_batch,
                    magnitude,
                    logits,
                    base_mask,
                    candidate_logits,
                    candidate_mask,
                )
                action_payloads.append((candidate, router_features))

        aligned = min(noisy.numel(), clean.numel(), base.numel())
        clean_np = tensor_to_numpy_mono(clean[:aligned])
        base_np = tensor_to_numpy_mono(base[:aligned].cpu())
        base_metrics = {
            "pesq": float(pesq_score(clean_np, base_np, sample_rate, bandwidth="wb")),
            "stoi": float(stoi_score(clean_np, base_np, sample_rate)),
            "sisdr": float(sisdr(clean_np, base_np)),
        }
        actions: list[dict[str, Any]] = []
        for low, (candidate, router_features) in zip(
            action_lows, action_payloads, strict=True
        ):
            candidate_np = tensor_to_numpy_mono(candidate[:aligned].cpu())
            candidate_metrics = {
                "pesq": float(
                    pesq_score(clean_np, candidate_np, sample_rate, bandwidth="wb")
                ),
                "stoi": float(stoi_score(clean_np, candidate_np, sample_rate)),
                "sisdr": float(sisdr(clean_np, candidate_np)),
            }
            actions.append(
                {
                    "low": low,
                    "features": router_features.squeeze(0).float().cpu().tolist(),
                    **candidate_metrics,
                    "delta_pesq": candidate_metrics["pesq"] - base_metrics["pesq"],
                }
            )
        records.append(
            {
                "token": Path(row.noisy).stem,
                "base": base_metrics,
                "actions": actions,
            }
        )
        if progress_callback and (
            index == 1 or index == len(rows) or index % 32 == 0
        ):
            progress_callback(
                f"T9 labeled {index}/{len(rows)} from {Path(manifest_path).name}"
            )
    return records


def _predict(ridge: dict[str, Any], features: np.ndarray) -> np.ndarray:
    mean = np.asarray(ridge["feature_mean"], dtype=np.float64)
    scale = np.asarray(ridge["feature_scale"], dtype=np.float64)
    weights = np.asarray(ridge["weights"], dtype=np.float64)
    return ((features - mean) / scale) @ weights + float(ridge["bias"])


def _aggregate(
    records: list[dict[str, Any]], decisions: np.ndarray
) -> dict[str, Any]:
    chosen = np.asarray(decisions, dtype=np.int64)
    if chosen.shape != (len(records),):
        raise ValueError("T9 decision count does not match records.")
    result: dict[str, Any] = {}
    for metric in ("pesq", "stoi", "sisdr"):
        values = []
        for row, action in zip(records, chosen, strict=True):
            values.append(
                float(
                    row["base"][metric]
                    if action < 0
                    else row["actions"][int(action)][metric]
                )
            )
        result[f"{metric}_mean"] = float(np.mean(values))
    result["base_count"] = int(np.sum(chosen < 0))
    result["action_counts"] = {
        str(index): int(np.sum(chosen == index))
        for index in range(len(T9_ACTION_LOWS))
    }
    result["nonbase_action_count"] = int(
        sum(count > 0 for count in result["action_counts"].values())
    )
    return result


def run_t9_multi_router_search(
    *,
    teacher_checkpoint: str | Path,
    identities_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    fit_count: int = T9_FIT_SUPPORT,
    calibration_count: int = T9_CALIBRATION_SUPPORT,
    fit_start: int = T9_FIT_START,
    lows: Iterable[float] = T9_ACTION_LOWS,
    lambdas: Iterable[float] = T9_RIDGE_LAMBDAS,
    thresholds: Iterable[float] = T9_THRESHOLDS,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T9 multi-action router search is CUDA-only.")
    action_lows = tuple(float(value) for value in lows)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    support = prepare_t9_support_manifests(
        identities_path,
        root / "support",
        fit_count=fit_count,
        calibration_count=calibration_count,
        fit_start=fit_start,
    )
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    fit_records = collect_multi_action_records(
        model,
        support["fit"]["manifest"],
        device=device,
        lows=action_lows,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )
    calibration_records = collect_multi_action_records(
        model,
        support["calibration"]["manifest"],
        device=device,
        lows=action_lows,
        max_files=max_eval_files,
        progress_callback=progress_callback,
    )

    ridges: list[dict[str, Any]] = []
    prediction_columns: list[np.ndarray] = []
    for action_index in range(len(action_lows)):
        x_fit = np.asarray(
            [row["actions"][action_index]["features"] for row in fit_records],
            dtype=np.float64,
        )
        y_fit = np.asarray(
            [row["actions"][action_index]["delta_pesq"] for row in fit_records],
            dtype=np.float64,
        )
        ridge = fit_ridge_router(x_fit, y_fit, lambdas=lambdas)
        ridges.append(ridge)
        x_calibration = np.asarray(
            [
                row["actions"][action_index]["features"]
                for row in calibration_records
            ],
            dtype=np.float64,
        )
        prediction_columns.append(_predict(ridge, x_calibration))
    predictions = np.stack(prediction_columns, axis=1)
    best_actions = np.argmax(predictions, axis=1)
    best_predictions = predictions[
        np.arange(predictions.shape[0]), best_actions
    ]

    baseline_calibration = _aggregate(
        calibration_records,
        np.full(len(calibration_records), -1, dtype=np.int64),
    )
    true_deltas = np.asarray(
        [
            [action["delta_pesq"] for action in row["actions"]]
            for row in calibration_records
        ],
        dtype=np.float64,
    )
    oracle_actions = np.argmax(true_deltas, axis=1)
    oracle_best = true_deltas[
        np.arange(true_deltas.shape[0]), oracle_actions
    ]
    oracle_decisions = np.where(oracle_best > 0.0, oracle_actions, -1)
    oracle_calibration = _aggregate(calibration_records, oracle_decisions)
    oracle_gain = (
        oracle_calibration["pesq_mean"] - baseline_calibration["pesq_mean"]
    )

    threshold_rows: list[dict[str, Any]] = []
    for threshold in tuple(float(value) for value in thresholds):
        decisions = np.where(best_predictions >= threshold, best_actions, -1)
        metrics = _aggregate(calibration_records, decisions)
        deltas = {
            key: float(metrics[key]) - float(baseline_calibration[key])
            for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
        }
        checks = {
            "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
            "stoi_drop_at_most_0_0015": deltas["stoi_mean"] >= -0.0015,
            "sisdr_drop_at_most_0_15": deltas["sisdr_mean"] >= -0.15,
            "at_least_two_nonbase_actions": metrics["nonbase_action_count"] >= 2,
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
    eligible = [row for row in threshold_rows if row["eligible"]]
    selected_threshold = (
        max(eligible, key=lambda row: row["metrics"]["pesq_mean"])
        if eligible
        else max(threshold_rows, key=lambda row: row["metrics"]["pesq_mean"])
    )
    configure_multi_action_router(
        model,
        ridges=ridges,
        threshold=float(selected_threshold["threshold"]),
        lows=action_lows,
    )
    checkpoint = root / "T9-MULTI-ROUTED.pt"
    save_checkpoint_package(
        checkpoint,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T9-TRAIN-ONLY-MULTI-ACTION-ROUTER",
            "action_lows": list(action_lows),
            "decision_threshold": selected_threshold["threshold"],
            "test_read": False,
        },
    )
    reloaded, reloaded_package = load_model_from_checkpoint(checkpoint, device=device)
    reloaded_target = (
        reloaded.base_model if hasattr(reloaded, "base_model") else reloaded
    )
    roundtrip = bool(
        getattr(reloaded_target, "multi_router_enabled", False)
        and tuple(getattr(reloaded_target, "multi_router_lows", ())) == action_lows
        and len(getattr(reloaded_target, "multi_router_weights", ()))
        == len(action_lows)
    )
    finite_ridges = all(
        math.isfinite(float(value))
        for ridge in ridges
        for value in (ridge["fit_mse"], ridge["fit_pearson"], ridge["bias"])
    )
    prevalidation_checks = {
        "oracle_gain_at_least_0_025": oracle_gain >= 0.025,
        "learned_threshold_eligible": bool(selected_threshold["eligible"]),
        "finite_ridges": finite_ridges,
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
                key: float(select_metrics[key])
                - float(baseline_select_metrics[key])
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
            and int(fit_count) == T9_FIT_SUPPORT
            and int(calibration_count) == T9_CALIBRATION_SUPPORT
            and int(fit_start) == T9_FIT_START
            and action_lows == T9_ACTION_LOWS
            and tuple(float(value) for value in lambdas) == T9_RIDGE_LAMBDAS
            and tuple(float(value) for value in thresholds) == T9_THRESHOLDS
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T9-TRAIN-ONLY-MULTI-ACTION-ROUTER",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "support": support,
        "action_lows": list(action_lows),
        "ridges": ridges,
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
