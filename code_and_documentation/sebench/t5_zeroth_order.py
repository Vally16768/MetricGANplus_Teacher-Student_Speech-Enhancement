"""True-PESQ zeroth-order frequency-curve search for T5."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as functional

from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_training import clone_state_dict, sha256_file
from sebench.training import evaluate_manifest


T5_KNOT_COUNT = 8
T5_INITIAL_COEFFICIENT = -0.10
T5_COEFFICIENT_BOUNDS = (-0.20, 0.05)
T5_COORDINATE_STEPS = (0.08, 0.04, 0.02)
T5_FIT_SUPPORT = 96
T5_CALIBRATION_SUPPORT = 96


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


def frequency_curve_from_knots(
    coefficients: Iterable[float],
    *,
    output_bins: int = 257,
) -> torch.Tensor:
    """Return a smooth linear-frequency bias curve from frozen knots."""
    values = tuple(float(item) for item in coefficients)
    if len(values) != T5_KNOT_COUNT:
        raise ValueError(f"T5 requires exactly {T5_KNOT_COUNT} frequency knots.")
    lower, upper = T5_COEFFICIENT_BOUNDS
    if any(not math.isfinite(item) or item < lower or item > upper for item in values):
        raise ValueError("T5 frequency coefficients violate the frozen bounds.")
    if int(output_bins) < 2:
        raise ValueError("T5 output_bins must be at least two.")
    source = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)
    return functional.interpolate(
        source,
        size=int(output_bins),
        mode="linear",
        align_corners=True,
    ).reshape(-1)


def apply_frequency_logit_curve(
    model: torch.nn.Module,
    coefficients: Iterable[float],
) -> torch.Tensor:
    """Fold one smooth frequency curve into the official output bias."""
    target = model.base_model if hasattr(model, "base_model") else model
    generator = getattr(target, "mask_generator", None)
    linear = getattr(generator, "linear2", None)
    if not isinstance(linear, torch.nn.Linear) or linear.bias is None:
        raise TypeError("T5 requires the official MetricGAN mask linear2 bias.")
    curve = frequency_curve_from_knots(
        coefficients,
        output_bins=int(linear.bias.numel()),
    ).to(device=linear.bias.device, dtype=linear.bias.dtype)
    with torch.no_grad():
        linear.bias.add_(curve)
    return curve.detach().cpu()


def prepare_t5_support_manifests(
    identities_path: str | Path,
    output_dir: str | Path,
    *,
    fit_count: int = T5_FIT_SUPPORT,
    calibration_count: int = T5_CALIBRATION_SUPPORT,
) -> dict[str, Any]:
    """Freeze disjoint fit/cal manifests from T3 train identities only."""
    identities = json.loads(Path(identities_path).read_text(encoding="utf-8"))
    train = [
        row for row in identities["records"] if row.get("partition") == "train"
    ]
    required = int(fit_count) + int(calibration_count)
    if fit_count < 1 or calibration_count < 1 or len(train) < required:
        raise ValueError("T5 support does not contain the required train identities.")
    fit = train[: int(fit_count)]
    calibration = train[int(fit_count) : required]
    fit_tokens = {str(row["token"]) for row in fit}
    calibration_tokens = {str(row["token"]) for row in calibration}
    fit_clean = {str(row["clean_token"]) for row in fit}
    calibration_clean = {str(row["clean_token"]) for row in calibration}
    if fit_tokens & calibration_tokens or fit_clean & calibration_clean:
        raise ValueError("T5 fit/calibration support is not pair/clean disjoint.")
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
        "source_partition": "train",
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


def _checks(
    metrics: dict[str, float],
    baseline: dict[str, Any],
) -> dict[str, bool]:
    return {
        "finite_metrics": all(math.isfinite(value) for value in metrics.values()),
        "stoi_guardrail": metrics["stoi_mean"]
        >= float(baseline["stoi_mean"]) - 0.002,
        "sisdr_guardrail": metrics["sisdr_mean"]
        >= float(baseline["sisdr_mean"]) - 0.25,
    }


def run_t5_frequency_search(
    *,
    teacher_checkpoint: str | Path,
    identities_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    steps: Iterable[float] = T5_COORDINATE_STEPS,
    fit_count: int = T5_FIT_SUPPORT,
    calibration_count: int = T5_CALIBRATION_SUPPORT,
    coordinate_limit: int | None = None,
    max_eval_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run deterministic fit/cal/rank/select zeroth-order search."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T5 true-PESQ search is CUDA-only.")
    normalized_steps = tuple(float(item) for item in steps)
    if not normalized_steps or any(item <= 0.0 for item in normalized_steps):
        raise ValueError("T5 coordinate steps must be positive.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    support = prepare_t5_support_manifests(
        identities_path,
        root / "support",
        fit_count=fit_count,
        calibration_count=calibration_count,
    )
    fit_manifest = support["fit"]["manifest"]
    calibration_manifest = support["calibration"]["manifest"]
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    base_state = clone_state_dict(model)

    def evaluate(
        coefficients: list[float] | None,
        manifest: str,
    ) -> dict[str, float]:
        model.load_state_dict(base_state)
        if coefficients is not None:
            apply_frequency_logit_curve(model, coefficients)
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

    baseline_fit = evaluate(None, fit_manifest)
    baseline_calibration = evaluate(None, calibration_manifest)
    if max_eval_files is not None:
        baseline_rank_metrics = evaluate(None, str(val_rank_manifest))
        baseline_select_metrics = evaluate(None, str(val_select_manifest))
    coefficients = [T5_INITIAL_COEFFICIENT] * T5_KNOT_COUNT
    current_fit = evaluate(coefficients, fit_manifest)
    initial_calibration = evaluate(coefficients, calibration_manifest)
    sweeps: list[dict[str, Any]] = []
    coordinate_count = (
        T5_KNOT_COUNT
        if coordinate_limit is None
        else min(T5_KNOT_COUNT, max(1, int(coordinate_limit)))
    )
    lower, upper = T5_COEFFICIENT_BOUNDS
    for sweep_index, step in enumerate(normalized_steps, start=1):
        decisions: list[dict[str, Any]] = []
        for coordinate in range(coordinate_count):
            options = [
                {
                    "offset": 0.0,
                    "coefficients": list(coefficients),
                    "fit_metrics": current_fit,
                    "checks": _checks(current_fit, baseline_fit),
                }
            ]
            for offset in (-step, step):
                trial = list(coefficients)
                trial[coordinate] += offset
                if trial[coordinate] < lower or trial[coordinate] > upper:
                    continue
                metrics = evaluate(trial, fit_manifest)
                options.append(
                    {
                        "offset": offset,
                        "coefficients": trial,
                        "fit_metrics": metrics,
                        "checks": _checks(metrics, baseline_fit),
                    }
                )
                if progress_callback:
                    progress_callback(
                        f"T5 sweep={sweep_index}/{len(normalized_steps)} "
                        f"coordinate={coordinate + 1}/{coordinate_count} "
                        f"offset={offset:+.3f} pesq={metrics['pesq_mean']:.6f}"
                    )
            for option in options:
                option["eligible"] = all(option["checks"].values())
            selected = max(
                (option for option in options if option["eligible"]),
                key=lambda option: float(option["fit_metrics"]["pesq_mean"]),
            )
            coefficients = list(selected["coefficients"])
            current_fit = dict(selected["fit_metrics"])
            decisions.append(
                {
                    "coordinate": coordinate,
                    "step": step,
                    "options": options,
                    "selected_offset": float(selected["offset"]),
                    "selected_coefficients": list(coefficients),
                }
            )
        calibration_metrics = evaluate(coefficients, calibration_manifest)
        calibration_checks = _checks(calibration_metrics, baseline_calibration)
        model.load_state_dict(base_state)
        apply_frequency_logit_curve(model, coefficients)
        checkpoint = root / "sweeps" / f"sweep-{sweep_index}.pt"
        save_checkpoint_package(
            checkpoint,
            model,
            model_family=str(package["model_family"]),
            variant=str(package.get("variant", "base")),
            extra={
                "strategy": "T5-TRUE-PESQ-FREQUENCY-CURVE",
                "sweep": sweep_index,
                "step": step,
                "coefficients": list(coefficients),
                "selection_role": "fit_then_calibration",
                "test_read": False,
            },
        )
        sweeps.append(
            {
                "sweep": sweep_index,
                "step": step,
                "coefficients": list(coefficients),
                "fit_metrics": dict(current_fit),
                "calibration_metrics": calibration_metrics,
                "calibration_checks": calibration_checks,
                "calibration_eligible": all(calibration_checks.values()),
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "decisions": decisions,
            }
        )
        _atomic_json(
            root / "progress.json",
            {
                "schema_version": 1,
                "support": support,
                "baseline_fit": baseline_fit,
                "baseline_calibration": baseline_calibration,
                "initial_calibration": initial_calibration,
                "sweeps": sweeps,
                "test_read": False,
            },
        )
    rank_candidates: list[dict[str, Any]] = []
    candidate_specs: list[tuple[str, list[float] | None]] = [
        ("T0", None),
        ("UNIFORM-NEG-0.10", [T5_INITIAL_COEFFICIENT] * T5_KNOT_COUNT),
        *[
            (f"SWEEP-{row['sweep']}", list(row["coefficients"]))
            for row in sweeps
            if row["calibration_eligible"]
        ],
    ]
    for name, candidate_coefficients in candidate_specs:
        metrics = (
            _triplet(baseline_rank_metrics)
            if name == "T0" and max_eval_files is None
            else evaluate(candidate_coefficients, str(val_rank_manifest))
        )
        checks = _checks(metrics, baseline_rank_metrics)
        rank_candidates.append(
            {
                "name": name,
                "coefficients": candidate_coefficients,
                "val_rank_metrics": metrics,
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
        if progress_callback:
            progress_callback(f"T5 rank {name} pesq={metrics['pesq_mean']:.6f}")
    selected = max(
        (row for row in rank_candidates if row["eligible"]),
        key=lambda row: float(row["val_rank_metrics"]["pesq_mean"]),
    )
    selected_coefficients = selected["coefficients"]
    model.load_state_dict(base_state)
    if selected_coefficients is not None:
        curve = apply_frequency_logit_curve(model, selected_coefficients)
    else:
        curve = torch.zeros(257)
    selected_path = root / "T5-SELECTED.pt"
    save_checkpoint_package(
        selected_path,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "strategy": "T5-TRUE-PESQ-FREQUENCY-CURVE",
            "selected_name": selected["name"],
            "coefficients": selected_coefficients,
            "curve_min": float(curve.min()),
            "curve_max": float(curve.max()),
            "selection_split": "val_rank",
            "test_read": False,
        },
    )
    if selected_coefficients is None:
        select_metrics = _triplet(baseline_select_metrics)
    else:
        select_metrics = evaluate(selected_coefficients, str(val_select_manifest))
    deltas = {
        key: float(select_metrics[key]) - float(baseline_select_metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }
    gate_checks = {
        "nonzero_candidate": selected_coefficients is not None,
        "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
        "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
        "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
        "production_support": (
            max_eval_files is None
            and int(fit_count) == T5_FIT_SUPPORT
            and int(calibration_count) == T5_CALIBRATION_SUPPORT
            and coordinate_limit is None
            and normalized_steps == T5_COORDINATE_STEPS
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T5-TRUE-PESQ-FREQUENCY-CURVE",
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "support": support,
        "baseline_fit_metrics": baseline_fit,
        "baseline_calibration_metrics": baseline_calibration,
        "initial_calibration_metrics": initial_calibration,
        "sweeps": sweeps,
        "rank_candidates": rank_candidates,
        "selected_name": selected["name"],
        "selected_coefficients": selected_coefficients,
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
