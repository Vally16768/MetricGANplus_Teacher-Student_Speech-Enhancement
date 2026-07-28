"""Deterministic PMSQE-primary micro-step backtracking for T4-B."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from sebench.checkpoints import (
    load_checkpoint_package,
    load_model_from_checkpoint,
    save_checkpoint_package,
)
from sebench.t3_perceptual import T3LossBreakdown, T3TeacherObjective
from sebench.t3_training import clone_state_dict, sha256_file
from sebench.teacher_cache import TeacherCacheDataset
from sebench.training import evaluate_manifest


T4_MICRO_HORIZONS = (1, 4, 16, 64, 256)
T4_BACKTRACK_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)
T4_MICRO_LR = 1e-6
T4_SUPERVISED_CONSTRAINT_SCALE = 0.10


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _triplet(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }


def t4_microstep_loss(
    breakdown: T3LossBreakdown,
    *,
    anchor_weight: float,
    pmsqe_weight: float,
    constraint_scale: float = T4_SUPERVISED_CONSTRAINT_SCALE,
) -> torch.Tensor:
    """Make PMSQE primary while retaining supervised/anchor constraints."""
    if not 0.0 < float(constraint_scale) <= 1.0:
        raise ValueError("T4 constraint scale must be in (0, 1].")
    if float(anchor_weight) < 0.0 or float(pmsqe_weight) <= 0.0:
        raise ValueError("T4 requires non-negative anchor and positive PMSQE weights.")
    constrained = (
        breakdown.mrstft
        + 0.10 * breakdown.sisdr
        + float(anchor_weight) * breakdown.anchor
    )
    return (
        float(constraint_scale) * constrained
        + float(pmsqe_weight) * breakdown.pmsqe
    )


def interpolate_state_dict(
    base: dict[str, torch.Tensor],
    proposal: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Interpolate a proposal back toward exact T0."""
    coefficient = float(alpha)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("T4 backtracking alpha must be in [0, 1].")
    if base.keys() != proposal.keys():
        raise ValueError("T4 interpolation state keys do not match.")
    result: dict[str, torch.Tensor] = {}
    for key in base:
        left = base[key]
        right = proposal[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"T4 interpolation tensor contract mismatch: {key}")
        if left.is_floating_point() or left.is_complex():
            result[key] = left + coefficient * (right - left)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"T4 cannot interpolate changed integer tensor: {key}")
            result[key] = left.clone()
    return result


def _train_micro_trajectory(
    *,
    teacher_checkpoint: str | Path,
    teacher_cache_manifest: str | Path,
    weights: dict[str, Any],
    horizon: int,
    seed: int,
    device: str,
    progress_callback: Callable[[str], None] | None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, float]]:
    if int(horizon) < 1:
        raise ValueError("T4 micro-step horizon must be positive.")
    _seed_everything(seed)
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    model.train()
    anchor_weight = float(weights["frozen_weights"]["anchor"])
    pmsqe_weight = float(weights["frozen_weights"]["pmsqe"])
    objective = T3TeacherObjective(
        branch="E2-PMSQE",
        anchor_weight=anchor_weight,
        pmsqe_weight=pmsqe_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=T4_MICRO_LR)
    dataset = TeacherCacheDataset(
        teacher_cache_manifest,
        segment_len=32_000,
        sample_rate=16_000,
        n_fft=512,
        hop_length=256,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        generator=generator,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    totals = {
        "loss": 0.0,
        "mrstft": 0.0,
        "sisdr": 0.0,
        "anchor": 0.0,
        "pmsqe": 0.0,
        "grad_norm": 0.0,
    }
    completed = 0
    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        teacher_t0 = batch["teacher_wav"].to(device, non_blocking=True)
        lengths = batch["length"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        candidate = model(noisy.unsqueeze(1)).squeeze(1)
        breakdown = objective(
            candidate.float(),
            clean.float(),
            teacher_t0.float(),
            lengths=lengths,
        )
        loss = t4_microstep_loss(
            breakdown,
            anchor_weight=anchor_weight,
            pmsqe_weight=pmsqe_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("T4-B encountered a non-finite objective.")
        loss.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise FloatingPointError("T4-B encountered a non-finite gradient.")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("T4-B encountered a non-finite gradient norm.")
        optimizer.step()
        completed += 1
        totals["loss"] += float(loss.detach())
        totals["mrstft"] += float(breakdown.mrstft.detach())
        totals["sisdr"] += float(breakdown.sisdr.detach())
        totals["anchor"] += float(breakdown.anchor.detach())
        totals["pmsqe"] += float(breakdown.pmsqe.detach())
        totals["grad_norm"] += float(grad_norm.detach())
        if progress_callback and (
            completed == 1 or completed == horizon or completed % 32 == 0
        ):
            progress_callback(f"T4-B horizon={horizon} trained {completed}/{horizon} steps")
        if completed >= horizon:
            break
    if completed != horizon:
        raise RuntimeError("T4-B cache did not provide enough batches for its horizon.")
    return (
        model,
        package,
        {key: value / completed for key, value in totals.items()},
    )


def run_t4_microstep_backtracking(
    *,
    teacher_checkpoint: str | Path,
    teacher_cache_manifest: str | Path,
    identities_path: str | Path,
    weights_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    baseline_rank_metrics: dict[str, Any],
    baseline_select_metrics: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    seed: int = 3003,
    horizons: Iterable[int] = T4_MICRO_HORIZONS,
    alphas: Iterable[float] = T4_BACKTRACK_ALPHAS,
    max_eval_files: int | None = None,
    resume: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train bounded prefixes from T0 and rank backtracked proposals."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T4-B teacher training is CUDA-only.")
    normalized_horizons = tuple(int(item) for item in horizons)
    normalized_alphas = tuple(float(item) for item in alphas)
    if (
        not normalized_horizons
        or tuple(sorted(set(normalized_horizons))) != normalized_horizons
        or any(item < 1 for item in normalized_horizons)
    ):
        raise ValueError("T4-B horizons must be unique, positive and increasing.")
    if (
        not normalized_alphas
        or any(not 0.0 < item <= 1.0 for item in normalized_alphas)
        or 1.0 not in normalized_alphas
    ):
        raise ValueError("T4-B alphas must be in (0, 1] and include 1.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identities = json.loads(Path(identities_path).read_text(encoding="utf-8"))
    weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    teacher_hash = sha256_file(teacher_checkpoint)
    if teacher_hash != str(weights["teacher_checkpoint_sha256"]):
        raise ValueError("T4-B teacher/weights identity mismatch.")
    if sha256_file(identities_path) != str(weights["identities_sha256"]):
        raise ValueError("T4-B identities/weights mismatch.")
    contract = {
        "teacher_checkpoint_sha256": teacher_hash,
        "teacher_cache_manifest_sha256": sha256_file(teacher_cache_manifest),
        "identities_sha256": sha256_file(identities_path),
        "weights_sha256": sha256_file(weights_path),
        "val_rank_manifest_sha256": sha256_file(val_rank_manifest),
        "val_select_manifest_sha256": sha256_file(val_select_manifest),
        "seed": int(seed),
        "horizons": list(normalized_horizons),
        "alphas": list(normalized_alphas),
        "optimizer": "Adam",
        "lr": T4_MICRO_LR,
        "constraint_scale": T4_SUPERVISED_CONSTRAINT_SCALE,
        "max_eval_files": max_eval_files,
        "test_read": False,
    }
    progress_path = root / "progress.json"
    progress: dict[str, Any] = {"schema_version": 1, "contract": contract, "horizons": []}
    if resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != contract:
            raise ValueError("T4-B resume contract mismatch.")
        for record in progress.get("horizons", []):
            checkpoint = Path(str(record["proposal_checkpoint"]))
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != record["proposal_checkpoint_sha256"]
            ):
                raise ValueError("T4-B resume proposal artifact mismatch.")
    completed = {int(row["horizon"]) for row in progress["horizons"]}
    stopped_unsafe = any(
        not bool(row["full_proposal_safe"]) for row in progress["horizons"]
    )
    base_model, base_package = load_model_from_checkpoint(
        teacher_checkpoint, device=device
    )
    base_state = clone_state_dict(base_model)
    if max_eval_files is not None:
        baseline_rank_metrics = _triplet(
            evaluate_manifest(
                base_model,
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
        baseline_select_metrics = _triplet(
            evaluate_manifest(
                base_model,
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
    del base_model
    torch.cuda.empty_cache()
    for horizon in normalized_horizons:
        if stopped_unsafe:
            break
        if horizon in completed:
            continue
        model, package, train_metrics = _train_micro_trajectory(
            teacher_checkpoint=teacher_checkpoint,
            teacher_cache_manifest=teacher_cache_manifest,
            weights=weights,
            horizon=horizon,
            seed=seed,
            device=device,
            progress_callback=progress_callback,
        )
        proposal_state = clone_state_dict(model)
        proposal_path = root / "proposals" / f"horizon-{horizon:03d}.pt"
        save_checkpoint_package(
            proposal_path,
            model,
            model_family=str(package["model_family"]),
            variant=str(package.get("variant", "base")),
            extra={
                "strategy": "T4-B-PMSQE-MICROSTEP",
                "horizon": horizon,
                "seed": seed,
                "contract": contract,
            },
        )
        candidates: list[dict[str, Any]] = []
        for alpha in normalized_alphas:
            model.load_state_dict(
                interpolate_state_dict(base_state, proposal_state, alpha)
            )
            metrics = _triplet(
                evaluate_manifest(
                    model,
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
            checks = {
                "finite_metrics": all(math.isfinite(value) for value in metrics.values()),
                "pesq_no_regression": metrics["pesq_mean"]
                >= float(baseline_rank_metrics["pesq_mean"]) - 0.005,
                "stoi_guardrail": metrics["stoi_mean"]
                >= float(baseline_rank_metrics["stoi_mean"]) - 0.002,
                "sisdr_guardrail": metrics["sisdr_mean"]
                >= float(baseline_rank_metrics["sisdr_mean"]) - 0.25,
            }
            candidates.append(
                {
                    "horizon": horizon,
                    "alpha": alpha,
                    "val_rank_metrics": metrics,
                    "checks": checks,
                    "eligible": all(checks.values()),
                }
            )
            if progress_callback:
                progress_callback(
                    f"T4-B horizon={horizon} alpha={alpha:.4f} "
                    f"pesq={metrics['pesq_mean']:.6f}"
                )
        full = next(item for item in candidates if item["alpha"] == 1.0)
        record = {
            "horizon": horizon,
            "train_metrics": train_metrics,
            "proposal_checkpoint": proposal_path.as_posix(),
            "proposal_checkpoint_sha256": sha256_file(proposal_path),
            "candidates": candidates,
            "full_proposal_safe": bool(full["eligible"]),
        }
        progress["horizons"].append(record)
        _atomic_json(progress_path, progress)
        del model
        torch.cuda.empty_cache()
        if not full["eligible"]:
            stopped_unsafe = True
            break
    all_candidates = [
        candidate
        for record in progress["horizons"]
        for candidate in record["candidates"]
        if candidate["eligible"]
    ]
    fallback = {
        "horizon": 0,
        "alpha": 0.0,
        "val_rank_metrics": _triplet(baseline_rank_metrics),
        "checks": {
            "finite_metrics": True,
            "pesq_no_regression": True,
            "stoi_guardrail": True,
            "sisdr_guardrail": True,
        },
        "eligible": True,
    }
    selected = max(
        [fallback, *all_candidates],
        key=lambda item: float(item["val_rank_metrics"]["pesq_mean"]),
    )
    selected_horizon = int(selected["horizon"])
    selected_alpha = float(selected["alpha"])
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    if selected_horizon:
        record = next(
            row for row in progress["horizons"] if int(row["horizon"]) == selected_horizon
        )
        proposal_package = load_checkpoint_package(
            record["proposal_checkpoint"], map_location="cpu"
        )
        model.load_state_dict(
            interpolate_state_dict(
                base_state,
                dict(proposal_package["state_dict"]),
                selected_alpha,
            )
        )
    selected_path = root / "T4-B-SELECTED.pt"
    save_checkpoint_package(
        selected_path,
        model,
        model_family=str(base_package["model_family"]),
        variant=str(base_package.get("variant", "base")),
        extra={
            "strategy": "T4-B-PMSQE-MICROSTEP",
            "selected_horizon": selected_horizon,
            "selected_alpha": selected_alpha,
            "selection_split": "val_rank",
            "contract": contract,
            "test_read": False,
        },
    )
    if selected_horizon == 0:
        select_metrics = _triplet(baseline_select_metrics)
    else:
        select_metrics = _triplet(
            evaluate_manifest(
                model,
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
    deltas = {
        key: float(select_metrics[key]) - float(baseline_select_metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }
    gate_checks = {
        "nonzero_candidate": selected_horizon > 0,
        "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
        "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
        "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
        "production_support": max_eval_files is None,
    }
    summary = {
        "schema_version": 1,
        "status": "passed" if all(gate_checks.values()) else "failed",
        "strategy": "T4-B-PMSQE-MICROSTEP",
        "contract": contract,
        "completed_horizons": [
            int(record["horizon"]) for record in progress["horizons"]
        ],
        "stopped_at_first_unsafe_horizon": stopped_unsafe,
        "selected_horizon": selected_horizon,
        "selected_alpha": selected_alpha,
        "selected_val_rank_metrics": selected["val_rank_metrics"],
        "selected_val_select_metrics": select_metrics,
        "val_select_deltas": deltas,
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "selected_checkpoint": selected_path.as_posix(),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "progress_sha256": sha256_file(progress_path),
        "test_read": False,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
