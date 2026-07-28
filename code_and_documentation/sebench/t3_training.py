"""Resumable matched E1/E2 teacher pilot for the predeclared T3 study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from metrics.pesq import pesq_score
from sebench.audio import load_mono_audio
from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.t3_perceptual import (
    DifferentiablePESQInspiredLoss,
    T3TeacherObjective,
)
from sebench.teacher_cache import TeacherCacheDataset
from sebench.training import evaluate_manifest


class PlannedT3Interruption(RuntimeError):
    """Controlled post-evaluation interruption used by exact-resume tests."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def capture_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def build_t3_training_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    branch: str,
    proposal: int,
    accepted_epoch: int,
    best_epoch: int,
    best_score: float,
    best_model_state: dict[str, torch.Tensor],
    epochs_without_improve: int,
    consecutive_rejections: int,
    history: list[dict[str, Any]],
    provenance: dict[str, Any],
    status: str = "post_evaluation",
) -> dict[str, Any]:
    """Return the only resume boundary accepted by the production trainer."""
    return {
        "schema_version": 1,
        "status": status,
        "branch": branch,
        "proposal": int(proposal),
        "accepted_epoch": int(accepted_epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "epochs_without_improve": int(epochs_without_improve),
        "consecutive_rejections": int(consecutive_rejections),
        "model_state": clone_state_dict(model),
        "best_model_state": {
            key: value.detach().cpu().clone()
            for key, value in best_model_state.items()
        },
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": copy.deepcopy(scheduler.state_dict()),
        "rng_state": capture_rng_state(),
        "history": copy.deepcopy(history),
        "provenance": copy.deepcopy(provenance),
    }


def restore_t3_training_state(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    expected_branch: str,
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("status") != "post_evaluation":
        raise ValueError("T3 resumes only from a complete post-evaluation boundary.")
    if payload.get("branch") != expected_branch:
        raise ValueError("T3 resume branch mismatch.")
    if payload.get("provenance") != expected_provenance:
        raise ValueError("T3 resume provenance mismatch.")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload.get("rng_state"))
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _metric_triplet(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }


def _input_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = noisy - clean
    return float(
        10.0
        * torch.log10(
            clean.square().mean().clamp_min(1e-12)
            / noise.square().mean().clamp_min(1e-12)
        ).item()
    )


@torch.inference_mode()
def audit_current_pmsqe_direction(
    *,
    model: torch.nn.Module,
    identity_records: list[dict[str, Any]],
    device: str,
    max_records: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Recheck local true-PESQ direction around the current generator."""
    selected = [
        row for row in identity_records if row.get("partition") == "calibration"
    ]
    if max_records is not None:
        selected = selected[: int(max_records)]
    if not selected:
        raise ValueError("T3 current-direction audit has no calibration identities.")
    pmsqe = DifferentiablePESQInspiredLoss().to(device)
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    try:
        for index, row in enumerate(selected, start=1):
            noisy, _ = load_mono_audio(str(row["noisy"]), 16_000)
            clean, _ = load_mono_audio(str(row["clean"]), 16_000)
            length = min(noisy.numel(), clean.numel())
            noisy = noisy[:length]
            clean = clean[:length]
            variants = model.forward_mask_logit_variants(
                noisy.to(device).reshape(1, 1, -1),
                (-0.02, 0.0, 0.02),
            )[:, 0, 0]
            clean_batch = clean.to(device).reshape(1, -1).expand(3, -1)
            pmsqe_values = (
                pmsqe.loss(clean_batch.float(), variants.float())
                .detach()
                .cpu()
                .numpy()
            )
            variants_cpu = variants.detach().cpu().numpy()
            true_scores = [
                float(pesq_score(clean.numpy(), variant, 16_000, bandwidth="wb"))
                for variant in variants_cpu
            ]
            snr = _input_snr(clean, noisy)
            for candidate_index in (0, 2):
                delta_true = true_scores[candidate_index] - true_scores[1]
                delta_predicted = -float(
                    pmsqe_values[candidate_index] - pmsqe_values[1]
                )
                if (
                    math.isfinite(delta_true)
                    and math.isfinite(delta_predicted)
                    and abs(delta_true) >= 1e-4
                ):
                    rows.append(
                        {
                            "delta_true": delta_true,
                            "delta_predicted": delta_predicted,
                            "snr": snr,
                        }
                    )
            if progress_callback and (
                index == 1 or index == len(selected) or index % 25 == 0
            ):
                progress_callback(
                    f"current-direction {index}/{len(selected)} identities"
                )
    finally:
        if was_training:
            model.train()
    true = np.asarray([row["delta_true"] for row in rows], dtype=np.float64)
    predicted = np.asarray(
        [row["delta_predicted"] for row in rows], dtype=np.float64
    )
    snr = np.asarray([row["snr"] for row in rows], dtype=np.float64)
    agreement = np.sign(true) == np.sign(predicted)
    quartiles: list[dict[str, Any]] = []
    if len(rows):
        bins = np.digitize(snr, np.quantile(snr, [0.25, 0.50, 0.75]))
        for index in range(4):
            chosen = bins == index
            quartiles.append(
                {
                    "quartile": index + 1,
                    "count": int(chosen.sum()),
                    "sign_agreement": (
                        float(agreement[chosen].mean())
                        if chosen.any()
                        else float("nan")
                    ),
                }
            )
    minimum_quartile = min(
        (float(row["sign_agreement"]) for row in quartiles),
        default=float("nan"),
    )
    required_pairs = 2 if max_records is not None and max_records < 100 else 200
    summary = {
        "identity_count": len(selected),
        "eligible_pairs": len(rows),
        "sign_agreement": float(agreement.mean()) if len(rows) else float("nan"),
        "delta_spearman": (
            float(spearmanr(true, predicted).statistic)
            if len(rows) > 1
            else float("nan")
        ),
        "snr_quartiles": quartiles,
    }
    checks = {
        "eligible_pairs": len(rows) >= required_pairs,
        "sign_agreement": summary["sign_agreement"] >= 0.70,
        "delta_spearman": summary["delta_spearman"] >= 0.60,
        "snr_quartiles": minimum_quartile >= 0.55,
    }
    return {**summary, "checks": checks, "passed": all(checks.values())}


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    objective: T3TeacherObjective,
    optimizer: torch.optim.Optimizer,
    dataset: TeacherCacheDataset,
    device: str,
    batch_size: int,
    seed: int,
    max_rows: int | None,
    grad_clip: float,
    progress_callback: Callable[[str], None] | None,
) -> dict[str, float]:
    _seed_everything(seed)
    active_dataset: Any = dataset
    if max_rows is not None:
        active_dataset = Subset(dataset, range(min(int(max_rows), len(dataset))))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        active_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        generator=generator,
    )
    model.train()
    totals = {"loss": 0.0, "mrstft": 0.0, "sisdr": 0.0, "anchor": 0.0, "pmsqe": 0.0}
    parameter_tensors = [item for item in model.parameters() if item.requires_grad]
    steps = 0
    for step, batch in enumerate(loader, start=1):
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
        if not torch.isfinite(breakdown.total):
            raise FloatingPointError("T3 encountered a non-finite objective.")
        breakdown.total.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameter_tensors
        ):
            raise FloatingPointError("T3 encountered a non-finite parameter gradient.")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameter_tensors, grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("T3 encountered a non-finite gradient norm.")
        optimizer.step()
        for key in totals:
            totals[key] += float(getattr(breakdown, key if key != "loss" else "total").detach())
        steps += 1
        if progress_callback and (step == 1 or step == len(loader) or step % 100 == 0):
            progress_callback(f"train {step}/{len(loader)} batches")
    return {key: value / max(steps, 1) for key, value in totals.items()}


def run_t3_branch(
    *,
    branch: str,
    teacher_checkpoint: str | Path,
    teacher_cache_manifest: str | Path,
    identities_path: str | Path,
    weights_path: str | Path,
    val_rank_manifest: str | Path,
    val_select_manifest: str | Path,
    output_dir: str | Path,
    baseline_rank_metrics: dict[str, Any],
    device: str = "cuda",
    seed: int = 3003,
    max_accepted_epochs: int = 10,
    batch_size: int = 1,
    smoke: bool = False,
    resume: bool = True,
    interrupt_after_proposal: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    normalized_branch = branch.upper()
    if normalized_branch not in {"E1-SUP", "E2-PMSQE"}:
        raise ValueError("branch must be E1-SUP or E2-PMSQE.")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T3 teacher training is CUDA-only.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "training_state.pt"
    history_path = root / "history.json"
    checkpoint_path = root / "selected.pt"
    identities = json.loads(Path(identities_path).read_text(encoding="utf-8"))
    weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    expected_checkpoint_hash = str(weights["teacher_checkpoint_sha256"])
    if sha256_file(teacher_checkpoint) != expected_checkpoint_hash:
        raise ValueError("T3 teacher checkpoint identity mismatch.")
    if sha256_file(identities_path) != str(weights["identities_sha256"]):
        raise ValueError("T3 identities/weights mismatch.")
    provenance = {
        "branch": normalized_branch,
        "seed": int(seed),
        "teacher_checkpoint_sha256": expected_checkpoint_hash,
        "teacher_cache_manifest_sha256": sha256_file(teacher_cache_manifest),
        "identities_sha256": sha256_file(identities_path),
        "weights_sha256": sha256_file(weights_path),
        "val_rank_manifest_sha256": sha256_file(val_rank_manifest),
        "val_select_manifest_sha256": sha256_file(val_select_manifest),
        "max_accepted_epochs": int(max_accepted_epochs),
        "optimizer": "Adam",
        "lr": 1e-6,
        "batch_size": int(batch_size),
        "segment_samples": 32_000,
        "smoke": bool(smoke),
    }
    _seed_everything(seed)
    model, package = load_model_from_checkpoint(teacher_checkpoint, device=device)
    model.train()
    objective = T3TeacherObjective(
        branch=normalized_branch,
        anchor_weight=float(weights["frozen_weights"]["anchor"]),
        pmsqe_weight=(
            float(weights["frozen_weights"]["pmsqe"])
            if normalized_branch == "E2-PMSQE"
            else 0.0
        ),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )
    dataset = TeacherCacheDataset(
        teacher_cache_manifest,
        segment_len=32_000,
        sample_rate=16_000,
        n_fft=512,
        hop_length=256,
    )
    proposal = 0
    accepted_epoch = 0
    best_epoch = 0
    best_score = float(baseline_rank_metrics["pesq_mean"])
    best_model_state = clone_state_dict(model)
    epochs_without_improve = 0
    consecutive_rejections = 0
    history: list[dict[str, Any]] = []
    if resume and state_path.is_file():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        payload = restore_t3_training_state(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_branch=normalized_branch,
            expected_provenance=provenance,
        )
        proposal = int(payload["proposal"])
        accepted_epoch = int(payload["accepted_epoch"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_score"])
        best_model_state = payload["best_model_state"]
        epochs_without_improve = int(payload["epochs_without_improve"])
        consecutive_rejections = int(payload["consecutive_rejections"])
        history = list(payload["history"])
    max_proposals = max_accepted_epochs + 6
    stop_reason = "maximum_accepted_epochs"
    while accepted_epoch < max_accepted_epochs and proposal < max_proposals:
        proposal += 1
        before_model = clone_state_dict(model)
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        before_scheduler = copy.deepcopy(scheduler.state_dict())
        before_rng = capture_rng_state()
        started = time.monotonic()
        failure: str | None = None
        try:
            train_metrics = _train_one_epoch(
                model=model,
                objective=objective,
                optimizer=optimizer,
                dataset=dataset,
                device=device,
                batch_size=batch_size,
                seed=seed + (accepted_epoch + 1) * 100_003,
                max_rows=2 if smoke else None,
                grad_clip=5.0,
                progress_callback=progress_callback,
            )
            rank_metrics = evaluate_manifest(
                model,
                str(val_rank_manifest),
                device,
                sample_rate=16_000,
                bandwidth="wb",
                compute_dnsmos=False,
                compute_composite=False,
                max_files=2 if smoke else None,
                batch_size=1,
                progress_callback=progress_callback,
            )
            local_gate: dict[str, Any] | None = None
            if normalized_branch == "E2-PMSQE":
                local_gate = audit_current_pmsqe_direction(
                    model=model,
                    identity_records=list(identities["records"]),
                    device=device,
                    max_records=4 if smoke else None,
                    progress_callback=progress_callback,
                )
            last_accepted = next(
                (item for item in reversed(history) if item.get("accepted")),
                None,
            )
            previous_pesq = (
                float(last_accepted["rank_metrics"]["pesq_mean"])
                if last_accepted is not None
                else float(baseline_rank_metrics["pesq_mean"])
            )
            checks = {
                "finite_metrics": all(
                    math.isfinite(float(rank_metrics[key]))
                    for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
                ),
                "pesq_rollback": float(rank_metrics["pesq_mean"])
                >= previous_pesq - 0.005,
                "stoi_guardrail": float(rank_metrics["stoi_mean"])
                >= float(baseline_rank_metrics["stoi_mean"]) - 0.002,
                "sisdr_guardrail": float(rank_metrics["sisdr_mean"])
                >= float(baseline_rank_metrics["sisdr_mean"]) - 0.25,
                "local_direction": local_gate is None or bool(local_gate["passed"]),
            }
        except FloatingPointError as exc:
            failure = f"{exc.__class__.__name__}: {exc}"
            train_metrics = {}
            rank_metrics = {}
            local_gate = None
            checks = {"training_exception": False}
        accepted = all(checks.values())
        row = {
            "proposal": proposal,
            "accepted_epoch_before": accepted_epoch,
            "accepted": accepted,
            "lr_before_decision": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.monotonic() - started,
            "train_metrics": train_metrics,
            "rank_metrics": _metric_triplet(rank_metrics) if rank_metrics else {},
            "local_direction_gate": local_gate,
            "checks": checks,
            "failure": failure,
        }
        if accepted:
            accepted_epoch += 1
            consecutive_rejections = 0
            score = float(rank_metrics["pesq_mean"])
            scheduler.step(score)
            if score > best_score:
                best_score = score
                best_epoch = accepted_epoch
                best_model_state = clone_state_dict(model)
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
            row["accepted_epoch_after"] = accepted_epoch
            row["lr_after_decision"] = float(optimizer.param_groups[0]["lr"])
        else:
            model.load_state_dict(before_model)
            optimizer.load_state_dict(before_optimizer)
            scheduler.load_state_dict(before_scheduler)
            restore_rng_state(before_rng)
            for group in optimizer.param_groups:
                group["lr"] = max(float(group["lr"]) * 0.5, 1e-7)
            scheduler._last_lr = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            consecutive_rejections += 1
            row["accepted_epoch_after"] = accepted_epoch
            row["lr_after_decision"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        state = build_t3_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            branch=normalized_branch,
            proposal=proposal,
            accepted_epoch=accepted_epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            best_model_state=best_model_state,
            epochs_without_improve=epochs_without_improve,
            consecutive_rejections=consecutive_rejections,
            history=history,
            provenance=provenance,
        )
        atomic_torch_save(state, state_path)
        _atomic_json(history_path, history)
        if interrupt_after_proposal == proposal:
            raise PlannedT3Interruption(f"planned interruption after proposal {proposal}")
        if consecutive_rejections >= 3:
            stop_reason = "three_consecutive_rollbacks"
            break
        if epochs_without_improve >= 3:
            stop_reason = "early_stopping"
            break
    if proposal >= max_proposals and accepted_epoch < max_accepted_epochs:
        stop_reason = "maximum_proposals"
    model.load_state_dict(best_model_state)
    save_checkpoint_package(
        checkpoint_path,
        model,
        model_family=str(package["model_family"]),
        variant=str(package.get("variant", "base")),
        extra={
            "branch": normalized_branch,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_rank_pesq": best_score,
            "t3_provenance": provenance,
        },
    )
    select_metrics = evaluate_manifest(
        model,
        str(val_select_manifest),
        device,
        sample_rate=16_000,
        bandwidth="wb",
        compute_dnsmos=False,
        compute_composite=False,
        max_files=2 if smoke else None,
        batch_size=1,
        progress_callback=progress_callback,
    )
    summary = {
        "schema_version": 1,
        "status": "verification_only" if smoke else "complete",
        "valid_for_promotion": not smoke,
        "branch": normalized_branch,
        "seed": seed,
        "accepted_epochs": accepted_epoch,
        "proposals": proposal,
        "best_epoch": best_epoch,
        "best_val_rank_pesq": best_score,
        "stop_reason": stop_reason,
        "selected_checkpoint": checkpoint_path.as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "val_select_metrics": _metric_triplet(select_metrics),
        "provenance": provenance,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
