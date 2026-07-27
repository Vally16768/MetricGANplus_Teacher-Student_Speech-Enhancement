"""Alternating MetricGAN+ discriminator refresh with Desktop-local replay.

This module follows the defining SpeechBrain recipe order before each
generator epoch: current clean/enhanced/noisy updates, historical enhanced
replay, then the same current updates again. Only generated enhanced tensors
and metric labels are cached; VoiceBank+DEMAND inputs remain referenced in
place and are never copied.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from metrics.pesq import pesq_score
from sebench.audio import load_mono_audio, manifest_hash
from sebench.bandwidth import resolve_bandwidth
from sebench.data import read_pair_manifest
from sebench.losses import (
    SpeechBrainMetricDiscriminator,
    save_pesq_proxy_checkpoint,
)


def normalize_pesq(score: float) -> float:
    """Map the PESQ range used by MetricGAN from [-0.5, 4.5] to [0, 1]."""
    return (float(score) + 0.5) / 5.0


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    temporary.replace(path)


def _row_token(noisy: Path, clean: Path) -> str:
    identity = f"{noisy.resolve()}|{clean.resolve()}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:20]


def _aligned_inputs(
    noisy_path: str | Path,
    clean_path: str | Path,
    *,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    noisy, _ = load_mono_audio(noisy_path, sample_rate)
    clean, _ = load_mono_audio(clean_path, sample_rate)
    length = min(noisy.numel(), clean.numel())
    return noisy[:length].contiguous(), clean[:length].contiguous()


@torch.inference_mode()
def build_current_teacher_replay(
    *,
    generator: torch.nn.Module,
    train_manifest: str | Path,
    replay_root: str | Path,
    epoch: int,
    device: str,
    bandwidth: str = "wb",
    max_rows: int = 100,
    seed: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate and score the current teacher distribution once per epoch."""
    profile = resolve_bandwidth(bandwidth)
    if profile.name != "wb":
        raise ValueError("The teacher MetricGAN discriminator is WB-only.")
    epoch_root = Path(replay_root) / f"epoch_{int(epoch):04d}"
    index_path = epoch_root / "index.json"
    if index_path.is_file():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            existing.get("train_manifest_sha256") == manifest_hash(train_manifest)
            and int(existing.get("epoch", -1)) == int(epoch)
        ):
            return existing

    rows = list(read_pair_manifest(train_manifest))
    rng = random.Random(int(seed) + int(epoch))
    rng.shuffle(rows)
    if int(max_rows) > 0:
        rows = rows[: int(max_rows)]

    noisy_score_path = Path(replay_root) / "noisy_scores.json"
    noisy_scores = (
        json.loads(noisy_score_path.read_text(encoding="utf-8"))
        if noisy_score_path.is_file()
        else {}
    )
    records: list[dict[str, Any]] = []
    generator_was_training = generator.training
    generator.eval()
    try:
        for index, row in enumerate(rows, start=1):
            noisy, clean = _aligned_inputs(
                row.noisy,
                row.clean,
                sample_rate=profile.sample_rate,
            )
            enhanced = generator.denoise_single(
                noisy.unsqueeze(0).to(device)
            ).squeeze(0).detach().cpu()
            length = min(noisy.numel(), clean.numel(), enhanced.numel())
            noisy = noisy[:length]
            clean = clean[:length]
            enhanced = enhanced[:length].contiguous()
            enhanced_pesq = pesq_score(
                clean.numpy(),
                enhanced.numpy(),
                profile.sample_rate,
                bandwidth=profile.name,
            )
            if not math.isfinite(enhanced_pesq):
                continue
            token = _row_token(row.noisy, row.clean)
            cache_path = epoch_root / "enhanced" / f"{token}.pt"
            _atomic_tensor(cache_path, enhanced.to(dtype=torch.float16))
            noisy_key = f"{token}|{profile.name}"
            noisy_pesq = noisy_scores.get(noisy_key)
            if noisy_pesq is None:
                noisy_pesq = pesq_score(
                    clean.numpy(),
                    noisy.numpy(),
                    profile.sample_rate,
                    bandwidth=profile.name,
                )
                if not math.isfinite(noisy_pesq):
                    cache_path.unlink(missing_ok=True)
                    continue
                noisy_scores[noisy_key] = float(noisy_pesq)
            records.append(
                {
                    "epoch": int(epoch),
                    "token": token,
                    "noisy": row.noisy.as_posix(),
                    "clean": row.clean.as_posix(),
                    "enhanced": cache_path.as_posix(),
                    "enhanced_pesq": float(enhanced_pesq),
                    "enhanced_target": normalize_pesq(float(enhanced_pesq)),
                    "noisy_pesq": float(noisy_pesq),
                    "noisy_target": normalize_pesq(float(noisy_pesq)),
                }
            )
            if progress_callback and (
                index == 1 or index == len(rows) or index % 32 == 0
            ):
                progress_callback(
                    f"MetricGAN D current cache {index}/{len(rows)} "
                    f"(epoch={epoch})"
                )
    finally:
        generator.train(generator_was_training)

    _atomic_json(noisy_score_path, noisy_scores)
    payload = {
        "schema_version": 1,
        "epoch": int(epoch),
        "bandwidth": profile.name,
        "sample_rate": profile.sample_rate,
        "storage_dtype": "float16",
        "cache_inputs": False,
        "train_manifest": str(Path(train_manifest)),
        "train_manifest_sha256": manifest_hash(train_manifest),
        "record_count": len(records),
        "records": records,
    }
    _atomic_json(index_path, payload)
    return payload


def _load_record(
    record: dict[str, Any],
    *,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy, clean = _aligned_inputs(
        record["noisy"],
        record["clean"],
        sample_rate=sample_rate,
    )
    enhanced = torch.load(
        record["enhanced"],
        map_location="cpu",
        weights_only=True,
    ).float()
    length = min(noisy.numel(), clean.numel(), enhanced.numel())
    return noisy[:length], clean[:length], enhanced[:length]


def _update_discriminator(
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    candidate: torch.Tensor,
    clean: torch.Tensor,
    target: float,
    *,
    device: str,
    grad_clip: float,
) -> tuple[float, float]:
    candidate_batch = candidate.unsqueeze(0).to(device)
    clean_batch = clean.unsqueeze(0).to(device)
    target_batch = torch.tensor([float(target)], device=device)
    prediction = discriminator.normalized_score(candidate_batch, clean_batch)
    loss = torch.mean((prediction - target_batch) ** 2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(grad_clip))
    optimizer.step()
    return float(loss.detach().cpu()), float(prediction.detach().cpu())


def _current_pass(
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    records: list[dict[str, Any]],
    *,
    device: str,
    sample_rate: int,
    grad_clip: float,
) -> list[float]:
    losses: list[float] = []
    for record in records:
        noisy, clean, enhanced = _load_record(record, sample_rate=sample_rate)
        for candidate, target in (
            (clean, 1.0),
            (enhanced, float(record["enhanced_target"])),
            (noisy, float(record["noisy_target"])),
        ):
            loss, _ = _update_discriminator(
                discriminator,
                optimizer,
                candidate,
                clean,
                target,
                device=device,
                grad_clip=grad_clip,
            )
            losses.append(loss)
    return losses


@torch.inference_mode()
def _calibrate_current(
    discriminator: SpeechBrainMetricDiscriminator,
    records: list[dict[str, Any]],
    *,
    device: str,
    sample_rate: int,
) -> dict[str, float]:
    targets: list[float] = []
    predictions: list[float] = []
    discriminator.eval()
    for record in records:
        noisy, clean, enhanced = _load_record(record, sample_rate=sample_rate)
        for candidate, raw_target in (
            (enhanced, float(record["enhanced_pesq"])),
            (noisy, float(record["noisy_pesq"])),
        ):
            normalized = discriminator.normalized_score(
                candidate.unsqueeze(0).to(device),
                clean.unsqueeze(0).to(device),
            )
            predictions.append(float((5.0 * normalized - 0.5).cpu()))
            targets.append(raw_target)
    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    errors = prediction_array - target_array
    correlation = (
        float(np.corrcoef(target_array, prediction_array)[0, 1])
        if len(target_array) > 1
        and np.std(target_array) > 0
        and np.std(prediction_array) > 0
        else float("nan")
    )
    return {
        "count": float(len(targets)),
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
        "mse": float(np.mean(errors**2)) if len(errors) else float("nan"),
        "pearson": correlation,
    }


def refresh_metricgan_discriminator(
    *,
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    generator: torch.nn.Module,
    train_manifest: str | Path,
    replay_root: str | Path,
    checkpoint_out: str | Path,
    epoch: int,
    device: str,
    max_rows: int,
    history_portion: float = 0.2,
    seed: int = 0,
    grad_clip: float = 5.0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Refresh D in the official current/history/current order, then freeze it."""
    if not 0.0 <= float(history_portion) <= 1.0:
        raise ValueError("history_portion must be in [0, 1].")
    current = build_current_teacher_replay(
        generator=generator,
        train_manifest=train_manifest,
        replay_root=replay_root,
        epoch=epoch,
        device=device,
        bandwidth="wb",
        max_rows=max_rows,
        seed=seed,
        progress_callback=progress_callback,
    )
    current_records = list(current["records"])
    if not current_records:
        raise RuntimeError("No finite current PESQ records were generated.")

    history_records: list[dict[str, Any]] = []
    for index_path in sorted(Path(replay_root).glob("epoch_*/index.json")):
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        history_records.extend(payload.get("records") or [])
    history_count = max(1, round(len(history_records) * float(history_portion)))
    rng = random.Random(int(seed) + int(epoch) * 1009)
    rng.shuffle(history_records)
    history_records = history_records[:history_count]

    discriminator.train()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    if progress_callback:
        progress_callback("MetricGAN D pass 1/3: current clean/enhanced/noisy")
    current_first = _current_pass(
        discriminator,
        optimizer,
        current_records,
        device=device,
        sample_rate=16_000,
        grad_clip=grad_clip,
    )
    if progress_callback:
        progress_callback(
            f"MetricGAN D pass 2/3: historical enhanced "
            f"({len(history_records)} records)"
        )
    historical_losses: list[float] = []
    for record in history_records:
        _, clean, enhanced = _load_record(record, sample_rate=16_000)
        loss, _ = _update_discriminator(
            discriminator,
            optimizer,
            enhanced,
            clean,
            float(record["enhanced_target"]),
            device=device,
            grad_clip=grad_clip,
        )
        historical_losses.append(loss)
    if progress_callback:
        progress_callback("MetricGAN D pass 3/3: current clean/enhanced/noisy")
    current_second = _current_pass(
        discriminator,
        optimizer,
        current_records,
        device=device,
        sample_rate=16_000,
        grad_clip=grad_clip,
    )

    calibration = _calibrate_current(
        discriminator,
        current_records,
        device=device,
        sample_rate=16_000,
    )
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    save_pesq_proxy_checkpoint(checkpoint_out, discriminator)
    summary = {
        "schema_version": 1,
        "epoch": int(epoch),
        "strategy": "speechbrain_current_historical_current",
        "current_record_count": len(current_records),
        "historical_record_count": len(history_records),
        "current_first_mse": float(np.mean(current_first)),
        "historical_mse": float(np.mean(historical_losses)),
        "current_second_mse": float(np.mean(current_second)),
        "calibration": calibration,
        "checkpoint": str(Path(checkpoint_out)),
        "replay_index": str(
            Path(replay_root) / f"epoch_{int(epoch):04d}" / "index.json"
        ),
        "cache_inputs": False,
        "storage_dtype": "float16",
    }
    _atomic_json(
        Path(replay_root) / f"epoch_{int(epoch):04d}" / "refresh_summary.json",
        summary,
    )
    return summary
