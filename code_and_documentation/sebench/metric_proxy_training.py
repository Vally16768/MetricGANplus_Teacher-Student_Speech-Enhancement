"""Build and validate bandwidth-specific PESQ proxy checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from metrics.pesq import pesq_score
from sebench.audio import load_mono_audio, manifest_hash, resample_mono_audio
from sebench.bandwidth import resolve_bandwidth
from sebench.checkpoints import load_model_from_checkpoint
from sebench.data import read_pair_manifest
from sebench.losses import PESQProxyRegressor, save_pesq_proxy_checkpoint


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for index, count in enumerate(counts):
        if count > 1:
            positions = np.flatnonzero(inverse == index)
            ranks[positions] = float(np.mean(ranks[positions]))
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


class ProxyRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], sample_rate: int) -> None:
        self.records = records
        self.sample_rate = int(sample_rate)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        noisy, _ = load_mono_audio(record["noisy"], self.sample_rate)
        clean, _ = load_mono_audio(record["clean"], self.sample_rate)
        candidate = torch.load(
            record["candidate"],
            map_location="cpu",
            weights_only=True,
        ).float()
        length = min(noisy.numel(), clean.numel(), candidate.numel())
        return {
            "noisy": noisy[:length],
            "clean": clean[:length],
            "candidate": candidate[:length],
            "target": torch.tensor(float(record["pesq"]), dtype=torch.float32),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_length = max(item["noisy"].numel() for item in batch)

    def pad(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(value, (0, max_length - value.numel()))

    return {
        "noisy": torch.stack([pad(item["noisy"]) for item in batch]),
        "clean": torch.stack([pad(item["clean"]) for item in batch]),
        "candidate": torch.stack([pad(item["candidate"]) for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }


def _candidate_token(split: str, row_index: int, source: str) -> str:
    raw = f"{split}|{row_index}|{source}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


@torch.inference_mode()
def build_proxy_records(
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output_dir: str | Path,
    bandwidth: str,
    candidate_teacher_checkpoint: str | Path | None,
    teacher_sample_rate: int = 16_000,
    max_train_rows: int = 256,
    max_validation_rows: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> dict[str, Any]:
    profile = resolve_bandwidth(bandwidth)
    root = Path(output_dir)
    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))

    teacher = None
    if candidate_teacher_checkpoint:
        teacher, _ = load_model_from_checkpoint(
            candidate_teacher_checkpoint,
            device=device,
        )
        teacher.eval()

    records: list[dict[str, Any]] = []
    split_specs = (
        ("train", Path(train_manifest), int(max_train_rows)),
        ("validation", Path(validation_manifest), int(max_validation_rows)),
    )
    for split, manifest_path, max_rows in split_specs:
        rows = list(read_pair_manifest(manifest_path))
        rng.shuffle(rows)
        rows = rows[:max_rows]
        for row_index, row in enumerate(rows):
            noisy, _ = load_mono_audio(row.noisy, profile.sample_rate)
            clean, _ = load_mono_audio(row.clean, profile.sample_rate)
            length = min(noisy.numel(), clean.numel())
            noisy = noisy[:length]
            clean = clean[:length]
            candidates: list[tuple[str, torch.Tensor]] = [
                ("noisy", noisy),
                ("blend25", 0.75 * noisy + 0.25 * clean),
                ("blend50", 0.50 * noisy + 0.50 * clean),
                ("blend75", 0.25 * noisy + 0.75 * clean),
                ("clean", clean),
            ]
            if teacher is not None:
                teacher_noisy, _ = load_mono_audio(row.noisy, teacher_sample_rate)
                teacher_candidate = teacher.denoise_single(
                    teacher_noisy.unsqueeze(0).to(device)
                ).squeeze(0).cpu()
                if teacher_sample_rate != profile.sample_rate:
                    teacher_candidate = resample_mono_audio(
                        teacher_candidate,
                        teacher_sample_rate,
                        profile.sample_rate,
                    )
                teacher_candidate = teacher_candidate[:length]
                candidates.append(("teacher", teacher_candidate))

            for source, candidate in candidates:
                aligned = min(length, candidate.numel())
                score = pesq_score(
                    clean[:aligned].numpy(),
                    candidate[:aligned].numpy(),
                    profile.sample_rate,
                    bandwidth=profile.name,
                )
                if not math.isfinite(score):
                    continue
                token = _candidate_token(split, row_index, source)
                candidate_path = candidate_dir / f"{token}.pt"
                torch.save(candidate[:aligned].contiguous(), candidate_path)
                records.append(
                    {
                        "split": split,
                        "source": source,
                        "noisy": row.noisy.as_posix(),
                        "clean": row.clean.as_posix(),
                        "candidate": candidate_path.as_posix(),
                        "pesq": float(score),
                    }
                )

    payload = {
        "schema_version": 1,
        "bandwidth": profile.name,
        "sample_rate": profile.sample_rate,
        "pesq_mode": profile.pesq_mode,
        "train_manifest": str(Path(train_manifest)),
        "train_manifest_sha256": manifest_hash(train_manifest),
        "validation_manifest": str(Path(validation_manifest)),
        "validation_manifest_sha256": manifest_hash(validation_manifest),
        "candidate_teacher_checkpoint": str(candidate_teacher_checkpoint or ""),
        "records": records,
    }
    records_path = root / "records.json"
    records_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["records_path"] = records_path.as_posix()
    return payload


def _evaluate_proxy(
    model: PESQProxyRegressor,
    loader: DataLoader,
    *,
    device: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    targets: list[float] = []
    predictions: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predicted = model(
                batch["noisy"].to(device),
                batch["candidate"].to(device),
                batch["clean"].to(device),
            )
            predictions.extend(float(value) for value in predicted.cpu())
            targets.extend(float(value) for value in batch["target"])
    target_np = np.asarray(targets, dtype=np.float64)
    prediction_np = np.asarray(predictions, dtype=np.float64)
    errors = prediction_np - target_np
    metrics = {
        "count": float(len(targets)),
        "mse": float(np.mean(errors**2)) if len(errors) else float("nan"),
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
        "pearson": _correlation(target_np, prediction_np),
        "spearman": _correlation(_rankdata(target_np), _rankdata(prediction_np)),
        "target_min": float(np.min(target_np)) if len(target_np) else float("nan"),
        "target_max": float(np.max(target_np)) if len(target_np) else float("nan"),
        "prediction_min": float(np.min(prediction_np)) if len(prediction_np) else float("nan"),
        "prediction_max": float(np.max(prediction_np)) if len(prediction_np) else float("nan"),
    }
    rows = [
        {"target": float(target), "prediction": float(prediction)}
        for target, prediction in zip(targets, predictions)
    ]
    return metrics, rows


def train_metric_proxy(
    records_payload: dict[str, Any],
    *,
    output_dir: str | Path,
    device: str,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1e-3,
    hidden_channels: int = 32,
    projection_dim: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    profile = resolve_bandwidth(
        str(records_payload["bandwidth"]),
        sample_rate=int(records_payload["sample_rate"]),
    )
    records = list(records_payload["records"])
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [
        record for record in records if record["split"] == "validation"
    ]
    if not train_records or not validation_records:
        raise ValueError("Metric proxy requires non-empty train and validation records.")

    train_loader = DataLoader(
        ProxyRecordDataset(train_records, profile.sample_rate),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        ProxyRecordDataset(validation_records, profile.sample_rate),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    model = PESQProxyRegressor(
        sample_rate=profile.sample_rate,
        n_fft=profile.n_fft,
        hop_length=profile.hop_length,
        win_length=profile.win_length,
        hidden_channels=int(hidden_channels),
        projection_dim=int(projection_dim),
        bandwidth=profile.name,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    history: list[dict[str, float]] = []
    best_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            prediction = model(
                batch["noisy"].to(device),
                batch["candidate"].to(device),
                batch["clean"].to(device),
            )
            loss = loss_fn(prediction, batch["target"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_metrics, _ = _evaluate_proxy(
            model,
            validation_loader,
            device=device,
        )
        row = {
            "epoch": float(epoch),
            "train_mse": float(np.mean(losses)),
            "validation_mse": validation_metrics["mse"],
            "validation_mae": validation_metrics["mae"],
            "validation_pearson": validation_metrics["pearson"],
            "validation_spearman": validation_metrics["spearman"],
        }
        history.append(row)
        if validation_metrics["mse"] < best_mse:
            best_mse = validation_metrics["mse"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Metric proxy training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics, calibration = _evaluate_proxy(
        model,
        validation_loader,
        device=device,
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "metric_proxy.pt"
    save_pesq_proxy_checkpoint(checkpoint_path, model)
    history_path = root / "history.json"
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_path = root / "calibration.csv"
    with calibration_path.open("w", encoding="utf-8") as handle:
        handle.write("target,prediction\n")
        for row in calibration:
            handle.write(f"{row['target']},{row['prediction']}\n")

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        [row["target"] for row in calibration],
        [row["prediction"] for row in calibration],
        s=12,
        alpha=0.7,
    )
    axis.plot([-0.5, 4.5], [-0.5, 4.5], linestyle="--", color="black")
    axis.set_xlabel("True PESQ")
    axis.set_ylabel("Predicted PESQ")
    axis.set_title(f"{profile.name.upper()} PESQ proxy calibration")
    figure.tight_layout()
    calibration_plot = root / "calibration.png"
    figure.savefig(calibration_plot, dpi=160)
    plt.close(figure)

    summary = {
        "schema_version": 1,
        "bandwidth": profile.name,
        "sample_rate": profile.sample_rate,
        "pesq_mode": profile.pesq_mode,
        "checkpoint": checkpoint_path.as_posix(),
        "records_path": records_payload.get("records_path"),
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "validation": validation_metrics,
        "history": history_path.as_posix(),
        "calibration_csv": calibration_path.as_posix(),
        "calibration_plot": calibration_plot.as_posix(),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = summary_path.as_posix()
    return summary
