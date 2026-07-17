from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .audio import load_audio_num_frames, load_mono_audio, load_mono_audio_window


@dataclass(frozen=True)
class ManifestRow:
    noisy: Path
    clean: Path


def read_pair_manifest(csv_path: str | Path) -> list[ManifestRow]:
    path = Path(csv_path)
    rows: list[ManifestRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "noisy" not in (reader.fieldnames or ()) or "clean" not in (reader.fieldnames or ()):
            raise ValueError(f"Manifest {path} must contain columns: noisy, clean")
        for row in reader:
            rows.append(ManifestRow(noisy=Path(row["noisy"]), clean=Path(row["clean"])))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_pair_manifest(csv_path: str | Path, rows: Sequence[ManifestRow]) -> str:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["noisy", "clean"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"noisy": row.noisy.as_posix(), "clean": row.clean.as_posix()})
    return path.as_posix()


def pair_key(row: ManifestRow) -> str:
    return f"{row.noisy.as_posix().lower()}|{row.clean.as_posix().lower()}"


def unique_manifest_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    dedup: dict[str, ManifestRow] = {}
    for row in rows:
        key = pair_key(row)
        if key not in dedup:
            dedup[key] = row
    return list(dedup.values())


def shuffled_manifest_rows(rows: Sequence[ManifestRow], *, seed: int) -> list[ManifestRow]:
    ordered = list(rows)
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    return ordered


def partition_manifest_rows(rows: Sequence[ManifestRow], *, shard_size: int, seed: int) -> list[list[ManifestRow]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")
    ordered = shuffled_manifest_rows(unique_manifest_rows(rows), seed=seed)
    return [ordered[index:index + shard_size] for index in range(0, len(ordered), shard_size)]


class VoiceBankDemandDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        segment_len: int = 16000 * 2,
        sample_rate: int = 16000,
        rows: Iterable[ManifestRow] | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.rows = list(rows) if rows is not None else read_pair_manifest(csv_path)
        self._frame_cache: dict[str, int] = {}
        if not self.rows:
            raise ValueError(f"No rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _num_frames(self, path: Path) -> int:
        key = path.as_posix()
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        num_frames, _ = load_audio_num_frames(path)
        self._frame_cache[key] = int(num_frames)
        return int(num_frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        noisy_total = self._num_frames(row.noisy)
        clean_total = self._num_frames(row.clean)
        total = min(noisy_total, clean_total)
        segment = self.segment_len
        if total >= segment:
            start = torch.randint(0, total - segment + 1, (1,)).item()
            noisy, _ = load_mono_audio_window(
                row.noisy,
                target_sr=self.sample_rate,
                frame_offset=int(start),
                num_frames=int(segment),
            )
            clean, _ = load_mono_audio_window(
                row.clean,
                target_sr=self.sample_rate,
                frame_offset=int(start),
                num_frames=int(segment),
            )
        else:
            noisy, _ = load_mono_audio(row.noisy, self.sample_rate)
            clean, _ = load_mono_audio(row.clean, self.sample_rate)
            pad = segment - total
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))

        return noisy, clean
