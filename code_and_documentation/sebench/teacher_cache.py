from __future__ import annotations

import csv
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from sebench.audio import (
    TARGET_SAMPLE_RATE,
    load_audio_num_frames,
    load_mono_audio,
    load_mono_audio_window,
    resample_mono_audio,
    stable_hash_text,
)
from sebench.data import ManifestRow, read_pair_manifest
from sebench.stm32_models import (
    STM32_HOP_LENGTH,
    compute_spectral_gating_guidance,
    frontend_defaults_for_sample_rate,
    padded_frame_count,
    waveform_to_erb_mask,
)


@dataclass(frozen=True)
class TeacherCacheRow:
    noisy: Path
    clean: Path
    teacher_wav: Path
    teacher_mask_erb: Path
    guidance_sg: Path | None
    noisy_cache: Path | None = None
    clean_cache: Path | None = None


@dataclass(frozen=True)
class TeacherCacheTarget:
    name: str
    sample_rate: int
    erb_bands: int = 32
    guidance_classic: str = "none"


def _row_key(row: ManifestRow) -> str:
    return stable_hash_text([row.noisy.as_posix(), row.clean.as_posix()])[:16]


class _TeacherCacheBuildDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        *,
        teacher_sample_rate: int,
        target_sample_rate: int,
        row_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.teacher_sample_rate = int(teacher_sample_rate)
        self.target_sample_rate = int(target_sample_rate)
        if row_indices is None:
            self.row_indices = list(range(len(rows)))
        else:
            self.row_indices = [int(index) for index in row_indices]

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = self.row_indices[index]
        row = self.rows[row_index]
        noisy_teacher, _ = load_mono_audio(row.noisy, self.teacher_sample_rate)
        clean_teacher, _ = load_mono_audio(row.clean, self.teacher_sample_rate)
        return {
            "index": int(row_index),
            "row": row,
            "row_key": _row_key(row),
            "noisy_teacher": noisy_teacher,
            "clean_teacher": clean_teacher,
        }


def _teacher_cache_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _atomic_torch_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(tensor, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _cache_entry_paths(*, row: ManifestRow, row_key: str, dirs: dict[str, Path], guidance_required: bool) -> dict[str, Path | None]:
    guidance_path = dirs["guidance_sg"] / f"{row_key}.pt" if guidance_required else None
    return {
        "noisy": row.noisy,
        "clean": row.clean,
        "teacher_wav": dirs["teacher_wav"] / f"{row_key}.pt",
        "teacher_mask_erb": dirs["teacher_mask_erb"] / f"{row_key}.pt",
        "guidance_sg": guidance_path,
        "noisy_cache": dirs["noisy_cache"] / f"{row_key}.pt",
        "clean_cache": dirs["clean_cache"] / f"{row_key}.pt",
    }


def _unlink_cache_paths(paths: dict[str, Path | None]) -> None:
    for key in ("guidance_sg", "noisy_cache", "clean_cache", "teacher_wav", "teacher_mask_erb"):
        path = paths.get(key)
        if path is None:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _load_existing_cache_payload(
    *,
    row: ManifestRow,
    row_key: str,
    dirs: dict[str, Path],
    guidance_required: bool,
    validate_existing: bool,
) -> dict[str, str] | None:
    payload = _cache_entry_paths(row=row, row_key=row_key, dirs=dirs, guidance_required=guidance_required)
    required_paths = [
        payload["teacher_wav"],
        payload["teacher_mask_erb"],
        payload["noisy_cache"],
        payload["clean_cache"],
    ]
    if guidance_required:
        required_paths.append(payload["guidance_sg"])
    for path in required_paths:
        if path is None or not path.is_file() or path.stat().st_size <= 0:
            _unlink_cache_paths(payload)
            return None
    if validate_existing:
        try:
            for path in required_paths:
                loaded = torch.load(path, map_location="cpu")
                if not isinstance(loaded, torch.Tensor):
                    raise TypeError(f"Expected tensor payload in {path}")
                del loaded
        except Exception:
            _unlink_cache_paths(payload)
            return None
    return {
        "noisy": row.noisy.as_posix(),
        "clean": row.clean.as_posix(),
        "teacher_wav": payload["teacher_wav"].as_posix(),
        "teacher_mask_erb": payload["teacher_mask_erb"].as_posix(),
        "guidance_sg": payload["guidance_sg"].as_posix() if payload["guidance_sg"] is not None else "",
        "noisy_cache": payload["noisy_cache"].as_posix(),
        "clean_cache": payload["clean_cache"].as_posix(),
    }


def _save_teacher_cache_entry(
    *,
    index: int,
    row: ManifestRow,
    row_key: str,
    teacher_wav_tensor: torch.Tensor,
    teacher_mask_tensor: torch.Tensor,
    guidance_tensor: torch.Tensor | None,
    noisy_cache_tensor: torch.Tensor | None,
    clean_cache_tensor: torch.Tensor | None,
    wav_dir: Path,
    mask_dir: Path,
    guidance_dir: Path,
    noisy_dir: Path | None,
    clean_dir: Path | None,
) -> tuple[int, dict[str, str]]:
    guidance_path: Path | None = None
    if guidance_tensor is not None:
        guidance_path = guidance_dir / f"{row_key}.pt"
        _atomic_torch_save(guidance_tensor, guidance_path)

    noisy_cache_path: Path | None = None
    if noisy_cache_tensor is not None and noisy_dir is not None:
        noisy_cache_path = noisy_dir / f"{row_key}.pt"
        _atomic_torch_save(noisy_cache_tensor, noisy_cache_path)

    clean_cache_path: Path | None = None
    if clean_cache_tensor is not None and clean_dir is not None:
        clean_cache_path = clean_dir / f"{row_key}.pt"
        _atomic_torch_save(clean_cache_tensor, clean_cache_path)

    teacher_wav_path = wav_dir / f"{row_key}.pt"
    teacher_mask_path = mask_dir / f"{row_key}.pt"
    _atomic_torch_save(teacher_wav_tensor, teacher_wav_path)
    _atomic_torch_save(teacher_mask_tensor, teacher_mask_path)
    return (
        int(index),
        {
            "noisy": row.noisy.as_posix(),
            "clean": row.clean.as_posix(),
            "teacher_wav": teacher_wav_path.as_posix(),
            "teacher_mask_erb": teacher_mask_path.as_posix(),
            "guidance_sg": guidance_path.as_posix() if guidance_path is not None else "",
            "noisy_cache": noisy_cache_path.as_posix() if noisy_cache_path is not None else "",
            "clean_cache": clean_cache_path.as_posix() if clean_cache_path is not None else "",
        },
    )


def _save_teacher_cache_target_entry(
    *,
    target_name: str,
    index: int,
    row: ManifestRow,
    row_key: str,
    teacher_wav_tensor: torch.Tensor,
    teacher_mask_tensor: torch.Tensor,
    guidance_tensor: torch.Tensor | None,
    noisy_cache_tensor: torch.Tensor | None,
    clean_cache_tensor: torch.Tensor | None,
    wav_dir: Path,
    mask_dir: Path,
    guidance_dir: Path,
    noisy_dir: Path | None,
    clean_dir: Path | None,
) -> tuple[str, int, dict[str, str]]:
    saved_index, payload = _save_teacher_cache_entry(
        index=index,
        row=row,
        row_key=row_key,
        teacher_wav_tensor=teacher_wav_tensor,
        teacher_mask_tensor=teacher_mask_tensor,
        guidance_tensor=guidance_tensor,
        noisy_cache_tensor=noisy_cache_tensor,
        clean_cache_tensor=clean_cache_tensor,
        wav_dir=wav_dir,
        mask_dir=mask_dir,
        guidance_dir=guidance_dir,
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
    )
    return str(target_name), int(saved_index), payload


def _target_dirs(out_root: Path, target_name: str) -> dict[str, Path]:
    root = out_root if target_name == "default" else out_root / target_name
    paths = {
        "root": root,
        "teacher_wav": root / "teacher_wav",
        "teacher_mask_erb": root / "teacher_mask_erb",
        "guidance_sg": root / "guidance_sg",
        "noisy_cache": root / "noisy_cache",
        "clean_cache": root / "clean_cache",
    }
    for key, path in paths.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_teacher_cache_rows(csv_path: Path, payload_rows: list[dict[str, str] | None]) -> str:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("noisy", "clean", "teacher_wav", "teacher_mask_erb", "guidance_sg", "noisy_cache", "clean_cache"),
        )
        writer.writeheader()
        writer.writerows([row for row in payload_rows if row is not None])
    return csv_path.as_posix()


def build_multi_target_teacher_cache(
    manifest_path: str | Path,
    teacher_model: torch.nn.Module,
    *,
    out_dir: str | Path,
    device: str,
    teacher_sample_rate: int = TARGET_SAMPLE_RATE,
    targets: list[TeacherCacheTarget],
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    write_workers: int = 0,
    resume: bool = False,
    validate_existing: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    if not targets:
        raise ValueError("targets must not be empty")

    rows = read_pair_manifest(manifest_path)
    out_root = Path(out_dir)
    stem = Path(manifest_path).stem
    target_specs = [
        TeacherCacheTarget(
            name=str(target.name),
            sample_rate=int(target.sample_rate),
            erb_bands=int(target.erb_bands),
            guidance_classic=str(target.guidance_classic),
        )
        for target in targets
    ]
    target_frontends = {
        target.name: frontend_defaults_for_sample_rate(target.sample_rate)
        for target in target_specs
    }
    target_dirs = {target.name: _target_dirs(out_root, target.name) for target in target_specs}
    manifest_outputs = {
        target.name: (out_root / f"{stem}_teacher_cache_{target.name}.csv")
        for target in target_specs
    }

    payload_rows_by_target: dict[str, list[dict[str, str] | None]] = {
        target.name: [None] * len(rows)
        for target in target_specs
    }
    processed = 0
    pending_row_indices: set[int] = set(range(len(rows)))
    if resume:
        pending_row_indices = set()
        checked_entries = 0
        total_entries = len(rows) * len(target_specs)
        for row_index, row in enumerate(rows):
            row_key = _row_key(row)
            row_pending = False
            for target in target_specs:
                payload = _load_existing_cache_payload(
                    row=row,
                    row_key=row_key,
                    dirs=target_dirs[target.name],
                    guidance_required=(target.guidance_classic == "spectral_gating"),
                    validate_existing=validate_existing,
                )
                checked_entries += 1
                if payload is None:
                    row_pending = True
                else:
                    payload_rows_by_target[target.name][row_index] = payload
                    processed += 1
                if progress_callback is not None and (checked_entries == 1 or checked_entries == total_entries or checked_entries % 100 == 0):
                    progress_callback(f"resume validation {checked_entries}/{total_entries} from {Path(manifest_path).name}")
            if row_pending:
                pending_row_indices.add(row_index)
        if progress_callback is not None and processed > 0:
            progress_callback(f"resume found {processed}/{len(rows) * len(target_specs)} valid entries from {Path(manifest_path).name}")

    teacher_model.eval()
    dataset = _TeacherCacheBuildDataset(
        rows,
        teacher_sample_rate=teacher_sample_rate,
        target_sample_rate=teacher_sample_rate,
        row_indices=sorted(pending_row_indices),
    )
    worker_count = max(0, int(num_workers))
    loader_kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory and str(device).startswith("cuda")),
        "collate_fn": _teacher_cache_collate,
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(dataset, **loader_kwargs)
    write_worker_count = max(0, int(write_workers))
    pending_writes: set[Any] = set()
    max_pending_writes = max(1, write_worker_count * max(1, int(loader_kwargs["batch_size"])) * max(1, len(target_specs)))
    write_pool = ThreadPoolExecutor(max_workers=write_worker_count) if write_worker_count > 0 else None

    def _drain_pending(wait_for_one: bool) -> None:
        nonlocal pending_writes, processed
        if not pending_writes:
            return
        if wait_for_one:
            done, not_done = wait(pending_writes, return_when=FIRST_COMPLETED)
        else:
            done, not_done = wait(pending_writes, timeout=0, return_when=FIRST_COMPLETED)
            if not done:
                return
        pending_writes = set(not_done)
        for future in done:
            target_name, index, payload = future.result()
            payload_rows_by_target[target_name][index] = payload
            processed += 1
            if progress_callback is not None:
                total_entries = len(rows) * len(target_specs)
                if processed == total_entries or processed == 1 or processed % 100 == 0:
                    progress_callback(f"teacher cache {processed}/{total_entries} from {Path(manifest_path).name}")

    try:
        with torch.no_grad():
            for batch_items in loader:
                grouped_items: dict[tuple[int, int], list[dict[str, Any]]] = {}
                for item in batch_items:
                    group_key = (int(item["noisy_teacher"].shape[-1]), int(item["clean_teacher"].shape[-1]))
                    grouped_items.setdefault(group_key, []).append(item)

                batch_outputs: dict[tuple[str, int], dict[str, torch.Tensor | None]] = {}
                for group in grouped_items.values():
                    noisy_teacher_batch = torch.stack([item["noisy_teacher"] for item in group], dim=0).to(device, non_blocking=True)
                    clean_teacher_batch = torch.stack([item["clean_teacher"] for item in group], dim=0)
                    teacher_wav_batch = teacher_model.denoise_single(noisy_teacher_batch)

                    for target in target_specs:
                        n_fft, hop_length, win_length = target_frontends[target.name]
                        if target.sample_rate != teacher_sample_rate:
                            noisy_target_batch = resample_mono_audio(noisy_teacher_batch, teacher_sample_rate, target.sample_rate)
                            clean_target_batch = resample_mono_audio(clean_teacher_batch, teacher_sample_rate, target.sample_rate)
                            teacher_wav_target = resample_mono_audio(teacher_wav_batch, teacher_sample_rate, target.sample_rate)
                        else:
                            noisy_target_batch = noisy_teacher_batch
                            clean_target_batch = clean_teacher_batch
                            teacher_wav_target = teacher_wav_batch

                        noisy_target_cpu = noisy_target_batch.cpu()
                        clean_target_cpu = clean_target_batch.cpu()
                        target_length = int(noisy_target_batch.shape[-1])
                        if int(teacher_wav_target.shape[-1]) > target_length:
                            teacher_wav_target = teacher_wav_target[..., :target_length]
                        elif int(teacher_wav_target.shape[-1]) < target_length:
                            teacher_wav_target = torch.nn.functional.pad(
                                teacher_wav_target,
                                (0, target_length - int(teacher_wav_target.shape[-1])),
                            )
                        teacher_mask_batch = waveform_to_erb_mask(
                            noisy_target_batch,
                            teacher_wav_target,
                            erb_bands=target.erb_bands,
                            sample_rate=target.sample_rate,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            win_length=win_length,
                        ).cpu()
                        guidance_batch: torch.Tensor | None = None
                        if target.guidance_classic == "spectral_gating":
                            guidance_batch = compute_spectral_gating_guidance(
                                noisy_target_batch,
                                erb_bands=target.erb_bands,
                                sample_rate=target.sample_rate,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                            ).cpu()
                        teacher_wav_cpu = teacher_wav_target.cpu()
                        for offset, item in enumerate(group):
                            batch_outputs[(target.name, int(item["index"]))] = {
                                "teacher_wav": teacher_wav_cpu[offset].clone(),
                                "teacher_mask": teacher_mask_batch[offset].clone(),
                                "guidance": None if guidance_batch is None else guidance_batch[offset].clone(),
                                "noisy_cache": noisy_target_cpu[offset].clone(),
                                "clean_cache": clean_target_cpu[offset].clone(),
                            }

                for item in batch_items:
                    row = item["row"]
                    row_key = str(item["row_key"])
                    row_index = int(item["index"])
                    for target in target_specs:
                        if payload_rows_by_target[target.name][row_index] is not None:
                            continue
                        tensors = batch_outputs[(target.name, row_index)]
                        dirs = target_dirs[target.name]
                        if write_pool is None:
                            _, payload = _save_teacher_cache_entry(
                                index=row_index,
                                row=row,
                                row_key=row_key,
                                teacher_wav_tensor=tensors["teacher_wav"],
                                teacher_mask_tensor=tensors["teacher_mask"],
                                guidance_tensor=tensors["guidance"],
                                noisy_cache_tensor=tensors["noisy_cache"],
                                clean_cache_tensor=tensors["clean_cache"],
                                wav_dir=dirs["teacher_wav"],
                                mask_dir=dirs["teacher_mask_erb"],
                                guidance_dir=dirs["guidance_sg"],
                                noisy_dir=dirs["noisy_cache"],
                                clean_dir=dirs["clean_cache"],
                            )
                            payload_rows_by_target[target.name][row_index] = payload
                            processed += 1
                            if progress_callback is not None:
                                total_entries = len(rows) * len(target_specs)
                                if processed == total_entries or processed == 1 or processed % 100 == 0:
                                    progress_callback(f"teacher cache {processed}/{total_entries} from {Path(manifest_path).name}")
                        else:
                            pending_writes.add(
                                write_pool.submit(
                                    _save_teacher_cache_target_entry,
                                    target_name=target.name,
                                    index=row_index,
                                    row=row,
                                    row_key=row_key,
                                    teacher_wav_tensor=tensors["teacher_wav"],
                                    teacher_mask_tensor=tensors["teacher_mask"],
                                    guidance_tensor=tensors["guidance"],
                                    noisy_cache_tensor=tensors["noisy_cache"],
                                    clean_cache_tensor=tensors["clean_cache"],
                                    wav_dir=dirs["teacher_wav"],
                                    mask_dir=dirs["teacher_mask_erb"],
                                    guidance_dir=dirs["guidance_sg"],
                                    noisy_dir=dirs["noisy_cache"],
                                    clean_dir=dirs["clean_cache"],
                                )
                            )
                if write_pool is not None:
                    _drain_pending(wait_for_one=False)
                    while len(pending_writes) >= max_pending_writes:
                        _drain_pending(wait_for_one=True)
        while pending_writes:
            _drain_pending(wait_for_one=True)
    finally:
        if write_pool is not None:
            write_pool.shutdown(wait=True)

    return {
        target.name: _write_teacher_cache_rows(manifest_outputs[target.name], payload_rows_by_target[target.name])
        for target in target_specs
    }


def build_teacher_cache(
    manifest_path: str | Path,
    teacher_model: torch.nn.Module,
    *,
    out_dir: str | Path,
    device: str,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    teacher_sample_rate: int = TARGET_SAMPLE_RATE,
    erb_bands: int = 32,
    guidance_classic: str = "none",
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    write_workers: int = 0,
    resume: bool = False,
    validate_existing: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    manifests = build_multi_target_teacher_cache(
        manifest_path,
        teacher_model,
        out_dir=out_dir,
        device=device,
        teacher_sample_rate=teacher_sample_rate,
        targets=[
            TeacherCacheTarget(
                name="default",
                sample_rate=int(target_sample_rate),
                erb_bands=int(erb_bands),
                guidance_classic=str(guidance_classic),
            )
        ],
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        write_workers=write_workers,
        resume=resume,
        validate_existing=validate_existing,
        progress_callback=progress_callback,
    )
    return manifests["default"]


def read_teacher_cache_manifest(csv_path: str | Path) -> list[TeacherCacheRow]:
    path = Path(csv_path)
    rows: list[TeacherCacheRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                TeacherCacheRow(
                    noisy=Path(row["noisy"]),
                    clean=Path(row["clean"]),
                    teacher_wav=Path(row["teacher_wav"]),
                    teacher_mask_erb=Path(row["teacher_mask_erb"]),
                    guidance_sg=Path(row["guidance_sg"]) if row.get("guidance_sg") else None,
                    noisy_cache=Path(row["noisy_cache"]) if row.get("noisy_cache") else None,
                    clean_cache=Path(row["clean_cache"]) if row.get("clean_cache") else None,
                )
            )
    if not rows:
        raise ValueError(f"Teacher cache manifest is empty: {path}")
    return rows


def write_teacher_cache_manifest(csv_path: str | Path, rows: list[TeacherCacheRow]) -> str:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("noisy", "clean", "teacher_wav", "teacher_mask_erb", "guidance_sg", "noisy_cache", "clean_cache"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "noisy": row.noisy.as_posix(),
                    "clean": row.clean.as_posix(),
                    "teacher_wav": row.teacher_wav.as_posix(),
                    "teacher_mask_erb": row.teacher_mask_erb.as_posix(),
                    "guidance_sg": row.guidance_sg.as_posix() if row.guidance_sg is not None else "",
                    "noisy_cache": row.noisy_cache.as_posix() if row.noisy_cache is not None else "",
                    "clean_cache": row.clean_cache.as_posix() if row.clean_cache is not None else "",
                }
            )
    return path.as_posix()


def filter_teacher_cache_manifest(
    csv_path: str | Path,
    *,
    allowed_pairs: set[tuple[str, str]],
    out_path: str | Path,
) -> str:
    filtered = [
        row
        for row in read_teacher_cache_manifest(csv_path)
        if (row.noisy.as_posix().lower(), row.clean.as_posix().lower()) in allowed_pairs
    ]
    if not filtered:
        raise ValueError(f"No rows matched teacher cache filter for {csv_path}")
    return write_teacher_cache_manifest(out_path, filtered)


class TeacherCacheDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        *,
        segment_len: int,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_fft: int = 512,
        hop_length: int = STM32_HOP_LENGTH,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.rows = read_teacher_cache_manifest(csv_path)
        self._frame_cache: dict[str, int] = {}

    @staticmethod
    def _load_cached_waveform(path: Path | None, fallback: Path, sample_rate: int) -> torch.Tensor:
        if path is not None:
            return torch.load(path, map_location="cpu").float()
        wav, _ = load_mono_audio(fallback, sample_rate)
        return wav

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        teacher_wav = torch.load(row.teacher_wav, map_location="cpu").float()
        teacher_mask = torch.load(row.teacher_mask_erb, map_location="cpu").float()
        guidance = torch.load(row.guidance_sg, map_location="cpu").float() if row.guidance_sg is not None else None
        noisy_full = self._load_cached_waveform(row.noisy_cache, row.noisy, self.sample_rate)
        clean_full = self._load_cached_waveform(row.clean_cache, row.clean, self.sample_rate)
        total = min(int(noisy_full.shape[-1]), int(clean_full.shape[-1]), int(teacher_wav.shape[-1]))
        segment = self.segment_len
        if total >= segment:
            max_start = total - segment
            aligned_max = max_start // self.hop_length
            frame_start = torch.randint(0, aligned_max + 1, (1,)).item() if aligned_max > 0 else 0
            start = frame_start * self.hop_length
            noisy = noisy_full[start:start + segment]
            clean = clean_full[start:start + segment]
            teacher_wav = teacher_wav[start:start + segment]
        else:
            noisy = noisy_full[:total]
            clean = clean_full[:total]
            teacher_wav = teacher_wav[:total]
            pad = segment - total
            start = 0
            frame_start = 0
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))
            teacher_wav = torch.nn.functional.pad(teacher_wav, (0, pad))

        if noisy.shape[-1] < segment:
            noisy = torch.nn.functional.pad(noisy, (0, segment - int(noisy.shape[-1])))
        if clean.shape[-1] < segment:
            clean = torch.nn.functional.pad(clean, (0, segment - int(clean.shape[-1])))
        if teacher_wav.shape[-1] < segment:
            teacher_wav = torch.nn.functional.pad(teacher_wav, (0, segment - int(teacher_wav.shape[-1])))

        segment_frames = padded_frame_count(segment, n_fft=self.n_fft, hop_length=self.hop_length)
        full_teacher_mask = teacher_mask
        teacher_frame_start = min(int(frame_start), max(0, int(full_teacher_mask.shape[-1]) - 1))
        teacher_mask = full_teacher_mask[:, teacher_frame_start:teacher_frame_start + segment_frames]
        if teacher_mask.shape[-1] == 0:
            teacher_mask = full_teacher_mask[:, -1:] if full_teacher_mask.shape[-1] > 0 else torch.ones((32, 1), dtype=torch.float32)
        if teacher_mask.shape[-1] < segment_frames:
            teacher_mask = torch.nn.functional.pad(teacher_mask, (0, segment_frames - teacher_mask.shape[-1]), mode="replicate")
        if guidance is not None:
            full_guidance = guidance
            guidance_frame_start = min(int(frame_start), max(0, int(full_guidance.shape[-1]) - 1))
            guidance = full_guidance[:, guidance_frame_start:guidance_frame_start + segment_frames]
            if guidance.shape[-1] == 0:
                guidance = full_guidance[:, -1:] if full_guidance.shape[-1] > 0 else torch.zeros((32, 1), dtype=torch.float32)
            if guidance.shape[-1] < segment_frames:
                guidance = torch.nn.functional.pad(guidance, (0, segment_frames - guidance.shape[-1]), mode="replicate")

        sample = {
            "noisy": noisy.contiguous().clone(),
            "clean": clean.contiguous().clone(),
            "teacher_wav": teacher_wav.contiguous().clone(),
            "teacher_mask_erb": teacher_mask.contiguous().clone(),
        }
        if guidance is not None:
            sample["guidance_sg"] = guidance.contiguous().clone()
        return sample
