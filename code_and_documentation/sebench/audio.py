from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000
_AUDIO_INFO_CACHE: dict[str, tuple[int, int]] = {}


def resample_mono_audio(wav: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return wav
    needs_unsqueeze = wav.ndim == 1
    wav_2d = wav.unsqueeze(0) if needs_unsqueeze else wav
    resampled = torchaudio.functional.resample(wav_2d, source_sr, target_sr)
    return resampled.squeeze(0) if needs_unsqueeze else resampled


def load_mono_audio(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0), target_sr


def load_audio_num_frames(path: str | Path) -> tuple[int, int]:
    key = Path(path).as_posix()
    cached = _AUDIO_INFO_CACHE.get(key)
    if cached is not None:
        return cached
    info_obj = None
    if hasattr(torchaudio, "info"):
        try:
            info_obj = torchaudio.info(str(path))
        except Exception:
            info_obj = None
    if info_obj is None:
        try:
            from torchaudio.backend import sox_io_backend  # type: ignore

            info_obj = sox_io_backend.info(str(path))
        except Exception:
            info_obj = None
    if info_obj is not None:
        value = (int(getattr(info_obj, "num_frames", 0) or 0), int(getattr(info_obj, "sample_rate", TARGET_SAMPLE_RATE)))
    else:
        wav, sr = torchaudio.load(str(path))
        value = (int(wav.shape[-1]), int(sr))
    _AUDIO_INFO_CACHE[key] = value
    return value


def load_mono_audio_window(
    path: str | Path,
    *,
    target_sr: int = TARGET_SAMPLE_RATE,
    frame_offset: int = 0,
    num_frames: int | None = None,
) -> tuple[torch.Tensor, int]:
    offset = max(int(frame_offset), 0)
    frames = None if num_frames is None else max(int(num_frames), 0)
    if frames == 0:
        return torch.zeros(0, dtype=torch.float32), target_sr

    _, source_sr = load_audio_num_frames(path)

    # Fast path: decode only the requested window at native sample rate.
    if source_sr == target_sr:
        kwargs: dict[str, int] = {"frame_offset": offset}
        if frames is not None:
            kwargs["num_frames"] = frames
        wav, sr = torchaudio.load(str(path), **kwargs)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0), int(sr)

    # Fallback path for sample-rate mismatch: decode full waveform, resample, then crop.
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, int(sr), target_sr)
    if frames is None:
        return wav.squeeze(0), target_sr
    end = offset + frames
    return wav.squeeze(0)[offset:end], target_sr


def save_mono_audio(path: str | Path, wav: torch.Tensor, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(out_path.as_posix(), wav, sample_rate)


def crop_or_pad(wav: torch.Tensor, length: int, start: int | None = None) -> torch.Tensor:
    if wav.shape[-1] >= length:
        if start is None:
            start = 0
        return wav[..., start:start + length]
    pad = length - wav.shape[-1]
    return torch.nn.functional.pad(wav, (0, pad))


def loop_to_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.shape[-1] == 0:
        raise ValueError("Cannot loop an empty waveform.")
    if wav.shape[-1] >= length:
        return wav[..., :length]
    repeats = (length + wav.shape[-1] - 1) // wav.shape[-1]
    tiled = wav.repeat(repeats)
    return tiled[:length]


def tensor_to_numpy_mono(wav: torch.Tensor) -> np.ndarray:
    return wav.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False)


def stable_hash_text(values: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_hash(csv_path: str | Path) -> str:
    path = Path(csv_path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        values = [f"{row['noisy']}|{row['clean']}" for row in reader]
    return stable_hash_text(values)
