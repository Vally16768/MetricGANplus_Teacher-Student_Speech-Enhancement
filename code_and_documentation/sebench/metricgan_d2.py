"""Fixed-support preparation and diagnostics for the T2 MetricGAN D2 critic."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics.pesq import pesq_score
from sebench.audio import load_mono_audio
from sebench.metricgan_alternating import normalize_pesq


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token(noisy: str, clean: str) -> str:
    return hashlib.sha256(f"{noisy}|{clean}".encode("utf-8")).hexdigest()[:20]


def _speaker_id(row: dict[str, str]) -> str | None:
    for key in ("speaker", "speaker_id", "talker", "talker_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    for key in ("clean", "noisy"):
        match = re.search(r"(?:^|[/_])(p\d{3})(?:[_/.]|$)", str(row.get(key) or ""))
        if match:
            return match.group(1)
    return None


def _load_aligned(
    noisy_path: str | Path,
    clean_path: str | Path,
    teacher_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy, _ = load_mono_audio(noisy_path, 16_000)
    clean, _ = load_mono_audio(clean_path, 16_000)
    teacher = torch.load(
        Path(teacher_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(teacher, torch.Tensor):
        raise TypeError(f"Expected tensor teacher target: {teacher_path}")
    if teacher.dtype != torch.float16:
        raise TypeError(f"Expected FP16 teacher target: {teacher_path}")
    teacher = teacher.float().reshape(-1)
    length = min(noisy.numel(), clean.numel(), teacher.numel())
    if length < 512:
        raise ValueError("D2 support waveform is too short for the WB frontend.")
    return (
        noisy[:length].contiguous(),
        clean[:length].contiguous(),
        teacher[:length].contiguous(),
    )


def _estimated_input_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = noisy - clean
    signal_power = float(torch.mean(clean.square()).clamp_min(1e-12))
    noise_power = float(torch.mean(noise.square()).clamp_min(1e-12))
    return float(10.0 * math.log10(signal_power / noise_power))


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"partitions": {}, "candidate_types": {}}
    for partition in ("train", "calibration", "audit"):
        selected = [row for row in records if row["partition"] == partition]
        result["partitions"][partition] = {
            "count": len(selected),
            "enhanced_pesq": _distribution(
                [float(row["enhanced_pesq"]) for row in selected]
            ),
            "noisy_pesq": _distribution(
                [float(row["noisy_pesq"]) for row in selected]
            ),
            "estimated_input_snr_db": _distribution(
                [float(row["estimated_input_snr_db"]) for row in selected]
            ),
        }
    result["candidate_types"] = {
        "clean": _distribution([4.5] * len(records)),
        "enhanced": _distribution(
            [float(row["enhanced_pesq"]) for row in records]
        ),
        "noisy": _distribution([float(row["noisy_pesq"]) for row in records]),
    }
    return result


def _write_coverage_plot(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for partition, color in (
        ("train", "tab:blue"),
        ("calibration", "tab:orange"),
        ("audit", "tab:green"),
    ):
        selected = [row for row in records if row["partition"] == partition]
        axes[0].hist(
            [float(row["enhanced_pesq"]) for row in selected],
            bins=24,
            alpha=0.45,
            label=partition,
            color=color,
        )
    axes[0].set_title("T0 enhanced PESQ-WB")
    axes[0].set_xlabel("PESQ-WB")
    axes[0].set_ylabel("utterances")
    axes[0].legend()
    axes[1].hist(
        [float(row["noisy_pesq"]) for row in records],
        bins=24,
        alpha=0.65,
        label="noisy",
    )
    axes[1].hist(
        [float(row["enhanced_pesq"]) for row in records],
        bins=24,
        alpha=0.55,
        label="T0 enhanced",
    )
    axes[1].set_title("D2 candidate score support")
    axes[1].set_xlabel("PESQ-WB")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def prepare_d2_support(
    *,
    train_manifest: str | Path,
    teacher_cache_manifest: str | Path,
    teacher_cache_metadata: str | Path,
    output_dir: str | Path,
    expected_teacher_sha256: str,
    train_rows: int = 1000,
    calibration_rows: int = 200,
    audit_rows: int = 200,
    seed: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Prepare a resumable, generated-only D2 support without copying inputs."""
    sizes = {
        "train": int(train_rows),
        "calibration": int(calibration_rows),
        "audit": int(audit_rows),
    }
    if any(value <= 0 for value in sizes.values()):
        raise ValueError("All D2 support partitions must be non-empty.")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    support_path = output_root / "support.json"
    progress_path = output_root / "progress.json"
    coverage_path = output_root / "coverage.json"
    plot_path = output_root / "coverage.png"

    source_paths = {
        "train_manifest": Path(train_manifest),
        "teacher_cache_manifest": Path(teacher_cache_manifest),
        "teacher_cache_metadata": Path(teacher_cache_metadata),
    }
    source_hashes_before = {
        key: _sha256(path) for key, path in source_paths.items()
    }
    metadata = json.loads(
        source_paths["teacher_cache_metadata"].read_text(encoding="utf-8")
    )
    if metadata.get("status") != "complete":
        raise ValueError("Teacher cache is not complete.")
    if bool(metadata.get("cache_inputs", True)):
        raise ValueError("D2 refuses a teacher cache that copied source inputs.")
    if str(metadata.get("storage_dtype")) != "float16":
        raise ValueError("D2 requires FP16 regenerable teacher outputs.")
    observed_teacher_hash = str(metadata.get("teacher_checkpoint_sha256") or "")
    if observed_teacher_hash != str(expected_teacher_sha256):
        raise ValueError("Teacher cache does not belong to the canonical T0 hash.")
    if str(metadata.get("train_manifest_sha256") or "") != source_hashes_before[
        "train_manifest"
    ]:
        raise ValueError("Teacher cache/train manifest identity mismatch.")

    with source_paths["teacher_cache_manifest"].open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    required = sum(sizes.values())
    if len(rows) < required:
        raise ValueError(
            f"D2 support requires {required} cache rows, found {len(rows)}."
        )
    identities = {
        (str(row.get("noisy") or ""), str(row.get("clean") or ""))
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("Teacher cache contains duplicate pair identities.")

    rng = random.Random(int(seed))
    rng.shuffle(rows)
    selected = rows[:required]
    partition_by_token: dict[str, str] = {}
    offset = 0
    for partition in ("train", "calibration", "audit"):
        for row in selected[offset : offset + sizes[partition]]:
            token = _token(str(row["noisy"]), str(row["clean"]))
            partition_by_token[token] = partition
        offset += sizes[partition]
    if len(partition_by_token) != required:
        raise RuntimeError("D2 partition identities are not disjoint.")

    speaker_ids = [_speaker_id(row) for row in selected]
    speaker_metadata_available = all(value is not None for value in speaker_ids)
    existing_records = []
    if progress_path.is_file():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            existing.get("source_hashes") == source_hashes_before
            and int(existing.get("seed", -1)) == int(seed)
            and existing.get("sizes") == sizes
        ):
            existing_records = list(existing.get("records") or [])
    records_by_token = {str(row["token"]): row for row in existing_records}

    for index, row in enumerate(selected, start=1):
        noisy_path = str(row["noisy"])
        clean_path = str(row["clean"])
        teacher_path = str(row["teacher_wav"])
        token = _token(noisy_path, clean_path)
        if token in records_by_token:
            continue
        noisy, clean, enhanced = _load_aligned(
            noisy_path,
            clean_path,
            teacher_path,
        )
        enhanced_pesq = float(
            pesq_score(
                clean.numpy(),
                enhanced.numpy(),
                16_000,
                bandwidth="wb",
            )
        )
        noisy_pesq = float(
            pesq_score(
                clean.numpy(),
                noisy.numpy(),
                16_000,
                bandwidth="wb",
            )
        )
        if not math.isfinite(enhanced_pesq) or not math.isfinite(noisy_pesq):
            raise RuntimeError(f"Non-finite PESQ in D2 support row {token}.")
        records_by_token[token] = {
            "token": token,
            "partition": partition_by_token[token],
            "noisy": noisy_path,
            "clean": clean_path,
            "enhanced": teacher_path,
            "sample_count": int(enhanced.numel()),
            "enhanced_pesq": enhanced_pesq,
            "enhanced_target": normalize_pesq(enhanced_pesq),
            "noisy_pesq": noisy_pesq,
            "noisy_target": normalize_pesq(noisy_pesq),
            "estimated_input_snr_db": _estimated_input_snr(clean, noisy),
            "speaker_id": _speaker_id(row),
        }
        if index == 1 or index == required or index % 25 == 0:
            progress = {
                "schema_version": 1,
                "status": "running",
                "seed": int(seed),
                "sizes": sizes,
                "source_hashes": source_hashes_before,
                "completed": len(records_by_token),
                "records": list(records_by_token.values()),
            }
            _atomic_json(progress_path, progress)
            if progress_callback:
                progress_callback(
                    f"D2 support PESQ {len(records_by_token)}/{required}"
                )

    records = [
        records_by_token[_token(str(row["noisy"]), str(row["clean"]))]
        for row in selected
    ]
    counts = {
        partition: sum(row["partition"] == partition for row in records)
        for partition in sizes
    }
    if counts != sizes:
        raise RuntimeError(f"D2 partition count mismatch: {counts} != {sizes}")
    source_hashes_after = {
        key: _sha256(path) for key, path in source_paths.items()
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A D2 source manifest changed during preparation.")

    coverage = _coverage(records)
    _atomic_json(coverage_path, coverage)
    _write_coverage_plot(records, plot_path)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "dataset": "VoiceBank+DEMAND",
        "bandwidth": "wb",
        "sample_rate": 16_000,
        "pesq_mode": "wb",
        "teacher_checkpoint_sha256": observed_teacher_hash,
        "train_manifest_sha256": source_hashes_before["train_manifest"],
        "teacher_cache_manifest_sha256": source_hashes_before[
            "teacher_cache_manifest"
        ],
        "teacher_cache_metadata_sha256": source_hashes_before[
            "teacher_cache_metadata"
        ],
        "seed": int(seed),
        "sizes": sizes,
        "counts": counts,
        "utterance_disjoint": len({row["token"] for row in records})
        == len(records),
        "speaker_metadata_available": speaker_metadata_available,
        "speaker_disjoint_verified": False,
        "speaker_limitation": (
            "Canonical content-addressed staging did not preserve original "
            "VoiceBank speaker IDs; partitions are strictly pair/utterance "
            "disjoint and this limitation is retained in claims."
            if not speaker_metadata_available
            else "Speaker IDs were detected but this fixed support preserves "
            "the predeclared deterministic utterance partition."
        ),
        "noise_condition_metadata_available": False,
        "cache_inputs": False,
        "storage_dtype": "float16",
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "coverage": coverage,
        "coverage_json": coverage_path.as_posix(),
        "coverage_plot": plot_path.as_posix(),
        "records": records,
    }
    _atomic_json(support_path, payload)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "status": "complete",
            "seed": int(seed),
            "sizes": sizes,
            "source_hashes": source_hashes_before,
            "completed": len(records),
            "support": support_path.as_posix(),
        },
    )
    return {**payload, "support_path": support_path.as_posix()}


def audit_d2_support(run_dir: str | Path) -> dict[str, Any]:
    """Independently reconcile a prepared D2 support package."""
    root = Path(run_dir).expanduser().resolve()
    support_path = root / "support" / "support.json"
    issues: list[str] = []
    if not support_path.is_file():
        return {
            "schema_version": 1,
            "valid": False,
            "issues": ["missing support/support.json"],
        }
    payload = json.loads(support_path.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    sizes = {
        key: int(value)
        for key, value in dict(payload.get("sizes") or {}).items()
    }
    observed_counts = {
        partition: sum(row.get("partition") == partition for row in records)
        for partition in ("train", "calibration", "audit")
    }
    if payload.get("status") != "complete":
        issues.append("support status is not complete")
    if observed_counts != sizes:
        issues.append("partition counts do not match declared sizes")
    tokens = [str(row.get("token") or "") for row in records]
    if not tokens or len(tokens) != len(set(tokens)) or any(not token for token in tokens):
        issues.append("support tokens are missing or duplicated")
    if not bool(payload.get("utterance_disjoint")):
        issues.append("support is not declared utterance-disjoint")
    if payload.get("source_hashes_before") != payload.get("source_hashes_after"):
        issues.append("source hashes changed during preparation")
    if bool(payload.get("cache_inputs", True)):
        issues.append("support declares copied source inputs")
    if str(payload.get("storage_dtype")) != "float16":
        issues.append("support storage dtype is not FP16")
    for directory in (root / "support" / "noisy", root / "support" / "clean"):
        if directory.exists():
            issues.append(f"unexpected copied-input directory: {directory.name}")
    for artifact in (
        root / "support" / "coverage.json",
        root / "support" / "coverage.png",
    ):
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            issues.append(f"missing coverage artifact: {artifact.name}")
    missing_enhanced = 0
    wrong_dtype = 0
    nonfinite = 0
    for row in records:
        enhanced_path = Path(str(row.get("enhanced") or ""))
        if not enhanced_path.is_file():
            missing_enhanced += 1
            continue
        tensor = torch.load(
            enhanced_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16:
            wrong_dtype += 1
        for key in ("enhanced_pesq", "noisy_pesq", "estimated_input_snr_db"):
            if not math.isfinite(float(row.get(key, float("nan")))):
                nonfinite += 1
    if missing_enhanced:
        issues.append(f"missing enhanced targets: {missing_enhanced}")
    if wrong_dtype:
        issues.append(f"non-FP16 enhanced targets: {wrong_dtype}")
    if nonfinite:
        issues.append(f"non-finite metric fields: {nonfinite}")
    return {
        "schema_version": 1,
        "valid": not issues,
        "issues": issues,
        "support_sha256": _sha256(support_path),
        "record_count": len(records),
        "counts": observed_counts,
        "unique_tokens": len(set(tokens)),
        "missing_enhanced": missing_enhanced,
        "wrong_dtype": wrong_dtype,
        "nonfinite_metric_fields": nonfinite,
        "source_hashes_unchanged": payload.get("source_hashes_before")
        == payload.get("source_hashes_after"),
        "speaker_disjoint_verified": bool(
            payload.get("speaker_disjoint_verified")
        ),
        "speaker_limitation": payload.get("speaker_limitation"),
    }
