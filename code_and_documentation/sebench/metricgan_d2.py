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
from sebench.losses import (
    SpeechBrainMetricDiscriminator,
    load_pesq_proxy_checkpoint,
    save_pesq_proxy_checkpoint,
)
from sebench.metricgan_alternating import (
    evaluate_calibration_gate,
    normalize_pesq,
)


class PlannedD2Interruption(RuntimeError):
    """Controlled interruption used to verify exact D2 resume behavior."""


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
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


def _range_candidate_waveforms(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    enhanced: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Predeclared train-only score-widening candidates around T0."""
    phase = torch.linspace(
        0.0,
        2.0 * math.pi,
        enhanced.numel(),
        dtype=enhanced.dtype,
    )
    bounded_mask = 1.0 + 0.05 * torch.sin(phase)
    return {
        "noisy_to_enhanced_050": 0.50 * noisy + 0.50 * enhanced,
        "enhanced_to_noisy_005": 0.95 * enhanced + 0.05 * noisy,
        "enhanced_to_clean_005": 0.95 * enhanced + 0.05 * clean,
        "enhanced_to_clean_025": 0.75 * enhanced + 0.25 * clean,
        "enhanced_to_clean_050": 0.50 * enhanced + 0.50 * clean,
        "output_mask_sine_005": enhanced * bounded_mask,
    }


def _pesq_bin(score: float, edges: tuple[float, ...]) -> int:
    return int(np.digitize([float(score)], np.asarray(edges)[1:-1])[0])


def _write_range_coverage(
    candidates: list[dict[str, Any]],
    *,
    edges: tuple[float, ...],
    json_path: Path,
    plot_path: Path,
) -> dict[str, Any]:
    by_type: dict[str, Any] = {}
    by_bin: dict[str, int] = {}
    for row in candidates:
        kind = str(row["candidate_type"])
        by_type.setdefault(kind, []).append(float(row["pesq"]))
        key = str(row["pesq_bin"])
        by_bin[key] = by_bin.get(key, 0) + 1
    coverage = {
        "candidate_count": len(candidates),
        "pesq_edges": list(edges),
        "by_type": {
            key: _distribution(values) for key, values in sorted(by_type.items())
        },
        "by_bin": by_bin,
        "pesq": _distribution([float(row["pesq"]) for row in candidates]),
    }
    _atomic_json(json_path, coverage)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, values in sorted(by_type.items()):
        axes[0].hist(values, bins=list(edges), alpha=0.35, label=kind)
    axes[0].set_title("D2-RANGE train-only PESQ-WB")
    axes[0].set_xlabel("PESQ-WB")
    axes[0].set_ylabel("candidates")
    axes[0].legend(fontsize=7)
    indices = list(range(len(edges) - 1))
    axes[1].bar(indices, [by_bin.get(str(index), 0) for index in indices])
    axes[1].set_xticks(
        indices,
        [f"{edges[index]:.1f}–{edges[index + 1]:.1f}" for index in indices],
        rotation=35,
        ha="right",
    )
    axes[1].set_title("Raw candidate availability by PESQ bin")
    axes[1].set_ylabel("candidates")
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return coverage


def prepare_d2_range_support(
    *,
    base_support_path: str | Path,
    output_dir: str | Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Add deterministic train-only variants without touching the fixed audit."""
    base_path = Path(base_support_path).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("status") != "complete":
        raise ValueError("D2-RANGE requires a complete fixed D2 support.")
    records = list(base.get("records") or [])
    train_records = [row for row in records if row.get("partition") == "train"]
    if not train_records:
        raise ValueError("D2-RANGE requires non-empty train support.")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_root = output_root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    support_path = output_root / "support.json"
    progress_path = output_root / "progress.json"
    coverage_path = output_root / "coverage.json"
    plot_path = output_root / "coverage.png"
    edges = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)

    existing: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("base_support_sha256") == _sha256(base_path):
            existing = {
                str(row["candidate_token"]): row
                for row in progress.get("range_candidates") or []
            }

    expected = len(train_records) * (1 + len(_range_candidate_waveforms(
        torch.zeros(512), torch.zeros(512), torch.zeros(512)
    )))
    completed_parents = 0
    for parent in train_records:
        noisy, clean, enhanced = _load_support_record(parent)
        variants: dict[str, torch.Tensor | None] = {
            "t0_enhanced": None,
            **_range_candidate_waveforms(noisy, clean, enhanced),
        }
        for kind, waveform in variants.items():
            candidate_token = hashlib.sha256(
                f"{parent['token']}|{kind}".encode("utf-8")
            ).hexdigest()[:24]
            if candidate_token in existing:
                continue
            if waveform is None:
                candidate_path = str(parent["enhanced"])
                score = float(parent["enhanced_pesq"])
                storage = "base_t0_fp16_reference"
            else:
                waveform = waveform.contiguous()
                candidate_path_obj = candidates_root / f"{candidate_token}.pt"
                _atomic_torch(candidate_path_obj, waveform.half().cpu())
                candidate_path = candidate_path_obj.as_posix()
                score = float(
                    pesq_score(
                        clean.numpy(),
                        waveform.numpy(),
                        16_000,
                        bandwidth="wb",
                    )
                )
                storage = "derived_fp16"
            if not math.isfinite(score):
                raise RuntimeError(
                    f"Non-finite D2-RANGE PESQ for {candidate_token}."
                )
            existing[candidate_token] = {
                "candidate_token": candidate_token,
                "parent_token": parent["token"],
                "candidate_type": kind,
                "candidate": candidate_path,
                "storage": storage,
                "sample_count": int(enhanced.numel()),
                "pesq": score,
                "target": normalize_pesq(score),
                "pesq_bin": _pesq_bin(score, edges),
            }
        completed_parents += 1
        if (
            completed_parents == 1
            or completed_parents == len(train_records)
            or completed_parents % 25 == 0
        ):
            _atomic_json(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "running",
                    "base_support_sha256": _sha256(base_path),
                    "completed_parents": completed_parents,
                    "expected_candidates": expected,
                    "range_candidates": list(existing.values()),
                },
            )
            if progress_callback:
                progress_callback(
                    "D2-RANGE support "
                    f"{completed_parents}/{len(train_records)} parents, "
                    f"{len(existing)}/{expected} candidates"
                )
    candidates = sorted(
        existing.values(),
        key=lambda row: (str(row["parent_token"]), str(row["candidate_type"])),
    )
    if len(candidates) != expected:
        raise RuntimeError(
            f"D2-RANGE candidate count {len(candidates)} != {expected}."
        )
    coverage = _write_range_coverage(
        candidates,
        edges=edges,
        json_path=coverage_path,
        plot_path=plot_path,
    )
    audit_tokens = sorted(
        str(row["token"]) for row in records if row["partition"] == "audit"
    )
    payload = {
        **base,
        "schema_version": 2,
        "strategy": "D2-RANGE",
        "base_support_path": base_path.as_posix(),
        "base_support_sha256": _sha256(base_path),
        "range_train_only": True,
        "range_candidate_storage_dtype": "float16",
        "range_candidate_count": len(candidates),
        "range_candidate_types": sorted(
            {str(row["candidate_type"]) for row in candidates}
        ),
        "range_pesq_edges": list(edges),
        "range_coverage": coverage,
        "range_coverage_json": coverage_path.as_posix(),
        "range_coverage_plot": plot_path.as_posix(),
        "fixed_audit_tokens_sha256": hashlib.sha256(
            "\n".join(audit_tokens).encode("utf-8")
        ).hexdigest(),
        "range_candidates": candidates,
    }
    _atomic_json(support_path, payload)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "status": "complete",
            "base_support_sha256": _sha256(base_path),
            "completed_parents": len(train_records),
            "candidate_count": len(candidates),
            "support": support_path.as_posix(),
        },
    )
    return {**payload, "support_path": support_path.as_posix()}


def audit_d2_range_support(run_dir: str | Path) -> dict[str, Any]:
    """Verify that D2-RANGE widens train only and preserves the fixed audit."""
    root = Path(run_dir).expanduser().resolve()
    path = root / "support" / "support.json"
    issues: list[str] = []
    if not path.is_file():
        return {"schema_version": 1, "valid": False, "issues": ["missing support"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = list(payload.get("range_candidates") or [])
    records = list(payload.get("records") or [])
    train_tokens = {
        str(row["token"]) for row in records if row.get("partition") == "train"
    }
    audit_tokens = sorted(
        str(row["token"]) for row in records if row.get("partition") == "audit"
    )
    if payload.get("strategy") != "D2-RANGE":
        issues.append("support strategy is not D2-RANGE")
    if not bool(payload.get("range_train_only")):
        issues.append("range support is not declared train-only")
    if not candidates:
        issues.append("range candidate list is empty")
    if any(str(row.get("parent_token")) not in train_tokens for row in candidates):
        issues.append("a range candidate is not derived from train")
    if len({str(row.get("candidate_token")) for row in candidates}) != len(candidates):
        issues.append("range candidate tokens are duplicated")
    expected_audit_hash = hashlib.sha256(
        "\n".join(audit_tokens).encode("utf-8")
    ).hexdigest()
    if payload.get("fixed_audit_tokens_sha256") != expected_audit_hash:
        issues.append("fixed audit identity mismatch")
    wrong_dtype = 0
    missing = 0
    nonfinite = 0
    for row in candidates:
        if not math.isfinite(float(row.get("pesq", float("nan")))):
            nonfinite += 1
        if row.get("storage") != "derived_fp16":
            continue
        candidate_path = Path(str(row.get("candidate") or ""))
        if not candidate_path.is_file():
            missing += 1
            continue
        tensor = torch.load(candidate_path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16:
            wrong_dtype += 1
    if missing:
        issues.append(f"missing derived candidates: {missing}")
    if wrong_dtype:
        issues.append(f"non-FP16 derived candidates: {wrong_dtype}")
    if nonfinite:
        issues.append(f"non-finite range labels: {nonfinite}")
    for directory in (root / "support" / "noisy", root / "support" / "clean"):
        if directory.exists():
            issues.append(f"unexpected copied-input directory: {directory.name}")
    return {
        "schema_version": 1,
        "valid": not issues,
        "issues": issues,
        "support_sha256": _sha256(path),
        "candidate_count": len(candidates),
        "candidate_types": sorted(
            {str(row.get("candidate_type")) for row in candidates}
        ),
        "train_parent_count": len(
            {str(row.get("parent_token")) for row in candidates}
        ),
        "fixed_audit_count": len(audit_tokens),
        "missing_candidates": missing,
        "wrong_dtype": wrong_dtype,
        "nonfinite_labels": nonfinite,
    }


def _load_support_record(
    record: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _load_aligned(
        record["noisy"],
        record["clean"],
        record["enhanced"],
    )


def _d2_update(
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    candidate: torch.Tensor,
    clean: torch.Tensor,
    target: float,
    *,
    device: str,
    grad_clip: float,
) -> float:
    prediction = discriminator.normalized_score(
        candidate.unsqueeze(0).to(device),
        clean.unsqueeze(0).to(device),
    )
    target_tensor = torch.tensor([float(target)], device=device)
    loss = torch.mean((prediction - target_tensor) ** 2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(grad_clip))
    optimizer.step()
    return float(loss.detach().cpu())


def _d2_current_pass(
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    records: list[dict[str, Any]],
    *,
    device: str,
    grad_clip: float,
) -> list[float]:
    losses: list[float] = []
    for record in records:
        noisy, clean, enhanced = _load_support_record(record)
        for candidate, target in (
            (clean, 1.0),
            (enhanced, float(record["enhanced_target"])),
            (noisy, float(record["noisy_target"])),
        ):
            losses.append(
                _d2_update(
                    discriminator,
                    optimizer,
                    candidate,
                    clean,
                    target,
                    device=device,
                    grad_clip=grad_clip,
                )
            )
    return losses


def _load_range_candidate(
    candidate_record: dict[str, Any],
    parent_records: dict[str, dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    parent = parent_records[str(candidate_record["parent_token"])]
    noisy, clean, enhanced = _load_support_record(parent)
    if candidate_record["candidate_type"] == "t0_enhanced":
        candidate = enhanced
    else:
        stored = torch.load(
            Path(candidate_record["candidate"]),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(stored, torch.Tensor) or stored.dtype != torch.float16:
            raise TypeError("D2-RANGE candidate must be a FP16 tensor.")
        candidate = stored.float().reshape(-1)
        length = min(noisy.numel(), clean.numel(), candidate.numel())
        noisy = noisy[:length]
        clean = clean[:length]
        candidate = candidate[:length]
    return noisy, clean, candidate, float(candidate_record["target"])


def _d2_range_pass(
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    candidates: list[dict[str, Any]],
    parent_records: dict[str, dict[str, Any]],
    *,
    device: str,
    grad_clip: float,
) -> list[float]:
    losses: list[float] = []
    for record in candidates:
        noisy, clean, candidate, target = _load_range_candidate(
            record,
            parent_records,
        )
        parent = parent_records[str(record["parent_token"])]
        for waveform, normalized_target in (
            (clean, 1.0),
            (candidate, target),
            (noisy, float(parent["noisy_target"])),
        ):
            losses.append(
                _d2_update(
                    discriminator,
                    optimizer,
                    waveform,
                    clean,
                    normalized_target,
                    device=device,
                    grad_clip=grad_clip,
                )
            )
    return losses


def _balanced_range_sample(
    candidates: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Deterministically sample near-equally from every non-empty PESQ bin."""
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        by_bin.setdefault(int(row["pesq_bin"]), []).append(row)
    for rows in by_bin.values():
        rng.shuffle(rows)
    bins = sorted(by_bin)
    if not bins:
        raise ValueError("D2-RANGE has no populated PESQ bins.")
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in bins}
    while len(selected) < int(count):
        progressed = False
        for key in bins:
            rows = by_bin[key]
            offset = offsets[key]
            if offset >= len(rows):
                continue
            selected.append(rows[offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == int(count):
                break
        if not progressed:
            break
    if len(selected) != int(count):
        raise ValueError(
            f"D2-RANGE requested {count} balanced candidates, found "
            f"{len(selected)}."
        )
    rng.shuffle(selected)
    return selected


@torch.inference_mode()
def _evaluate_d2(
    discriminator: SpeechBrainMetricDiscriminator,
    records: list[dict[str, Any]],
    *,
    device: str,
) -> dict[str, Any]:
    discriminator.eval()
    targets: list[float] = []
    predictions: list[float] = []
    snr_values: list[float] = []
    for record in records:
        _, clean, enhanced = _load_support_record(record)
        normalized = discriminator.normalized_score(
            enhanced.unsqueeze(0).to(device),
            clean.unsqueeze(0).to(device),
        )
        targets.append(float(record["enhanced_pesq"]))
        predictions.append(float((5.0 * normalized - 0.5).cpu()))
        snr_values.append(float(record["estimated_input_snr_db"]))
    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    errors = prediction_array - target_array

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        output = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            output[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return output

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if (
            len(left) < 2
            or float(np.std(left)) == 0.0
            or float(np.std(right)) == 0.0
        ):
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan")
    return {
        "record_count": len(records),
        "count": len(records),
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
        "normalized_mae": (
            float(np.mean(np.abs(errors))) / 5.0
            if len(errors)
            else float("nan")
        ),
        "rmse": rmse,
        "normalized_rmse": rmse / 5.0,
        "mse": float(np.mean(errors**2)) if len(errors) else float("nan"),
        "pearson": correlation(target_array, prediction_array),
        "spearman": correlation(ranks(target_array), ranks(prediction_array)),
        "target_min": float(np.min(target_array)),
        "target_max": float(np.max(target_array)),
        "target_std": float(np.std(target_array)),
        "prediction_min": float(np.min(prediction_array)),
        "prediction_max": float(np.max(prediction_array)),
        "prediction_std": float(np.std(prediction_array)),
        "targets": targets,
        "predictions": predictions,
        "estimated_input_snr_db": snr_values,
    }


def _calibration_selection_score(metrics: dict[str, Any]) -> float:
    """Minimized D2 selection score; audit thresholds remain unchanged."""
    pearson = float(metrics.get("pearson", float("nan")))
    spearman = float(metrics.get("spearman", float("nan")))
    if not math.isfinite(pearson):
        pearson = -1.0
    if not math.isfinite(spearman):
        spearman = -1.0
    range_excess = max(
        0.0,
        float(metrics["target_min"]) - float(metrics["prediction_min"]),
        float(metrics["prediction_max"]) - float(metrics["target_max"]),
    )
    return float(
        float(metrics["normalized_mae"])
        + 0.10 * max(0.0, 0.80 - pearson)
        + 0.10 * max(0.0, 0.80 - spearman)
        + 0.01 * range_excess
    )


def _snr_subgroups(
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    snr = np.asarray(metrics["estimated_input_snr_db"], dtype=np.float64)
    targets = np.asarray(metrics["targets"], dtype=np.float64)
    predictions = np.asarray(metrics["predictions"], dtype=np.float64)
    if not len(snr):
        return []
    boundaries = np.quantile(snr, [0.0, 0.25, 0.5, 0.75, 1.0])
    groups = []
    for index in range(4):
        lower = float(boundaries[index])
        upper = float(boundaries[index + 1])
        mask = (snr >= lower) & (
            snr <= upper if index == 3 else snr < upper
        )
        selected_targets = targets[mask]
        selected_predictions = predictions[mask]
        errors = selected_predictions - selected_targets
        if not len(selected_targets):
            groups.append(
                {
                    "quartile": index + 1,
                    "snr_min": lower,
                    "snr_max": upper,
                    "count": 0,
                    "mae": float("nan"),
                    "pearson": float("nan"),
                }
            )
            continue
        correlation = (
            float(np.corrcoef(selected_targets, selected_predictions)[0, 1])
            if len(selected_targets) > 1
            and np.std(selected_targets) > 0
            and np.std(selected_predictions) > 0
            else float("nan")
        )
        groups.append(
            {
                "quartile": index + 1,
                "snr_min": lower,
                "snr_max": upper,
                "count": int(np.sum(mask)),
                "mae": float(np.mean(np.abs(errors))),
                "pearson": correlation,
            }
        )
    return groups


def _rank_correlation(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        len(left_array) < 2
        or np.std(left_array) == 0
        or np.std(right_array) == 0
    ):
        return float("nan")
    left_rank = np.argsort(np.argsort(left_array, kind="mergesort"))
    right_rank = np.argsort(np.argsort(right_array, kind="mergesort"))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


@torch.inference_mode()
def _directional_audit(
    discriminator: SpeechBrainMetricDiscriminator,
    records: list[dict[str, Any]],
    *,
    device: str,
    alpha: float = 0.05,
    true_delta_floor: float = 0.005,
) -> dict[str, Any]:
    discriminator.eval()
    rows: list[dict[str, Any]] = []
    for record in records:
        noisy, clean, enhanced = _load_support_record(record)
        base_true = float(record["enhanced_pesq"])
        base_prediction = float(
            (
                5.0
                * discriminator.normalized_score(
                    enhanced.unsqueeze(0).to(device),
                    clean.unsqueeze(0).to(device),
                )
                - 0.5
            ).cpu()
        )
        variants = {
            "toward_clean": (1.0 - alpha) * enhanced + alpha * clean,
            "toward_noisy": (1.0 - alpha) * enhanced + alpha * noisy,
        }
        for name, candidate in variants.items():
            true_score = float(
                pesq_score(
                    clean.numpy(),
                    candidate.numpy(),
                    16_000,
                    bandwidth="wb",
                )
            )
            prediction = float(
                (
                    5.0
                    * discriminator.normalized_score(
                        candidate.unsqueeze(0).to(device),
                        clean.unsqueeze(0).to(device),
                    )
                    - 0.5
                ).cpu()
            )
            true_delta = true_score - base_true
            predicted_delta = prediction - base_prediction
            eligible = abs(true_delta) >= float(true_delta_floor)
            rows.append(
                {
                    "token": record["token"],
                    "variant": name,
                    "true_delta": true_delta,
                    "predicted_delta": predicted_delta,
                    "eligible": eligible,
                    "sign_match": (
                        bool(np.sign(true_delta) == np.sign(predicted_delta))
                        if eligible
                        else None
                    ),
                }
            )
    eligible_rows = [row for row in rows if row["eligible"]]
    sign_agreement = (
        float(np.mean([bool(row["sign_match"]) for row in eligible_rows]))
        if eligible_rows
        else float("nan")
    )
    spearman = _rank_correlation(
        [float(row["true_delta"]) for row in eligible_rows],
        [float(row["predicted_delta"]) for row in eligible_rows],
    )
    gate = {
        "passed": bool(
            len(eligible_rows) >= len(records)
            and math.isfinite(sign_agreement)
            and sign_agreement >= 0.70
            and math.isfinite(spearman)
            and spearman >= 0.60
        ),
        "checks": {
            "eligible_pair_count": len(eligible_rows) >= len(records),
            "sign_agreement": math.isfinite(sign_agreement)
            and sign_agreement >= 0.70,
            "spearman": math.isfinite(spearman) and spearman >= 0.60,
        },
        "thresholds": {
            "minimum_eligible_pairs": len(records),
            "minimum_sign_agreement": 0.70,
            "minimum_spearman": 0.60,
            "true_delta_floor": float(true_delta_floor),
            "alpha": float(alpha),
        },
    }
    return {
        "pair_count": len(rows),
        "eligible_pair_count": len(eligible_rows),
        "sign_agreement": sign_agreement,
        "spearman": spearman,
        "gate": gate,
        "pairs": rows,
    }


def _write_d2_plots(
    *,
    output_root: Path,
    history: list[dict[str, Any]],
    audit: dict[str, Any],
    directional: dict[str, Any],
) -> dict[str, str]:
    history_plot = output_root / "reports" / "training_history.png"
    calibration_plot = output_root / "reports" / "audit_calibration.png"
    directional_plot = output_root / "reports" / "directional_deltas.png"
    history_plot.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        [row["epoch"] for row in history],
        [row["calibration"]["normalized_mae"] for row in history],
        marker="o",
    )
    axes[0].axhline(0.06, linestyle="--", color="black")
    axes[0].set_title("D2 calibration normalized MAE")
    axes[0].set_xlabel("epoch")
    axes[1].plot(
        [row["epoch"] for row in history],
        [row["calibration"]["pearson"] for row in history],
        marker="o",
        label="Pearson",
    )
    axes[1].plot(
        [row["epoch"] for row in history],
        [row["calibration"]["spearman"] for row in history],
        marker="o",
        label="Spearman",
    )
    axes[1].axhline(0.80, linestyle="--", color="black")
    axes[1].set_title("D2 calibration correlation")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(history_plot, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(audit["targets"], audit["predictions"], s=14, alpha=0.65)
    lower = min(audit["target_min"], audit["prediction_min"])
    upper = max(audit["target_max"], audit["prediction_max"])
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    axis.set_xlabel("true PESQ-WB")
    axis.set_ylabel("D2 PESQ-WB")
    axis.set_title("Untouched D2 audit")
    figure.tight_layout()
    figure.savefig(calibration_plot, dpi=160)
    plt.close(figure)

    eligible = [row for row in directional["pairs"] if row["eligible"]]
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        [row["true_delta"] for row in eligible],
        [row["predicted_delta"] for row in eligible],
        s=14,
        alpha=0.65,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("true PESQ-WB delta")
    axis.set_ylabel("D2 predicted delta")
    axis.set_title("Local directional audit")
    figure.tight_layout()
    figure.savefig(directional_plot, dpi=160)
    plt.close(figure)
    return {
        "history_plot": history_plot.as_posix(),
        "calibration_plot": calibration_plot.as_posix(),
        "directional_plot": directional_plot.as_posix(),
    }


def fit_d2_official(
    *,
    support_path: str | Path,
    output_dir: str | Path,
    device: str,
    max_epochs: int = 20,
    current_rows: int = 100,
    lr: float = 5e-4,
    history_portion: float = 0.20,
    lr_factor: float = 0.5,
    lr_patience: int = 2,
    min_lr: float = 1e-6,
    early_stop_patience: int = 5,
    grad_clip: float = 5.0,
    seed: int = 0,
    resume: bool = False,
    interrupt_after_epoch: int | None = None,
    base_channels: int = 15,
    evaluation_limit: int | None = None,
    run_directional_audit: bool = True,
    strict_gate: bool = True,
    strategy: str = "D2-OFFICIAL",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit D2 with official batch-1 passes and select without reading audit."""
    if strategy not in {"D2-OFFICIAL", "D2-RANGE"}:
        raise ValueError(f"Unsupported D2 strategy: {strategy}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(support_path).read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    train_records = [row for row in records if row["partition"] == "train"]
    calibration_records = [
        row for row in records if row["partition"] == "calibration"
    ]
    audit_records = [row for row in records if row["partition"] == "audit"]
    range_candidates = list(payload.get("range_candidates") or [])
    if strategy == "D2-RANGE" and not range_candidates:
        raise ValueError("D2-RANGE fitting requires range candidates.")
    if evaluation_limit is not None:
        calibration_records = calibration_records[: int(evaluation_limit)]
        audit_records = audit_records[: int(evaluation_limit)]
    if not train_records or not calibration_records or not audit_records:
        raise ValueError("D2 requires non-empty train/calibration/audit support.")
    available_current = (
        len(range_candidates) if strategy == "D2-RANGE" else len(train_records)
    )
    if int(current_rows) > available_current:
        raise ValueError("D2 current_rows exceeds the fixed train support.")

    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    discriminator = SpeechBrainMetricDiscriminator(
        base_channels=int(base_channels)
    ).to(device)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_factor),
        patience=int(lr_patience),
        min_lr=float(min_lr),
    )
    state_path = output_root / "training_state.pt"
    selected_checkpoint = output_root / "models" / f"{strategy}.pt"
    history_path = output_root / "metrics" / "history.json"
    history: list[dict[str, Any]] = []
    seen_tokens: list[str] = []
    best_score = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    start_epoch = 1

    if resume:
        if not state_path.is_file():
            raise FileNotFoundError("D2 resume state does not exist.")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("support_sha256") != _sha256(support_path):
            raise ValueError("D2 resume support identity mismatch.")
        if state.get("strategy", "D2-OFFICIAL") != strategy:
            raise ValueError("D2 resume strategy mismatch.")
        discriminator.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(device)
        scheduler.load_state_dict(state["scheduler_state"])
        history = list(state["history"])
        seen_tokens = list(state["seen_tokens"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improve = int(state["epochs_without_improve"])
        start_epoch = int(state["epoch"]) + 1

    records_by_token = {str(row["token"]): row for row in train_records}
    range_by_token = {
        str(row["candidate_token"]): row for row in range_candidates
    }
    stopped_early = False
    for epoch in range(start_epoch, int(max_epochs) + 1):
        rng = random.Random(int(seed) + epoch * 1009)
        if strategy == "D2-RANGE":
            current = _balanced_range_sample(
                range_candidates,
                int(current_rows),
                rng,
            )
            current_tokens = [str(row["candidate_token"]) for row in current]
        else:
            current = rng.sample(train_records, int(current_rows))
            current_tokens = [str(row["token"]) for row in current]
        seen_tokens = list(dict.fromkeys([*seen_tokens, *current_tokens]))
        history_count = max(1, round(len(seen_tokens) * float(history_portion)))
        historical_tokens = list(seen_tokens)
        rng.shuffle(historical_tokens)
        historical = [
            (
                range_by_token[token]
                if strategy == "D2-RANGE"
                else records_by_token[token]
            )
            for token in historical_tokens[:history_count]
        ]

        discriminator.train()
        if progress_callback:
            progress_callback(
                f"D2 epoch {epoch}/{max_epochs} pass 1/3 current "
                f"({len(current)} records)"
            )
        if strategy == "D2-RANGE":
            current_first = _d2_range_pass(
                discriminator,
                optimizer,
                current,
                records_by_token,
                device=device,
                grad_clip=grad_clip,
            )
        else:
            current_first = _d2_current_pass(
                discriminator,
                optimizer,
                current,
                device=device,
                grad_clip=grad_clip,
            )
        if progress_callback:
            progress_callback(
                f"D2 epoch {epoch}/{max_epochs} pass 2/3 historical "
                f"({len(historical)} records)"
            )
        historical_losses = []
        for record in historical:
            if strategy == "D2-RANGE":
                _, clean, candidate, target = _load_range_candidate(
                    record,
                    records_by_token,
                )
            else:
                _, clean, candidate = _load_support_record(record)
                target = float(record["enhanced_target"])
            historical_losses.append(
                _d2_update(
                    discriminator,
                    optimizer,
                    candidate,
                    clean,
                    target,
                    device=device,
                    grad_clip=grad_clip,
                )
            )
        if progress_callback:
            progress_callback(
                f"D2 epoch {epoch}/{max_epochs} pass 3/3 current"
            )
        if strategy == "D2-RANGE":
            current_second = _d2_range_pass(
                discriminator,
                optimizer,
                current,
                records_by_token,
                device=device,
                grad_clip=grad_clip,
            )
        else:
            current_second = _d2_current_pass(
                discriminator,
                optimizer,
                current,
                device=device,
                grad_clip=grad_clip,
            )
        calibration = _evaluate_d2(
            discriminator,
            calibration_records,
            device=device,
        )
        selection_score = _calibration_selection_score(calibration)
        improved = selection_score < best_score - 1e-12
        if improved:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improve = 0
            save_pesq_proxy_checkpoint(selected_checkpoint, discriminator)
        else:
            epochs_without_improve += 1
        scheduler.step(selection_score)
        row = {
            "epoch": epoch,
            "current_record_count": len(current),
            "historical_record_count": len(historical),
            "current_first_mse": float(np.mean(current_first)),
            "historical_mse": float(np.mean(historical_losses)),
            "current_second_mse": float(np.mean(current_second)),
            "calibration": calibration,
            "selection_score": selection_score,
            "improved": improved,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "epochs_without_improve": epochs_without_improve,
            "lr_after_eval": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        _atomic_json(history_path, history)
        state = {
            "schema_version": 1,
            "strategy": strategy,
            "epoch": epoch,
            "support_sha256": _sha256(support_path),
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in discriminator.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "seen_tokens": seen_tokens,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "epochs_without_improve": epochs_without_improve,
            "selected_checkpoint": selected_checkpoint.as_posix(),
        }
        _atomic_torch(state_path, state)
        if progress_callback:
            progress_callback(
                f"D2 epoch {epoch}: nMAE={calibration['normalized_mae']:.4f} "
                f"r={calibration['pearson']:.4f} "
                f"rho={calibration['spearman']:.4f} "
                f"best={best_epoch} lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        if interrupt_after_epoch is not None and epoch == int(
            interrupt_after_epoch
        ):
            raise PlannedD2Interruption(
                f"Planned D2 interruption after epoch {epoch}."
            )
        if epochs_without_improve >= int(early_stop_patience):
            stopped_early = True
            break

    if not selected_checkpoint.is_file():
        raise RuntimeError("D2 fitting did not produce a selected checkpoint.")
    selected = load_pesq_proxy_checkpoint(
        selected_checkpoint,
        device=device,
        freeze=True,
    )
    if not isinstance(selected, SpeechBrainMetricDiscriminator):
        raise TypeError("Selected D2 checkpoint has the wrong model kind.")
    audit = _evaluate_d2(selected, audit_records, device=device)
    audit_gate = evaluate_calibration_gate(
        audit,
        min_records=(len(audit_records) if not strict_gate else 200),
        max_normalized_mae=(0.06 if strict_gate else 1.0),
        min_pearson=(0.80 if strict_gate else -1.0),
        min_spearman=(0.80 if strict_gate else -1.0),
        min_prediction_std=(0.02 if strict_gate else 0.0),
        range_tolerance_raw=(0.30 if strict_gate else 5.0),
    )
    audit["subgroups_by_estimated_snr"] = _snr_subgroups(audit)
    directional = (
        _directional_audit(selected, audit_records, device=device)
        if run_directional_audit
        else {
            "pair_count": 0,
            "eligible_pair_count": 0,
            "sign_agreement": float("nan"),
            "spearman": float("nan"),
            "gate": {
                "passed": not strict_gate,
                "checks": {},
                "thresholds": {},
            },
            "pairs": [],
        }
    )
    if not strict_gate:
        directional["gate"]["verification_only_observed_passed"] = bool(
            directional["gate"]["passed"]
        )
        directional["gate"]["passed"] = True
    passed = bool(audit_gate["passed"] and directional["gate"]["passed"])
    plots = _write_d2_plots(
        output_root=output_root,
        history=history,
        audit=audit,
        directional=directional,
    )
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "strategy": strategy,
        "support_path": str(Path(support_path)),
        "support_sha256": _sha256(support_path),
        "max_epochs": int(max_epochs),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "stopped_early": stopped_early,
        "selected_checkpoint": selected_checkpoint.as_posix(),
        "selected_checkpoint_sha256": _sha256(selected_checkpoint),
        "training_state": state_path.as_posix(),
        "history": history_path.as_posix(),
        "audit": audit,
        "audit_gate": audit_gate,
        "directional": directional,
        "passed": passed,
        "plots": plots,
    }
    _atomic_json(output_root / "metrics" / "summary.json", result)
    return result
