"""Fresh, T2-disjoint support and train-only weight calibration for T3."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from metrics.pesq import pesq_score
from sebench.audio import load_mono_audio
from sebench.checkpoints import load_model_from_checkpoint
from sebench.t3_perceptual import (
    DifferentiablePESQInspiredLoss,
    calibrate_t3_gradient_weights,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_token(noisy: str, clean: str) -> str:
    return hashlib.sha256(f"{noisy}|{clean}".encode("utf-8")).hexdigest()[:20]


def _clean_token(clean: str) -> str:
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:20]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_t2_exclusions(paths: Iterable[str | Path]) -> tuple[set[str], set[str], dict[str, str]]:
    pair_tokens: set[str] = set()
    clean_paths: set[str] = set()
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = list(payload.get("records") or [])
        if not records:
            raise ValueError(f"T2 exclusion support has no records: {path}")
        hashes[path.name + ":" + _sha256(path)[:12]] = _sha256(path)
        for row in records:
            noisy = str(row.get("noisy") or "")
            clean = str(row.get("clean") or "")
            if not noisy or not clean:
                raise ValueError(f"Malformed T2 exclusion record in {path}.")
            pair_tokens.add(_pair_token(noisy, clean))
            clean_paths.add(clean)
    if not pair_tokens:
        raise ValueError("At least one T2 exclusion support is required.")
    return pair_tokens, clean_paths, hashes


def prepare_t3_identities(
    *,
    train_manifest: str | Path,
    teacher_cache_manifest: str | Path,
    teacher_cache_metadata: str | Path,
    t2_support_paths: Iterable[str | Path],
    output_dir: str | Path,
    expected_teacher_sha256: str,
    train_rows: int = 1_000,
    calibration_rows: int = 200,
    audit_rows: int = 200,
    seed: int = 3_003,
) -> dict[str, Any]:
    """Freeze fresh T3 identities without reading validation or test."""

    sizes = {
        "train": int(train_rows),
        "calibration": int(calibration_rows),
        "audit": int(audit_rows),
    }
    if any(value <= 0 for value in sizes.values()):
        raise ValueError("Every T3 direction partition must be non-empty.")
    source_paths = {
        "train_manifest": Path(train_manifest).expanduser().resolve(),
        "teacher_cache_manifest": Path(teacher_cache_manifest).expanduser().resolve(),
        "teacher_cache_metadata": Path(teacher_cache_metadata).expanduser().resolve(),
    }
    source_hashes_before = {key: _sha256(path) for key, path in source_paths.items()}
    metadata = json.loads(
        source_paths["teacher_cache_metadata"].read_text(encoding="utf-8")
    )
    if metadata.get("status") != "complete":
        raise ValueError("T3 requires a complete teacher cache.")
    if bool(metadata.get("cache_inputs", True)):
        raise ValueError("T3 refuses teacher caches that duplicate input audio.")
    if str(metadata.get("storage_dtype")) != "float16":
        raise ValueError("T3 requires local FP16 teacher outputs.")
    if str(metadata.get("teacher_checkpoint_sha256") or "") != str(
        expected_teacher_sha256
    ):
        raise ValueError("T3 teacher-cache checkpoint identity mismatch.")
    if str(metadata.get("train_manifest_sha256") or "") != source_hashes_before[
        "train_manifest"
    ]:
        raise ValueError("T3 teacher-cache/train-manifest identity mismatch.")
    with source_paths["teacher_cache_manifest"].open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    excluded_pairs, excluded_clean, exclusion_hashes = _load_t2_exclusions(
        t2_support_paths
    )
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    required = sum(sizes.values())
    selected: list[dict[str, str]] = []
    selected_clean: set[str] = set()
    for row in rows:
        noisy = str(row.get("noisy") or "")
        clean = str(row.get("clean") or "")
        teacher_wav = str(row.get("teacher_wav") or "")
        if not noisy or not clean or not teacher_wav:
            raise ValueError("Teacher cache contains a malformed row.")
        if _pair_token(noisy, clean) in excluded_pairs or clean in excluded_clean:
            continue
        if clean in selected_clean:
            continue
        selected.append(row)
        selected_clean.add(clean)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(
            f"Only {len(selected)} T2-disjoint clean identities available; "
            f"{required} required."
        )
    records: list[dict[str, Any]] = []
    offset = 0
    for partition in ("train", "calibration", "audit"):
        for row in selected[offset : offset + sizes[partition]]:
            noisy = str(row["noisy"])
            clean = str(row["clean"])
            teacher_path = Path(str(row["teacher_wav"]))
            teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
            if not isinstance(teacher, torch.Tensor) or teacher.dtype != torch.float16:
                raise TypeError(f"T3 expected an FP16 teacher tensor: {teacher_path}")
            records.append(
                {
                    "token": _pair_token(noisy, clean),
                    "clean_token": _clean_token(clean),
                    "partition": partition,
                    "noisy": noisy,
                    "clean": clean,
                    "teacher_t0": teacher_path.as_posix(),
                    "sample_count": int(teacher.numel()),
                }
            )
        offset += sizes[partition]
    source_hashes_after = {key: _sha256(path) for key, path in source_paths.items()}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A source manifest changed during T3 identity selection.")
    tokens = [str(row["token"]) for row in records]
    clean_tokens = [str(row["clean_token"]) for row in records]
    payload = {
        "schema_version": 1,
        "status": "identities_frozen",
        "dataset": "VoiceBank+DEMAND",
        "bandwidth": "wb",
        "sample_rate": 16_000,
        "pesq_mode": "wb",
        "seed": int(seed),
        "sizes": sizes,
        "counts": {
            partition: sum(row["partition"] == partition for row in records)
            for partition in sizes
        },
        "teacher_checkpoint_sha256": str(expected_teacher_sha256),
        "train_manifest_sha256": source_hashes_before["train_manifest"],
        "teacher_cache_manifest_sha256": source_hashes_before[
            "teacher_cache_manifest"
        ],
        "teacher_cache_metadata_sha256": source_hashes_before[
            "teacher_cache_metadata"
        ],
        "t2_exclusion_support_hashes": exclusion_hashes,
        "t2_excluded_pair_count": len(excluded_pairs),
        "t2_excluded_clean_count": len(excluded_clean),
        "pair_disjoint_from_t2": not bool(set(tokens) & excluded_pairs),
        "clean_disjoint_from_t2": not bool(selected_clean & excluded_clean),
        "pair_disjoint_within_t3": len(tokens) == len(set(tokens)),
        "clean_disjoint_within_t3": len(clean_tokens) == len(set(clean_tokens)),
        "cache_inputs": False,
        "storage_dtype": "float16",
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "records": records,
    }
    output_path = Path(output_dir).expanduser().resolve() / "identities.json"
    _atomic_json(output_path, payload)
    return {**payload, "identities_path": output_path.as_posix()}


def audit_t3_identities(path: str | Path) -> dict[str, Any]:
    identity_path = Path(path).expanduser().resolve()
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    issues: list[str] = []
    declared = dict(payload.get("sizes") or {})
    observed = {
        partition: sum(row.get("partition") == partition for row in records)
        for partition in ("train", "calibration", "audit")
    }
    if payload.get("status") != "identities_frozen":
        issues.append("identity status is not frozen")
    if observed != {key: int(value) for key, value in declared.items()}:
        issues.append("partition counts do not match")
    for key in (
        "pair_disjoint_from_t2",
        "clean_disjoint_from_t2",
        "pair_disjoint_within_t3",
        "clean_disjoint_within_t3",
    ):
        if not bool(payload.get(key)):
            issues.append(f"{key} is false")
    if payload.get("source_hashes_before") != payload.get("source_hashes_after"):
        issues.append("source hashes changed")
    missing_t0 = 0
    wrong_dtype = 0
    for row in records:
        path_value = Path(str(row.get("teacher_t0") or ""))
        if not path_value.is_file():
            missing_t0 += 1
            continue
        tensor = torch.load(path_value, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16:
            wrong_dtype += 1
    if missing_t0:
        issues.append(f"missing T0 tensors: {missing_t0}")
    if wrong_dtype:
        issues.append(f"non-FP16 T0 tensors: {wrong_dtype}")
    return {
        "schema_version": 1,
        "valid": not issues,
        "issues": issues,
        "identity_sha256": _sha256(identity_path),
        "record_count": len(records),
        "counts": observed,
        "missing_t0": missing_t0,
        "wrong_dtype": wrong_dtype,
    }


def calibrate_t3_weights(
    *,
    identities_path: str | Path,
    teacher_checkpoint: str | Path,
    expected_teacher_sha256: str,
    output_path: str | Path,
    device: str = "cuda",
    rows: int = 16,
    segment_samples: int = 32_000,
    logit_delta: float = 0.02,
) -> dict[str, Any]:
    """Freeze E1/E2 weights from T3 train identities only."""

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T3 gradient calibration requires CUDA.")
    identity_path = Path(identities_path).expanduser().resolve()
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    train_records = [
        row for row in payload.get("records") or [] if row.get("partition") == "train"
    ]
    selected = train_records[: int(rows)]
    if len(selected) != int(rows):
        raise ValueError(f"T3 weight calibration requires {rows} train identities.")
    checkpoint_path = Path(teacher_checkpoint).expanduser().resolve()
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != str(expected_teacher_sha256):
        raise ValueError("T3 calibration checkpoint SHA-256 mismatch.")
    model, package = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    observed: list[dict[str, float | str | int]] = []
    parameter_gradient_evidence: dict[str, Any] | None = None
    for row in selected:
        noisy, _ = load_mono_audio(str(row["noisy"]), 16_000)
        clean, _ = load_mono_audio(str(row["clean"]), 16_000)
        length = min(noisy.numel(), clean.numel(), int(segment_samples))
        noisy_batch = noisy[:length].to(device).reshape(1, 1, -1)
        clean_batch = clean[:length].to(device).reshape(1, -1)
        variants = model.forward_mask_logit_variants(
            noisy_batch,
            (0.0, float(logit_delta)),
        )
        teacher_t0 = variants[0, :, 0].detach()
        candidate = variants[1, :, 0]
        candidate.retain_grad()
        calibration = calibrate_t3_gradient_weights(
            candidate=candidate,
            clean=clean_batch,
            teacher_t0=teacher_t0,
        )
        if parameter_gradient_evidence is None:
            surrogate = DifferentiablePESQInspiredLoss().to(device)(
                candidate,
                clean_batch,
            )
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            gradients = torch.autograd.grad(
                surrogate,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            present = [gradient for gradient in gradients if gradient is not None]
            aggregate = torch.sqrt(
                sum(gradient.detach().float().square().sum() for gradient in present)
            )
            parameter_gradient_evidence = {
                "token": str(row["token"]),
                "parameter_tensors_total": len(parameters),
                "parameter_tensors_with_gradient": len(present),
                "all_finite": bool(
                    present
                    and all(torch.isfinite(gradient).all() for gradient in present)
                ),
                "aggregate_l2_norm": float(aggregate.item()),
            }
        observed.append(
            {
                "token": str(row["token"]),
                "samples": int(length),
                "supervised_norm": calibration.supervised_norm,
                "anchor_norm": calibration.anchor_norm,
                "pmsqe_norm": calibration.pmsqe_norm,
            }
        )
        del variants, candidate, teacher_t0, noisy_batch, clean_batch
        model.zero_grad(set_to_none=True)
    supervised_norm = float(np.median([float(row["supervised_norm"]) for row in observed]))
    anchor_norm = float(np.median([float(row["anchor_norm"]) for row in observed]))
    pmsqe_norm = float(np.median([float(row["pmsqe_norm"]) for row in observed]))
    if min(supervised_norm, anchor_norm, pmsqe_norm) <= 1e-12:
        raise RuntimeError("T3 aggregate gradient calibration contains a vanishing term.")
    anchor_contribution = supervised_norm
    anchor_weight = anchor_contribution / anchor_norm
    pre_pmsqe = supervised_norm + anchor_contribution
    pmsqe_contribution = (0.10 / 0.90) * pre_pmsqe
    pmsqe_weight = pmsqe_contribution / pmsqe_norm
    result = {
        "schema_version": 1,
        "status": "weights_frozen",
        "dataset_role": "train_fit/T3-direction-train only",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "identities_sha256": _sha256(identity_path),
        "teacher_checkpoint_sha256": checkpoint_hash,
        "teacher_family": package["model_family"],
        "rows": int(rows),
        "segment_samples": int(segment_samples),
        "mask_logit_delta": float(logit_delta),
        "aggregation": "median component waveform-gradient L2 norm",
        "median_gradient_norms": {
            "supervised": supervised_norm,
            "anchor": anchor_norm,
            "pmsqe": pmsqe_norm,
        },
        "frozen_weights": {
            "anchor": anchor_weight,
            "pmsqe": pmsqe_weight,
            "sisdr": 0.10,
        },
        "component_fraction_bounds": {
            "anchor": 0.50,
            "pmsqe": 0.10,
        },
        "parameter_gradient_evidence": parameter_gradient_evidence,
        "records": observed,
    }
    _atomic_json(Path(output_path).expanduser().resolve(), result)
    return result


def _input_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    signal = float(torch.mean(clean.square()).clamp_min(1e-12))
    noise = float(torch.mean((noisy - clean).square()).clamp_min(1e-12))
    return float(10.0 * np.log10(signal / noise))


def generate_t3_mask_candidates(
    *,
    identities_path: str | Path,
    teacher_checkpoint: str | Path,
    expected_teacher_sha256: str,
    output_dir: str | Path,
    device: str = "cuda",
    logit_deltas: tuple[float, ...] = (-0.04, -0.02, 0.02, 0.04),
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Generate resumable FP16 mask-logit candidates and true PESQ labels."""

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("T3 teacher-manifold candidate generation requires CUDA.")
    identity_path = Path(identities_path).expanduser().resolve()
    identities = json.loads(identity_path.read_text(encoding="utf-8"))
    records = list(identities.get("records") or [])
    if identities.get("status") != "identities_frozen" or not records:
        raise ValueError("T3 candidates require frozen non-empty identities.")
    checkpoint_path = Path(teacher_checkpoint).expanduser().resolve()
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != str(expected_teacher_sha256):
        raise ValueError("T3 candidate checkpoint SHA-256 mismatch.")
    deltas = tuple(float(value) for value in logit_deltas)
    if deltas != (-0.04, -0.02, 0.02, 0.04):
        raise ValueError("T3 mask-logit deltas must remain -0.04,-0.02,+0.02,+0.04.")
    root = Path(output_dir).expanduser().resolve()
    candidates_root = root / "waveforms"
    candidates_root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    output_path = root / "candidates.json"
    existing: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("identities_sha256") == _sha256(identity_path)
            and progress.get("teacher_checkpoint_sha256") == checkpoint_hash
        ):
            existing = {
                str(row["token"]): row for row in progress.get("parents") or []
            }
    model, package = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    pmsqe = DifferentiablePESQInspiredLoss().to(device)
    parity_errors: list[float] = []
    for index, row in enumerate(records, start=1):
        token = str(row["token"])
        if token in existing:
            continue
        noisy, _ = load_mono_audio(str(row["noisy"]), 16_000)
        clean, _ = load_mono_audio(str(row["clean"]), 16_000)
        cached = torch.load(
            str(row["teacher_t0"]),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(cached, torch.Tensor) or cached.dtype != torch.float16:
            raise TypeError("T3 candidate generation requires an FP16 T0 cache.")
        cached = cached.float().reshape(-1)
        length = min(noisy.numel(), clean.numel(), cached.numel())
        noisy = noisy[:length].contiguous()
        clean = clean[:length].contiguous()
        cached = cached[:length].contiguous()
        with torch.no_grad():
            variants = model.forward_mask_logit_variants(
                noisy.to(device).reshape(1, 1, -1),
                (0.0, *deltas),
            )[:, 0, 0]
            parity_mae = float(
                torch.mean(torch.abs(variants[0].cpu() - cached)).item()
            )
            if parity_mae > 5e-4:
                raise RuntimeError(
                    f"T3 zero-delta/cache parity MAE {parity_mae:.6g} exceeds 5e-4."
                )
            parity_errors.append(parity_mae)
            clean_gpu = clean.to(device).reshape(1, -1)
            clean_batch = clean_gpu.expand(variants.shape[0], -1).contiguous()
            with torch.autocast(device_type="cuda", enabled=False):
                pmsqe_values = (
                    pmsqe.loss(clean_batch.float(), variants.float())
                    .detach()
                    .cpu()
                    .numpy()
                )
        variant_cpu = variants.detach().cpu()
        base_pesq = float(
            pesq_score(clean.numpy(), variant_cpu[0].numpy(), 16_000, bandwidth="wb")
        )
        candidates: list[dict[str, Any]] = []
        for variant_index, delta in enumerate(deltas, start=1):
            candidate_token = hashlib.sha256(
                f"{token}|mask_logit|{delta:+.3f}".encode("utf-8")
            ).hexdigest()[:24]
            candidate_path = candidates_root / f"{candidate_token}.pt"
            torch.save(variant_cpu[variant_index].half(), candidate_path)
            score = float(
                pesq_score(
                    clean.numpy(),
                    variant_cpu[variant_index].numpy(),
                    16_000,
                    bandwidth="wb",
                )
            )
            candidates.append(
                {
                    "candidate_token": candidate_token,
                    "mask_logit_delta": delta,
                    "candidate": candidate_path.as_posix(),
                    "storage_dtype": "float16",
                    "pesq": score,
                    "pmsqe": float(pmsqe_values[variant_index]),
                    "delta_pesq": score - base_pesq,
                    "delta_pmsqe": float(pmsqe_values[variant_index] - pmsqe_values[0]),
                }
            )
        existing[token] = {
            "token": token,
            "partition": row["partition"],
            "sample_count": length,
            "estimated_input_snr_db": _input_snr(clean, noisy),
            "t0_pesq": base_pesq,
            "t0_pmsqe": float(pmsqe_values[0]),
            "zero_delta_cache_mae": parity_mae,
            "candidates": candidates,
        }
        if index == 1 or index == len(records) or index % 25 == 0:
            _atomic_json(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "running",
                    "identities_sha256": _sha256(identity_path),
                    "teacher_checkpoint_sha256": checkpoint_hash,
                    "completed": len(existing),
                    "parents": list(existing.values()),
                },
            )
            if progress_callback:
                progress_callback(f"T3 mask candidates {len(existing)}/{len(records)}")
    ordered = [existing[str(row["token"])] for row in records]
    payload = {
        "schema_version": 1,
        "status": "candidates_complete",
        "dataset": "VoiceBank+DEMAND",
        "bandwidth": "wb",
        "sample_rate": 16_000,
        "pesq_mode": "wb",
        "identities_sha256": _sha256(identity_path),
        "teacher_checkpoint_sha256": checkpoint_hash,
        "teacher_family": package["model_family"],
        "mask_logit_deltas": list(deltas),
        "parent_count": len(ordered),
        "candidate_count": len(ordered) * len(deltas),
        "storage_dtype": "float16",
        "cache_inputs": False,
        "zero_delta_cache_mae_max": max(parity_errors) if parity_errors else 0.0,
        "parents": ordered,
    }
    _atomic_json(output_path, payload)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "status": "complete",
            "identities_sha256": _sha256(identity_path),
            "teacher_checkpoint_sha256": checkpoint_hash,
            "completed": len(ordered),
            "candidates_sha256": _sha256(output_path),
        },
    )
    return {**payload, "candidates_path": output_path.as_posix()}


def audit_t3_direction(
    *,
    candidates_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Apply the one-shot true-PESQ local direction gate."""

    path = Path(candidates_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = json.loads(
        Path(weights_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    parents = list(payload.get("parents") or [])
    issues: list[str] = []
    if payload.get("status") != "candidates_complete":
        issues.append("candidates are not complete")
    rows_by_partition: dict[str, list[dict[str, float]]] = {
        "train": [],
        "calibration": [],
        "audit": [],
    }
    missing = 0
    wrong_dtype = 0
    for parent in parents:
        partition = str(parent["partition"])
        for candidate in parent.get("candidates") or []:
            candidate_path = Path(str(candidate["candidate"]))
            if not candidate_path.is_file():
                missing += 1
                continue
            tensor = torch.load(candidate_path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16:
                wrong_dtype += 1
            delta_true = float(candidate["delta_pesq"])
            delta_predicted = -float(candidate["delta_pmsqe"])
            if np.isfinite(delta_true) and np.isfinite(delta_predicted) and abs(delta_true) >= 1e-4:
                rows_by_partition[partition].append(
                    {
                        "delta_true": delta_true,
                        "delta_predicted": delta_predicted,
                        "snr": float(parent["estimated_input_snr_db"]),
                    }
                )
    if missing:
        issues.append(f"missing candidate tensors: {missing}")
    if wrong_dtype:
        issues.append(f"non-FP16 candidate tensors: {wrong_dtype}")

    def summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
        true = np.asarray([row["delta_true"] for row in rows], dtype=np.float64)
        predicted = np.asarray(
            [row["delta_predicted"] for row in rows],
            dtype=np.float64,
        )
        snr = np.asarray([row["snr"] for row in rows], dtype=np.float64)
        agreement = np.sign(true) == np.sign(predicted)
        rho = float(spearmanr(true, predicted).statistic) if len(rows) > 1 else float("nan")
        quartiles: list[dict[str, Any]] = []
        if len(rows):
            edges = np.quantile(snr, [0.25, 0.50, 0.75])
            bins = np.digitize(snr, edges)
            for index in range(4):
                selected = bins == index
                quartiles.append(
                    {
                        "quartile": index + 1,
                        "count": int(selected.sum()),
                        "sign_agreement": float(agreement[selected].mean())
                        if selected.any()
                        else float("nan"),
                    }
                )
        return {
            "eligible_pairs": len(rows),
            "sign_agreement": float(agreement.mean()) if len(rows) else float("nan"),
            "delta_spearman": rho,
            "snr_quartiles": quartiles,
        }

    summaries = {key: summarize(value) for key, value in rows_by_partition.items()}
    audit = summaries["audit"]
    min_quartile = min(
        float(row["sign_agreement"]) for row in audit["snr_quartiles"]
    ) if audit["snr_quartiles"] else float("nan")
    gate_checks = {
        "eligible_pairs_at_least_200": int(audit["eligible_pairs"]) >= 200,
        "sign_agreement_at_least_0_70": float(audit["sign_agreement"]) >= 0.70,
        "delta_spearman_at_least_0_60": float(audit["delta_spearman"]) >= 0.60,
        "every_snr_quartile_at_least_0_55": min_quartile >= 0.55,
        "finite_nonvanishing_waveform_and_parameter_gradients": bool(
            weights.get("status") == "weights_frozen"
            and all(
                np.isfinite(float(value)) and float(value) > 0.0
                for value in dict(weights.get("median_gradient_norms") or {}).values()
            )
            and dict(weights.get("parameter_gradient_evidence") or {}).get(
                "all_finite"
            )
            and int(
                dict(weights.get("parameter_gradient_evidence") or {}).get(
                    "parameter_tensors_with_gradient",
                    0,
                )
            )
            == int(
                dict(weights.get("parameter_gradient_evidence") or {}).get(
                    "parameter_tensors_total",
                    -1,
                )
            )
            and float(
                dict(weights.get("parameter_gradient_evidence") or {}).get(
                    "aggregate_l2_norm",
                    0.0,
                )
            )
            > 0.0
        ),
        "candidate_artifacts_valid": not issues,
    }
    passed = all(gate_checks.values())
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_rows = rows_by_partition["audit"]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        [row["delta_true"] for row in audit_rows],
        [row["delta_predicted"] for row in audit_rows],
        s=8,
        alpha=0.45,
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axvline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("true delta PESQ-WB")
    axis.set_ylabel("predicted improvement (-delta PMSQE)")
    axis.set_title("T3 untouched direction audit")
    figure.tight_layout()
    plot_path = root / "direction_audit.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "valid": not issues,
        "passed": passed,
        "eligible_threshold": 1e-4,
        "candidates_sha256": _sha256(path),
        "weights_sha256": _sha256(weights_path),
        "summaries": summaries,
        "audit_min_snr_quartile_sign_agreement": min_quartile,
        "gate_checks": gate_checks,
        "issues": issues,
        "direction_plot": plot_path.as_posix(),
    }
    _atomic_json(root / "direction_audit.json", result)
    return result
