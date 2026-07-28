#!/usr/bin/env python3
"""Canonical VoiceBank-only MetricGAN+ WB teacher -> WB/NB students campaign."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.bandwidth import PROFILES, resolve_bandwidth  # noqa: E402
from sebench.checkpoints import load_model_from_checkpoint  # noqa: E402
from sebench.data import read_pair_manifest  # noqa: E402
from sebench.metric_proxy_training import (  # noqa: E402
    build_proxy_records,
    train_metric_proxy,
)
from sebench.metricgan_d2 import (  # noqa: E402
    PlannedD2Interruption,
    audit_d2_range_support,
    audit_d2_support,
    fit_d2_official,
    prepare_d2_range_support,
    prepare_d2_support,
)
from sebench.runtime import require_shared_venv, require_training_cuda  # noqa: E402
from sebench.t3_support import (  # noqa: E402
    audit_t3_direction,
    audit_t3_identities,
    calibrate_t3_weights,
    generate_t3_mask_candidates,
    prepare_t3_identities,
)
from sebench.t3_training import run_t3_branch  # noqa: E402
from sebench.t4_calibration import run_t4_logit_bias_scan  # noqa: E402
from sebench.t4_microstep import run_t4_microstep_backtracking  # noqa: E402
from sebench.t5_zeroth_order import run_t5_frequency_search  # noqa: E402
from sebench.teacher_cache import (  # noqa: E402
    TeacherCacheTarget,
    build_multi_target_teacher_cache,
)
from sebench.training import (  # noqa: E402
    ExperimentConfig,
    PlannedTrainingInterruption,
    evaluate_manifest,
    run_experiment,
)


CELL_ORDER = (
    "T0-WB-OFFICIAL",
    "S0-WB",
    "S0-NB",
    "T1-WB-BASE",
    "T1-WB-METRIC",
    "S1-WB",
    "S1-NB",
)

BASELINE_CELL_ORDER = CELL_ORDER[:3]
BASELINE_SCOPE = "official_teacher_students_baseline"
TWO_STAGE_SCOPE = "teacher_improvement_two_stage"
STUDENT_CONTINUATION_CELL_ORDER = ("S0-WB", "S0-NB")
STUDENT_CONTINUATION_SCOPE = "official_student_training_continuation"
CONVERGED_BASELINE_SCOPE = "official_teacher_students_converged_baseline"
TEACHER_CALIBRATION_SCOPE = "teacher_current_output_calibration"
TEACHER_TRIAL_SCOPE = "teacher_only_metric_improvement"
T3_DIRECT_SCOPE = "t3_direct_perceptual_pilot"
TEACHER_CALIBRATION_CELL_ORDER = ("E0-T0", "E0-D-CAL")
TEACHER_TRIAL_CELL_ORDER = ("E0-T0", "E1-CONTROL", "E2-PESQ")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_artifact_path(
    run_root: Path,
    value: Any,
    *,
    default: str | Path | None = None,
) -> Path | None:
    """Resolve portable run-relative artifact references for audit."""
    raw = value if value not in {None, ""} else default
    if raw in {None, ""}:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else run_root / path


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    config = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    unresolved: list[str] = []

    def scan(value: Any, key: str = "") -> None:
        if isinstance(value, str) and "${" in value:
            unresolved.append(f"{key}={value}")
        elif isinstance(value, dict):
            for child_key, child in value.items():
                scan(child, f"{key}.{child_key}".strip("."))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, f"{key}[{index}]")

    scan(config)
    if unresolved:
        raise ValueError(
            "Unresolved campaign environment variables: " + ", ".join(unresolved)
        )
    return config


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_campaign_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = dict(config["dataset"])
    if dataset.get("name") != "VoiceBank+DEMAND":
        raise ValueError("Canonical campaign accepts only VoiceBank+DEMAND.")
    if not bool(dataset.get("source_read_only", False)):
        raise ValueError("dataset.source_read_only must be true.")
    manifests = {
        name: Path(str(dataset[name])).expanduser().resolve()
        for name in ("train_fit", "val_rank", "val_select", "test")
    }
    for name, path in manifests.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} manifest: {path}")
    run_root = Path(str(config["runtime"]["run_root"])).expanduser().resolve()
    manifest_root = Path(str(dataset["manifest_root"])).expanduser().resolve()
    if _is_within(run_root, manifest_root):
        raise ValueError("runtime.run_root must be outside the manifest input root.")
    if str(config["runtime"].get("device", "")).lower() == "cpu":
        raise ValueError("Canonical campaign training is GPU-only.")
    teacher_checkpoint_sha256 = str(
        config.get("model", {}).get("teacher_checkpoint_sha256") or ""
    )
    if (
        len(teacher_checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in teacher_checkpoint_sha256)
    ):
        raise ValueError("model.teacher_checkpoint_sha256 must be canonical SHA-256.")
    cache_root = Path(
        str(config.get("teacher_cache", {}).get("root") or "")
    ).expanduser().resolve()
    if not str(config.get("teacher_cache", {}).get("root") or "").strip():
        raise ValueError("teacher_cache.root is required.")
    if _is_within(cache_root, manifest_root):
        raise ValueError("teacher_cache.root must be outside the manifest input root.")
    if not _is_within(cache_root, run_root.parent):
        raise ValueError(
            "teacher_cache.root must remain inside the Desktop-local runtime area."
        )
    if bool(config["teacher_cache"].get("cache_inputs", True)):
        raise ValueError(
            "Canonical cache must not duplicate noisy/clean dataset inputs."
        )
    if str(config["teacher_cache"].get("storage_dtype")) != "float16":
        raise ValueError("Canonical cache storage_dtype must be float16.")
    training = dict(config["training"])
    if int(training.get("student_epochs", 0)) != 50:
        raise ValueError("Canonical student training must use a 50-epoch ceiling.")
    if int(training.get("metric_discriminator_rows", 0)) < 100:
        raise ValueError(
            "Canonical teacher refresh requires at least 100 training outputs."
        )
    if int(training.get("metric_discriminator_calibration_rows", 0)) < 100:
        raise ValueError(
            "Canonical teacher calibration requires at least 100 held-out outputs."
        )
    if int(training.get("metric_discriminator_min_calibration_records", 0)) < 100:
        raise ValueError(
            "Canonical teacher calibration gate requires at least 100 records."
        )
    if int(training.get("teacher_trial_epochs", 0)) != 10:
        raise ValueError("Canonical teacher-only pilot ceiling must be 10 epochs.")
    if float(training.get("teacher_trial_lr", 0.0)) != 1e-6:
        raise ValueError("Canonical teacher-only pilot LR must be 1e-6.")
    for key, minimum in (
        ("d2_train_rows", 1000),
        ("d2_calibration_rows", 200),
        ("d2_audit_rows", 200),
    ):
        if int(training.get(key, 0)) < minimum:
            raise ValueError(f"training.{key} must be at least {minimum}.")
    if int(training.get("d2_max_epochs", 0)) != 20:
        raise ValueError("training.d2_max_epochs must be 20.")
    if int(training.get("d2_current_rows", 0)) != 100:
        raise ValueError("training.d2_current_rows must be 100.")
    if int(training.get("d2_early_stop_patience", 0)) != 5:
        raise ValueError("training.d2_early_stop_patience must be 5.")
    if int(training.get("d2_lr_patience", 0)) != 2:
        raise ValueError("training.d2_lr_patience must be 2.")
    if float(training.get("d2_lr_factor", 0.0)) != 0.5:
        raise ValueError("training.d2_lr_factor must be 0.5.")
    if float(training.get("d2_min_lr", 0.0)) != 1e-6:
        raise ValueError("training.d2_min_lr must be 1e-6.")
    for key, minimum in (
        ("t3_direction_train_rows", 1000),
        ("t3_direction_calibration_rows", 200),
        ("t3_direction_audit_rows", 200),
    ):
        if int(training.get(key, 0)) < minimum:
            raise ValueError(f"training.{key} must be at least {minimum}.")
    if int(training.get("t3_direction_seed", -1)) != 3003:
        raise ValueError("training.t3_direction_seed must be the frozen seed 3003.")
    if int(training.get("t3_weight_calibration_rows", 0)) != 16:
        raise ValueError("T3 gradient calibration must use 16 train identities.")
    if int(training.get("t3_weight_segment_samples", 0)) != 32000:
        raise ValueError("T3 gradient calibration segments must be 32000 samples.")
    if float(training.get("t3_weight_mask_logit_delta", 0.0)) != 0.02:
        raise ValueError("T3 gradient calibration mask-logit delta must be 0.02.")
    student_lr_patience = int(training.get("student_lr_patience", 0))
    student_early_stop_patience = int(
        training.get("student_early_stop_patience", 0)
    )
    if student_lr_patience < 1:
        raise ValueError("training.student_lr_patience must be positive.")
    if student_early_stop_patience <= student_lr_patience:
        raise ValueError(
            "Student early-stop patience must exceed LR-reduction patience."
        )
    student_lr_factor = float(training.get("student_lr_factor", 0.0))
    if not 0.0 < student_lr_factor < 1.0:
        raise ValueError("training.student_lr_factor must be between 0 and 1.")
    if float(training.get("student_min_lr", 0.0)) <= 0.0:
        raise ValueError("training.student_min_lr must be positive.")

    sets: dict[str, dict[str, set[str]]] = {}
    for name, path in manifests.items():
        rows = read_pair_manifest(path)
        pairs = {f"{row.noisy}|{row.clean}" for row in rows}
        clean = {row.clean.as_posix() for row in rows}
        if len(rows) != len(pairs) or len(rows) != len(clean):
            raise ValueError(f"Duplicate pair/clean identity in {name}: {path}")
        sets[name] = {"pair": pairs, "clean": clean}
    overlaps: dict[str, int] = {}
    names = list(manifests)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for identity in ("pair", "clean"):
                key = f"{left}__{right}_{identity}_overlap"
                overlaps[key] = len(sets[left][identity] & sets[right][identity])
    nonzero = {key: value for key, value in overlaps.items() if value}
    if nonzero:
        raise ValueError(f"Campaign split leakage: {nonzero}")
    return {
        "dataset": "VoiceBank+DEMAND",
        "manifests": {
            name: {
                "path": path.as_posix(),
                "rows": len(read_pair_manifest(path)),
                "sha256": sha256(path),
            }
            for name, path in manifests.items()
        },
        "overlaps": overlaps,
        "teacher_cache_root": cache_root.as_posix(),
        "valid": True,
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_details(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    details: dict[str, Any] = {}
    for key in (
        "bandwidth",
        "checkpoint",
        "checkpoint_out",
        "pesq_mode",
        "sample_rate",
        "summary_path",
        "train_record_count",
        "validation_record_count",
    ):
        if key in result:
            details[key] = result[key]
    for split in ("val_rank_metrics", "val_select_metrics", "test_metrics"):
        metrics = dict(result.get(split) or {})
        if metrics:
            details[split] = {
                key: metrics.get(key)
                for key in (
                    "bandwidth",
                    "sample_rate",
                    "pesq_mode",
                    "count",
                    "pesq_mean",
                    "stoi_mean",
                    "sisdr_mean",
                    "delta_snr_mean",
                )
                if key in metrics
            }
    if "validation" in result:
        details["validation"] = result["validation"]
    return details


def _mark_stage(
    run_root: Path,
    *,
    stage: str,
    status: str,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    progress_path = run_root / "tracking" / "campaign_progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {
            "schema_version": 1,
            "run_id": run_root.name,
            "started_utc": _utc_now(),
            "stages": {},
        }
    stage_payload = dict(progress["stages"].get(stage) or {})
    if status == "running":
        stage_payload["started_utc"] = _utc_now()
    if status in {"completed", "failed"}:
        stage_payload["finished_utc"] = _utc_now()
    stage_payload["status"] = status
    if details:
        stage_payload["details"] = details
    if error:
        stage_payload["error"] = error
    progress["stages"][stage] = stage_payload
    progress["current_stage"] = stage
    progress["updated_utc"] = _utc_now()
    if status == "failed":
        progress["status"] = "failed"
    elif status == "completed":
        progress["completed_stage_count"] = sum(
            item.get("status") == "completed"
            for item in progress["stages"].values()
        )
    else:
        progress["status"] = "running"
    _atomic_json(progress_path, progress)

    status_path = run_root / "status.json"
    current = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {}
    )
    current.update(
        {
            "status": "failed" if status == "failed" else "running",
            "current_stage": stage,
            "stage_status": status,
            "updated_utc": progress["updated_utc"],
            "valid_for_promotion": False,
        }
    )
    if error:
        current["error"] = error
    _atomic_json(status_path, current)


def monitor_campaign_run(run_dir: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser().resolve()
    status_path = run_root / "status.json"
    progress_path = run_root / "tracking" / "campaign_progress.json"
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_root}")
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {"status": "unknown"}
    )
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {"stages": {}}
    )
    cells: dict[str, Any] = {}
    cells_root = run_root / "cells"
    if cells_root.is_dir():
        for cell_root in sorted(path for path in cells_root.iterdir() if path.is_dir()):
            cell_progress_path = cell_root / "progress.json"
            cell_summary_path = cell_root / "summary.json"
            cells[cell_root.name] = {
                "progress": (
                    json.loads(cell_progress_path.read_text(encoding="utf-8"))
                    if cell_progress_path.is_file()
                    else None
                ),
                "summary": (
                    _stage_details(
                        json.loads(cell_summary_path.read_text(encoding="utf-8"))
                    )
                    if cell_summary_path.is_file()
                    else None
                ),
            }
    return {
        "run_id": run_root.name,
        "status": status,
        "current_stage": progress.get("current_stage"),
        "completed_stage_count": progress.get("completed_stage_count", 0),
        "stages": progress.get("stages", {}),
        "cells": cells,
    }


def _effective_training(config: dict[str, Any], mode: str) -> dict[str, Any]:
    values = dict(config["training"])
    if mode != "full":
        values.update(dict(config.get(mode) or {}))
    return values


def _student_schedule(
    effective: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    early_stop_patience = (
        int(effective["student_early_stop_patience"])
        if mode == "full"
        else int(
            effective.get(
                "early_stop_patience",
                0 if mode == "smoke" else 2,
            )
        )
    )
    return {
        "early_stop_patience": early_stop_patience,
        "lr_factor": float(effective["student_lr_factor"]),
        "lr_patience": int(effective["student_lr_patience"]),
        "min_lr": float(effective["student_min_lr"]),
    }


def _experiment_config(
    config: dict[str, Any],
    *,
    run_root: Path,
    cell: str,
    family: str,
    bandwidth: str,
    loss_recipe: str,
    epochs: int,
    lr: float,
    seed: int,
    init_checkpoint: str | None = None,
    proxy_checkpoint: str | None = None,
    teacher_cache_manifest: str | None = None,
    include_test: bool = True,
    evaluate_init_checkpoint: bool = False,
    frontend: dict[str, int] | None = None,
    alternating_metric_discriminator: bool = False,
    metric_discriminator_calibration_only: bool = False,
    resume_training_state: str | None = None,
    early_stop_patience: int | None = None,
    lr_factor: float | None = None,
    lr_patience: int | None = None,
    min_lr: float | None = None,
    mode: str,
) -> ExperimentConfig:
    profile = resolve_bandwidth(bandwidth)
    model_frontend = dict(frontend or profile.as_dict())
    effective = _effective_training(config, mode)
    evaluation = dict(config["evaluation"])
    if mode != "full":
        evaluation.update(dict(config.get(mode) or {}))
    dataset = dict(config["dataset"])
    cell_root = run_root / "cells" / cell
    cell_root.mkdir(parents=True, exist_ok=False)
    return ExperimentConfig(
        train_csv=str(dataset["train_fit"]),
        val_rank_csv=str(dataset["val_rank"]),
        val_select_csv=str(dataset["val_select"]),
        test_csv=str(dataset["test"]) if include_test else None,
        checkpoint_out=(cell_root / "model.pt").as_posix(),
        training_state_out=(cell_root / "training_state.pt").as_posix(),
        progress_json_out=(cell_root / "progress.json").as_posix(),
        model_family=family,
        variant=str(config["model"]["variant"]),
        loss_recipe=loss_recipe,
        run_name=cell,
        phase=f"verification_{mode}" if mode != "full" else "canonical_campaign",
        epochs=int(epochs),
        batch_size=int(effective["batch_size"]),
        grad_accum=1,
        lr=float(lr),
        segment_len=profile.sample_rate * 2,
        num_workers=int(effective["num_workers"]),
        prefetch_factor=2,
        persistent_workers=int(effective["num_workers"]) > 0,
        pin_memory=True,
        checkpoint_every_steps=0,
        checkpoint_every_minutes=0,
        checkpoint_keep_last=1,
        history_plot_every_epochs=1,
        history_plot_final_only=False,
        record_step_history=False,
        lr_factor=float(0.5 if lr_factor is None else lr_factor),
        lr_patience=int(2 if lr_patience is None else lr_patience),
        min_lr=float(1e-6 if min_lr is None else min_lr),
        early_stop_patience=int(
            (
                effective.get(
                    "early_stop_patience",
                    0 if mode == "smoke" else 5,
                )
                if early_stop_patience is None
                else early_stop_patience
            )
        ),
        min_epochs=1,
        eval_every=1,
        rank_eval_every=1,
        select_eval_every=1,
        grad_clip=float(effective["grad_clip"]),
        seed=int(seed),
        amp=bool(effective["amp"]),
        scheduler="plateau",
        device=str(config["runtime"]["device"]),
        mlflow_uri=(run_root / "tracking").as_posix(),
        mlflow_artifact_root=(run_root / "tracking" / "artifacts").as_posix(),
        experiment_name=f"voicebank_{run_root.name}",
        selection_metric="val_select/pesq_mean",
        evaluate_init_checkpoint=bool(evaluate_init_checkpoint),
        pesq_proxy_checkpoint=proxy_checkpoint,
        metric_proxy_weight=float(effective["metric_proxy_weight"]),
        teacher_anchor_weight=float(effective["teacher_anchor_weight"]),
        metric_discriminator_mode=(
            "alternating" if alternating_metric_discriminator else "frozen"
        ),
        metric_discriminator_lr=float(effective["metric_discriminator_lr"]),
        metric_discriminator_rows=int(effective["metric_discriminator_rows"]),
        metric_discriminator_calibration_rows=int(
            effective["metric_discriminator_calibration_rows"]
        ),
        metric_discriminator_history_portion=float(
            effective["metric_discriminator_history_portion"]
        ),
        metric_discriminator_replay_root=(
            (cell_root / "metricgan_replay").as_posix()
            if alternating_metric_discriminator
            else None
        ),
        metric_discriminator_calibration_only=bool(
            metric_discriminator_calibration_only
        ),
        metric_discriminator_min_calibration_records=int(
            effective["metric_discriminator_min_calibration_records"]
        ),
        metric_discriminator_max_normalized_mae=float(
            effective["metric_discriminator_max_normalized_mae"]
        ),
        metric_discriminator_min_pearson=float(
            effective["metric_discriminator_min_pearson"]
        ),
        metric_discriminator_min_spearman=float(
            effective["metric_discriminator_min_spearman"]
        ),
        metric_discriminator_min_prediction_std=float(
            effective["metric_discriminator_min_prediction_std"]
        ),
        metric_discriminator_range_tolerance_raw=float(
            effective["metric_discriminator_range_tolerance_raw"]
        ),
        eval_dnsmos=False,
        sample_count=int(evaluation["sample_count"]),
        benchmark_seconds=int(evaluation["benchmark_seconds"]),
        benchmark_repeats=int(evaluation["benchmark_repeats"]),
        eval_batch_size=int(evaluation["eval_batch_size"]),
        cache_eval_audio=True,
        rank_compute_composite=False,
        select_compute_composite=bool(evaluation["compute_composite"]),
        teacher_cache_manifest=teacher_cache_manifest,
        guidance_classic="none",
        erb_bands=int(config["model"]["erb_bands"]),
        context_frames=5,
        init_checkpoint=init_checkpoint,
        resume_training_state=resume_training_state,
        sample_rate=int(model_frontend["sample_rate"]),
        bandwidth=profile.name,
        n_fft=int(model_frontend["n_fft"]),
        hop_length=int(model_frontend["hop_length"]),
        win_length=int(model_frontend["win_length"]),
        log_torch_model=False,
        log_system_metrics=False,
    )


def _run_cell(**kwargs: Any) -> dict[str, Any]:
    run_root = Path(kwargs["run_root"])
    stage = str(kwargs["cell"])
    _mark_stage(run_root, stage=stage, status="running")
    try:
        experiment = _experiment_config(**kwargs)
        summary = run_experiment(experiment)
        _atomic_json(Path(experiment.checkpoint_out).parent / "summary.json", summary)
        _mark_stage(
            run_root,
            stage=stage,
            status="completed",
            details=_stage_details(summary),
        )
        return summary
    except BaseException as exc:
        _mark_stage(
            run_root,
            stage=stage,
            status="failed",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        raise


def _teacher_cache_identity(
    *,
    checkpoint_hash: str,
    manifest_hash: str,
    cache_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    contract = {
        "schema_version": 1,
        "teacher_sample_rate": 16_000,
        "targets": {
            "wb": {"sample_rate": 16_000, "erb_bands": 32},
            "nb": {"sample_rate": 8_000, "erb_bands": 32},
        },
        "cache_inputs": bool(cache_config["cache_inputs"]),
        "storage_dtype": str(cache_config["storage_dtype"]),
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    key = (
        f"teacher-{checkpoint_hash[:16]}-{manifest_hash[:16]}-"
        f"{contract_hash[:8]}"
    )
    return key, contract


def _build_cache(
    config: dict[str, Any],
    *,
    run_root: Path,
    teacher_checkpoint: str,
    cache_label: str,
    mode: str,
) -> dict[str, str]:
    stage = f"TEACHER-CACHE-{cache_label.upper()}-WB-NB"
    _mark_stage(run_root, stage=stage, status="running")
    try:
        teacher, _ = load_model_from_checkpoint(
            teacher_checkpoint,
            device=str(config["runtime"]["device"]),
        )
        effective = _effective_training(config, mode)
        checkpoint_hash = sha256(teacher_checkpoint)
        manifest_hash = sha256(str(config["dataset"]["train_fit"]))
        cache_config = dict(config["teacher_cache"])
        cache_key, cache_contract = _teacher_cache_identity(
            checkpoint_hash=checkpoint_hash,
            manifest_hash=manifest_hash,
            cache_config=cache_config,
        )
        cache_root = (
            Path(str(cache_config["root"])).expanduser().resolve()
            / cache_key
        )
        result = build_multi_target_teacher_cache(
            str(config["dataset"]["train_fit"]),
            teacher,
            out_dir=cache_root,
            device=str(config["runtime"]["device"]),
            teacher_sample_rate=16_000,
            targets=[
                TeacherCacheTarget(name="wb", sample_rate=16_000, erb_bands=32),
                TeacherCacheTarget(name="nb", sample_rate=8_000, erb_bands=32),
            ],
            batch_size=int(effective["batch_size"]),
            num_workers=int(effective["num_workers"]),
            pin_memory=True,
            persistent_workers=int(effective["num_workers"]) > 0,
            prefetch_factor=2,
            write_workers=0,
            resume=True,
            validate_existing=True,
            cache_inputs=bool(cache_config["cache_inputs"]),
            storage_dtype=str(cache_config["storage_dtype"]),
        )
        existing_metadata_path = cache_root / "cache_metadata.json"
        existing_metadata = (
            json.loads(existing_metadata_path.read_text(encoding="utf-8"))
            if existing_metadata_path.is_file()
            else {}
        )
        cache_labels = sorted(
            set(existing_metadata.get("cache_labels") or []) | {cache_label}
        )
        metadata = {
            **cache_contract,
            "cache_labels": cache_labels,
            "cache_key": cache_key,
            "teacher_checkpoint_sha256": checkpoint_hash,
            "train_manifest_sha256": manifest_hash,
            "manifests": result,
            "status": "complete",
        }
        _atomic_json(existing_metadata_path, metadata)
        _mark_stage(
            run_root,
            stage=stage,
            status="completed",
            details=metadata,
        )
        return result
    except BaseException as exc:
        _mark_stage(
            run_root,
            stage=stage,
            status="failed",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        raise


def _proxy(
    config: dict[str, Any],
    *,
    run_root: Path,
    bandwidth: str,
    candidate_teacher_checkpoint: str,
    mode: str,
) -> dict[str, Any]:
    stage = f"PROXY-{bandwidth.upper()}"
    _mark_stage(run_root, stage=stage, status="running")
    try:
        effective = _effective_training(config, mode)
        proxy_root = run_root / "metric_proxies" / bandwidth
        records = build_proxy_records(
            train_manifest=str(config["dataset"]["train_fit"]),
            validation_manifest=str(config["dataset"]["val_rank"]),
            output_dir=proxy_root / "records",
            bandwidth=bandwidth,
            candidate_teacher_checkpoint=candidate_teacher_checkpoint,
            max_train_rows=int(effective["proxy_train_rows"]),
            max_validation_rows=int(effective["proxy_validation_rows"]),
            device=str(config["runtime"]["device"]),
            seed=int(effective["seed"]),
        )
        result = train_metric_proxy(
            records,
            output_dir=proxy_root,
            device=str(config["runtime"]["device"]),
            epochs=int(effective["proxy_epochs"]),
            batch_size=int(effective["proxy_batch_size"]),
            lr=float(effective["proxy_lr"]),
            seed=int(effective["seed"]),
            model_kind="speechbrain_metric_discriminator",
        )
        _mark_stage(
            run_root,
            stage=stage,
            status="completed",
            details=_stage_details(result),
        )
        return result
    except BaseException as exc:
        _mark_stage(
            run_root,
            stage=stage,
            status="failed",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        raise


def _best_teacher(
    official: dict[str, Any],
    baseline: dict[str, Any],
    metric: dict[str, Any],
    *,
    config: dict[str, Any],
    verification_only: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    candidates = {"T1-WB-BASE": baseline, "T1-WB-METRIC": metric}
    name = max(
        candidates,
        key=lambda key: float(
            candidates[key].get("val_select_metrics", {}).get("pesq_mean")
            or float("-inf")
        ),
    )
    selected = candidates[name]
    official_metrics = dict(official.get("val_select_metrics") or {})
    selected_metrics = dict(selected.get("val_select_metrics") or {})
    training = dict(config["training"])
    deltas = {
        metric_name: float(selected_metrics.get(metric_name, float("nan")))
        - float(official_metrics.get(metric_name, float("nan")))
        for metric_name in (
            "pesq_mean",
            "stoi_mean",
            "sisdr_mean",
            "delta_snr_mean",
        )
    }
    checks = {
        "pesq_gain": deltas["pesq_mean"]
        >= float(training["teacher_min_pesq_gain"]),
        "stoi_guardrail": deltas["stoi_mean"]
        >= -float(training["teacher_max_stoi_drop"]),
        "sisdr_guardrail": deltas["sisdr_mean"]
        >= -float(training["teacher_max_sisdr_drop"]),
    }
    gate_passed = all(checks.values())
    gate = {
        "selected_candidate": name,
        "official_teacher": "T0-WB-OFFICIAL",
        "val_select_deltas": deltas,
        "checks": checks,
        "passed": gate_passed,
        "verification_override": bool(verification_only and not gate_passed),
    }
    if not gate_passed and not verification_only:
        raise RuntimeError(
            "No fine-tuned teacher passed the predeclared improvement gate: "
            f"{gate}"
        )
    if not gate_passed:
        gate["downstream_teacher"] = "T0-WB-OFFICIAL"
        return "T0-WB-OFFICIAL", official, gate
    gate["downstream_teacher"] = name
    return name, selected, gate


def _metric_rows(
    cells: dict[str, dict[str, Any]],
    *,
    cell_order: tuple[str, ...] = CELL_ORDER,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cell_order:
        summary = cells[cell]
        bandwidth = "nb" if "-NB" in cell else "wb"
        for split_key, split_label in (
            ("val_rank_metrics", "val_rank"),
            ("val_select_metrics", "val_select"),
            ("test_metrics", "test"),
        ):
            for metric, value in dict(summary.get(split_key) or {}).items():
                if isinstance(value, (int, float, str)) and metric != "sample_paths":
                    rows.append(
                        {
                            "cell": cell,
                            "bandwidth": bandwidth,
                            "split": split_label,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return rows


def _write_report(
    *,
    run_root: Path,
    cells: dict[str, dict[str, Any]],
    proxies: dict[str, dict[str, Any]],
    selected_teacher: str,
    teacher_gate: dict[str, Any] | None,
    mode: str,
    verification_only: bool,
    campaign_scope: str = TWO_STAGE_SCOPE,
    cell_order: tuple[str, ...] = CELL_ORDER,
    comparison_pairs: dict[str, tuple[str, str]] | None = None,
    baseline_contract: dict[str, Any] | None = None,
    student_continuation_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics_dir = run_root / "metrics"
    reports_dir = run_root / "reports"
    models_dir = run_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = _metric_rows(cells, cell_order=cell_order)
    csv_path = metrics_dir / "canonical_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cell", "bandwidth", "split", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    deltas: dict[str, dict[str, float]] = {}
    pairs = comparison_pairs
    if pairs is None:
        pairs = {
            "teacher_control_vs_official": (
                "T0-WB-OFFICIAL",
                "T1-WB-BASE",
            ),
            "teacher_metric_vs_official": (
                "T0-WB-OFFICIAL",
                "T1-WB-METRIC",
            ),
            "student_wb_after_teacher_upgrade": ("S0-WB", "S1-WB"),
            "student_nb_after_teacher_upgrade": ("S0-NB", "S1-NB"),
        }
    for label, (baseline, metric) in pairs.items():
        baseline_metrics = dict(cells[baseline].get("test_metrics") or {})
        metric_metrics = dict(cells[metric].get("test_metrics") or {})
        deltas[label] = {
            name: float(metric_metrics[name]) - float(baseline_metrics[name])
            for name in ("pesq_mean", "stoi_mean", "sisdr_mean", "delta_snr_mean")
            if name in baseline_metrics and name in metric_metrics
        }

    figure, axis = plt.subplots(figsize=(9, 4.5))
    pesq = [
        float(cells[cell].get("test_metrics", {}).get("pesq_mean", float("nan")))
        for cell in cell_order
    ]
    axis.bar(cell_order, pesq)
    axis.set_ylabel("PESQ")
    axis.set_title("VoiceBank+DEMAND campaign: profile-matched test PESQ")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure_path = reports_dir / "test_pesq_by_cell.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    model_inventory: dict[str, Any] = {}
    for cell, summary in cells.items():
        checkpoint = Path(str(summary["checkpoint_out"]))
        target = models_dir / f"{cell}.pt"
        shutil.copy2(checkpoint, target)
        model_inventory[cell] = {
            "path": target.as_posix(),
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
        }

    summary = {
        "schema_version": 1,
        "dataset": "VoiceBank+DEMAND",
        "campaign_scope": campaign_scope,
        "expected_cells": list(cell_order),
        "verification_only": verification_only,
        "selected_teacher": selected_teacher,
        "cells": cells,
        "metric_proxies": proxies,
        "paired_deltas": deltas,
        "model_inventory": model_inventory,
        "canonical_metrics_csv": csv_path.as_posix(),
        "test_pesq_plot": figure_path.as_posix(),
    }
    if teacher_gate is not None:
        summary["teacher_promotion_gate"] = teacher_gate
    if baseline_contract is not None:
        summary["baseline_contract"] = baseline_contract
    if student_continuation_contract is not None:
        summary["student_continuation_contract"] = student_continuation_contract

    lines = [
        "# VoiceBank MetricGAN+ campaign report",
        "",
        f"Status: {'verification-only ' + mode if verification_only else 'evaluated campaign'}",
        "",
        (
            "Official T0 checkpoint distilled into fresh WB and NB students."
            if campaign_scope == BASELINE_SCOPE
            else (
                "WB and NB students continued from immutable optimizer states."
                if campaign_scope == STUDENT_CONTINUATION_SCOPE
                else f"Selected teacher by val_select PESQ: `{selected_teacher}`."
            )
        ),
        "",
    ]
    if deltas:
        lines.extend(["## Paired stage deltas", ""])
        for label, values in deltas.items():
            rendered = ", ".join(
                f"{key}={value:+.4f}" for key, value in values.items()
            )
            lines.append(f"- {label}: {rendered or 'no comparable metrics'}")
    lines.extend(
        [
            "",
            "WB and NB PESQ are reported under separate protocols and are not pooled.",
            "Verification-only results are not publication evidence.",
            "",
        ]
    )
    report_path = reports_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary["report"] = report_path.as_posix()
    _atomic_json(metrics_dir / "campaign_summary.json", summary)
    return summary


def audit_campaign_run(run_dir: str | Path) -> dict[str, Any]:
    """Independently reconcile a completed campaign package."""
    run_root = Path(run_dir).expanduser().resolve()
    summary_path = run_root / "metrics" / "campaign_summary.json"
    status_path = run_root / "status.json"
    provenance_path = run_root / "provenance" / "provenance.json"
    issues: list[str] = []

    def load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            issues.append(f"missing required JSON: {path.relative_to(run_root)}")
            return {}
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"invalid JSON {path.relative_to(run_root)}: {exc}")
            return {}

    summary = load_json(summary_path)
    status = load_json(status_path)
    provenance = load_json(provenance_path)
    manifest_path = run_root / "import_manifest.json"
    if manifest_path.is_file():
        artifact_manifest = load_json(manifest_path)
        declared_artifacts: set[str] = set()
        for item in artifact_manifest.get("files") or []:
            entry = dict(item or {})
            relative = Path(str(entry.get("path") or ""))
            if (
                not relative.as_posix()
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                issues.append(f"invalid artifact-manifest path: {relative}")
                continue
            declared_artifacts.add(relative.as_posix())
            artifact = run_root / relative
            if not artifact.is_file():
                issues.append(f"missing manifested artifact: {relative}")
                continue
            if int(entry.get("bytes") or -1) != artifact.stat().st_size:
                issues.append(f"manifested artifact size mismatch: {relative}")
            if str(entry.get("sha256") or "") != sha256(artifact):
                issues.append(f"manifested artifact hash mismatch: {relative}")
        actual_artifacts = {
            path.relative_to(run_root).as_posix()
            for path in run_root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if declared_artifacts != actual_artifacts:
            missing = sorted(actual_artifacts - declared_artifacts)
            extra = sorted(declared_artifacts - actual_artifacts)
            issues.append(
                "artifact-manifest inventory mismatch: "
                f"unlisted={missing} nonexistent={extra}"
            )
    campaign_scope = str(
        summary.get("campaign_scope") or TWO_STAGE_SCOPE
    )
    if campaign_scope in {BASELINE_SCOPE, CONVERGED_BASELINE_SCOPE}:
        canonical_cells = BASELINE_CELL_ORDER
    elif campaign_scope == STUDENT_CONTINUATION_SCOPE:
        canonical_cells = STUDENT_CONTINUATION_CELL_ORDER
    elif campaign_scope == TEACHER_CALIBRATION_SCOPE:
        canonical_cells = TEACHER_CALIBRATION_CELL_ORDER
    elif campaign_scope == TEACHER_TRIAL_SCOPE:
        canonical_cells = TEACHER_TRIAL_CELL_ORDER
    else:
        canonical_cells = CELL_ORDER
    declared_cells = tuple(summary.get("expected_cells") or canonical_cells)
    if declared_cells != canonical_cells:
        issues.append(
            f"declared cell order mismatch for {campaign_scope}: "
            f"expected={list(canonical_cells)} actual={list(declared_cells)}"
        )
    expected_cells = canonical_cells
    cells = dict(summary.get("cells") or {})
    if set(cells) != set(expected_cells):
        issues.append(
            f"cell set mismatch: expected={sorted(expected_cells)} "
            f"actual={sorted(cells)}"
        )

    csv_values: dict[tuple[str, str, str], str] = {}
    csv_path = _run_artifact_path(
        run_root,
        summary.get("canonical_metrics_csv"),
    )
    if csv_path is None or not csv_path.is_file():
        issues.append("missing canonical metrics CSV")
    else:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                csv_values[(row["cell"], row["split"], row["metric"])] = row["value"]

    reported_samples: set[Path] = set()
    audited_splits = (
        (
            ("val_rank_metrics", "val_rank"),
            ("val_select_metrics", "val_select"),
        )
        if campaign_scope
        in {TEACHER_CALIBRATION_SCOPE, TEACHER_TRIAL_SCOPE}
        else (
            ("val_rank_metrics", "val_rank"),
            ("val_select_metrics", "val_select"),
            ("test_metrics", "test"),
        )
    )
    for cell in expected_cells:
        payload = dict(cells.get(cell) or {})
        bandwidth = "nb" if "-NB" in cell else "wb"
        profile = PROFILES[bandwidth]
        for split_key, split_label in audited_splits:
            metrics = dict(payload.get(split_key) or {})
            expected_metadata = {
                "bandwidth": bandwidth,
                "reference_bandwidth": bandwidth,
                "sample_rate": profile.sample_rate,
                "pesq_mode": profile.pesq_mode,
            }
            for key, expected in expected_metadata.items():
                if metrics.get(key) != expected:
                    issues.append(
                        f"{cell}/{split_label}: {key}={metrics.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            for key, value in metrics.items():
                if key == "sample_paths":
                    for raw_path in value or []:
                        sample = _run_artifact_path(run_root, raw_path)
                        if sample is None:
                            issues.append(
                                f"invalid reported sample: {cell}/{split_label}"
                            )
                            continue
                        reported_samples.add(sample)
                        if not sample.is_file():
                            issues.append(f"missing reported sample: {sample}")
                    continue
                observed = csv_values.get((cell, split_label, key))
                if observed is None:
                    issues.append(f"missing CSV metric: {cell}/{split_label}/{key}")
                elif isinstance(value, (int, float)):
                    try:
                        if abs(float(observed) - float(value)) > 1e-9:
                            issues.append(
                                f"CSV mismatch: {cell}/{split_label}/{key}"
                            )
                    except ValueError:
                        issues.append(
                            f"non-numeric CSV value: {cell}/{split_label}/{key}"
                        )
                elif str(observed) != str(value):
                    issues.append(f"CSV mismatch: {cell}/{split_label}/{key}")

    model_inventory = dict(summary.get("model_inventory") or {})
    for cell in expected_cells:
        model = dict(model_inventory.get(cell) or {})
        path = _run_artifact_path(run_root, model.get("path"))
        if path is None or not path.is_file():
            issues.append(f"missing model: {cell}")
            continue
        if int(model.get("bytes") or -1) != path.stat().st_size:
            issues.append(f"model size mismatch: {cell}")
        if str(model.get("sha256") or "") != sha256(path):
            issues.append(f"model hash mismatch: {cell}")

    report_defaults = {
        "test_pesq_plot": run_root / "reports" / "test_pesq_by_cell.png",
        "report": run_root / "reports" / "report.md",
    }
    for path_key, default_path in report_defaults.items():
        path = _run_artifact_path(
            run_root,
            summary.get(path_key),
            default=default_path,
        )
        if path is None or not path.is_file():
            issues.append(f"missing report artifact: {path_key}")

    verification_only = bool(summary.get("verification_only"))
    if bool(provenance.get("verification_only")) != verification_only:
        issues.append("verification_only mismatch between summary and provenance")
    if verification_only and bool(status.get("valid_for_promotion")):
        issues.append("verification-only run incorrectly marked promotable")
    teacher_gate = dict(summary.get("teacher_promotion_gate") or {})
    if campaign_scope in {BASELINE_SCOPE, CONVERGED_BASELINE_SCOPE}:
        baseline_contract = dict(summary.get("baseline_contract") or {})
        if summary.get("selected_teacher") != "T0-WB-OFFICIAL":
            issues.append("baseline selected teacher is not T0-WB-OFFICIAL")
        if not bool(baseline_contract.get("passed")):
            issues.append("official baseline contract did not pass")
        if set(summary.get("metric_proxies") or {}) != set():
            issues.append("baseline package must not contain metric proxies")
        if baseline_contract.get("students") != ["S0-WB", "S0-NB"]:
            issues.append("baseline student contract mismatch")
        teacher_model = dict(
            model_inventory.get("T0-WB-OFFICIAL") or {}
        )
        if (
            baseline_contract.get("teacher_checkpoint_sha256")
            != teacher_model.get("sha256")
        ):
            issues.append("baseline teacher checkpoint hash mismatch")
        if campaign_scope == CONVERGED_BASELINE_SCOPE:
            closure = dict(summary.get("baseline_closure_contract") or {})
            if not bool(closure.get("passed")):
                issues.append("converged baseline closure contract did not pass")
            if closure.get("source_cells") != {
                "T0-WB-OFFICIAL": "baseline",
                "S0-WB": "continuation",
                "S0-NB": "continuation",
            }:
                issues.append("converged baseline source-cell mapping mismatch")
            source_models = dict(closure.get("source_model_sha256") or {})
            for cell in BASELINE_CELL_ORDER:
                if source_models.get(cell) != dict(
                    model_inventory.get(cell) or {}
                ).get("sha256"):
                    issues.append(
                        f"converged baseline source model hash mismatch: {cell}"
                    )
            epoch20 = dict(closure.get("epoch20_val_select_pesq") or {})
            converged = dict(closure.get("converged_val_select_pesq") or {})
            declared_deltas = dict(closure.get("val_select_pesq_delta") or {})
            for cell in STUDENT_CONTINUATION_CELL_ORDER:
                try:
                    observed = float(converged[cell]) - float(epoch20[cell])
                    if abs(observed - float(declared_deltas[cell])) > 1e-9:
                        issues.append(
                            f"converged baseline delta mismatch: {cell}"
                        )
                    final_score = float(
                        dict(cells[cell].get("val_select_metrics") or {})[
                            "pesq_mean"
                        ]
                    )
                    if abs(final_score - float(converged[cell])) > 1e-9:
                        issues.append(
                            f"converged baseline final-score mismatch: {cell}"
                        )
                except (KeyError, TypeError, ValueError):
                    issues.append(
                        f"invalid converged baseline metric binding: {cell}"
                    )
            for artifact_key in (
                "convergence_plot",
                "epoch_comparison_csv",
            ):
                artifact_path = _run_artifact_path(
                    run_root,
                    summary.get(artifact_key),
                )
                if artifact_path is None or not artifact_path.is_file():
                    issues.append(
                        f"missing converged baseline artifact: {artifact_key}"
                    )
    elif campaign_scope == STUDENT_CONTINUATION_SCOPE:
        continuation = dict(summary.get("student_continuation_contract") or {})
        if continuation.get("students") != list(STUDENT_CONTINUATION_CELL_ORDER):
            issues.append("student continuation cell contract mismatch")
        if int(continuation.get("max_epochs") or 0) != 50:
            issues.append("student continuation ceiling is not 50 epochs")
        schedule = dict(continuation.get("schedule") or {})
        if schedule != {
            "early_stop_patience": 8,
            "lr_factor": 0.5,
            "lr_patience": 2,
            "min_lr": 1e-6,
            "scheduler": "plateau",
        }:
            issues.append("student continuation schedule contract mismatch")
        sources = dict(continuation.get("sources") or {})
        for cell in STUDENT_CONTINUATION_CELL_ORDER:
            source = dict(sources.get(cell) or {})
            state_path = Path(str(source.get("training_state") or ""))
            model_path = Path(str(source.get("model") or ""))
            if not state_path.is_file():
                issues.append(f"missing continuation source state: {cell}")
            elif source.get("training_state_sha256") != sha256(state_path):
                issues.append(f"continuation source state hash mismatch: {cell}")
            if not model_path.is_file():
                issues.append(f"missing continuation source model: {cell}")
            elif source.get("model_sha256") != sha256(model_path):
                issues.append(f"continuation source model hash mismatch: {cell}")
            cell_summary = dict(cells.get(cell) or {})
            source_epoch = int(source.get("epoch") or 0)
            if int(cell_summary.get("stop_epoch") or 0) <= source_epoch:
                issues.append(f"continuation did not advance beyond source: {cell}")
            if int(cell_summary.get("stop_epoch") or 0) > 50:
                issues.append(f"continuation exceeded epoch ceiling: {cell}")
    elif campaign_scope == TEACHER_CALIBRATION_SCOPE:
        calibration_gate = dict(summary.get("teacher_calibration_gate") or {})
        if not calibration_gate:
            issues.append("missing teacher calibration gate")
        calibration_cell = dict(cells.get("E0-D-CAL") or {})
        refreshes = list(
            calibration_cell.get("metric_discriminator_refresh_history") or []
        )
        declared_refreshes = summary.get(
            "teacher_calibration_refresh_count"
        )
        if not refreshes:
            issues.append("calibration package contains no discriminator refresh")
        elif (
            declared_refreshes is not None
            and int(declared_refreshes) != len(refreshes)
        ):
            issues.append("teacher calibration refresh-count mismatch")
        elif dict(refreshes[-1].get("calibration_gate") or {}) != calibration_gate:
            issues.append("teacher calibration gate binding mismatch")
        if int(
            calibration_cell.get("metric_discriminator_accepted_update_count")
            or 0
        ) != 0:
            issues.append("calibration-only run updated the generator")
    elif campaign_scope == TEACHER_TRIAL_SCOPE:
        if not teacher_gate:
            issues.append("missing teacher promotion gate")
        for cell in TEACHER_TRIAL_CELL_ORDER:
            payload = dict(cells.get(cell) or {})
            for split_key in ("val_rank_metrics", "val_select_metrics"):
                if not dict(payload.get(split_key) or {}):
                    issues.append(f"missing teacher trial split: {cell}/{split_key}")
        metric_cell = dict(cells.get("E2-PESQ") or {})
        refreshes = list(
            metric_cell.get("metric_discriminator_refresh_history") or []
        )
        accepted = int(
            metric_cell.get("metric_discriminator_accepted_update_count") or 0
        )
        observed_accepted = sum(
            bool(dict(item.get("calibration_gate") or {}).get("passed"))
            for item in refreshes
        )
        if accepted != observed_accepted:
            issues.append("teacher accepted-update count mismatch")
        if bool(teacher_gate.get("passed")) and not all(
            bool(dict(item.get("calibration_gate") or {}).get("passed"))
            for item in refreshes
        ):
            issues.append("promoted teacher contains an uncalibrated G epoch")
    else:
        if not teacher_gate:
            issues.append("missing teacher promotion gate")
        elif not verification_only and not bool(teacher_gate.get("passed")):
            issues.append("full run teacher promotion gate did not pass")

    result = {
        "schema_version": 1,
        "run_id": provenance.get("run_id"),
        "campaign_scope": campaign_scope,
        "valid": not issues,
        "verification_only": verification_only,
        "cell_count": len(cells),
        "reported_sample_count": len(reported_samples),
        "model_count": len(model_inventory),
        "issues": issues,
    }
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(reports_dir / "audit.json", result)
    return result


def _finish_run(
    *,
    run_root: Path,
    provenance: dict[str, Any],
    report: dict[str, Any],
    mode: str,
    git: dict[str, Any],
    promotion_gate_passed: bool,
) -> dict[str, Any]:
    provenance["status"] = "evaluated"
    provenance["selected_teacher"] = report["selected_teacher"]
    if "teacher_promotion_gate" in report:
        provenance["teacher_promotion_gate"] = report["teacher_promotion_gate"]
    if "baseline_contract" in report:
        provenance["baseline_contract"] = report["baseline_contract"]
    if "student_continuation_contract" in report:
        provenance["student_continuation_contract"] = report[
            "student_continuation_contract"
        ]
    provenance["report"] = report["report"]
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": "evaluated",
            "campaign_mode": mode,
            "campaign_scope": report["campaign_scope"],
            "current_stage": "AUDIT",
            "valid_for_promotion": False,
        },
    )
    _mark_stage(run_root, stage="AUDIT", status="running")
    package_audit = audit_campaign_run(run_root)
    _mark_stage(
        run_root,
        stage="AUDIT",
        status="completed" if package_audit["valid"] else "failed",
        details=_stage_details(package_audit),
        error=None if package_audit["valid"] else str(package_audit["issues"]),
    )
    if not package_audit["valid"]:
        raise RuntimeError(
            f"Campaign package audit failed: {package_audit['issues']}"
        )
    final_status = {
        "smoke": "smoke-passed",
        "pilot": "pilot-passed",
        "full": "audited",
    }[mode]
    provenance["status"] = final_status
    provenance["audit"] = package_audit
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": final_status,
            "campaign_mode": mode,
            "campaign_scope": report["campaign_scope"],
            "current_stage": None,
            "valid_for_promotion": bool(
                mode == "full" and not git["dirty"] and promotion_gate_passed
            ),
        },
    )
    progress_path = run_root / "tracking" / "campaign_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["status"] = final_status
    progress["finished_utc"] = _utc_now()
    progress["current_stage"] = None
    _atomic_json(progress_path, progress)
    return {"run_root": run_root.as_posix(), **report}


def _teacher_run_preflight(
    config: dict[str, Any],
    *,
    run_id: str,
    mode: str,
    scope: str,
    allow_dirty_smoke: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"] and not (mode == "smoke" and allow_dirty_smoke):
        raise RuntimeError(
            "Refusing teacher training from a dirty worktree. "
            "Only explicit dirty smoke verification is allowed."
        )
    require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
    config["runtime"]["device"] = require_training_cuda(
        str(config["runtime"]["device"])
    )
    run_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    )
    run_root.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "mode": mode,
        "campaign_scope": scope,
        "verification_only": True,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "dataset_audit": dataset_audit,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    (run_root / "provenance" / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _atomic_json(
        run_root / "status.json",
        {
            "status": "running",
            "campaign_mode": mode,
            "campaign_scope": scope,
            "valid_for_promotion": False,
        },
    )
    _mark_stage(
        run_root,
        stage="preflight",
        status="completed",
        details={"dataset": dataset_audit, "git": git},
    )
    return run_root, provenance, git, _effective_training(config, mode)


def _frozen_e0_checkpoint() -> tuple[Path, dict[str, Any]]:
    package_root = (
        REPO_ROOT
        / "experiments"
        / "runs"
        / "20260727-converged-s0-baseline-v2"
    )
    summary = json.loads(
        (package_root / "metrics" / "campaign_summary.json").read_text(
            encoding="utf-8"
        )
    )
    model = package_root / "models" / "T0-WB-OFFICIAL.pt"
    inventory = dict(summary["model_inventory"]["T0-WB-OFFICIAL"])
    observed = sha256(model)
    if observed != inventory["sha256"]:
        raise ValueError("Published E0 teacher hash mismatch.")
    if observed != summary["baseline_contract"]["teacher_checkpoint_sha256"]:
        raise ValueError("Published E0 teacher ancestry mismatch.")
    return model, {
        "source_run_id": package_root.name,
        "checkpoint_sha256": observed,
        "bandwidth": "wb",
        "sample_rate": 16_000,
        "pesq_mode": "wb",
        "official_revision": "a196ce26b3bdace6fa1d819017584bdbcce462a8",
    }


def _write_teacher_calibration_plot(
    run_root: Path,
    refreshes: list[dict[str, Any]],
) -> Path:
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    plotted = False
    for refresh in refreshes:
        calibration = dict(refresh.get("calibration") or {})
        targets = list(calibration.get("targets") or [])
        predictions = list(calibration.get("predictions") or [])
        if targets and len(targets) == len(predictions):
            axis.scatter(
                targets,
                predictions,
                alpha=0.65,
                label=f"refresh {refresh.get('epoch')}",
            )
            plotted = True
    axis.plot([-0.5, 4.5], [-0.5, 4.5], linestyle="--", color="black")
    axis.set_xlim(-0.5, 4.5)
    axis.set_ylim(-0.5, 4.5)
    axis.set_xlabel("True PESQ-WB")
    axis.set_ylabel("Predicted PESQ-WB")
    axis.set_title("Held-out current-output discriminator calibration")
    if plotted:
        axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = run_root / "reports" / "current_output_calibration.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def run_teacher_calibration(
    config: dict[str, Any],
    *,
    run_id: str,
    mode: str,
    allow_dirty_smoke: bool = False,
) -> dict[str, Any]:
    """Run one discriminator refresh against frozen E0 with no G update."""
    run_root, provenance, git, effective = _teacher_run_preflight(
        config,
        run_id=run_id,
        mode=mode,
        scope=TEACHER_CALIBRATION_SCOPE,
        allow_dirty_smoke=allow_dirty_smoke,
    )
    e0_checkpoint, e0_contract = _frozen_e0_checkpoint()
    provenance["e0_contract"] = e0_contract
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    teacher_common = {
        "config": config,
        "run_root": run_root,
        "family": str(config["model"]["teacher_family"]),
        "bandwidth": "wb",
        "lr": float(effective["teacher_trial_lr"]),
        "seed": int(effective["seed"]),
        "frontend": dict(config["model"]["teacher_frontend"]),
        "include_test": False,
        "mode": mode,
    }
    cells: dict[str, dict[str, Any]] = {}
    cells["E0-T0"] = _run_cell(
        **teacher_common,
        cell="E0-T0",
        loss_recipe="T0",
        epochs=0,
        init_checkpoint=e0_checkpoint.as_posix(),
        evaluate_init_checkpoint=True,
    )
    proxy = _proxy(
        config,
        run_root=run_root,
        bandwidth="wb",
        candidate_teacher_checkpoint=str(cells["E0-T0"]["checkpoint_out"]),
        mode=mode,
    )
    calibration_refreshes = int(
        effective.get("teacher_calibration_refreshes", 2)
    )
    if calibration_refreshes < 1 or calibration_refreshes > 2:
        raise ValueError("teacher_calibration_refreshes must be 1 or 2.")
    cells["E0-D-CAL"] = _run_cell(
        **teacher_common,
        cell="E0-D-CAL",
        loss_recipe="T0_PESQ",
        epochs=calibration_refreshes,
        init_checkpoint=str(cells["E0-T0"]["checkpoint_out"]),
        evaluate_init_checkpoint=True,
        proxy_checkpoint=str(proxy["checkpoint"]),
        alternating_metric_discriminator=True,
        metric_discriminator_calibration_only=True,
        early_stop_patience=0,
    )
    refreshes = list(
        cells["E0-D-CAL"].get("metric_discriminator_refresh_history") or []
    )
    if len(refreshes) != calibration_refreshes:
        raise RuntimeError(
            "Calibration-only diagnostic refresh count does not match "
            "the predeclared schedule."
        )
    calibration_gate = dict(refreshes[-1]["calibration_gate"])
    plot = _write_teacher_calibration_plot(run_root, refreshes)
    report = _write_report(
        run_root=run_root,
        cells=cells,
        proxies={"wb": proxy},
        selected_teacher="E0-T0",
        teacher_gate=None,
        mode=mode,
        verification_only=True,
        campaign_scope=TEACHER_CALIBRATION_SCOPE,
        cell_order=TEACHER_CALIBRATION_CELL_ORDER,
        comparison_pairs={},
    )
    report["teacher_calibration_gate"] = calibration_gate
    report["teacher_calibration_refresh_count"] = len(refreshes)
    report["e0_contract"] = e0_contract
    report["calibration_plot"] = plot.as_posix()
    _atomic_json(run_root / "metrics" / "campaign_summary.json", report)
    result = _finish_run(
        run_root=run_root,
        provenance=provenance,
        report=report,
        mode=mode,
        git=git,
        promotion_gate_passed=False,
    )
    return {
        **result,
        "teacher_calibration_gate": calibration_gate,
    }


def _teacher_trial_gate(
    official: dict[str, Any],
    control: dict[str, Any],
    metric: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    training = dict(config["training"])
    e0 = dict(official.get("val_select_metrics") or {})
    e1 = dict(control.get("val_select_metrics") or {})
    e2 = dict(metric.get("val_select_metrics") or {})
    deltas = {
        name: float(e2[name]) - float(e0[name])
        for name in ("pesq_mean", "stoi_mean", "sisdr_mean", "delta_snr_mean")
    }
    refreshes = list(
        metric.get("metric_discriminator_refresh_history") or []
    )
    accepted = int(metric.get("metric_discriminator_accepted_update_count") or 0)
    checks = {
        "pesq_gain": deltas["pesq_mean"]
        >= float(training["teacher_min_pesq_gain"]),
        "stoi_guardrail": deltas["stoi_mean"]
        >= -float(training["teacher_max_stoi_drop"]),
        "sisdr_guardrail": deltas["sisdr_mean"]
        >= -float(training["teacher_max_sisdr_drop"]),
        "metric_beats_control": float(e2["pesq_mean"]) > float(e1["pesq_mean"]),
        "accepted_metric_update": accepted > 0,
        "all_accepted_updates_calibrated": accepted
        == sum(
            bool(dict(item.get("calibration_gate") or {}).get("passed"))
            for item in refreshes
        ),
    }
    return {
        "official_teacher": "E0-T0",
        "control_teacher": "E1-CONTROL",
        "metric_teacher": "E2-PESQ",
        "selected_candidate": "E2-PESQ" if all(checks.values()) else "E0-T0",
        "downstream_teacher": "E2-PESQ" if all(checks.values()) else "E0-T0",
        "val_select_deltas": deltas,
        "e2_minus_e1_pesq": float(e2["pesq_mean"]) - float(e1["pesq_mean"]),
        "accepted_metric_update_count": accepted,
        "refresh_count": len(refreshes),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_teacher_trial(
    config: dict[str, Any],
    *,
    calibration_run_dir: str | Path,
    run_id: str,
    mode: str,
    allow_dirty_smoke: bool = False,
) -> dict[str, Any]:
    """Run E0/E1/E2 only; never build C1 or train students."""
    calibration_root = Path(calibration_run_dir).expanduser().resolve()
    calibration_audit = audit_campaign_run(calibration_root)
    calibration_summary = json.loads(
        (calibration_root / "metrics" / "campaign_summary.json").read_text(
            encoding="utf-8"
        )
    )
    calibration_gate = dict(
        calibration_summary.get("teacher_calibration_gate") or {}
    )
    if not calibration_audit["valid"] or not bool(calibration_gate.get("passed")):
        raise ValueError(
            "Teacher pilot requires an audited, passed calibration-only run."
        )
    calibration_cell = dict(calibration_summary["cells"]["E0-D-CAL"])
    discriminator_checkpoint = _run_artifact_path(
        calibration_root,
        calibration_cell["metric_discriminator_checkpoint"],
    )
    if discriminator_checkpoint is None or not discriminator_checkpoint.is_file():
        raise FileNotFoundError("Calibration discriminator checkpoint is missing.")

    run_root, provenance, git, effective = _teacher_run_preflight(
        config,
        run_id=run_id,
        mode=mode,
        scope=TEACHER_TRIAL_SCOPE,
        allow_dirty_smoke=allow_dirty_smoke,
    )
    e0_checkpoint, e0_contract = _frozen_e0_checkpoint()
    provenance["e0_contract"] = e0_contract
    provenance["calibration_source"] = {
        "run_id": calibration_root.name,
        "summary_sha256": sha256(
            calibration_root / "metrics" / "campaign_summary.json"
        ),
        "discriminator_sha256": sha256(discriminator_checkpoint),
    }
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    common = {
        "config": config,
        "run_root": run_root,
        "family": str(config["model"]["teacher_family"]),
        "bandwidth": "wb",
        "lr": float(effective["teacher_trial_lr"]),
        "seed": int(effective["seed"]),
        "frontend": dict(config["model"]["teacher_frontend"]),
        "include_test": False,
        "mode": mode,
    }
    cells: dict[str, dict[str, Any]] = {}
    cells["E0-T0"] = _run_cell(
        **common,
        cell="E0-T0",
        loss_recipe="T0",
        epochs=0,
        init_checkpoint=e0_checkpoint.as_posix(),
        evaluate_init_checkpoint=True,
    )
    anchor_cache = _build_cache(
        config,
        run_root=run_root,
        teacher_checkpoint=str(cells["E0-T0"]["checkpoint_out"]),
        cache_label="teacher-trial-anchor",
        mode=mode,
    )
    trial_epochs = int(effective["teacher_trial_epochs"])
    cells["E1-CONTROL"] = _run_cell(
        **common,
        cell="E1-CONTROL",
        loss_recipe="T0",
        epochs=trial_epochs,
        init_checkpoint=str(cells["E0-T0"]["checkpoint_out"]),
        evaluate_init_checkpoint=True,
        teacher_cache_manifest=anchor_cache["wb"],
        early_stop_patience=int(effective["teacher_trial_early_stop_patience"]),
    )
    cells["E2-PESQ"] = _run_cell(
        **common,
        cell="E2-PESQ",
        loss_recipe="T0_PESQ",
        epochs=trial_epochs,
        init_checkpoint=str(cells["E0-T0"]["checkpoint_out"]),
        evaluate_init_checkpoint=True,
        teacher_cache_manifest=anchor_cache["wb"],
        proxy_checkpoint=discriminator_checkpoint.as_posix(),
        alternating_metric_discriminator=True,
        early_stop_patience=int(effective["teacher_trial_early_stop_patience"]),
    )
    gate = _teacher_trial_gate(
        cells["E0-T0"],
        cells["E1-CONTROL"],
        cells["E2-PESQ"],
        config=config,
    )
    selected = "E2-PESQ" if gate["passed"] else "E0-T0"
    plot = _write_teacher_calibration_plot(
        run_root,
        list(
            cells["E2-PESQ"].get("metric_discriminator_refresh_history")
            or []
        ),
    )
    report = _write_report(
        run_root=run_root,
        cells=cells,
        proxies={},
        selected_teacher=selected,
        teacher_gate=gate,
        mode=mode,
        verification_only=True,
        campaign_scope=TEACHER_TRIAL_SCOPE,
        cell_order=TEACHER_TRIAL_CELL_ORDER,
        comparison_pairs={
            "control_vs_official": ("E0-T0", "E1-CONTROL"),
            "metric_vs_official": ("E0-T0", "E2-PESQ"),
            "metric_vs_control": ("E1-CONTROL", "E2-PESQ"),
        },
    )
    report["e0_contract"] = e0_contract
    report["calibration_source_gate"] = calibration_gate
    report["calibration_plot"] = plot.as_posix()
    _atomic_json(run_root / "metrics" / "campaign_summary.json", report)
    return _finish_run(
        run_root=run_root,
        provenance=provenance,
        report=report,
        mode=mode,
        git=git,
        promotion_gate_passed=bool(gate["passed"]),
    )


def run_t3_matched_pilot(
    config: dict[str, Any],
    *,
    support_run_dir: str | Path,
    teacher_checkpoint: str | Path,
    teacher_cache_manifest: str | Path,
    run_id: str,
    mode: str,
    allow_dirty_smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the matched T3 E1/E2 teacher branches; never read test."""
    planned_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    )
    planned_provenance_path = planned_root / "provenance" / "provenance.json"
    adopting_contract = (
        not resume
        and planned_provenance_path.is_file()
        and json.loads(
            planned_provenance_path.read_text(encoding="utf-8")
        ).get("status")
        == "planned"
    )
    if adopting_contract:
        dataset_audit = validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError("T3 run-contract adoption requires a clean worktree.")
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        config["runtime"]["device"] = require_training_cuda(
            str(config["runtime"]["device"])
        )
        run_root = planned_root
        provenance = json.loads(
            planned_provenance_path.read_text(encoding="utf-8")
        )
        if provenance.get("git_commit") != git["commit"]:
            raise ValueError("T3 run contract was created from another commit.")
        resolved_config = run_root / "provenance" / "config_resolved.yaml"
        if (
            not resolved_config.is_file()
            or sha256(resolved_config) != provenance.get("config_sha256")
        ):
            raise ValueError("T3 run-contract config identity mismatch.")
        provenance.update(
            {
                "status": "running",
                "mode": mode,
                "campaign_scope": T3_DIRECT_SCOPE,
                "verification_only": mode != "full",
                "dataset_audit": dataset_audit,
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device": torch.cuda.get_device_name(0),
                },
            }
        )
    elif resume:
        validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError("T3 production resume requires a clean worktree.")
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        config["runtime"]["device"] = require_training_cuda(
            str(config["runtime"]["device"])
        )
        run_root = (
            Path(str(config["runtime"]["run_root"])).expanduser().resolve()
            / run_id
        )
        provenance_path = run_root / "provenance" / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError("T3 resume provenance is missing.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("campaign_scope") != T3_DIRECT_SCOPE:
            raise ValueError("T3 resume campaign scope mismatch.")
        if provenance.get("git_commit") != git["commit"]:
            raise ValueError("T3 resume requires the original committed code snapshot.")
    else:
        run_root, provenance, git, _ = _teacher_run_preflight(
            config,
            run_id=run_id,
            mode=mode,
            scope=T3_DIRECT_SCOPE,
            allow_dirty_smoke=allow_dirty_smoke,
        )
    support_root = Path(support_run_dir).expanduser().resolve()
    provenance["verification_only"] = mode != "full"
    identities_path = support_root / "support" / "identities.json"
    weights_path = support_root / "support" / "weights.json"
    direction_path = support_root / "reports" / "direction_audit.json"
    direction = json.loads(direction_path.read_text(encoding="utf-8"))
    if not bool(direction.get("valid")) or not bool(direction.get("passed")):
        raise ValueError("T3 E1/E2 requires a valid passed direction audit.")
    expected_hash = str(config["model"]["teacher_checkpoint_sha256"])
    if sha256(teacher_checkpoint) != expected_hash:
        raise ValueError("T3 official teacher hash mismatch.")
    identities = json.loads(identities_path.read_text(encoding="utf-8"))
    if (
        sha256(teacher_cache_manifest)
        != str(identities["teacher_cache_manifest_sha256"])
    ):
        raise ValueError("T3 cache manifest does not match frozen identities.")
    support_contract = {
        "run_id": support_root.name,
        "identities_sha256": sha256(identities_path),
        "weights_sha256": sha256(weights_path),
        "direction_audit_sha256": sha256(direction_path),
        "teacher_checkpoint_sha256": expected_hash,
        "teacher_cache_manifest_sha256": sha256(teacher_cache_manifest),
    }
    if resume and dict(provenance.get("support") or {}) != support_contract:
        raise ValueError("T3 resume support contract mismatch.")
    provenance["support"] = support_contract
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    device = str(config["runtime"]["device"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dataset = dict(config["dataset"])
    smoke = mode == "smoke"

    def progress(message: str) -> None:
        print(f"[T3] {message}", file=sys.stderr, flush=True)

    baseline_path = run_root / "cells" / "E0-T0" / "summary.json"
    if resume and baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("checkpoint_sha256") != expected_hash:
            raise ValueError("T3 resume E0 checkpoint mismatch.")
    else:
        _mark_stage(run_root, stage="E0-T0", status="running")
        baseline_model, _ = load_model_from_checkpoint(
            teacher_checkpoint, device=device
        )
        baseline_rank = evaluate_manifest(
            baseline_model,
            str(dataset["val_rank"]),
            device,
            sample_rate=16_000,
            bandwidth="wb",
            compute_dnsmos=False,
            compute_composite=False,
            max_files=2 if smoke else None,
            batch_size=1,
            progress_callback=progress,
        )
        baseline_select = evaluate_manifest(
            baseline_model,
            str(dataset["val_select"]),
            device,
            sample_rate=16_000,
            bandwidth="wb",
            compute_dnsmos=False,
            compute_composite=False,
            max_files=2 if smoke else None,
            batch_size=1,
            progress_callback=progress,
        )
        del baseline_model
        torch.cuda.empty_cache()
        baseline = {
            "val_rank_metrics": {
                key: float(baseline_rank[key])
                for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
            },
            "val_select_metrics": {
                key: float(baseline_select[key])
                for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
            },
            "checkpoint_sha256": expected_hash,
        }
        _atomic_json(baseline_path, baseline)
        _mark_stage(run_root, stage="E0-T0", status="completed", details=baseline)

    branches: dict[str, dict[str, Any]] = {}
    for branch in ("E1-SUP", "E2-PMSQE"):
        branch_summary_path = run_root / "cells" / branch / "summary.json"
        if resume and branch_summary_path.is_file():
            branches[branch] = json.loads(
                branch_summary_path.read_text(encoding="utf-8")
            )
            branch_contract = dict(branches[branch].get("provenance") or {})
            checkpoint_path = Path(
                str(branches[branch].get("selected_checkpoint") or "")
            )
            if (
                branches[branch].get("status") != "complete"
                or branches[branch].get("branch") != branch
                or branch_contract.get("teacher_checkpoint_sha256")
                != expected_hash
                or branch_contract.get("identities_sha256")
                != support_contract["identities_sha256"]
                or not checkpoint_path.is_file()
                or sha256(checkpoint_path)
                != branches[branch].get("selected_checkpoint_sha256")
            ):
                raise ValueError(f"T3 resume completed-branch audit failed: {branch}")
            continue
        _mark_stage(run_root, stage=branch, status="running")
        try:
            branches[branch] = run_t3_branch(
                branch=branch,
                teacher_checkpoint=teacher_checkpoint,
                teacher_cache_manifest=teacher_cache_manifest,
                identities_path=identities_path,
                weights_path=weights_path,
                val_rank_manifest=dataset["val_rank"],
                val_select_manifest=dataset["val_select"],
                output_dir=run_root / "cells" / branch,
                baseline_rank_metrics=baseline["val_rank_metrics"],
                device=device,
                seed=int(config["training"]["t3_direction_seed"]),
                max_accepted_epochs=1 if smoke else 10,
                batch_size=1,
                smoke=smoke,
                resume=True,
                progress_callback=progress,
            )
        except BaseException as exc:
            _mark_stage(
                run_root,
                stage=branch,
                status="failed",
                error=f"{exc.__class__.__name__}: {exc}",
            )
            provenance["status"] = "failed"
            provenance["failure"] = {
                "stage": branch,
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(
                run_root / "provenance" / "provenance.json",
                provenance,
            )
            raise
        _mark_stage(
            run_root,
            stage=branch,
            status="completed",
            details=branches[branch],
        )
        torch.cuda.empty_cache()

    t0 = dict(baseline["val_select_metrics"])
    e1 = dict(branches["E1-SUP"]["val_select_metrics"])
    e2 = dict(branches["E2-PMSQE"]["val_select_metrics"])
    deltas = {
        key: float(e2[key]) - float(t0[key])
        for key in ("pesq_mean", "stoi_mean", "sisdr_mean")
    }
    checks = {
        "pesq_gain_at_least_0_01": deltas["pesq_mean"] >= 0.01,
        "stoi_drop_at_most_0_002": deltas["stoi_mean"] >= -0.002,
        "sisdr_drop_at_most_0_25": deltas["sisdr_mean"] >= -0.25,
        "e2_beats_e1_pesq": float(e2["pesq_mean"]) > float(e1["pesq_mean"]),
        "production_run": not smoke,
    }
    gate = {
        "passed": all(checks.values()),
        "checks": checks,
        "e2_minus_t0": deltas,
        "e2_minus_e1_pesq": float(e2["pesq_mean"]) - float(e1["pesq_mean"]),
        "selected": "E2-PMSQE" if all(checks.values()) else "E0-T0",
    }
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "campaign_scope": T3_DIRECT_SCOPE,
        "mode": mode,
        "verification_only": smoke,
        "test_read": False,
        "baseline": baseline,
        "branches": branches,
        "teacher_gate": gate,
        "teacher_improved_single_seed": bool(gate["passed"]),
        "requires_three_seed_confirmation": bool(gate["passed"]),
    }
    _atomic_json(run_root / "metrics" / "campaign_summary.json", summary)
    provenance["status"] = (
        "single_seed_gate_passed" if gate["passed"] else "direct_pilot_complete"
    )
    provenance["teacher_gate"] = gate
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": provenance["status"],
            "campaign_mode": mode,
            "campaign_scope": T3_DIRECT_SCOPE,
            "valid_for_promotion": False,
            "teacher_improved_single_seed": bool(gate["passed"]),
            "test_read": False,
        },
    )
    return {"run_root": run_root.as_posix(), **summary}


def run_t4_calibration(
    config: dict[str, Any],
    *,
    baseline_run_dir: str | Path,
    teacher_checkpoint: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Run deterministic T4-A selection; test and students remain blocked."""
    dataset_audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("T4 calibration requires a clean committed snapshot.")
    require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
    device = require_training_cuda(str(config["runtime"]["device"]))
    run_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    )
    provenance_path = run_root / "provenance" / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("T4 requires a planned run contract.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "planned":
        raise ValueError("T4 new run must adopt a planned contract.")
    if provenance.get("git_commit") != git["commit"]:
        raise ValueError("T4 run contract commit mismatch.")
    resolved_config = run_root / "provenance" / "config_resolved.yaml"
    if (
        not resolved_config.is_file()
        or sha256(resolved_config) != provenance.get("config_sha256")
    ):
        raise ValueError("T4 run contract config mismatch.")
    baseline_root = Path(baseline_run_dir).expanduser().resolve()
    baseline_path = baseline_root / "metrics" / "campaign_summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_hash = str(config["model"]["teacher_checkpoint_sha256"])
    if (
        sha256(teacher_checkpoint) != expected_hash
        or baseline["baseline"]["checkpoint_sha256"] != expected_hash
    ):
        raise ValueError("T4 baseline teacher identity mismatch.")
    if bool(baseline.get("test_read")):
        raise ValueError("T4 baseline source must not read test.")
    provenance.update(
        {
            "status": "running",
            "campaign_scope": "t4_uniform_logit_calibration",
            "verification_only": False,
            "dataset_audit": dataset_audit,
            "baseline_source": {
                "run_id": baseline_root.name,
                "summary_sha256": sha256(baseline_path),
                "teacher_checkpoint_sha256": expected_hash,
            },
        }
    )
    _atomic_json(provenance_path, provenance)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    def progress(message: str) -> None:
        print(f"[T4] {message}", file=sys.stderr, flush=True)

    try:
        result = run_t4_logit_bias_scan(
            teacher_checkpoint=teacher_checkpoint,
            val_rank_manifest=config["dataset"]["val_rank"],
            val_select_manifest=config["dataset"]["val_select"],
            baseline_rank_metrics=baseline["baseline"]["val_rank_metrics"],
            baseline_select_metrics=baseline["baseline"]["val_select_metrics"],
            output_dir=run_root / "cells" / "T4-A-LOGIT-BIAS",
            device=device,
            progress_callback=progress,
        )
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["failure"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(provenance_path, provenance)
        _atomic_json(
            run_root / "status.json",
            {"status": "failed", "valid_for_promotion": False},
        )
        raise
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "campaign_scope": "t4_uniform_logit_calibration",
        "baseline_source": provenance["baseline_source"],
        "test_read": False,
        "result": result,
        "teacher_gate": result["gate"],
    }
    _atomic_json(run_root / "metrics" / "campaign_summary.json", summary)
    provenance["status"] = (
        "candidate_gate_passed" if result["gate"]["passed"] else "complete_failed_gate"
    )
    provenance["result_summary_sha256"] = sha256(
        run_root / "cells" / "T4-A-LOGIT-BIAS" / "summary.json"
    )
    _atomic_json(provenance_path, provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": provenance["status"],
            "campaign_scope": "t4_uniform_logit_calibration",
            "teacher_gate_passed": bool(result["gate"]["passed"]),
            "valid_for_promotion": False,
            "test_read": False,
        },
    )
    return {"run_root": run_root.as_posix(), **summary}


def run_t4_microstep_trial(
    config: dict[str, Any],
    *,
    baseline_run_dir: str | Path,
    t4a_run_dir: str | Path,
    support_run_dir: str | Path,
    teacher_checkpoint: str | Path,
    teacher_cache_manifest: str | Path,
    run_id: str,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run T4-B PMSQE-primary micro-steps; never read test or train students."""
    dataset_audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("T4-B requires a clean committed snapshot.")
    require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
    device = require_training_cuda(str(config["runtime"]["device"]))
    run_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    )
    provenance_path = run_root / "provenance" / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("T4-B requires a planned run contract.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if resume:
        if (
            provenance.get("campaign_scope") != "t4_pmsqe_microstep_backtracking"
            or provenance.get("status") not in {"running", "failed"}
        ):
            raise ValueError("T4-B resume provenance is not resumable.")
    elif provenance.get("status") != "planned":
        raise ValueError("T4-B new run must adopt a planned contract.")
    if provenance.get("git_commit") != git["commit"]:
        raise ValueError("T4-B run contract commit mismatch.")
    resolved_config = run_root / "provenance" / "config_resolved.yaml"
    if (
        not resolved_config.is_file()
        or sha256(resolved_config) != provenance.get("config_sha256")
    ):
        raise ValueError("T4-B run contract config mismatch.")
    baseline_root = Path(baseline_run_dir).expanduser().resolve()
    baseline_path = baseline_root / "metrics" / "campaign_summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    t4a_root = Path(t4a_run_dir).expanduser().resolve()
    t4a_path = t4a_root / "cells" / "T4-A-LOGIT-BIAS" / "summary.json"
    t4a = json.loads(t4a_path.read_text(encoding="utf-8"))
    if (
        t4a.get("status") != "failed"
        or bool(t4a["gate"]["passed"])
        or not bool(t4a["gate"]["checks"]["stoi_drop_at_most_0_002"])
        or not bool(t4a["gate"]["checks"]["sisdr_drop_at_most_0_25"])
        or float(t4a["val_select_deltas"]["pesq_mean"]) >= 0.01
    ):
        raise ValueError("T4-B requires a safe but below-threshold T4-A result.")
    expected_hash = str(config["model"]["teacher_checkpoint_sha256"])
    if (
        sha256(teacher_checkpoint) != expected_hash
        or baseline["baseline"]["checkpoint_sha256"] != expected_hash
        or t4a["teacher_checkpoint_sha256"] != expected_hash
    ):
        raise ValueError("T4-B baseline teacher identity mismatch.")
    if bool(baseline.get("test_read")) or bool(t4a.get("test_read")):
        raise ValueError("T4-B sources must not read test.")
    support_root = Path(support_run_dir).expanduser().resolve()
    identities_path = support_root / "support" / "identities.json"
    weights_path = support_root / "support" / "weights.json"
    source_contract = {
        "baseline_run_id": baseline_root.name,
        "baseline_summary_sha256": sha256(baseline_path),
        "t4a_run_id": t4a_root.name,
        "t4a_summary_sha256": sha256(t4a_path),
        "support_run_id": support_root.name,
        "identities_sha256": sha256(identities_path),
        "weights_sha256": sha256(weights_path),
        "teacher_checkpoint_sha256": expected_hash,
        "teacher_cache_manifest_sha256": sha256(teacher_cache_manifest),
    }
    if resume and provenance.get("source_contract") != source_contract:
        raise ValueError("T4-B resume source contract mismatch.")
    provenance.update(
        {
            "status": "running",
            "campaign_scope": "t4_pmsqe_microstep_backtracking",
            "verification_only": bool(smoke),
            "dataset_audit": dataset_audit,
            "source_contract": source_contract,
        }
    )
    _atomic_json(provenance_path, provenance)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    def progress(message: str) -> None:
        print(f"[T4-B] {message}", file=sys.stderr, flush=True)

    try:
        result = run_t4_microstep_backtracking(
            teacher_checkpoint=teacher_checkpoint,
            teacher_cache_manifest=teacher_cache_manifest,
            identities_path=identities_path,
            weights_path=weights_path,
            val_rank_manifest=config["dataset"]["val_rank"],
            val_select_manifest=config["dataset"]["val_select"],
            baseline_rank_metrics=baseline["baseline"]["val_rank_metrics"],
            baseline_select_metrics=baseline["baseline"]["val_select_metrics"],
            output_dir=run_root / "cells" / "T4-B-MICROSTEP",
            device=device,
            seed=int(config["training"]["t3_direction_seed"]),
            horizons=(1,) if smoke else (1, 4, 16, 64, 256),
            alphas=(1.0, 0.5) if smoke else (1.0, 0.5, 0.25, 0.125, 0.0625),
            max_eval_files=2 if smoke else None,
            resume=True,
            progress_callback=progress,
        )
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["failure"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(provenance_path, provenance)
        _atomic_json(
            run_root / "status.json",
            {"status": "failed", "valid_for_promotion": False},
        )
        raise
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "campaign_scope": "t4_pmsqe_microstep_backtracking",
        "verification_only": bool(smoke),
        "source_contract": source_contract,
        "test_read": False,
        "result": result,
        "teacher_gate": result["gate"],
    }
    _atomic_json(run_root / "metrics" / "campaign_summary.json", summary)
    provenance["status"] = (
        "verification_complete"
        if smoke
        else (
            "candidate_gate_passed"
            if result["gate"]["passed"]
            else "complete_failed_gate"
        )
    )
    provenance["result_summary_sha256"] = sha256(
        run_root / "cells" / "T4-B-MICROSTEP" / "summary.json"
    )
    _atomic_json(provenance_path, provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": provenance["status"],
            "campaign_scope": "t4_pmsqe_microstep_backtracking",
            "teacher_gate_passed": bool(result["gate"]["passed"]),
            "valid_for_promotion": False,
            "verification_only": bool(smoke),
            "test_read": False,
        },
    )
    return {"run_root": run_root.as_posix(), **summary}


def run_t5_frequency_trial(
    config: dict[str, Any],
    *,
    baseline_run_dir: str | Path,
    t4b_run_dir: str | Path,
    support_run_dir: str | Path,
    teacher_checkpoint: str | Path,
    run_id: str,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run guarded true-PESQ frequency-curve search; never read test."""
    dataset_audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("T5 requires a clean committed snapshot.")
    require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
    device = require_training_cuda(str(config["runtime"]["device"]))
    run_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    )
    provenance_path = run_root / "provenance" / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("T5 requires a planned run contract.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "planned":
        raise ValueError("T5 new run must adopt a planned contract.")
    if provenance.get("git_commit") != git["commit"]:
        raise ValueError("T5 run contract commit mismatch.")
    resolved_config = run_root / "provenance" / "config_resolved.yaml"
    if (
        not resolved_config.is_file()
        or sha256(resolved_config) != provenance.get("config_sha256")
    ):
        raise ValueError("T5 run contract config mismatch.")
    baseline_root = Path(baseline_run_dir).expanduser().resolve()
    baseline_path = baseline_root / "metrics" / "campaign_summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    t4b_root = Path(t4b_run_dir).expanduser().resolve()
    t4b_path = t4b_root / "cells" / "T4-B-MICROSTEP" / "summary.json"
    t4b = json.loads(t4b_path.read_text(encoding="utf-8"))
    if (
        t4b.get("status") != "failed"
        or bool(t4b["gate"]["passed"])
        or float(t4b["val_select_deltas"]["pesq_mean"]) >= 0.01
    ):
        raise ValueError("T5 requires the terminal below-threshold T4-B result.")
    expected_hash = str(config["model"]["teacher_checkpoint_sha256"])
    if (
        sha256(teacher_checkpoint) != expected_hash
        or baseline["baseline"]["checkpoint_sha256"] != expected_hash
        or t4b["contract"]["teacher_checkpoint_sha256"] != expected_hash
    ):
        raise ValueError("T5 baseline teacher identity mismatch.")
    if bool(baseline.get("test_read")) or bool(t4b.get("test_read")):
        raise ValueError("T5 sources must not read test.")
    support_root = Path(support_run_dir).expanduser().resolve()
    identities_path = support_root / "support" / "identities.json"
    source_contract = {
        "baseline_run_id": baseline_root.name,
        "baseline_summary_sha256": sha256(baseline_path),
        "t4b_run_id": t4b_root.name,
        "t4b_summary_sha256": sha256(t4b_path),
        "support_run_id": support_root.name,
        "identities_sha256": sha256(identities_path),
        "teacher_checkpoint_sha256": expected_hash,
    }
    provenance.update(
        {
            "status": "running",
            "campaign_scope": "t5_true_pesq_frequency_curve",
            "verification_only": bool(smoke),
            "dataset_audit": dataset_audit,
            "source_contract": source_contract,
        }
    )
    _atomic_json(provenance_path, provenance)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    def progress(message: str) -> None:
        print(f"[T5] {message}", file=sys.stderr, flush=True)

    try:
        result = run_t5_frequency_search(
            teacher_checkpoint=teacher_checkpoint,
            identities_path=identities_path,
            val_rank_manifest=config["dataset"]["val_rank"],
            val_select_manifest=config["dataset"]["val_select"],
            baseline_rank_metrics=baseline["baseline"]["val_rank_metrics"],
            baseline_select_metrics=baseline["baseline"]["val_select_metrics"],
            output_dir=run_root / "cells" / "T5-FREQUENCY-CURVE",
            device=device,
            steps=(0.04,) if smoke else (0.08, 0.04, 0.02),
            fit_count=2 if smoke else 96,
            calibration_count=2 if smoke else 96,
            coordinate_limit=1 if smoke else None,
            max_eval_files=2 if smoke else None,
            progress_callback=progress,
        )
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["failure"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(provenance_path, provenance)
        _atomic_json(
            run_root / "status.json",
            {"status": "failed", "valid_for_promotion": False},
        )
        raise
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "campaign_scope": "t5_true_pesq_frequency_curve",
        "verification_only": bool(smoke),
        "source_contract": source_contract,
        "test_read": False,
        "result": result,
        "teacher_gate": result["gate"],
    }
    _atomic_json(run_root / "metrics" / "campaign_summary.json", summary)
    provenance["status"] = (
        "verification_complete"
        if smoke
        else (
            "candidate_gate_passed"
            if result["gate"]["passed"]
            else "complete_failed_gate"
        )
    )
    provenance["result_summary_sha256"] = sha256(
        run_root / "cells" / "T5-FREQUENCY-CURVE" / "summary.json"
    )
    _atomic_json(provenance_path, provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": provenance["status"],
            "campaign_scope": "t5_true_pesq_frequency_curve",
            "teacher_gate_passed": bool(result["gate"]["passed"]),
            "valid_for_promotion": False,
            "verification_only": bool(smoke),
            "test_read": False,
        },
    )
    return {"run_root": run_root.as_posix(), **summary}


def run_all(
    config: dict[str, Any],
    *,
    run_id: str,
    mode: str,
    allow_dirty_smoke: bool,
    baseline_only: bool = False,
) -> dict[str, Any]:
    if mode not in {"smoke", "pilot", "full"}:
        raise ValueError(f"Unsupported campaign mode: {mode}")
    audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"] and not (mode == "smoke" and allow_dirty_smoke):
        raise RuntimeError(
            "Refusing campaign training from a dirty worktree. "
            "Only an explicit --allow-dirty-smoke verification run is allowed."
        )
    shared_venv = Path(str(config["runtime"]["shared_venv"]))
    require_shared_venv(shared_venv)
    device = require_training_cuda(str(config["runtime"]["device"]))
    config["runtime"]["device"] = device
    run_root = Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    verification_only = mode != "full"

    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "mode": mode,
        "campaign_scope": BASELINE_SCOPE if baseline_only else TWO_STAGE_SCOPE,
        "verification_only": verification_only,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "dataset_audit": audit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    (run_root / "provenance" / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _atomic_json(
        run_root / "status.json",
        {
            "status": "running",
            "campaign_mode": mode,
            "current_stage": "preflight",
            "valid_for_promotion": False,
        },
    )
    _mark_stage(
        run_root,
        stage="preflight",
        status="completed",
        details={"dataset": audit, "git": git, "device": device},
    )
    effective = _effective_training(config, mode)
    student_schedule = _student_schedule(effective, mode=mode)
    cells: dict[str, dict[str, Any]] = {}
    proxies: dict[str, dict[str, Any]] = {}

    def execute(stage: str, function: Any) -> Any:
        _mark_stage(run_root, stage=stage, status="running")
        try:
            result = function()
        except BaseException as exc:
            _mark_stage(
                run_root,
                stage=stage,
                status="failed",
                error=f"{exc.__class__.__name__}: {exc}",
            )
            raise
        _mark_stage(
            run_root,
            stage=stage,
            status="completed",
            details=_stage_details(result),
        )
        return result

    teacher_frontend = dict(config["model"]["teacher_frontend"])
    common_official_teacher = {
        "config": config,
        "run_root": run_root,
        "family": str(config["model"]["teacher_family"]),
        "bandwidth": "wb",
        "lr": float(effective["teacher_branch_lr"]),
        "seed": int(effective["seed"]),
        "frontend": teacher_frontend,
        "include_test": True,
        "mode": mode,
    }
    cells["T0-WB-OFFICIAL"] = _run_cell(
        **common_official_teacher,
        cell="T0-WB-OFFICIAL",
        loss_recipe="T0",
        epochs=0,
        evaluate_init_checkpoint=True,
    )
    official_cache_manifests = _build_cache(
        config,
        run_root=run_root,
        teacher_checkpoint=str(cells["T0-WB-OFFICIAL"]["checkpoint_out"]),
        cache_label="official",
        mode=mode,
    )

    for bandwidth in ("wb", "nb"):
        cell = f"S0-{bandwidth.upper()}"
        cells[cell] = _run_cell(
            config=config,
            run_root=run_root,
            cell=cell,
            family=str(config["model"][f"student_{bandwidth}_family"]),
            bandwidth=bandwidth,
            loss_recipe="D1",
            epochs=int(effective["student_epochs"]),
            lr=float(effective["student_lr"]),
            seed=int(effective["seed"]),
            teacher_cache_manifest=official_cache_manifests[bandwidth],
            **student_schedule,
            mode=mode,
        )

    if baseline_only:
        baseline_contract = {
            "passed": True,
            "teacher": "T0-WB-OFFICIAL",
            "teacher_checkpoint_sha256": sha256(
                str(cells["T0-WB-OFFICIAL"]["checkpoint_out"])
            ),
            "cache_manifests": official_cache_manifests,
            "students": ["S0-WB", "S0-NB"],
        }
        _mark_stage(
            run_root,
            stage="BASELINE-SELECTION",
            status="completed",
            details=baseline_contract,
        )
        report = execute(
            "REPORT",
            lambda: _write_report(
                run_root=run_root,
                cells=cells,
                proxies=proxies,
                selected_teacher="T0-WB-OFFICIAL",
                teacher_gate=None,
                mode=mode,
                verification_only=verification_only,
                campaign_scope=BASELINE_SCOPE,
                cell_order=BASELINE_CELL_ORDER,
                comparison_pairs={},
                baseline_contract=baseline_contract,
            ),
        )
        return _finish_run(
            run_root=run_root,
            provenance=provenance,
            report=report,
            mode=mode,
            git=git,
            promotion_gate_passed=True,
        )

    proxies["wb"] = execute(
        "PROXY-WB",
        lambda: _proxy(
            config,
            run_root=run_root,
            bandwidth="wb",
            candidate_teacher_checkpoint=str(
                cells["T0-WB-OFFICIAL"]["checkpoint_out"]
            ),
            mode=mode,
        ),
    )
    common_finetuned_teacher = {
        **common_official_teacher,
        "epochs": int(effective["teacher_branch_epochs"]),
        "init_checkpoint": str(cells["T0-WB-OFFICIAL"]["checkpoint_out"]),
        "evaluate_init_checkpoint": True,
        "teacher_cache_manifest": official_cache_manifests["wb"],
    }
    cells["T1-WB-BASE"] = _run_cell(
        **common_finetuned_teacher,
        cell="T1-WB-BASE",
        loss_recipe="T0",
    )
    cells["T1-WB-METRIC"] = _run_cell(
        **common_finetuned_teacher,
        cell="T1-WB-METRIC",
        loss_recipe="T0_PESQ",
        proxy_checkpoint=str(proxies["wb"]["checkpoint"]),
        alternating_metric_discriminator=True,
    )
    selected_teacher, selected_summary, teacher_gate = _best_teacher(
        cells["T0-WB-OFFICIAL"],
        cells["T1-WB-BASE"],
        cells["T1-WB-METRIC"],
        config=config,
        verification_only=verification_only,
    )
    _mark_stage(
        run_root,
        stage="TEACHER-SELECTION",
        status="completed",
        details={
            "selected_teacher": selected_teacher,
            "val_select_pesq": selected_summary.get("best_val_select_pesq"),
            "promotion_gate": teacher_gate,
        },
    )
    improved_cache_manifests = _build_cache(
        config,
        run_root=run_root,
        teacher_checkpoint=str(selected_summary["checkpoint_out"]),
        cache_label="improved",
        mode=mode,
    )

    for bandwidth in ("wb", "nb"):
        cell = f"S1-{bandwidth.upper()}"
        cells[cell] = _run_cell(
            config=config,
            run_root=run_root,
            cell=cell,
            family=str(config["model"][f"student_{bandwidth}_family"]),
            bandwidth=bandwidth,
            loss_recipe="D1",
            epochs=int(effective["student_epochs"]),
            lr=float(effective["student_lr"]),
            seed=int(effective["seed"]),
            teacher_cache_manifest=improved_cache_manifests[bandwidth],
            **student_schedule,
            mode=mode,
        )

    report = execute(
        "REPORT",
        lambda: _write_report(
            run_root=run_root,
            cells=cells,
            proxies=proxies,
            selected_teacher=selected_teacher,
            teacher_gate=teacher_gate,
            mode=mode,
            verification_only=verification_only,
        ),
    )
    return _finish_run(
        run_root=run_root,
        provenance=provenance,
        report=report,
        mode=mode,
        git=git,
        promotion_gate_passed=bool(teacher_gate["passed"]),
    )


def continue_official_students(
    config: dict[str, Any],
    *,
    source_run_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Continue both official-cache students from immutable epoch states."""
    dataset_audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("Refusing student continuation from a dirty worktree.")
    shared_venv = Path(str(config["runtime"]["shared_venv"]))
    require_shared_venv(shared_venv)
    device = require_training_cuda(str(config["runtime"]["device"]))
    config["runtime"]["device"] = device

    source_root = Path(source_run_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source run does not exist: {source_root}")
    source_status_path = source_root / "status.json"
    source_provenance_path = source_root / "provenance" / "provenance.json"
    if not source_status_path.is_file() or not source_provenance_path.is_file():
        raise FileNotFoundError("Source run is missing status/provenance.")
    source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
    source_provenance = json.loads(
        source_provenance_path.read_text(encoding="utf-8")
    )
    if source_status.get("status") != "audited":
        raise ValueError("Student continuation requires an audited full source run.")
    if source_provenance.get("campaign_scope") != BASELINE_SCOPE:
        raise ValueError("Student continuation requires an official baseline source.")

    run_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    )
    run_root.mkdir(parents=True, exist_ok=False)
    effective = _effective_training(config, "full")
    schedule = _student_schedule(effective, mode="full")
    max_epochs = int(effective["student_epochs"])
    source_contract: dict[str, dict[str, Any]] = {}
    source_payloads: dict[str, dict[str, Any]] = {}

    for cell in STUDENT_CONTINUATION_CELL_ORDER:
        cell_root = source_root / "cells" / cell
        summary_path = cell_root / "summary.json"
        state_path = cell_root / "training_state.pt"
        model_path = cell_root / "model.pt"
        if (
            not summary_path.is_file()
            or not state_path.is_file()
            or not model_path.is_file()
        ):
            raise FileNotFoundError(f"Source artifacts are incomplete for {cell}.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        source_epoch = int(state.get("epoch") or 0)
        if source_epoch >= max_epochs:
            raise ValueError(
                f"{cell} source epoch {source_epoch} is not below {max_epochs}."
            )
        source_config = dict(state.get("config") or {})
        expected_bandwidth = "nb" if cell.endswith("-NB") else "wb"
        if source_config.get("bandwidth") != expected_bandwidth:
            raise ValueError(f"{cell} source state bandwidth mismatch.")
        expected_family = str(
            config["model"][f"student_{expected_bandwidth}_family"]
        )
        if source_config.get("model_family") != expected_family:
            raise ValueError(f"{cell} source state model-family mismatch.")
        if source_config.get("loss_recipe") != "D1":
            raise ValueError(f"{cell} source state loss-recipe mismatch.")
        for key, dataset_key in (
            ("train_csv", "train_fit"),
            ("val_rank_csv", "val_rank"),
            ("val_select_csv", "val_select"),
            ("test_csv", "test"),
        ):
            source_manifest = Path(str(source_config.get(key) or ""))
            current_manifest = Path(str(config["dataset"][dataset_key]))
            if (
                not source_manifest.is_file()
                or sha256(source_manifest) != sha256(current_manifest)
            ):
                raise ValueError(f"{cell} source manifest mismatch: {key}.")
        teacher_cache_manifest = Path(
            str(source_config.get("teacher_cache_manifest") or "")
        )
        if not teacher_cache_manifest.is_file():
            raise FileNotFoundError(
                f"{cell} source teacher-cache manifest is missing."
            )
        source_payloads[cell] = {
            "state_path": state_path,
            "model_path": model_path,
            "teacher_cache_manifest": teacher_cache_manifest,
        }
        source_contract[cell] = {
            "epoch": source_epoch,
            "best_epoch": int(summary.get("best_epoch") or 0),
            "best_score": float(summary.get("best_score") or float("nan")),
            "training_state": state_path.as_posix(),
            "training_state_sha256": sha256(state_path),
            "model": model_path.as_posix(),
            "model_sha256": sha256(model_path),
        }

    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "mode": "full",
        "campaign_scope": STUDENT_CONTINUATION_SCOPE,
        "verification_only": False,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "dataset_audit": dataset_audit,
        "source_run": {
            "run_id": source_root.name,
            "git_commit": source_provenance.get("git_commit"),
            "path": source_root.as_posix(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    (run_root / "provenance" / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _atomic_json(
        run_root / "status.json",
        {
            "status": "running",
            "campaign_mode": "full",
            "campaign_scope": STUDENT_CONTINUATION_SCOPE,
            "current_stage": "preflight",
            "valid_for_promotion": False,
        },
    )
    _mark_stage(
        run_root,
        stage="preflight",
        status="completed",
        details={
            "dataset": dataset_audit,
            "git": git,
            "device": device,
            "source_run_id": source_root.name,
        },
    )

    cells: dict[str, dict[str, Any]] = {}
    for cell in STUDENT_CONTINUATION_CELL_ORDER:
        bandwidth = "nb" if cell.endswith("-NB") else "wb"
        source = source_payloads[cell]
        _mark_stage(run_root, stage=cell, status="running")
        try:
            experiment = _experiment_config(
                config,
                run_root=run_root,
                cell=cell,
                family=str(config["model"][f"student_{bandwidth}_family"]),
                bandwidth=bandwidth,
                loss_recipe="D1",
                epochs=max_epochs,
                lr=float(effective["student_lr"]),
                seed=int(effective["seed"]),
                teacher_cache_manifest=source[
                    "teacher_cache_manifest"
                ].as_posix(),
                resume_training_state=source["state_path"].as_posix(),
                **schedule,
                mode="full",
            )
            shutil.copy2(source["model_path"], experiment.checkpoint_out)
            summary = run_experiment(experiment)
            summary["continued_from"] = source_contract[cell]
            _atomic_json(
                Path(experiment.checkpoint_out).parent / "summary.json",
                summary,
            )
            cells[cell] = summary
            _mark_stage(
                run_root,
                stage=cell,
                status="completed",
                details=_stage_details(summary),
            )
        except BaseException as exc:
            _mark_stage(
                run_root,
                stage=cell,
                status="failed",
                error=f"{exc.__class__.__name__}: {exc}",
            )
            raise

    continuation_contract = {
        "passed": True,
        "source_run_id": source_root.name,
        "students": list(STUDENT_CONTINUATION_CELL_ORDER),
        "max_epochs": max_epochs,
        "schedule": {
            "scheduler": "plateau",
            **schedule,
        },
        "sources": source_contract,
    }
    report = _write_report(
        run_root=run_root,
        cells=cells,
        proxies={},
        selected_teacher="T0-WB-OFFICIAL",
        teacher_gate=None,
        mode="full",
        verification_only=False,
        campaign_scope=STUDENT_CONTINUATION_SCOPE,
        cell_order=STUDENT_CONTINUATION_CELL_ORDER,
        comparison_pairs={},
        student_continuation_contract=continuation_contract,
    )
    return _finish_run(
        run_root=run_root,
        provenance=provenance,
        report=report,
        mode="full",
        git=git,
        promotion_gate_passed=True,
    )


def close_converged_baseline(
    *,
    baseline_run_dir: str | Path,
    continuation_run_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Merge an audited epoch-20 baseline and its audited continuation."""
    baseline_root = Path(baseline_run_dir).expanduser().resolve()
    continuation_root = Path(continuation_run_dir).expanduser().resolve()
    if baseline_root.parent != continuation_root.parent:
        raise ValueError("Baseline and continuation must share one local run root.")
    run_root = baseline_root.parent / run_id
    if run_root.exists():
        raise FileExistsError(f"Closure run already exists: {run_root}")

    baseline_audit = audit_campaign_run(baseline_root)
    continuation_audit = audit_campaign_run(continuation_root)
    if not baseline_audit["valid"]:
        raise ValueError(f"Baseline source audit failed: {baseline_audit['issues']}")
    if not continuation_audit["valid"]:
        raise ValueError(
            f"Continuation source audit failed: {continuation_audit['issues']}"
        )

    def load_json(path: Path) -> dict[str, Any]:
        return dict(json.loads(path.read_text(encoding="utf-8")))

    baseline_summary = load_json(
        baseline_root / "metrics" / "campaign_summary.json"
    )
    continuation_summary = load_json(
        continuation_root / "metrics" / "campaign_summary.json"
    )
    baseline_provenance = load_json(
        baseline_root / "provenance" / "provenance.json"
    )
    continuation_provenance = load_json(
        continuation_root / "provenance" / "provenance.json"
    )
    if baseline_summary.get("campaign_scope") != BASELINE_SCOPE:
        raise ValueError("The baseline source is not an official baseline package.")
    if continuation_summary.get("campaign_scope") != STUDENT_CONTINUATION_SCOPE:
        raise ValueError("The continuation source is not a student continuation.")
    continuation_contract = dict(
        continuation_summary.get("student_continuation_contract") or {}
    )
    if continuation_contract.get("source_run_id") != baseline_root.name:
        raise ValueError("Continuation ancestry does not name the baseline source.")

    baseline_cells = dict(baseline_summary.get("cells") or {})
    continuation_cells = dict(continuation_summary.get("cells") or {})
    cells = {
        "T0-WB-OFFICIAL": copy.deepcopy(
            baseline_cells["T0-WB-OFFICIAL"]
        ),
        "S0-WB": copy.deepcopy(continuation_cells["S0-WB"]),
        "S0-NB": copy.deepcopy(continuation_cells["S0-NB"]),
    }
    baseline_inventory = dict(baseline_summary.get("model_inventory") or {})
    continuation_inventory = dict(
        continuation_summary.get("model_inventory") or {}
    )
    source_hashes = {
        "T0-WB-OFFICIAL": dict(
            baseline_inventory["T0-WB-OFFICIAL"]
        )["sha256"],
        "S0-WB": dict(continuation_inventory["S0-WB"])["sha256"],
        "S0-NB": dict(continuation_inventory["S0-NB"])["sha256"],
    }
    sources = dict(continuation_contract.get("sources") or {})
    for cell in STUDENT_CONTINUATION_CELL_ORDER:
        source_hash = dict(sources.get(cell) or {}).get("model_sha256")
        original_hash = dict(baseline_inventory.get(cell) or {}).get("sha256")
        if source_hash != original_hash:
            raise ValueError(
                f"Continuation source hash does not match baseline: {cell}"
            )

    epoch20 = {
        cell: float(
            dict(baseline_cells[cell].get("val_select_metrics") or {})[
                "pesq_mean"
            ]
        )
        for cell in STUDENT_CONTINUATION_CELL_ORDER
    }
    converged = {
        cell: float(
            dict(continuation_cells[cell].get("val_select_metrics") or {})[
                "pesq_mean"
            ]
        )
        for cell in STUDENT_CONTINUATION_CELL_ORDER
    }
    deltas = {
        cell: converged[cell] - epoch20[cell]
        for cell in STUDENT_CONTINUATION_CELL_ORDER
    }
    closure_contract = {
        "passed": True,
        "baseline_run_id": baseline_root.name,
        "continuation_run_id": continuation_root.name,
        "source_cells": {
            "T0-WB-OFFICIAL": "baseline",
            "S0-WB": "continuation",
            "S0-NB": "continuation",
        },
        "source_model_sha256": source_hashes,
        "epoch20_val_select_pesq": epoch20,
        "converged_val_select_pesq": converged,
        "val_select_pesq_delta": deltas,
        "selection": {
            cell: {
                "best_epoch": int(cells[cell]["best_epoch"]),
                "stop_epoch": int(cells[cell]["stop_epoch"]),
                "stop_reason": str(cells[cell]["stop_reason"]),
                "ceiling_limited": bool(
                    int(cells[cell]["best_epoch"]) >= 50
                ),
            }
            for cell in STUDENT_CONTINUATION_CELL_ORDER
        },
    }
    baseline_contract = {
        "passed": True,
        "teacher": "T0-WB-OFFICIAL",
        "students": list(STUDENT_CONTINUATION_CELL_ORDER),
        "teacher_checkpoint_sha256": source_hashes["T0-WB-OFFICIAL"],
    }
    git = _git_state()
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "evaluated",
        "mode": "full",
        "campaign_scope": CONVERGED_BASELINE_SCOPE,
        "verification_only": False,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "sources": {
            "baseline": {
                "run_id": baseline_root.name,
                "git_commit": baseline_provenance.get("git_commit"),
                "summary_sha256": sha256(
                    baseline_root / "metrics" / "campaign_summary.json"
                ),
            },
            "continuation": {
                "run_id": continuation_root.name,
                "git_commit": continuation_provenance.get("git_commit"),
                "summary_sha256": sha256(
                    continuation_root / "metrics" / "campaign_summary.json"
                ),
            },
        },
        "baseline_closure_contract": closure_contract,
    }
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    source_config = continuation_root / "provenance" / "config_resolved.yaml"
    (run_root / "provenance").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, run_root / "provenance" / "config_resolved.yaml")

    report = _write_report(
        run_root=run_root,
        cells=cells,
        proxies={},
        selected_teacher="T0-WB-OFFICIAL",
        teacher_gate=None,
        mode="full",
        verification_only=False,
        campaign_scope=CONVERGED_BASELINE_SCOPE,
        cell_order=BASELINE_CELL_ORDER,
        comparison_pairs={},
        baseline_contract=baseline_contract,
    )
    report["baseline_closure_contract"] = closure_contract

    comparison_csv = run_root / "metrics" / "epoch20_vs_converged.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell",
                "pesq_mode",
                "epoch20_best_epoch",
                "epoch20_val_select_pesq",
                "converged_best_epoch",
                "converged_stop_epoch",
                "converged_val_select_pesq",
                "delta",
                "stop_reason",
                "ceiling_limited",
            ],
        )
        writer.writeheader()
        for cell in STUDENT_CONTINUATION_CELL_ORDER:
            writer.writerow(
                {
                    "cell": cell,
                    "pesq_mode": "nb" if cell.endswith("-NB") else "wb",
                    "epoch20_best_epoch": baseline_cells[cell]["best_epoch"],
                    "epoch20_val_select_pesq": epoch20[cell],
                    "converged_best_epoch": cells[cell]["best_epoch"],
                    "converged_stop_epoch": cells[cell]["stop_epoch"],
                    "converged_val_select_pesq": converged[cell],
                    "delta": deltas[cell],
                    "stop_reason": cells[cell]["stop_reason"],
                    "ceiling_limited": closure_contract["selection"][cell][
                        "ceiling_limited"
                    ],
                }
            )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, cell, color in zip(
        axes,
        STUDENT_CONTINUATION_CELL_ORDER,
        ("tab:blue", "tab:orange"),
    ):
        history_path = (
            continuation_root / "cells" / cell / "training_history.csv"
        )
        epochs: list[int] = []
        scores: list[float] = []
        with history_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("val_select_pesq") not in {None, ""}:
                    epochs.append(int(row["epoch"]))
                    scores.append(float(row["val_select_pesq"]))
        axis.plot(epochs, scores, color=color, label="val_select PESQ")
        axis.axvline(20, color="black", linestyle="--", label="old ceiling")
        axis.axvline(
            int(cells[cell]["best_epoch"]),
            color="green",
            linestyle=":",
            label="selected",
        )
        axis.set_title(
            f"{cell} ({'PESQ-NB' if cell.endswith('-NB') else 'PESQ-WB'})"
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("PESQ")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("S0 continuation: epoch-20 baseline to early stopping")
    figure.tight_layout()
    convergence_plot = run_root / "reports" / "convergence_comparison.png"
    figure.savefig(convergence_plot, dpi=160)
    plt.close(figure)

    lines = [
        "# Converged official-teacher S0 baseline closure",
        "",
        "Evidence status: **reproduced and audited candidate**.",
        "",
        "The official MetricGAN+ WB teacher is unchanged. The two students were "
        "continued from immutable epoch-20 optimizer states under the declared "
        "max-50, plateau-LR and early-stopping policy.",
        "",
        "## Epoch-20 versus converged selection",
        "",
        "| Cell | Protocol | Epoch-20 | Converged | Delta | Stop |",
        "|---|---|---:|---:|---:|---|",
    ]
    for cell in STUDENT_CONTINUATION_CELL_ORDER:
        protocol = "PESQ-NB" if cell.endswith("-NB") else "PESQ-WB"
        lines.append(
            f"| {cell} | {protocol} | {epoch20[cell]:.6f} | "
            f"{converged[cell]:.6f} | {deltas[cell]:+.6f} | "
            f"best {cells[cell]['best_epoch']}, "
            f"{cells[cell]['stop_reason']} at {cells[cell]['stop_epoch']} |"
        )
    lines.extend(
        [
            "",
            "## Final profile-matched metrics",
            "",
            "| Cell | Split | PESQ | STOI | SI-SDR | Delta-SNR | Support |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in BASELINE_CELL_ORDER:
        for key, label in (
            ("val_rank_metrics", "val_rank"),
            ("val_select_metrics", "val_select"),
            ("test_metrics", "test"),
        ):
            metrics = dict(cells[cell].get(key) or {})
            lines.append(
                f"| {cell} | {label} | "
                f"{float(metrics.get('pesq_mean', float('nan'))):.6f} | "
                f"{float(metrics.get('stoi_mean', float('nan'))):.6f} | "
                f"{float(metrics.get('sisdr_mean', float('nan'))):.6f} | "
                f"{float(metrics.get('delta_snr_mean', float('nan'))):.6f} | "
                f"{int(metrics.get('count') or 0)} |"
            )
    lines.extend(
        [
            "",
            "WB and NB PESQ use different bandwidth protocols and are never "
            "pooled or directly ranked against one another.",
            "",
            "Both students stopped through early stopping and neither selected "
            "the epoch-50 ceiling. The result is one-seed evidence; uncertainty "
            "across seeds is not established.",
            "",
            "Source packages:",
            f"- baseline: `{baseline_root.name}`",
            f"- continuation: `{continuation_root.name}`",
            "",
        ]
    )
    report_path = run_root / "reports" / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    report["report"] = report_path.as_posix()
    report["convergence_plot"] = convergence_plot.as_posix()
    report["epoch_comparison_csv"] = comparison_csv.as_posix()
    _atomic_json(run_root / "metrics" / "campaign_summary.json", report)

    provenance["report"] = report_path.as_posix()
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": "evaluated",
            "campaign_mode": "full",
            "campaign_scope": CONVERGED_BASELINE_SCOPE,
            "current_stage": "AUDIT",
            "valid_for_promotion": False,
        },
    )
    audit = audit_campaign_run(run_root)
    provenance["status"] = "audited" if audit["valid"] else "failed"
    provenance["audit"] = audit
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": "audited" if audit["valid"] else "failed",
            "campaign_mode": "full",
            "campaign_scope": CONVERGED_BASELINE_SCOPE,
            "current_stage": None,
            "valid_for_promotion": bool(audit["valid"] and not git["dirty"]),
        },
    )
    if not audit["valid"]:
        raise RuntimeError(f"Converged baseline audit failed: {audit['issues']}")
    return {
        "run_root": run_root.as_posix(),
        "audit": audit,
        "baseline_closure_contract": closure_contract,
    }


def _portable_metric_payload(value: Any) -> Any:
    """Remove regenerable sample paths while preserving aggregate evidence."""
    if isinstance(value, dict):
        return {
            key: ([] if key == "sample_paths" else _portable_metric_payload(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_metric_payload(item) for item in value]
    return value


def _portable_baseline_cell(
    cell: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    scalar_keys = (
        "audit_only",
        "best_epoch",
        "best_score",
        "best_val_rank_pesq",
        "best_val_select_dnsmos_ovr",
        "best_val_select_pesq",
        "context_frames",
        "early_stopped",
        "erb_bands",
        "global_step",
        "guidance_classic",
        "inference_seconds_10s",
        "loss_recipe",
        "metric_discriminator_mode",
        "model_family",
        "postfilter_mode",
        "postfilter_preset",
        "qat",
        "quantize_dynamic",
        "run_name",
        "seed",
        "selection_guardrail_metric",
        "selection_guardrail_min",
        "selection_metric",
        "spectral_native_gate",
        "stop_epoch",
        "stop_reason",
        "target_floor",
        "threshold_met",
        "train_postfilter",
        "variant",
    )
    portable = {
        key: copy.deepcopy(payload[key])
        for key in scalar_keys
        if key in payload
    }
    for key in (
        "val_rank_metrics",
        "val_rank_metrics_by_split",
        "val_select_metrics",
        "val_select_metrics_by_split",
        "test_metrics",
        "test_metrics_by_split",
    ):
        if key in payload:
            portable[key] = _portable_metric_payload(payload[key])
    if "continued_from" in payload:
        source = dict(payload.get("continued_from") or {})
        portable["continued_from"] = {
            key: source[key]
            for key in (
                "epoch",
                "best_epoch",
                "best_score",
                "model_sha256",
                "training_state_sha256",
            )
            if key in source
        }
    portable["checkpoint_out"] = f"models/{cell}.pt"
    if cell in STUDENT_CONTINUATION_CELL_ORDER:
        portable["history_csv"] = f"metrics/training/{cell}.csv"
        portable["history_json"] = f"metrics/training/{cell}.json"
        portable["history_plot"] = f"reports/training/{cell}.png"
    return portable


def _artifact_manifest(run_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if relative == "import_manifest.json":
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }


def _write_portable_history(source: Path, target: Path) -> None:
    """Copy a training history without machine-bound cache/manifest paths."""
    private_fields = {"teacher_cache_manifest", "train_manifest"}
    if source.suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [
                field
                for field in (reader.fieldnames or [])
                if field not in private_fields
            ]
            rows = [
                {key: value for key, value in row.items() if key in fieldnames}
                for row in reader
            ]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))

        def scrub(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: scrub(item)
                    for key, item in value.items()
                    if key not in private_fields
                }
            if isinstance(value, list):
                return [scrub(item) for item in value]
            return value

        _atomic_json(target, scrub(payload))
        return
    shutil.copy2(source, target)


def _public_package_issues(run_root: Path) -> list[str]:
    issues: list[str] = []
    forbidden_text = (
        "/home/",
        "/media/",
        "/mnt/",
        "/srv/",
        "\\\\Users\\\\",
        "kingston",
    )
    forbidden_suffixes = {".wav", ".flac", ".mp3", ".ogg"}
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if path.suffix.lower() in forbidden_suffixes:
            issues.append(f"audio artifact is not publicable: {relative}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            issues.append(f"artifact exceeds 100 MiB: {relative}")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".png"}:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        for marker in forbidden_text:
            if marker.lower() in text:
                issues.append(f"private location marker {marker!r}: {relative}")
    return issues


def promote_converged_baseline(
    *,
    source_run_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Export an audited converged S0 baseline as a portable Git package."""
    source_root = Path(source_run_dir).expanduser().resolve()
    destination = REPO_ROOT / "experiments" / "runs" / run_id
    if destination.exists():
        raise FileExistsError(f"Promotion destination already exists: {destination}")
    promotion_git = _git_state()
    if promotion_git["dirty"]:
        raise ValueError(
            "Refusing to promote a baseline from a dirty worktree."
        )
    source_status = json.loads(
        (source_root / "status.json").read_text(encoding="utf-8")
    )
    source_audit = audit_campaign_run(source_root)
    if not source_audit["valid"] or not bool(
        source_status.get("valid_for_promotion")
    ):
        raise ValueError("Source baseline is not audited and promotable.")
    source_summary = json.loads(
        (source_root / "metrics" / "campaign_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_provenance = json.loads(
        (source_root / "provenance" / "provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if source_summary.get("campaign_scope") != CONVERGED_BASELINE_SCOPE:
        raise ValueError("Only a converged S0 baseline may be promoted here.")

    for name in (
        "provenance",
        "logs",
        "metrics",
        "metrics/training",
        "models",
        "reports",
        "reports/training",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)

    source_inventory = dict(source_summary["model_inventory"])
    model_inventory: dict[str, Any] = {}
    for cell in BASELINE_CELL_ORDER:
        source_model = _run_artifact_path(
            source_root,
            dict(source_inventory[cell])["path"],
        )
        if source_model is None or not source_model.is_file():
            raise FileNotFoundError(f"Missing selected source model: {cell}")
        target_model = destination / "models" / f"{cell}.pt"
        shutil.copy2(source_model, target_model)
        torch.load(target_model, map_location="cpu", weights_only=True)
        observed_hash = sha256(target_model)
        if observed_hash != dict(source_inventory[cell])["sha256"]:
            raise ValueError(f"Selected model hash changed during copy: {cell}")
        model_inventory[cell] = {
            "path": f"models/{cell}.pt",
            "sha256": observed_hash,
            "bytes": target_model.stat().st_size,
        }

    copies = {
        "metrics/canonical_metrics.csv": source_summary["canonical_metrics_csv"],
        "metrics/epoch20_vs_converged.csv": source_summary[
            "epoch_comparison_csv"
        ],
        "reports/test_pesq_by_cell.png": source_summary["test_pesq_plot"],
        "reports/convergence_comparison.png": source_summary[
            "convergence_plot"
        ],
        "reports/report.md": source_summary["report"],
    }
    for target_relative, source_value in copies.items():
        source_path = _run_artifact_path(source_root, source_value)
        if source_path is None or not source_path.is_file():
            raise FileNotFoundError(f"Missing promotion source: {target_relative}")
        shutil.copy2(source_path, destination / target_relative)

    closure = copy.deepcopy(source_summary["baseline_closure_contract"])
    continuation_id = str(closure["continuation_run_id"])
    continuation_root = source_root.parent / continuation_id
    for cell in STUDENT_CONTINUATION_CELL_ORDER:
        source_cell = continuation_root / "cells" / cell
        for suffix, target_relative in (
            ("training_history.csv", f"metrics/training/{cell}.csv"),
            ("training_history.json", f"metrics/training/{cell}.json"),
            ("training_history.png", f"reports/training/{cell}.png"),
        ):
            source_path = source_cell / suffix
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing training evidence: {source_path}")
            _write_portable_history(source_path, destination / target_relative)

    cells = {
        cell: _portable_baseline_cell(
            cell,
            dict(source_summary["cells"][cell]),
        )
        for cell in BASELINE_CELL_ORDER
    }
    portable_summary = {
        "schema_version": 1,
        "dataset": "VoiceBank+DEMAND",
        "campaign_scope": CONVERGED_BASELINE_SCOPE,
        "expected_cells": list(BASELINE_CELL_ORDER),
        "verification_only": False,
        "selected_teacher": "T0-WB-OFFICIAL",
        "cells": cells,
        "metric_proxies": {},
        "paired_deltas": {},
        "model_inventory": model_inventory,
        "canonical_metrics_csv": "metrics/canonical_metrics.csv",
        "test_pesq_plot": "reports/test_pesq_by_cell.png",
        "report": "reports/report.md",
        "convergence_plot": "reports/convergence_comparison.png",
        "epoch_comparison_csv": "metrics/epoch20_vs_converged.csv",
        "baseline_contract": copy.deepcopy(
            source_summary["baseline_contract"]
        ),
        "baseline_closure_contract": closure,
        "public_artifact_policy": {
            "included": [
                "selected model weights",
                "aggregate metrics",
                "student training histories",
                "figures and report",
                "portable configuration and hash provenance",
            ],
            "excluded": [
                "VoiceBank+DEMAND audio",
                "generated evaluation audio",
                "teacher cache",
                "training-state checkpoints",
                "replay buffers",
            ],
        },
    }
    if "corrective_evaluation" in source_summary:
        portable_summary["corrective_evaluation"] = copy.deepcopy(
            source_summary["corrective_evaluation"]
        )
    _atomic_json(
        destination / "metrics" / "campaign_summary.json",
        portable_summary,
    )
    _atomic_json(destination / "metrics" / "summary.json", portable_summary)

    source_config_path = source_root / "provenance" / "config_resolved.yaml"
    config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    config["dataset"].update(
        {
            "manifest_root": "${METRICGAN_MANIFEST_ROOT}",
            "train_fit": "${METRICGAN_MANIFEST_ROOT}/train_fit.csv",
            "val_rank": "${METRICGAN_MANIFEST_ROOT}/val_rank.csv",
            "val_select": "${METRICGAN_MANIFEST_ROOT}/val_select.csv",
            "test": "${METRICGAN_MANIFEST_ROOT}/test.csv",
        }
    )
    config["runtime"].update(
        {
            "run_root": "${METRICGAN_RUN_ROOT}",
            "shared_venv": "${METRICGAN_SHARED_VENV}",
        }
    )
    config["teacher_cache"]["root"] = (
        "${METRICGAN_RUN_ROOT}/../teacher_cache_store"
    )
    config_path = destination / "provenance" / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    manifest_hashes: dict[str, str] = {}
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    for key in ("train_fit", "val_rank", "val_select", "test"):
        manifest = Path(str(source_config["dataset"][key]))
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing source manifest: {key}")
        manifest_hashes[key] = sha256(manifest)
    command = (
        "METRICGAN_MANIFEST_ROOT=<voicebank-manifests> "
        "METRICGAN_RUN_ROOT=<desktop-local-runs> "
        "METRICGAN_SHARED_VENV=<desktop-shared-venv> "
        "python campaign.py run-baseline --run-id "
        f"{closure['baseline_run_id']} && "
        "python campaign.py continue-students --source-run-dir "
        f"<local-runs>/{closure['baseline_run_id']} --run-id "
        f"{closure['continuation_run_id']}"
    )
    (destination / "provenance" / "command.txt").write_text(
        command + "\n",
        encoding="utf-8",
    )
    (destination / "provenance" / "environment.txt").write_text(
        "\n".join(
            (
                f"python={platform.python_version()}",
                f"torch={torch.__version__}",
                "device=cuda",
                "dataset=VoiceBank+DEMAND (external, read-only)",
                "",
            )
        ),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "audited",
        "evidence_status": "reproduced",
        "campaign_scope": CONVERGED_BASELINE_SCOPE,
        "verification_only": False,
        "git_commit": source_provenance.get("git_commit"),
        "git_dirty": False,
        "promotion_git_commit": promotion_git["commit"],
        "promotion_git_dirty": promotion_git["dirty"],
        "seed": 0,
        "config_sha256": sha256(config_path),
        "manifest_sha256": manifest_hashes,
        "source_package": {
            "run_id": source_root.name,
            "summary_sha256": sha256(
                source_root / "metrics" / "campaign_summary.json"
            ),
            "audit_sha256": sha256(source_root / "reports" / "audit.json"),
        },
        "sources": copy.deepcopy(source_provenance.get("sources") or {}),
        "selected_model_sha256": {
            cell: model_inventory[cell]["sha256"]
            for cell in BASELINE_CELL_ORDER
        },
        "path_binding": (
            "Dataset, run-root and shared-venv paths are supplied through "
            "environment variables; no machine-local path is published."
        ),
    }
    if "corrective_evaluation" in source_summary:
        provenance["corrective_evaluation"] = copy.deepcopy(
            source_summary["corrective_evaluation"]
        )
    _atomic_json(
        destination / "provenance" / "provenance.json",
        provenance,
    )
    _atomic_json(
        destination / "status.json",
        {
            "status": "valid",
            "campaign_mode": "full",
            "campaign_scope": CONVERGED_BASELINE_SCOPE,
            "valid_for_promotion": True,
        },
    )
    (destination / "logs" / "promotion.log").write_text(
        "Portable baseline package created from independently audited S0 "
        "closure; no dataset, cache, replay, audio, or training state copied.\n",
        encoding="utf-8",
    )

    audit = audit_campaign_run(destination)
    privacy_issues = _public_package_issues(destination)
    if not audit["valid"] or privacy_issues:
        raise RuntimeError(
            "Promoted package validation failed: "
            f"audit={audit['issues']} privacy={privacy_issues}"
        )
    manifest = _artifact_manifest(destination)
    manifest.update(
        {
            "run_id": run_id,
            "source_run_id": source_root.name,
            "copy_mode": "selected-and-hash-verified",
            "excluded_classes": portable_summary[
                "public_artifact_policy"
            ]["excluded"],
        }
    )
    _atomic_json(destination / "import_manifest.json", manifest)
    return {
        "run_root": destination.as_posix(),
        "audit": audit,
        "privacy_issues": privacy_issues,
        "artifact_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def reevaluate_converged_baseline(
    config: dict[str, Any],
    *,
    source_run_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Re-evaluate selected S0 models with padding-invariant inference."""
    source_root = Path(source_run_dir).expanduser().resolve()
    source_audit = audit_campaign_run(source_root)
    if not source_audit["valid"]:
        raise ValueError(f"Source package audit failed: {source_audit['issues']}")
    source_summary = json.loads(
        (source_root / "metrics" / "campaign_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if source_summary.get("campaign_scope") != CONVERGED_BASELINE_SCOPE:
        raise ValueError("Corrective evaluation requires a converged S0 package.")
    dataset_audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("Refusing baseline re-evaluation from a dirty worktree.")
    require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
    config["runtime"]["device"] = require_training_cuda(
        str(config["runtime"]["device"])
    )
    run_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve() / run_id
    )
    run_root.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "mode": "full",
        "campaign_scope": CONVERGED_BASELINE_SCOPE,
        "verification_only": False,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "dataset_audit": dataset_audit,
        "corrective_evaluation": {
            "cause": (
                "Variable-length bidirectional teacher inference previously "
                "treated right-padding as future context."
            ),
            "protocol": "per-utterance true-length inference",
            "source_run_id": source_root.name,
            "source_summary_sha256": sha256(
                source_root / "metrics" / "campaign_summary.json"
            ),
        },
    }
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    (run_root / "provenance" / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _atomic_json(
        run_root / "status.json",
        {
            "status": "running",
            "campaign_mode": "full",
            "campaign_scope": CONVERGED_BASELINE_SCOPE,
            "valid_for_promotion": False,
        },
    )
    _mark_stage(
        run_root,
        stage="preflight",
        status="completed",
        details={"dataset": dataset_audit, "git": git},
    )

    source_cells = dict(source_summary["cells"])
    source_inventory = dict(source_summary["model_inventory"])
    cells: dict[str, dict[str, Any]] = {}
    for cell in BASELINE_CELL_ORDER:
        source_payload = dict(source_cells[cell])
        source_model = _run_artifact_path(
            source_root,
            dict(source_inventory[cell])["path"],
        )
        if source_model is None or not source_model.is_file():
            raise FileNotFoundError(f"Missing corrective-evaluation model: {cell}")
        bandwidth = "nb" if cell.endswith("-NB") else "wb"
        frontend = (
            dict(config["model"]["teacher_frontend"])
            if cell == "T0-WB-OFFICIAL"
            else None
        )
        evaluated = _run_cell(
            config=config,
            run_root=run_root,
            cell=cell,
            family=str(source_payload["model_family"]),
            bandwidth=bandwidth,
            loss_recipe=str(source_payload["loss_recipe"]),
            epochs=0,
            lr=0.0,
            seed=int(source_payload["seed"]),
            init_checkpoint=source_model.as_posix(),
            include_test=True,
            evaluate_init_checkpoint=True,
            frontend=frontend,
            mode="full",
        )
        for key in (
            "best_epoch",
            "stop_epoch",
            "stop_reason",
            "early_stopped",
            "global_step",
            "continued_from",
        ):
            if key in source_payload:
                evaluated[key] = copy.deepcopy(source_payload[key])
        shutil.copy2(source_model, Path(str(evaluated["checkpoint_out"])))
        _atomic_json(
            Path(str(evaluated["checkpoint_out"])).parent / "summary.json",
            evaluated,
        )
        cells[cell] = evaluated

    baseline_contract = copy.deepcopy(source_summary["baseline_contract"])
    closure_contract = copy.deepcopy(
        source_summary["baseline_closure_contract"]
    )
    for cell in STUDENT_CONTINUATION_CELL_ORDER:
        corrected = float(cells[cell]["val_select_metrics"]["pesq_mean"])
        closure_contract["converged_val_select_pesq"][cell] = corrected
        closure_contract["val_select_pesq_delta"][cell] = corrected - float(
            closure_contract["epoch20_val_select_pesq"][cell]
        )
    report = _write_report(
        run_root=run_root,
        cells=cells,
        proxies={},
        selected_teacher="T0-WB-OFFICIAL",
        teacher_gate=None,
        mode="full",
        verification_only=False,
        campaign_scope=CONVERGED_BASELINE_SCOPE,
        cell_order=BASELINE_CELL_ORDER,
        comparison_pairs={},
        baseline_contract=baseline_contract,
    )
    report["baseline_closure_contract"] = closure_contract
    comparison_target = run_root / "metrics" / "epoch20_vs_converged.csv"
    convergence_target = run_root / "reports" / "convergence_comparison.png"
    comparison_source = _run_artifact_path(
        source_root,
        source_summary["epoch_comparison_csv"],
    )
    if comparison_source is None or not comparison_source.is_file():
        raise FileNotFoundError("Missing source artifact: epoch_comparison_csv")
    with comparison_source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        comparison_rows = list(reader)
        comparison_fields = list(reader.fieldnames or [])
    for row in comparison_rows:
        cell = str(row["cell"])
        corrected = closure_contract["converged_val_select_pesq"][cell]
        row["converged_val_select_pesq"] = str(corrected)
        row["delta"] = str(
            closure_contract["val_select_pesq_delta"][cell]
        )
    with comparison_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    convergence_source = _run_artifact_path(
        source_root,
        source_summary["convergence_plot"],
    )
    if convergence_source is None or not convergence_source.is_file():
        raise FileNotFoundError("Missing source artifact: convergence_plot")
    shutil.copy2(convergence_source, convergence_target)
    report["epoch_comparison_csv"] = comparison_target.as_posix()
    report["convergence_plot"] = convergence_target.as_posix()
    report["corrective_evaluation"] = provenance["corrective_evaluation"]
    _atomic_json(run_root / "metrics" / "campaign_summary.json", report)
    return _finish_run(
        run_root=run_root,
        provenance=provenance,
        report=report,
        mode="full",
        git=git,
        promotion_gate_passed=True,
    )


def _resume_equivalence_issues(
    control_state: dict[str, Any],
    resumed_state: dict[str, Any],
    *,
    control_checkpoint: Path,
    resumed_checkpoint: Path,
) -> list[str]:
    issues: list[str] = []
    for key in ("best_epoch", "epochs_without_improve", "global_step"):
        if int(control_state.get(key, -1)) != int(resumed_state.get(key, -1)):
            issues.append(f"state mismatch: {key}")
    if abs(
        float(control_state.get("best_score", float("nan")))
        - float(resumed_state.get("best_score", float("nan")))
    ) > 1e-9:
        issues.append("state mismatch: best_score")

    control_lrs = [
        float(group["lr"])
        for group in dict(control_state["optimizer_state"])["param_groups"]
    ]
    resumed_lrs = [
        float(group["lr"])
        for group in dict(resumed_state["optimizer_state"])["param_groups"]
    ]
    if control_lrs != resumed_lrs:
        issues.append("optimizer LR mismatch")
    if control_state.get("scheduler_state") != resumed_state.get(
        "scheduler_state"
    ):
        issues.append("scheduler state mismatch")

    def tensor_mapping_equal(
        left: dict[str, torch.Tensor],
        right: dict[str, torch.Tensor],
    ) -> bool:
        return set(left) == set(right) and all(
            torch.equal(left[key].cpu(), right[key].cpu())
            for key in left
        )

    if not tensor_mapping_equal(
        dict(control_state["model_state"]),
        dict(resumed_state["model_state"]),
    ):
        issues.append("final model state mismatch")
    control_package = torch.load(
        control_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    resumed_package = torch.load(
        resumed_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if not tensor_mapping_equal(
        dict(control_package["state_dict"]),
        dict(resumed_package["state_dict"]),
    ):
        issues.append("best checkpoint state mismatch")

    control_history = [
        {
            key: row.get(key)
            for key in (
                "row_type",
                "epoch",
                "global_step",
                "selection_score",
                "improved",
                "epochs_without_improve",
                "early_stop_triggered",
                "lr_after_eval",
            )
        }
        for row in list(control_state.get("history_rows") or [])
        if row.get("row_type") in {"init", "epoch"}
    ]
    resumed_history = [
        {
            key: row.get(key)
            for key in (
                "row_type",
                "epoch",
                "global_step",
                "selection_score",
                "improved",
                "epochs_without_improve",
                "early_stop_triggered",
                "lr_after_eval",
            )
        }
        for row in list(resumed_state.get("history_rows") or [])
        if row.get("row_type") in {"init", "epoch"}
    ]
    if control_history != resumed_history:
        issues.append("post-evaluation history mismatch")
    return issues


def smoke_resume_equivalence(
    config: dict[str, Any],
    *,
    source_run_dir: str | Path,
    run_id: str,
    cell: str = "S0-NB",
) -> dict[str, Any]:
    """Fault-inject one post-evaluation interruption and compare with control."""
    if cell not in STUDENT_CONTINUATION_CELL_ORDER:
        raise ValueError(f"Unsupported resume-smoke cell: {cell}")
    dataset_audit = validate_campaign_config(config)
    git = _git_state()
    if git["dirty"]:
        raise RuntimeError("Resume smoke requires a clean committed snapshot.")
    require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
    device = require_training_cuda(str(config["runtime"]["device"]))
    config["runtime"]["device"] = device

    source_root = Path(source_run_dir).expanduser().resolve()
    source_audit = audit_campaign_run(source_root)
    if not source_audit["valid"]:
        raise ValueError(f"Resume-smoke source audit failed: {source_audit['issues']}")
    source_summary = json.loads(
        (source_root / "metrics" / "campaign_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_cell = dict(source_summary["cells"][cell])
    source_state_path = source_root / "cells" / cell / "training_state.pt"
    source_model_path = source_root / "models" / f"{cell}.pt"
    if not source_state_path.is_file() or not source_model_path.is_file():
        raise FileNotFoundError(f"Missing source state/model for {cell}")
    source_state = torch.load(
        source_state_path,
        map_location="cpu",
        weights_only=True,
    )
    source_config = dict(source_state.get("config") or {})
    for state_key, dataset_key in (
        ("train_csv", "train_fit"),
        ("val_rank_csv", "val_rank"),
        ("val_select_csv", "val_select"),
    ):
        state_manifest = Path(str(source_config.get(state_key) or ""))
        current_manifest = Path(str(config["dataset"][dataset_key]))
        if (
            not state_manifest.is_file()
            or sha256(state_manifest) != sha256(current_manifest)
        ):
            raise ValueError(f"Resume-smoke source manifest mismatch: {state_key}")
    teacher_cache_manifest = Path(
        str(source_cell.get("teacher_cache_manifest") or "")
    )
    if not teacher_cache_manifest.is_file():
        raise FileNotFoundError("Resume-smoke teacher cache is missing.")

    run_root = (
        Path(str(config["runtime"]["run_root"])).expanduser().resolve()
        / run_id
    )
    run_root.mkdir(parents=True, exist_ok=False)
    source_epoch = int(source_state.get("epoch") or 0)
    target_epochs = source_epoch + 2
    bandwidth = "nb" if cell.endswith("-NB") else "wb"
    effective = _effective_training(config, "smoke")
    common = {
        "config": config,
        "run_root": run_root,
        "family": str(source_cell["model_family"]),
        "bandwidth": bandwidth,
        "loss_recipe": str(source_cell["loss_recipe"]),
        "epochs": target_epochs,
        "lr": float(source_config.get("lr") or effective["student_lr"]),
        "seed": int(source_cell["seed"]),
        "teacher_cache_manifest": teacher_cache_manifest.as_posix(),
        "include_test": False,
        "resume_training_state": source_state_path.as_posix(),
        "early_stop_patience": 0,
        "lr_factor": float(config["training"]["student_lr_factor"]),
        "lr_patience": int(config["training"]["student_lr_patience"]),
        "min_lr": float(config["training"]["student_min_lr"]),
        "mode": "smoke",
    }
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "campaign_scope": "resume_equivalence_smoke",
        "verification_only": True,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "dataset_audit": dataset_audit,
        "source": {
            "run_id": source_root.name,
            "cell": cell,
            "state_sha256": sha256(source_state_path),
            "model_sha256": sha256(source_model_path),
            "epoch": source_epoch,
        },
    }
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    (run_root / "provenance" / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    control = _experiment_config(
        **common,
        cell=f"{cell}-CONTROL",
    )
    interrupted = _experiment_config(
        **common,
        cell=f"{cell}-INTERRUPTED",
    )
    control.optimizer_lr_override_after_resume = 0.0
    interrupted.optimizer_lr_override_after_resume = 0.0
    control.min_lr = 0.0
    interrupted.min_lr = 0.0
    shutil.copy2(source_model_path, control.checkpoint_out)
    shutil.copy2(source_model_path, interrupted.checkpoint_out)
    control_summary = run_experiment(control)
    interrupted.interrupt_after_evaluation_epoch = source_epoch + 1
    try:
        run_experiment(interrupted)
    except PlannedTrainingInterruption:
        pass
    else:
        raise RuntimeError("Resume smoke did not trigger the planned interruption.")
    interrupted.interrupt_after_evaluation_epoch = None
    interrupted.resume_training_state = interrupted.training_state_out
    resumed_summary = run_experiment(interrupted)

    control_state = torch.load(
        control.training_state_out,
        map_location="cpu",
        weights_only=True,
    )
    resumed_state = torch.load(
        interrupted.training_state_out,
        map_location="cpu",
        weights_only=True,
    )
    issues = _resume_equivalence_issues(
        control_state,
        resumed_state,
        control_checkpoint=Path(control.checkpoint_out),
        resumed_checkpoint=Path(interrupted.checkpoint_out),
    )
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "valid": not issues,
        "verification_only": True,
        "cell": cell,
        "source_epoch": source_epoch,
        "interrupted_after_epoch": source_epoch + 1,
        "final_epoch": target_epochs,
        "optimizer_updates_disabled_for_equivalence": True,
        "control": {
            "best_epoch": control_summary["best_epoch"],
            "best_score": control_summary["best_score"],
            "model_sha256": sha256(control.checkpoint_out),
            "training_state_sha256": sha256(control.training_state_out),
        },
        "resumed": {
            "best_epoch": resumed_summary["best_epoch"],
            "best_score": resumed_summary["best_score"],
            "model_sha256": sha256(interrupted.checkpoint_out),
            "training_state_sha256": sha256(interrupted.training_state_out),
        },
        "issues": issues,
    }
    _atomic_json(run_root / "reports" / "resume_audit.json", result)
    provenance["status"] = "smoke-passed" if result["valid"] else "failed"
    provenance["resume_audit"] = result
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": provenance["status"],
            "campaign_scope": "resume_equivalence_smoke",
            "valid_for_promotion": False,
        },
    )
    if issues:
        raise RuntimeError(f"Resume equivalence smoke failed: {issues}")
    return {"run_root": run_root.as_posix(), **result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(REPO_ROOT / "configs" / "voicebank_campaign.yaml").as_posix(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    smoke = subparsers.add_parser("smoke-all")
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument("--allow-dirty-smoke", action="store_true")
    pilot = subparsers.add_parser("pilot-all")
    pilot.add_argument("--run-id", required=True)
    full = subparsers.add_parser("run-all")
    full.add_argument("--run-id", required=True)
    baseline_smoke = subparsers.add_parser("smoke-baseline")
    baseline_smoke.add_argument("--run-id", required=True)
    baseline_smoke.add_argument("--allow-dirty-smoke", action="store_true")
    baseline_pilot = subparsers.add_parser("pilot-baseline")
    baseline_pilot.add_argument("--run-id", required=True)
    baseline_full = subparsers.add_parser("run-baseline")
    baseline_full.add_argument("--run-id", required=True)
    continuation = subparsers.add_parser("continue-students")
    continuation.add_argument("--source-run-dir", required=True)
    continuation.add_argument("--run-id", required=True)
    closure = subparsers.add_parser("close-baseline")
    closure.add_argument("--baseline-run-dir", required=True)
    closure.add_argument("--continuation-run-dir", required=True)
    closure.add_argument("--run-id", required=True)
    promotion = subparsers.add_parser("promote-baseline")
    promotion.add_argument("--source-run-dir", required=True)
    promotion.add_argument("--run-id", required=True)
    reevaluate_baseline = subparsers.add_parser("reevaluate-baseline")
    reevaluate_baseline.add_argument("--source-run-dir", required=True)
    reevaluate_baseline.add_argument("--run-id", required=True)
    resume_smoke = subparsers.add_parser("smoke-resume")
    resume_smoke.add_argument("--source-run-dir", required=True)
    resume_smoke.add_argument("--run-id", required=True)
    resume_smoke.add_argument(
        "--cell",
        choices=STUDENT_CONTINUATION_CELL_ORDER,
        default="S0-NB",
    )
    teacher_calibration_smoke = subparsers.add_parser(
        "smoke-teacher-calibration"
    )
    teacher_calibration_smoke.add_argument("--run-id", required=True)
    teacher_calibration_smoke.add_argument(
        "--allow-dirty-smoke",
        action="store_true",
    )
    teacher_calibration = subparsers.add_parser("calibrate-teacher")
    teacher_calibration.add_argument("--run-id", required=True)
    teacher_smoke = subparsers.add_parser("smoke-teacher")
    teacher_smoke.add_argument("--calibration-run-dir", required=True)
    teacher_smoke.add_argument("--run-id", required=True)
    teacher_smoke.add_argument("--allow-dirty-smoke", action="store_true")
    teacher_pilot = subparsers.add_parser("pilot-teacher")
    teacher_pilot.add_argument("--calibration-run-dir", required=True)
    teacher_pilot.add_argument("--run-id", required=True)
    d2_support = subparsers.add_parser("prepare-d2-support")
    d2_support.add_argument("--run-id", required=True)
    d2_support.add_argument("--teacher-cache-manifest", required=True)
    d2_support.add_argument("--teacher-cache-metadata", required=True)
    d2_support_audit = subparsers.add_parser("audit-d2-support")
    d2_support_audit.add_argument("--run-dir", required=True)
    d2_smoke = subparsers.add_parser("smoke-d2")
    d2_smoke.add_argument("--support-run-dir", required=True)
    d2_smoke.add_argument("--run-id", required=True)
    d2_train = subparsers.add_parser("train-d2")
    d2_train.add_argument("--support-run-dir", required=True)
    d2_train.add_argument("--run-id", required=True)
    d2_range_support = subparsers.add_parser("prepare-d2-range-support")
    d2_range_support.add_argument("--base-support-run-dir", required=True)
    d2_range_support.add_argument("--run-id", required=True)
    d2_range_support_audit = subparsers.add_parser("audit-d2-range-support")
    d2_range_support_audit.add_argument("--run-dir", required=True)
    d2_range_smoke = subparsers.add_parser("smoke-d2-range")
    d2_range_smoke.add_argument("--support-run-dir", required=True)
    d2_range_smoke.add_argument("--run-id", required=True)
    d2_range_train = subparsers.add_parser("train-d2-range")
    d2_range_train.add_argument("--support-run-dir", required=True)
    d2_range_train.add_argument("--run-id", required=True)
    t3_support = subparsers.add_parser("prepare-t3-support")
    t3_support.add_argument("--run-id", required=True)
    t3_support.add_argument("--teacher-cache-manifest", required=True)
    t3_support.add_argument("--teacher-cache-metadata", required=True)
    t3_support.add_argument(
        "--t2-support-path",
        action="append",
        required=True,
        help="Repeat for every distinct T2 base support to exclude.",
    )
    t3_support_audit = subparsers.add_parser("audit-t3-support")
    t3_support_audit.add_argument("--run-dir", required=True)
    t3_weights = subparsers.add_parser("calibrate-t3-weights")
    t3_weights.add_argument("--support-run-dir", required=True)
    t3_weights.add_argument("--teacher-checkpoint", required=True)
    t3_candidates = subparsers.add_parser("generate-t3-candidates")
    t3_candidates.add_argument("--support-run-dir", required=True)
    t3_candidates.add_argument("--teacher-checkpoint", required=True)
    t3_direction_audit = subparsers.add_parser("audit-t3-direction")
    t3_direction_audit.add_argument("--support-run-dir", required=True)
    for command in ("smoke-t3-teacher", "train-t3-teacher"):
        t3_teacher = subparsers.add_parser(command)
        t3_teacher.add_argument("--support-run-dir", required=True)
        t3_teacher.add_argument("--teacher-checkpoint", required=True)
        t3_teacher.add_argument("--teacher-cache-manifest", required=True)
        t3_teacher.add_argument("--run-id", required=True)
        if command == "smoke-t3-teacher":
            t3_teacher.add_argument("--allow-dirty-smoke", action="store_true")
        else:
            t3_teacher.add_argument("--resume", action="store_true")
    t4_scan = subparsers.add_parser("scan-t4-logit-bias")
    t4_scan.add_argument("--baseline-run-dir", required=True)
    t4_scan.add_argument("--teacher-checkpoint", required=True)
    t4_scan.add_argument("--run-id", required=True)
    for command in ("smoke-t4-microstep", "train-t4-microstep"):
        t4_micro = subparsers.add_parser(command)
        t4_micro.add_argument("--baseline-run-dir", required=True)
        t4_micro.add_argument("--t4a-run-dir", required=True)
        t4_micro.add_argument("--support-run-dir", required=True)
        t4_micro.add_argument("--teacher-checkpoint", required=True)
        t4_micro.add_argument("--teacher-cache-manifest", required=True)
        t4_micro.add_argument("--run-id", required=True)
        if command == "train-t4-microstep":
            t4_micro.add_argument("--resume", action="store_true")
    for command in ("smoke-t5-frequency", "search-t5-frequency"):
        t5_search = subparsers.add_parser(command)
        t5_search.add_argument("--baseline-run-dir", required=True)
        t5_search.add_argument("--t4b-run-dir", required=True)
        t5_search.add_argument("--support-run-dir", required=True)
        t5_search.add_argument("--teacher-checkpoint", required=True)
        t5_search.add_argument("--run-id", required=True)
    audit = subparsers.add_parser("audit-run")
    audit.add_argument("--run-dir", required=True)
    monitor = subparsers.add_parser("monitor-run")
    monitor.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit-run":
        result = audit_campaign_run(args.run_dir)
    elif args.command == "audit-d2-support":
        result = audit_d2_support(args.run_dir)
    elif args.command == "audit-d2-range-support":
        result = audit_d2_range_support(args.run_dir)
    elif args.command == "audit-t3-support":
        result = audit_t3_identities(
            Path(args.run_dir).expanduser().resolve()
            / "support"
            / "identities.json"
        )
    elif args.command == "audit-t3-direction":
        run_root = Path(args.support_run_dir).expanduser().resolve()
        result = audit_t3_direction(
            candidates_path=run_root / "support" / "candidates" / "candidates.json",
            weights_path=run_root / "support" / "weights.json",
            output_dir=run_root / "reports",
        )
        provenance_path = run_root / "provenance" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["status"] = (
            "direction_gate_passed" if result["passed"] else "direction_gate_failed"
        )
        provenance["direction_audit_sha256"] = sha256(
            run_root / "reports" / "direction_audit.json"
        )
        provenance["direction_gate_passed"] = result["passed"]
        _atomic_json(provenance_path, provenance)
        _atomic_json(
            run_root / "status.json",
            {
                "status": provenance["status"],
                "campaign_scope": "t3_direction_support",
                "valid_for_promotion": False,
                "direction_gate_passed": result["passed"],
            },
        )
    elif args.command == "monitor-run":
        result = monitor_campaign_run(args.run_dir)
    elif args.command == "close-baseline":
        result = close_converged_baseline(
            baseline_run_dir=args.baseline_run_dir,
            continuation_run_dir=args.continuation_run_dir,
            run_id=args.run_id,
        )
    elif args.command == "promote-baseline":
        result = promote_converged_baseline(
            source_run_dir=args.source_run_dir,
            run_id=args.run_id,
        )
    elif args.command == "reevaluate-baseline":
        config = load_campaign_config(args.config)
        result = reevaluate_converged_baseline(
            config,
            source_run_dir=args.source_run_dir,
            run_id=args.run_id,
        )
    elif args.command == "prepare-d2-support":
        config = load_campaign_config(args.config)
        dataset_audit = validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError("D2 support preparation requires a clean snapshot.")
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        run_root = (
            Path(str(config["runtime"]["run_root"])).expanduser().resolve()
            / args.run_id
        )
        run_root.mkdir(parents=True, exist_ok=False)
        provenance = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "running",
            "campaign_scope": "t2_d2_fixed_support",
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "dataset_audit": dataset_audit,
            "teacher_cache_manifest_sha256": sha256(
                args.teacher_cache_manifest
            ),
            "teacher_cache_metadata_sha256": sha256(
                args.teacher_cache_metadata
            ),
        }
        _atomic_json(run_root / "provenance" / "provenance.json", provenance)
        (run_root / "provenance" / "config_resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        training = dict(config["training"])
        support = prepare_d2_support(
            train_manifest=config["dataset"]["train_fit"],
            teacher_cache_manifest=args.teacher_cache_manifest,
            teacher_cache_metadata=args.teacher_cache_metadata,
            output_dir=run_root / "support",
            expected_teacher_sha256=config["model"][
                "teacher_checkpoint_sha256"
            ],
            train_rows=int(training["d2_train_rows"]),
            calibration_rows=int(training["d2_calibration_rows"]),
            audit_rows=int(training["d2_audit_rows"]),
            seed=int(training["seed"]),
            progress_callback=print,
        )
        provenance["status"] = "prepared"
        provenance["support_path"] = support["support_path"]
        provenance["support_counts"] = support["counts"]
        _atomic_json(run_root / "provenance" / "provenance.json", provenance)
        _atomic_json(
            run_root / "status.json",
            {
                "status": "prepared",
                "campaign_scope": "t2_d2_fixed_support",
                "valid_for_promotion": False,
                "support_counts": support["counts"],
            },
        )
        result = {
            "run_root": run_root.as_posix(),
            "support_path": support["support_path"],
            "counts": support["counts"],
            "coverage": support["coverage"],
            "speaker_disjoint_verified": support[
                "speaker_disjoint_verified"
            ],
            "speaker_limitation": support["speaker_limitation"],
        }
    elif args.command == "prepare-t3-support":
        config = load_campaign_config(args.config)
        dataset_audit = validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError(
                "T3 identity selection requires a clean committed snapshot."
            )
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        training = dict(config["training"])
        run_root = (
            Path(str(config["runtime"]["run_root"])).expanduser().resolve()
            / args.run_id
        )
        run_root.mkdir(parents=True, exist_ok=False)
        support = prepare_t3_identities(
            train_manifest=config["dataset"]["train_fit"],
            teacher_cache_manifest=args.teacher_cache_manifest,
            teacher_cache_metadata=args.teacher_cache_metadata,
            t2_support_paths=args.t2_support_path,
            output_dir=run_root / "support",
            expected_teacher_sha256=config["model"][
                "teacher_checkpoint_sha256"
            ],
            train_rows=int(training["t3_direction_train_rows"]),
            calibration_rows=int(training["t3_direction_calibration_rows"]),
            audit_rows=int(training["t3_direction_audit_rows"]),
            seed=int(training["t3_direction_seed"]),
        )
        support_audit = audit_t3_identities(support["identities_path"])
        provenance = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "identities_frozen",
            "campaign_scope": "t3_direction_support",
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "dataset_audit": dataset_audit,
            "identity_sha256": support_audit["identity_sha256"],
            "support_counts": support_audit["counts"],
        }
        _atomic_json(run_root / "provenance" / "provenance.json", provenance)
        (run_root / "provenance" / "config_resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        _atomic_json(
            run_root / "status.json",
            {
                "status": "identities_frozen",
                "campaign_scope": "t3_direction_support",
                "valid_for_promotion": False,
                "audit_valid": support_audit["valid"],
            },
        )
        result = {
            "run_root": run_root.as_posix(),
            "identities_path": support["identities_path"],
            "counts": support["counts"],
            "audit": support_audit,
        }
    elif args.command == "calibrate-t3-weights":
        config = load_campaign_config(args.config)
        validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError("T3 gradient calibration requires a clean snapshot.")
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        device = require_training_cuda(str(config["runtime"]["device"]))
        training = dict(config["training"])
        run_root = Path(args.support_run_dir).expanduser().resolve()
        identities_path = run_root / "support" / "identities.json"
        identity_audit = audit_t3_identities(identities_path)
        if not identity_audit["valid"]:
            raise ValueError(
                f"T3 identity audit failed: {identity_audit['issues']}"
            )
        result = calibrate_t3_weights(
            identities_path=identities_path,
            teacher_checkpoint=args.teacher_checkpoint,
            expected_teacher_sha256=config["model"][
                "teacher_checkpoint_sha256"
            ],
            output_path=run_root / "support" / "weights.json",
            device=device,
            rows=int(training["t3_weight_calibration_rows"]),
            segment_samples=int(training["t3_weight_segment_samples"]),
            logit_delta=float(training["t3_weight_mask_logit_delta"]),
        )
        provenance_path = run_root / "provenance" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["status"] = "weights_frozen"
        provenance["weight_calibration_commit"] = git["commit"]
        provenance["weights_sha256"] = sha256(
            run_root / "support" / "weights.json"
        )
        _atomic_json(provenance_path, provenance)
        _atomic_json(
            run_root / "status.json",
            {
                "status": "weights_frozen",
                "campaign_scope": "t3_direction_support",
                "valid_for_promotion": False,
                "audit_valid": True,
            },
        )
    elif args.command == "generate-t3-candidates":
        config = load_campaign_config(args.config)
        validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError("T3 candidate generation requires a clean snapshot.")
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        device = require_training_cuda(str(config["runtime"]["device"]))
        run_root = Path(args.support_run_dir).expanduser().resolve()
        identities_path = run_root / "support" / "identities.json"
        if not (run_root / "support" / "weights.json").is_file():
            raise ValueError("T3 weights must be frozen before candidate generation.")
        result = generate_t3_mask_candidates(
            identities_path=identities_path,
            teacher_checkpoint=args.teacher_checkpoint,
            expected_teacher_sha256=config["model"][
                "teacher_checkpoint_sha256"
            ],
            output_dir=run_root / "support" / "candidates",
            device=device,
            progress_callback=print,
        )
        provenance_path = run_root / "provenance" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["status"] = "candidates_complete"
        provenance["candidate_generation_commit"] = git["commit"]
        provenance["candidates_sha256"] = sha256(result["candidates_path"])
        _atomic_json(provenance_path, provenance)
        _atomic_json(
            run_root / "status.json",
            {
                "status": "candidates_complete",
                "campaign_scope": "t3_direction_support",
                "valid_for_promotion": False,
                "candidate_count": result["candidate_count"],
            },
        )
    elif args.command == "prepare-d2-range-support":
        config = load_campaign_config(args.config)
        dataset_audit = validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError(
                "D2-RANGE support preparation requires a clean snapshot."
            )
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        base_root = Path(args.base_support_run_dir).expanduser().resolve()
        base_audit = audit_d2_support(base_root)
        if not base_audit["valid"]:
            raise ValueError(f"Base D2 support failed audit: {base_audit['issues']}")
        base_path = base_root / "support" / "support.json"
        run_root = (
            Path(str(config["runtime"]["run_root"])).expanduser().resolve()
            / args.run_id
        )
        run_root.mkdir(parents=True, exist_ok=False)
        provenance = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "running",
            "campaign_scope": "t2_d2_range_support",
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "dataset_audit": dataset_audit,
            "base_support": {
                "run_id": base_root.name,
                "support_sha256": sha256(base_path),
                "audit": base_audit,
            },
        }
        _atomic_json(run_root / "provenance" / "provenance.json", provenance)
        (run_root / "provenance" / "config_resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        support = prepare_d2_range_support(
            base_support_path=base_path,
            output_dir=run_root / "support",
            progress_callback=print,
        )
        support_audit = audit_d2_range_support(run_root)
        provenance["status"] = "prepared" if support_audit["valid"] else "failed"
        provenance["support_sha256"] = sha256(
            run_root / "support" / "support.json"
        )
        provenance["support_audit"] = support_audit
        _atomic_json(run_root / "provenance" / "provenance.json", provenance)
        _atomic_json(
            run_root / "status.json",
            {
                "status": provenance["status"],
                "campaign_scope": "t2_d2_range_support",
                "valid_for_promotion": False,
                "candidate_count": support["range_candidate_count"],
            },
        )
        result = {
            "run_root": run_root.as_posix(),
            "support_path": support["support_path"],
            "candidate_count": support["range_candidate_count"],
            "coverage": support["range_coverage"],
            "audit": support_audit,
        }
    elif args.command in {
        "smoke-d2",
        "train-d2",
        "smoke-d2-range",
        "train-d2-range",
    }:
        config = load_campaign_config(args.config)
        dataset_audit = validate_campaign_config(config)
        git = _git_state()
        if git["dirty"]:
            raise RuntimeError("D2 fitting requires a clean committed snapshot.")
        require_shared_venv(Path(str(config["runtime"]["shared_venv"])))
        device = require_training_cuda(str(config["runtime"]["device"]))
        support_root = Path(args.support_run_dir).expanduser().resolve()
        range_mode = args.command in {"smoke-d2-range", "train-d2-range"}
        support_audit = (
            audit_d2_range_support(support_root)
            if range_mode
            else audit_d2_support(support_root)
        )
        if not support_audit["valid"]:
            raise ValueError(f"D2 support audit failed: {support_audit['issues']}")
        support_path = support_root / "support" / "support.json"
        run_root = (
            Path(str(config["runtime"]["run_root"])).expanduser().resolve()
            / args.run_id
        )
        run_root.mkdir(parents=True, exist_ok=False)
        mode = "smoke" if args.command.startswith("smoke-") else "full"
        strategy = "D2-RANGE" if range_mode else "D2-OFFICIAL"
        provenance = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "running",
            "campaign_scope": (
                "t2_d2_range" if range_mode else "t2_d2_official"
            ),
            "strategy": strategy,
            "mode": mode,
            "verification_only": mode == "smoke",
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "dataset_audit": dataset_audit,
            "support_source": {
                "run_id": support_root.name,
                "support_sha256": sha256(support_path),
                "audit": support_audit,
            },
        }
        _atomic_json(run_root / "provenance" / "provenance.json", provenance)
        (run_root / "provenance" / "config_resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        training = dict(config["training"])
        try:
            result = fit_d2_official(
                support_path=support_path,
                output_dir=run_root,
                device=device,
                max_epochs=(
                    1 if mode == "smoke" else int(training["d2_max_epochs"])
                ),
                current_rows=(
                    2 if mode == "smoke" else int(training["d2_current_rows"])
                ),
                lr=float(training["metric_discriminator_lr"]),
                history_portion=float(
                    training["metric_discriminator_history_portion"]
                ),
                lr_factor=float(training["d2_lr_factor"]),
                lr_patience=int(training["d2_lr_patience"]),
                min_lr=float(training["d2_min_lr"]),
                early_stop_patience=int(training["d2_early_stop_patience"]),
                grad_clip=float(training["grad_clip"]),
                seed=int(training["seed"]),
                evaluation_limit=(2 if mode == "smoke" else None),
                run_directional_audit=True,
                strict_gate=mode == "full",
                strategy=strategy,
                progress_callback=print,
            )
        except BaseException as exc:
            provenance["status"] = "failed"
            provenance["failure"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(
                run_root / "provenance" / "provenance.json",
                provenance,
            )
            _atomic_json(
                run_root / "status.json",
                {
                    "status": "failed",
                    "campaign_scope": provenance["campaign_scope"],
                    "valid_for_promotion": False,
                },
            )
            raise
        provenance["status"] = result["status"]
        provenance["selected_checkpoint_sha256"] = result[
            "selected_checkpoint_sha256"
        ]
        provenance["gate_passed"] = result["passed"]
        _atomic_json(run_root / "provenance" / "provenance.json", provenance)
        _atomic_json(
            run_root / "status.json",
            {
                "status": result["status"],
                "campaign_scope": provenance["campaign_scope"],
                "valid_for_promotion": False,
                "verification_only": mode == "smoke",
                "gate_passed": result["passed"],
            },
        )
    elif args.command in {
        "smoke-teacher-calibration",
        "calibrate-teacher",
    }:
        config = load_campaign_config(args.config)
        result = run_teacher_calibration(
            config,
            run_id=args.run_id,
            mode=(
                "smoke"
                if args.command == "smoke-teacher-calibration"
                else "pilot"
            ),
            allow_dirty_smoke=bool(
                getattr(args, "allow_dirty_smoke", False)
            ),
        )
    elif args.command in {"smoke-teacher", "pilot-teacher"}:
        config = load_campaign_config(args.config)
        result = run_teacher_trial(
            config,
            calibration_run_dir=args.calibration_run_dir,
            run_id=args.run_id,
            mode="smoke" if args.command == "smoke-teacher" else "pilot",
            allow_dirty_smoke=bool(
                getattr(args, "allow_dirty_smoke", False)
            ),
        )
    elif args.command in {"smoke-t3-teacher", "train-t3-teacher"}:
        config = load_campaign_config(args.config)
        result = run_t3_matched_pilot(
            config,
            support_run_dir=args.support_run_dir,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_cache_manifest=args.teacher_cache_manifest,
            run_id=args.run_id,
            mode="smoke" if args.command == "smoke-t3-teacher" else "full",
            allow_dirty_smoke=bool(
                getattr(args, "allow_dirty_smoke", False)
            ),
            resume=bool(getattr(args, "resume", False)),
        )
    elif args.command == "scan-t4-logit-bias":
        config = load_campaign_config(args.config)
        result = run_t4_calibration(
            config,
            baseline_run_dir=args.baseline_run_dir,
            teacher_checkpoint=args.teacher_checkpoint,
            run_id=args.run_id,
        )
    elif args.command in {"smoke-t4-microstep", "train-t4-microstep"}:
        config = load_campaign_config(args.config)
        result = run_t4_microstep_trial(
            config,
            baseline_run_dir=args.baseline_run_dir,
            t4a_run_dir=args.t4a_run_dir,
            support_run_dir=args.support_run_dir,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_cache_manifest=args.teacher_cache_manifest,
            run_id=args.run_id,
            smoke=args.command == "smoke-t4-microstep",
            resume=bool(getattr(args, "resume", False)),
        )
    elif args.command in {"smoke-t5-frequency", "search-t5-frequency"}:
        config = load_campaign_config(args.config)
        result = run_t5_frequency_trial(
            config,
            baseline_run_dir=args.baseline_run_dir,
            t4b_run_dir=args.t4b_run_dir,
            support_run_dir=args.support_run_dir,
            teacher_checkpoint=args.teacher_checkpoint,
            run_id=args.run_id,
            smoke=args.command == "smoke-t5-frequency",
        )
    else:
        config = load_campaign_config(args.config)
    if args.command == "validate":
        result = validate_campaign_config(config)
    elif args.command in {
        "smoke-all",
        "pilot-all",
        "run-all",
        "smoke-baseline",
        "pilot-baseline",
        "run-baseline",
    }:
        try:
            mode = {
                "smoke-all": "smoke",
                "pilot-all": "pilot",
                "run-all": "full",
                "smoke-baseline": "smoke",
                "pilot-baseline": "pilot",
                "run-baseline": "full",
            }[args.command]
            result = run_all(
                config,
                run_id=args.run_id,
                mode=mode,
                baseline_only=args.command.endswith("-baseline"),
                allow_dirty_smoke=bool(
                    getattr(args, "allow_dirty_smoke", False)
                ),
            )
        except BaseException as exc:
            run_root = (
                Path(str(config["runtime"]["run_root"])).expanduser().resolve()
                / args.run_id
            )
            if run_root.exists():
                provenance_path = run_root / "provenance" / "provenance.json"
                provenance = (
                    json.loads(provenance_path.read_text(encoding="utf-8"))
                    if provenance_path.exists()
                    else {"run_id": args.run_id}
                )
                provenance["status"] = "failed"
                provenance["failure"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _atomic_json(provenance_path, provenance)
                _atomic_json(
                    run_root / "status.json",
                    {"status": "failed", "valid_for_promotion": False},
                )
            raise
    elif args.command == "continue-students":
        try:
            result = continue_official_students(
                config,
                source_run_dir=args.source_run_dir,
                run_id=args.run_id,
            )
        except BaseException as exc:
            run_root = (
                Path(str(config["runtime"]["run_root"])).expanduser().resolve()
                / args.run_id
            )
            if run_root.exists():
                provenance_path = run_root / "provenance" / "provenance.json"
                provenance = (
                    json.loads(provenance_path.read_text(encoding="utf-8"))
                    if provenance_path.exists()
                    else {"run_id": args.run_id}
                )
                provenance["status"] = "failed"
                provenance["failure"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _atomic_json(provenance_path, provenance)
                _atomic_json(
                    run_root / "status.json",
                    {"status": "failed", "valid_for_promotion": False},
                )
            raise
    elif args.command == "smoke-resume":
        result = smoke_resume_equivalence(
            config,
            source_run_dir=args.source_run_dir,
            run_id=args.run_id,
            cell=args.cell,
        )
    elif args.command not in {
        "audit-run",
        "close-baseline",
        "promote-baseline",
        "reevaluate-baseline",
        "smoke-teacher-calibration",
        "calibrate-teacher",
        "smoke-teacher",
        "pilot-teacher",
        "monitor-run",
        "prepare-d2-support",
        "audit-d2-support",
        "smoke-d2",
        "train-d2",
        "prepare-d2-range-support",
        "audit-d2-range-support",
        "smoke-d2-range",
        "train-d2-range",
        "prepare-t3-support",
        "audit-t3-support",
        "calibrate-t3-weights",
        "generate-t3-candidates",
        "audit-t3-direction",
        "smoke-t3-teacher",
        "train-t3-teacher",
        "scan-t4-logit-bias",
        "smoke-t4-microstep",
        "train-t4-microstep",
        "smoke-t5-frequency",
        "search-t5-frequency",
        "smoke-resume",
        "continue-students",
    }:
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command in {
        "audit-run",
        "audit-d2-support",
        "audit-d2-range-support",
        "audit-t3-support",
        "audit-t3-direction",
    } and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
