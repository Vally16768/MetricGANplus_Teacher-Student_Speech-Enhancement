#!/usr/bin/env python3
"""Canonical VoiceBank-only MetricGAN+ WB teacher -> WB/NB students campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
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
from sebench.runtime import require_shared_venv, require_training_cuda  # noqa: E402
from sebench.teacher_cache import (  # noqa: E402
    TeacherCacheTarget,
    build_multi_target_teacher_cache,
)
from sebench.training import ExperimentConfig, run_experiment  # noqa: E402


CELL_ORDER = (
    "T0-WB-OFFICIAL",
    "S0-WB",
    "S0-NB",
    "T1-WB-BASE",
    "T1-WB-METRIC",
    "S1-WB",
    "S1-NB",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        lr_factor=0.5,
        lr_patience=2,
        min_lr=1e-6,
        early_stop_patience=int(
            effective.get("early_stop_patience", 0 if mode == "smoke" else 5)
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
        metric_discriminator_history_portion=float(
            effective["metric_discriminator_history_portion"]
        ),
        metric_discriminator_replay_root=(
            (cell_root / "metricgan_replay").as_posix()
            if alternating_metric_discriminator
            else None
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


def _metric_rows(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
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
    teacher_gate: dict[str, Any],
    mode: str,
    verification_only: bool,
) -> dict[str, Any]:
    metrics_dir = run_root / "metrics"
    reports_dir = run_root / "reports"
    models_dir = run_root / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = _metric_rows(cells)
    csv_path = metrics_dir / "canonical_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cell", "bandwidth", "split", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    deltas: dict[str, dict[str, float]] = {}
    pairs = {
        "teacher_control_vs_official": ("T0-WB-OFFICIAL", "T1-WB-BASE"),
        "teacher_metric_vs_official": ("T0-WB-OFFICIAL", "T1-WB-METRIC"),
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
        for cell in CELL_ORDER
    ]
    axis.bar(CELL_ORDER, pesq)
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
        "verification_only": verification_only,
        "selected_teacher": selected_teacher,
        "teacher_promotion_gate": teacher_gate,
        "cells": cells,
        "metric_proxies": proxies,
        "paired_deltas": deltas,
        "model_inventory": model_inventory,
        "canonical_metrics_csv": csv_path.as_posix(),
        "test_pesq_plot": figure_path.as_posix(),
    }

    lines = [
        "# VoiceBank MetricGAN+ campaign report",
        "",
        f"Status: {'verification-only ' + mode if verification_only else 'evaluated campaign'}",
        "",
        f"Selected teacher by val_select PESQ: `{selected_teacher}`.",
        "",
        "## Paired stage deltas",
        "",
    ]
    for label, values in deltas.items():
        rendered = ", ".join(f"{key}={value:+.4f}" for key, value in values.items())
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
    cells = dict(summary.get("cells") or {})
    if set(cells) != set(CELL_ORDER):
        issues.append(
            f"cell set mismatch: expected={sorted(CELL_ORDER)} actual={sorted(cells)}"
        )

    csv_values: dict[tuple[str, str, str], str] = {}
    csv_path = Path(str(summary.get("canonical_metrics_csv") or ""))
    if not csv_path.is_file():
        issues.append("missing canonical metrics CSV")
    else:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                csv_values[(row["cell"], row["split"], row["metric"])] = row["value"]

    reported_samples: set[Path] = set()
    for cell in CELL_ORDER:
        payload = dict(cells.get(cell) or {})
        bandwidth = "nb" if "-NB" in cell else "wb"
        profile = PROFILES[bandwidth]
        for split_key, split_label in (
            ("val_rank_metrics", "val_rank"),
            ("val_select_metrics", "val_select"),
            ("test_metrics", "test"),
        ):
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
                        sample = Path(str(raw_path))
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
    for cell in CELL_ORDER:
        model = dict(model_inventory.get(cell) or {})
        path = Path(str(model.get("path") or ""))
        if not path.is_file():
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
        path = Path(str(summary.get(path_key) or default_path))
        if not path.is_file():
            issues.append(f"missing report artifact: {path_key}")

    verification_only = bool(summary.get("verification_only"))
    if bool(provenance.get("verification_only")) != verification_only:
        issues.append("verification_only mismatch between summary and provenance")
    if verification_only and bool(status.get("valid_for_promotion")):
        issues.append("verification-only run incorrectly marked promotable")
    teacher_gate = dict(summary.get("teacher_promotion_gate") or {})
    if not teacher_gate:
        issues.append("missing teacher promotion gate")
    elif not verification_only and not bool(teacher_gate.get("passed")):
        issues.append("full run teacher promotion gate did not pass")

    result = {
        "schema_version": 1,
        "run_id": provenance.get("run_id"),
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


def run_all(
    config: dict[str, Any],
    *,
    run_id: str,
    mode: str,
    allow_dirty_smoke: bool,
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
            mode=mode,
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
    provenance["status"] = "evaluated"
    provenance["selected_teacher"] = selected_teacher
    provenance["teacher_promotion_gate"] = teacher_gate
    provenance["report"] = report["report"]
    _atomic_json(run_root / "provenance" / "provenance.json", provenance)
    _atomic_json(
        run_root / "status.json",
        {
            "status": "evaluated",
            "campaign_mode": mode,
            "current_stage": "AUDIT",
            "valid_for_promotion": False,
        },
    )
    package_audit = execute("AUDIT", lambda: audit_campaign_run(run_root))
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
            "current_stage": None,
            "valid_for_promotion": bool(
                mode == "full" and not git["dirty"] and teacher_gate["passed"]
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
    audit = subparsers.add_parser("audit-run")
    audit.add_argument("--run-dir", required=True)
    monitor = subparsers.add_parser("monitor-run")
    monitor.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit-run":
        result = audit_campaign_run(args.run_dir)
    elif args.command == "monitor-run":
        result = monitor_campaign_run(args.run_dir)
    else:
        config = load_campaign_config(args.config)
    if args.command == "validate":
        result = validate_campaign_config(config)
    elif args.command in {"smoke-all", "pilot-all", "run-all"}:
        try:
            mode = {
                "smoke-all": "smoke",
                "pilot-all": "pilot",
                "run-all": "full",
            }[args.command]
            result = run_all(
                config,
                run_id=args.run_id,
                mode=mode,
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
    elif args.command not in {"audit-run", "monitor-run"}:
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "audit-run" and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
