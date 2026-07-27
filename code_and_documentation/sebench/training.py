from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import mlflow
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from metrics.dnsmos import dnsmos_wav
from metrics.composite import composite_scores
from metrics.pesq import pesq_score
from metrics.sisdr import sisdr
from metrics.snr import delta_snr
from metrics.stoi import stoi_score
from sebench.audio import load_mono_audio, loop_to_length, manifest_hash, save_mono_audio, tensor_to_numpy_mono
from sebench.bandwidth import resolve_bandwidth
from sebench.checkpoints import load_checkpoint_package, load_model_from_checkpoint, save_checkpoint_package
from sebench.data import ManifestRow, VoiceBankDemandDataset, read_pair_manifest
from sebench.losses import (
    CompositeEnhancementLoss,
    SpeechBrainMetricDiscriminator,
    load_pesq_proxy_checkpoint,
)
from sebench.metricgan_alternating import refresh_metricgan_discriminator
from sebench.mlflow_utils import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    flatten_params,
    log_dict_artifact,
    terminate_matching_runs,
)
from sebench.models import build_enhancer
from sebench.runtime import require_cuda_device
from sebench.teacher_cache import TeacherCacheDataset


@dataclass
class ExperimentConfig:
    train_csv: str
    checkpoint_out: str
    model_family: str = "metricgan_plus_teacher_wb"
    variant: str = "base"
    loss_recipe: str = "R1"
    val_rank_csv: str | None = None
    val_select_csv: str | None = None
    test_csv: str | None = None
    rank_eval_manifests: dict[str, str] | None = None
    select_eval_manifests: dict[str, str] | None = None
    test_eval_manifests: dict[str, str] | None = None
    train_csv_schedule: list[str] | None = None
    teacher_cache_schedule: list[str] | None = None
    run_name: str | None = None
    phase: str | None = None
    epochs: int = 30
    batch_size: int | None = None
    grad_accum: int | None = None
    lr: float = 1e-3
    segment_len: int = 32000
    num_workers: int | None = None
    prefetch_factor: int | None = None
    persistent_workers: bool | None = None
    pin_memory: bool | None = None
    checkpoint_every_steps: int = 500
    checkpoint_every_minutes: float = 5.0
    checkpoint_snapshot_every_periods: int = 1
    checkpoint_keep_last: int = 2
    training_state_out: str | None = None
    resume_training_state: str | None = None
    interrupt_after_evaluation_epoch: int | None = None
    history_plot_every_epochs: int = 1
    history_plot_final_only: bool = False
    history_persist_every_periods: int = 4
    record_step_history: bool = True
    enable_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    lr_factor: float = 0.5
    lr_patience: int = 2
    min_lr: float = 1e-6
    early_stop_patience: int = 5
    min_epochs: int = 10
    eval_every: int = 2
    rank_eval_every: int | None = None
    select_eval_every: int | None = None
    grad_clip: float = 5.0
    seed: int = 0
    amp: bool = True
    scheduler: str = "plateau"
    device: str = "cuda"
    gpu_ids: list[int] | None = None
    mlflow_uri: str = DEFAULT_TRACKING_URI
    mlflow_artifact_root: str = DEFAULT_ARTIFACT_ROOT
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    parent_run_id: str | None = None
    selection_metric: str = "val_select/pesq_mean"
    target_floor: float | None = None
    selection_guardrail_metric: str | None = None
    selection_guardrail_min: float | None = None
    evaluate_init_checkpoint: bool = False
    pesq_proxy_checkpoint: str | None = None
    eval_dnsmos: bool = True
    sample_count: int = 3
    benchmark_seconds: int = 10
    benchmark_repeats: int = 3
    max_eval_files: int | None = None
    rank_max_eval_files: int | None = None
    final_max_eval_files: int | None = None
    eval_batch_size: int | None = None
    cache_eval_audio: bool = True
    rank_compute_composite: bool = True
    select_compute_composite: bool = True
    postfilter_mode: str = "none"
    postfilter_preset: str = "medium"
    train_postfilter: bool = False
    spectral_native_gate: bool = False
    teacher_source_run_id: str | None = None
    teacher_variant: str | None = None
    audit_only: bool = False
    teacher_cache_manifest: str | None = None
    guidance_classic: str = "none"
    erb_bands: int = 32
    context_frames: int = 5
    qat: bool = False
    init_checkpoint: str | None = None
    progress_json_out: str | None = None
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 320
    log_torch_model: bool = False
    log_system_metrics: bool = False
    quantize_dynamic: bool = False
    deterministic: bool = False
    optimizer_lr_override_after_resume: float | None = None
    bandwidth: str | None = None
    metric_proxy_weight: float = 0.25
    teacher_anchor_weight: float = 0.75
    metric_discriminator_mode: str = "frozen"
    metric_discriminator_lr: float = 5e-4
    metric_discriminator_rows: int = 100
    metric_discriminator_calibration_rows: int = 100
    metric_discriminator_history_portion: float = 0.2
    metric_discriminator_replay_root: str | None = None
    metric_discriminator_calibration_only: bool = False
    metric_discriminator_min_calibration_records: int = 100
    metric_discriminator_max_normalized_mae: float = 0.06
    metric_discriminator_min_pearson: float = 0.80
    metric_discriminator_min_spearman: float = 0.80
    metric_discriminator_min_prediction_std: float = 0.02
    metric_discriminator_range_tolerance_raw: float = 0.30


def _normalize_runtime_devices(device: str, gpu_ids: list[int] | None) -> tuple[str, list[int]]:
    resolved_device = require_cuda_device(device)
    normalized_gpu_ids: list[int] = []

    if gpu_ids:
        for value in gpu_ids:
            gpu_id = int(value)
            if gpu_id not in normalized_gpu_ids:
                normalized_gpu_ids.append(gpu_id)

    device_obj = torch.device(resolved_device)
    if device_obj.type == "cuda" and device_obj.index is None:
        normalized_gpu_ids.insert(0, int(torch.cuda.current_device()))
    if device_obj.type == "cuda" and device_obj.index is not None and device_obj.index not in normalized_gpu_ids:
        normalized_gpu_ids.insert(0, int(device_obj.index))

    if not normalized_gpu_ids and device_obj.type == "cuda" and device_obj.index is not None:
        normalized_gpu_ids.append(int(device_obj.index))

    if normalized_gpu_ids:
        resolved_device = f"cuda:{normalized_gpu_ids[0]}"

    return resolved_device, normalized_gpu_ids


def _wrap_model_for_runtime(model: nn.Module, device: str, gpu_ids: list[int] | None) -> nn.Module:
    if not str(device).startswith("cuda"):
        return model
    runtime_gpu_ids = [int(value) for value in list(gpu_ids or [])]
    if len(runtime_gpu_ids) <= 1:
        return model
    return nn.DataParallel(model, device_ids=runtime_gpu_ids, output_device=runtime_gpu_ids[0])


def _unwrap_runtime_model(model: nn.Module) -> nn.Module:
    runtime_model = model
    while hasattr(runtime_model, "module"):
        runtime_model = runtime_model.module
    return runtime_model.base_model if hasattr(runtime_model, "base_model") else runtime_model


def _normalize_manifest_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _clean_key_from_path(value: str) -> str:
    path = _normalize_manifest_path(value)
    for marker in ("/clean_train/", "/clean_val/", "/clean_test/", "/clean_sources/"):
        if marker in path:
            return path.split(marker, 1)[1].lstrip("/")
    return path


def _manifest_keysets(csv_path: str) -> dict[str, Any]:
    rows = read_pair_manifest(csv_path)
    pair_set: set[str] = set()
    clean_set: set[str] = set()
    for row in rows:
        pair_set.add(f"{_normalize_manifest_path(row.noisy.as_posix())}|{_normalize_manifest_path(row.clean.as_posix())}")
        clean_set.add(_clean_key_from_path(row.clean.as_posix()))
    return {
        "path": Path(csv_path).resolve().as_posix(),
        "rows": len(rows),
        "pair_set": pair_set,
        "clean_set": clean_set,
        "duplicate_pairs": len(rows) - len(pair_set),
        "duplicate_clean_keys": len(rows) - len(clean_set),
    }


def _merge_manifest_map(primary_name: str, primary_csv: str | None, extra: dict[str, str] | None) -> dict[str, str]:
    manifests: dict[str, str] = {}
    if primary_csv:
        manifests[primary_name] = str(primary_csv)
    if extra:
        for name, path in extra.items():
            if path:
                manifests[str(name)] = str(path)
    return manifests


def _validate_manifest_integrity(config: ExperimentConfig) -> None:
    manifests: dict[str, str] = {"train": config.train_csv}
    for index, manifest in enumerate(config.train_csv_schedule or [], start=1):
        manifests[f"train_schedule_{index:03d}"] = manifest

    rank_manifests = _merge_manifest_map("val_rank", config.val_rank_csv, config.rank_eval_manifests)
    select_manifests = _merge_manifest_map("val_select", config.val_select_csv, config.select_eval_manifests)
    test_manifests = _merge_manifest_map("test", config.test_csv, config.test_eval_manifests)
    manifests.update(rank_manifests)
    manifests.update(select_manifests)
    manifests.update(test_manifests)

    stats = {name: _manifest_keysets(path) for name, path in manifests.items()}
    for name, payload in stats.items():
        if payload["duplicate_pairs"] > 0 or payload["duplicate_clean_keys"] > 0:
            raise ValueError(
                f"Manifest `{name}` has duplicates (pairs={payload['duplicate_pairs']} clean_keys={payload['duplicate_clean_keys']}): "
                f"{payload['path']}"
            )

    train_pairs: set[str] = set()
    train_clean: set[str] = set()
    for name, payload in stats.items():
        if name.startswith("train"):
            train_pairs.update(payload["pair_set"])
            train_clean.update(payload["clean_set"])

    val_pairs: set[str] = set()
    val_clean: set[str] = set()
    for name in (*rank_manifests.keys(), *select_manifests.keys()):
        if name in stats:
            val_pairs.update(stats[name]["pair_set"])
            val_clean.update(stats[name]["clean_set"])

    test_pairs: set[str] = set()
    test_clean: set[str] = set()
    for name in test_manifests:
        if name in stats:
            test_pairs.update(stats[name]["pair_set"])
            test_clean.update(stats[name]["clean_set"])

    overlap_train_val_pairs = len(train_pairs & val_pairs)
    overlap_train_val_clean = len(train_clean & val_clean)
    if overlap_train_val_pairs or overlap_train_val_clean:
        raise ValueError(
            f"Data leakage train<->val detected: pair_overlap={overlap_train_val_pairs} clean_overlap={overlap_train_val_clean}"
        )

    overlap_train_test_pairs = len(train_pairs & test_pairs)
    overlap_train_test_clean = len(train_clean & test_clean)
    if overlap_train_test_pairs or overlap_train_test_clean:
        raise ValueError(
            f"Data leakage train<->test detected: pair_overlap={overlap_train_test_pairs} clean_overlap={overlap_train_test_clean}"
        )

    overlap_val_test_pairs = len(val_pairs & test_pairs)
    overlap_val_test_clean = len(val_clean & test_clean)
    if overlap_val_test_pairs or overlap_val_test_clean:
        raise ValueError(
            f"Data leakage val<->test detected: pair_overlap={overlap_val_test_pairs} clean_overlap={overlap_val_test_clean}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataloader_seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def suggest_num_workers(cpu_count: int | None = None) -> int:
    cpu_total = cpu_count or os.cpu_count() or 4
    if cpu_total <= 2:
        return 0
    return min(6, max(2, cpu_total - 2))


def suggest_runtime_profile(model_family: str, variant: str, segment_len: int) -> dict[str, int]:
    short_segment = segment_len <= 16000

    if model_family == "metricgan_plus":
        batch_size = 12 if short_segment else 8
    elif model_family in {"metricgan_plus_native8k", "metricgan_plus_teacher_wb"}:
        batch_size = 12 if short_segment else 8
    elif model_family in {
        "metricgan_plus_native8k_causal_s",
        "metricgan_plus_student_wb",
        "metricgan_plus_student_nb",
    }:
        batch_size = 16 if short_segment else 12
    elif model_family in {
        "metricgan_plus_student_wb_causal_max",
        "metricgan_plus_student_nb_causal_max",
    }:
        batch_size = 8 if short_segment else 6
    elif model_family == "metricgan_plus_native8k_causal_xs":
        batch_size = 18 if short_segment else 14
    elif model_family == "metricgan_plus_native8k_causal_n6":
        batch_size = 12 if short_segment else 10
    elif variant == "small":
        batch_size = 8 if short_segment else 4
    else:
        batch_size = 6 if short_segment else 4

    target_effective_batch = 8
    grad_accum = max(1, (target_effective_batch + batch_size - 1) // batch_size)
    eval_batch_size = min(16, max(4, batch_size * 2))
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "num_workers": suggest_num_workers(),
        "eval_batch_size": eval_batch_size,
    }


def apply_runtime_profile(config: ExperimentConfig) -> None:
    profile = suggest_runtime_profile(config.model_family, config.variant, config.segment_len)
    if config.batch_size is None or config.batch_size <= 0:
        config.batch_size = profile["batch_size"]
    if config.grad_accum is None or config.grad_accum <= 0:
        config.grad_accum = profile["grad_accum"]
    if config.num_workers is None or config.num_workers < 0:
        config.num_workers = profile["num_workers"]
    if config.eval_batch_size is None or config.eval_batch_size <= 0:
        config.eval_batch_size = profile["eval_batch_size"]


def install_termination_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _raise_interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, _raise_interrupt)
    return previous


def restore_termination_handlers(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(destination)


def _atomic_write_history_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())

    preferred = [
        "row_type",
        "epoch",
        "step_in_epoch",
        "global_step",
        "lr",
        "epoch_seconds",
        "eval_performed",
        "selection_score",
        "selection_guardrail_value",
        "selection_guardrail_passed",
        "improved",
        "epochs_without_improve",
        "early_stop_triggered",
    ]
    ordered: list[str] = [key for key in preferred if key in keys]
    ordered.extend(sorted(key for key in keys if key.startswith("train/") and key not in ordered))
    ordered.extend(sorted(key for key in keys if key.startswith("val_rank/") and key not in ordered))
    ordered.extend(sorted(key for key in keys if key.startswith("best/") and key not in ordered))
    ordered.extend(sorted(key for key in keys if key not in ordered))

    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in ordered})
    temp_path.replace(destination)


def _append_history_jsonl(rows: list[dict[str, Any]], destination: Path) -> None:
    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _render_training_history_plot(rows: list[dict[str, Any]], destination: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    def _series(key: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            row_type = str(row.get("row_type", "epoch"))
            if row_type == "epoch":
                x_value = row.get("epoch")
            elif row_type == "step":
                x_value = row.get("global_step")
            else:
                continue
            value = row.get(key)
            if isinstance(x_value, (int, float)) and isinstance(value, (int, float)):
                xs.append(float(x_value))
                ys.append(float(value))
        return xs, ys

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = list(axes.flat)

    for key, label in (
        ("train/loss", "Train Loss"),
        ("train/wave_loss", "Wave Loss"),
        ("train/spectral_loss", "Spectral Loss"),
    ):
        xs, ys = _series(key)
        if xs:
            axes_flat[0].plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=label)
    axes_flat[0].set_title("Training Losses")
    axes_flat[0].set_xlabel("Epoch / Global Step")
    axes_flat[0].set_ylabel("Loss")
    axes_flat[0].grid(alpha=0.3, linestyle="--")
    if axes_flat[0].has_data():
        axes_flat[0].legend(loc="best")

    for key, label in (
        ("val_rank/pesq_mean", "PESQ"),
        ("val_rank/stoi_mean", "STOI"),
    ):
        xs, ys = _series(key)
        if xs:
            axes_flat[1].plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=label)
    axes_flat[1].set_title("Validation Rank Quality")
    axes_flat[1].set_xlabel("Epoch / Global Step")
    axes_flat[1].set_ylabel("Metric")
    axes_flat[1].grid(alpha=0.3, linestyle="--")
    if axes_flat[1].has_data():
        axes_flat[1].legend(loc="best")

    for key, label in (
        ("val_rank/sisdr_mean", "SI-SDR"),
        ("val_rank/delta_snr_mean", "Delta SNR"),
    ):
        xs, ys = _series(key)
        if xs:
            axes_flat[2].plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=label)
    axes_flat[2].set_title("Validation Rank Signal Metrics")
    axes_flat[2].set_xlabel("Epoch / Global Step")
    axes_flat[2].set_ylabel("Metric")
    axes_flat[2].grid(alpha=0.3, linestyle="--")
    if axes_flat[2].has_data():
        axes_flat[2].legend(loc="best")

    for key, label in (
        ("lr", "Learning Rate"),
        ("epochs_without_improve", "Epochs Without Improve"),
    ):
        xs, ys = _series(key)
        if xs:
            axes_flat[3].plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=label)
    axes_flat[3].set_title("Optimization Dynamics")
    axes_flat[3].set_xlabel("Epoch / Global Step")
    axes_flat[3].set_ylabel("Value")
    axes_flat[3].grid(alpha=0.3, linestyle="--")
    if axes_flat[3].has_data():
        axes_flat[3].legend(loc="best")

    fig.tight_layout()
    temp_path = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    fig.savefig(temp_path.as_posix(), dpi=150)
    plt.close(fig)
    temp_path.replace(destination)


def _save_training_history_artifacts(
    rows: list[dict[str, Any]],
    *,
    history_json_path: Path,
    history_csv_path: Path,
    history_plot_path: Path,
    write_plot: bool,
) -> None:
    history_payload = {
        "count": len(rows),
        "updated_unix": time.time(),
        "rows": rows,
    }
    _atomic_write_json(history_payload, history_json_path)
    _atomic_write_history_csv(rows, history_csv_path)
    if write_plot:
        _render_training_history_plot(rows, history_plot_path)


def _collate_teacher_cache_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty teacher-cache batch.")
    payload: dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        values = [item[key].contiguous() for item in batch if key in item]
        if not values:
            continue
        payload[key] = torch.stack(values, dim=0)
    return payload


def build_dataloader(csv_path: str, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    using_teacher_cache = bool(config.teacher_cache_manifest and shuffle)
    if using_teacher_cache:
        dataset = TeacherCacheDataset(
            config.teacher_cache_manifest,
            segment_len=config.segment_len,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
    else:
        dataset = VoiceBankDemandDataset(csv_path, segment_len=config.segment_len, sample_rate=config.sample_rate)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    requested_workers = max(int(config.num_workers or 0), 0)
    cpu_total = os.cpu_count() or 4
    dataset_cap = max(1, min(len(dataset), int(config.batch_size or 1) * 2)) if shuffle else min(len(dataset), 4)
    available_workers = max(1, cpu_total - 2 if cpu_total > 2 else 1)
    if using_teacher_cache:
        safe_worker_cap = max(1, min(available_workers, len(dataset)))
    else:
        loader_budget = max(dataset_cap, int(config.batch_size or 1) * 8, 4)
        safe_worker_cap = max(1, min(available_workers, loader_budget, len(dataset)))
    num_workers = min(requested_workers, safe_worker_cap) if requested_workers > 0 else 0
    pin_memory = config.device.startswith("cuda") if config.pin_memory is None else bool(config.pin_memory)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": shuffle,
        "pin_memory": pin_memory,
        "worker_init_fn": dataloader_seed_worker,
        "generator": generator,
    }
    if using_teacher_cache:
        loader_kwargs["collate_fn"] = _collate_teacher_cache_batch
    if num_workers > 0:
        if config.persistent_workers is None:
            loader_kwargs["persistent_workers"] = True
        else:
            loader_kwargs["persistent_workers"] = bool(config.persistent_workers)
        default_prefetch = 2 if using_teacher_cache else 1
        prefetch = int(config.prefetch_factor) if config.prefetch_factor is not None else default_prefetch
        loader_kwargs["prefetch_factor"] = max(prefetch, 1)
    if requested_workers != num_workers:
        print(
            f"[dataloader] capped num_workers from {requested_workers} to {num_workers} "
            f"for {config.model_family}/{config.variant} shuffle={shuffle} teacher_cache={using_teacher_cache}",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"[dataloader] csv={Path(csv_path).name} rows={len(dataset)} batch_size={config.batch_size} "
        f"shuffle={shuffle} num_workers={num_workers} pin_memory={pin_memory} "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)} "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 0)} teacher_cache={using_teacher_cache}",
        file=sys.stderr,
        flush=True,
    )
    return DataLoader(
        **loader_kwargs,
    )


@torch.inference_mode()
def benchmark_loader_throughput(
    config: ExperimentConfig,
    *,
    max_steps: int = 120,
    warmup_steps: int = 30,
) -> dict[str, float]:
    loader = build_dataloader(config.train_csv, config, shuffle=True)
    iterator = iter(loader)
    measured_steps = 0
    measured_samples = 0
    data_wait = 0.0
    transfer_time = 0.0
    device_is_cuda = config.device.startswith("cuda")
    start_window = 0.0

    for step_idx in range(max_steps):
        wait_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_step = time.perf_counter() - wait_start

        transfer_start = time.perf_counter()
        if isinstance(batch, dict):
            noisy = batch["noisy"]
            clean = batch["clean"]
            noisy = noisy.to(config.device, non_blocking=device_is_cuda)
            clean = clean.to(config.device, non_blocking=device_is_cuda)
            batch_size = int(noisy.size(0))
        else:
            noisy, clean = batch
            noisy = noisy.to(config.device, non_blocking=device_is_cuda)
            clean = clean.to(config.device, non_blocking=device_is_cuda)
            batch_size = int(noisy.size(0))
        if device_is_cuda:
            torch.cuda.synchronize()
        transfer_step = time.perf_counter() - transfer_start

        if step_idx + 1 == warmup_steps:
            start_window = time.perf_counter()
        if step_idx + 1 > warmup_steps:
            data_wait += data_wait_step
            transfer_time += transfer_step
            measured_steps += 1
            measured_samples += batch_size

    elapsed = max(time.perf_counter() - start_window, 1e-9) if measured_steps > 0 else 1e-9
    wait_fraction = data_wait / max(data_wait + transfer_time, 1e-9)
    return {
        "measured_steps": float(measured_steps),
        "measured_samples": float(measured_samples),
        "elapsed_seconds": float(elapsed),
        "batches_per_second": float(measured_steps / elapsed),
        "samples_per_second": float(measured_samples / elapsed),
        "data_wait_fraction": float(wait_fraction),
    }


def autotune_loader_profile(
    config: ExperimentConfig,
    *,
    candidates_num_workers: list[int],
    candidates_prefetch_factor: list[int],
    max_steps: int = 120,
    warmup_steps: int = 30,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not candidates_num_workers:
        raise ValueError("candidates_num_workers cannot be empty.")
    if not candidates_prefetch_factor:
        raise ValueError("candidates_prefetch_factor cannot be empty.")

    results: list[dict[str, Any]] = []
    for workers in sorted(set(int(v) for v in candidates_num_workers if int(v) >= 0)):
        for prefetch in sorted(set(int(v) for v in candidates_prefetch_factor if int(v) >= 1)):
            trial_cfg = replace(
                config,
                num_workers=workers,
                prefetch_factor=prefetch,
                persistent_workers=workers > 0,
            )
            metrics = benchmark_loader_throughput(
                trial_cfg,
                max_steps=max_steps,
                warmup_steps=warmup_steps,
            )
            row = {
                "num_workers": workers,
                "prefetch_factor": prefetch,
                **metrics,
            }
            results.append(row)
            if progress_callback is not None:
                progress_callback(
                    "loader autotune "
                    f"workers={workers} prefetch={prefetch} "
                    f"samples_per_second={metrics['samples_per_second']:.2f} "
                    f"wait_fraction={metrics['data_wait_fraction']:.3f}"
                )

    if not results:
        raise RuntimeError("No valid loader autotune results.")
    winner = max(results, key=lambda item: (float(item["samples_per_second"]), -float(item["data_wait_fraction"])))
    return {
        "winner": winner,
        "results": results,
        "max_steps": int(max_steps),
        "warmup_steps": int(warmup_steps),
    }


def _selection_score(metrics: dict[str, float]) -> float:
    if "pesq_mean" in metrics:
        return float(metrics["pesq_mean"])
    if "loss" in metrics:
        return float(-metrics["loss"])
    return float("-inf")


_COMPLETED_EPOCH_STATE_REASONS = frozenset(
    {"epoch", "evaluation", "final", "best"}
)


class PlannedTrainingInterruption(RuntimeError):
    """Controlled fault injection after a durable evaluation checkpoint."""


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(
                numpy_state[1].astype(np.int64, copy=True)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(payload: dict[str, Any] | None) -> None:
    state = dict(payload or {})
    if state.get("python") is not None:
        random.setstate(tuple(state["python"]))
    numpy_payload = dict(state.get("numpy") or {})
    if numpy_payload:
        numpy_tensor = numpy_payload["state"]
        np.random.set_state(
            (
                str(numpy_payload["bit_generator"]),
                numpy_tensor.cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_payload["position"]),
                int(numpy_payload["has_gauss"]),
                float(numpy_payload["cached_gaussian"]),
            )
        )
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state.get("torch_cuda") or [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def _resume_epoch_position(payload: dict[str, Any]) -> tuple[int, int]:
    """Return next epoch and last completed epoch for a saved state."""
    resume_epoch = int(payload.get("epoch", 0) or 0)
    reason = str(payload.get("reason") or "").strip().lower()
    if reason in _COMPLETED_EPOCH_STATE_REASONS:
        return resume_epoch + 1, resume_epoch
    start_epoch = max(1, resume_epoch)
    return start_epoch, max(0, start_epoch - 1)


def _training_control_state_fields(
    *,
    scheduler: Any,
    best_score: float,
    best_epoch: int,
    best_rank_metrics: dict[str, Any],
    best_select_metrics: dict[str, Any],
    best_rank_metrics_by_split: dict[str, dict[str, Any]],
    best_select_metrics_by_split: dict[str, dict[str, Any]],
    epochs_without_improve: int,
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the post-evaluation state shared by checkpoints and resume tests."""
    return {
        "scheduler_state": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_rank_metrics": dict(best_rank_metrics),
        "best_select_metrics": dict(best_select_metrics),
        "best_rank_metrics_by_split": dict(best_rank_metrics_by_split),
        "best_select_metrics_by_split": dict(best_select_metrics_by_split),
        "epochs_without_improve": int(epochs_without_improve),
        "history_rows": list(history_rows),
    }


def _metric_discriminator_state_fields(
    *,
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    refresh_history: list[dict[str, Any]],
    replay_root: str,
) -> dict[str, Any]:
    return {
        "metric_discriminator_state": discriminator.state_dict(),
        "metric_discriminator_optimizer_state": optimizer.state_dict(),
        "metric_discriminator_refresh_history": list(refresh_history),
        "metric_discriminator_replay_root": str(Path(replay_root).resolve()),
    }


def _restore_metric_discriminator_state(
    payload: dict[str, Any],
    *,
    discriminator: SpeechBrainMetricDiscriminator,
    optimizer: torch.optim.Optimizer,
    replay_root: str,
) -> list[dict[str, Any]]:
    expected_replay = str(Path(replay_root).resolve())
    observed_replay = str(payload.get("metric_discriminator_replay_root") or "")
    if observed_replay and observed_replay != expected_replay:
        raise ValueError(
            "Metric-discriminator replay identity changed across resume."
        )
    discriminator.load_state_dict(payload["metric_discriminator_state"])
    optimizer.load_state_dict(payload["metric_discriminator_optimizer_state"])
    return list(payload.get("metric_discriminator_refresh_history") or [])


def _resolve_eval_frequency(config: ExperimentConfig) -> tuple[int, int]:
    rank_every = int(config.rank_eval_every or config.eval_every or 0)
    select_every = int(config.select_eval_every if config.select_eval_every is not None else (config.eval_every or 0))
    return max(rank_every, 0), max(select_every, 0)


def _resolve_eval_manifest_groups(config: ExperimentConfig) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    rank_manifests = _merge_manifest_map("val_rank", config.val_rank_csv, config.rank_eval_manifests)
    select_manifests = _merge_manifest_map("val_select", config.val_select_csv, config.select_eval_manifests)
    test_manifests = _merge_manifest_map("test", config.test_csv, config.test_eval_manifests)
    return rank_manifests, select_manifests, test_manifests


def _flatten_split_metrics(prefix: str, payload: dict[str, dict[str, Any]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for label, metrics in payload.items():
        for key, value in metrics.items():
            if key == "sample_paths":
                continue
            if isinstance(value, (int, float)):
                flat[f"{label}/{key}"] = float(value)
                if label == prefix:
                    flat[f"{prefix}/{key}"] = float(value)
    return flat


def _metric_expression_value(metrics: dict[str, float], expression: str) -> float | None:
    normalized = str(expression).strip()
    reducers = {
        "mean": lambda values: sum(values) / float(len(values)),
        "min": min,
        "max": max,
        "harmonic_mean": lambda values: float(len(values)) / sum(1.0 / max(value, 1e-8) for value in values),
    }
    lowered = normalized.lower()
    for name, reducer in reducers.items():
        prefix = f"{name}("
        if not lowered.startswith(prefix) or not normalized.endswith(")"):
            continue
        inner = normalized[len(prefix):-1]
        parts = [part.strip() for part in inner.split(",") if part.strip()]
        if not parts:
            return None
        values: list[float] = []
        for part in parts:
            value = _metric_value_from_flat(metrics, part)
            if value is None:
                return None
            values.append(float(value))
        return float(reducer(values))
    return None


def _metric_value_from_flat(metrics: dict[str, float], metric_name: str | None) -> float | None:
    if not metric_name:
        return None
    metric_name = str(metric_name).strip()
    expression_value = _metric_expression_value(metrics, metric_name)
    if expression_value is not None:
        return expression_value
    if metric_name in metrics:
        return float(metrics[metric_name])
    normalized = metric_name.replace("_", "/")
    if normalized in metrics:
        return float(metrics[normalized])
    return None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CompositeEnhancementLoss,
    config: ExperimentConfig,
    epoch: int,
    scaler: torch.amp.GradScaler | None,
    step_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    model.train()
    running_total = 0.0
    running_wave = 0.0
    running_spec = 0.0
    running_sisdr = 0.0
    running_noise_gate = 0.0
    running_speech_preserve = 0.0
    running_teacher_mask = 0.0
    running_teacher_wave = 0.0
    running_pesq_proxy = 0.0
    running_predicted_pesq = 0.0
    seen = 0
    data_wait_seconds = 0.0
    step_compute_seconds = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress_enabled = sys.stderr.isatty()
    total_batches = len(loader)
    progress = tqdm(
        range(total_batches),
        desc=f"Epoch {epoch:03d} [train]",
        unit="batch",
        disable=not progress_enabled,
        mininterval=1.0,
    )
    autocast_enabled = config.amp and config.device.startswith("cuda")
    iterator = iter(loader)

    for step, _ in enumerate(progress, start=1):
        wait_start = time.perf_counter()
        batch = next(iterator)
        data_wait_seconds += time.perf_counter() - wait_start

        compute_start = time.perf_counter()
        guidance = None
        teacher_wav = None
        teacher_mask_erb = None
        if isinstance(batch, dict):
            noisy = batch["noisy"]
            clean = batch["clean"]
            teacher_wav = batch.get("teacher_wav")
            teacher_mask_erb = batch.get("teacher_mask_erb")
            guidance = batch.get("guidance_sg")
        else:
            noisy, clean = batch
        noisy = noisy.to(config.device, non_blocking=config.device.startswith("cuda")).unsqueeze(1)
        clean = clean.to(config.device, non_blocking=config.device.startswith("cuda")).unsqueeze(1)
        if teacher_wav is not None:
            teacher_wav = teacher_wav.to(config.device, non_blocking=config.device.startswith("cuda")).unsqueeze(1)
        if teacher_mask_erb is not None:
            teacher_mask_erb = teacher_mask_erb.to(config.device, non_blocking=config.device.startswith("cuda"))
        if guidance is not None:
            guidance = guidance.to(config.device, non_blocking=config.device.startswith("cuda"))

        with torch.autocast(device_type="cuda" if config.device.startswith("cuda") else "cpu", enabled=autocast_enabled):
            try:
                enhanced = model(noisy, guidance=guidance) if guidance is not None else model(noisy)
            except TypeError:
                enhanced = model(noisy)
            breakdown = loss_fn(
                enhanced,
                clean,
                noisy,
                epoch=epoch,
                total_epochs=config.epochs,
                teacher_wav=teacher_wav,
                teacher_mask_erb=teacher_mask_erb,
            )
            scaled_loss = breakdown.total / float(config.grad_accum)

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = step % config.grad_accum == 0 or step == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = noisy.size(0)
        running_total += breakdown.total.item() * batch_size
        running_wave += breakdown.wave.item() * batch_size
        running_spec += breakdown.spectral.item() * batch_size
        running_sisdr += breakdown.sisdr.item() * batch_size
        running_noise_gate += breakdown.noise_gate.item() * batch_size
        running_speech_preserve += breakdown.speech_preserve.item() * batch_size
        running_teacher_mask += breakdown.teacher_mask.item() * batch_size
        running_teacher_wave += breakdown.teacher_wave.item() * batch_size
        running_pesq_proxy += breakdown.pesq_proxy.item() * batch_size
        running_predicted_pesq += breakdown.predicted_pesq.item() * batch_size
        seen += batch_size
        if progress_enabled:
            progress.set_postfix(
                loss=f"{running_total / seen:.4f}",
                wave=f"{running_wave / seen:.4f}",
                spectral=f"{running_spec / seen:.4f}",
                sisdr=f"{running_sisdr / seen:.4f}",
                teacher=f"{running_teacher_mask / seen:.4f}",
                pesq=f"{running_predicted_pesq / max(seen, 1):.3f}",
            )
        if step_callback is not None:
            step_callback(
                step,
                {
                    "train/loss": float(running_total / max(seen, 1)),
                    "train/wave_loss": float(running_wave / max(seen, 1)),
                    "train/spectral_loss": float(running_spec / max(seen, 1)),
                    "train/sisdr_loss": float(running_sisdr / max(seen, 1)),
                    "train/noise_gate_loss": float(running_noise_gate / max(seen, 1)),
                    "train/speech_preserve_loss": float(running_speech_preserve / max(seen, 1)),
                    "train/teacher_mask_loss": float(running_teacher_mask / max(seen, 1)),
                    "train/teacher_wave_loss": float(running_teacher_wave / max(seen, 1)),
                    "train/pesq_proxy_loss": float(running_pesq_proxy / max(seen, 1)),
                    "train/predicted_pesq": float(running_predicted_pesq / max(seen, 1)),
                },
            )
        step_compute_seconds += time.perf_counter() - compute_start

    total_step_seconds = max(data_wait_seconds + step_compute_seconds, 1e-9)
    return {
        "loss": running_total / max(seen, 1),
        "wave_loss": running_wave / max(seen, 1),
        "spectral_loss": running_spec / max(seen, 1),
        "sisdr_loss": running_sisdr / max(seen, 1),
        "noise_gate_loss": running_noise_gate / max(seen, 1),
        "speech_preserve_loss": running_speech_preserve / max(seen, 1),
        "teacher_mask_loss": running_teacher_mask / max(seen, 1),
        "teacher_wave_loss": running_teacher_wave / max(seen, 1),
        "pesq_proxy_loss": running_pesq_proxy / max(seen, 1),
        "predicted_pesq": running_predicted_pesq / max(seen, 1),
        "data_wait_seconds": data_wait_seconds,
        "step_compute_seconds": step_compute_seconds,
        "data_wait_fraction": float(data_wait_seconds / total_step_seconds),
    }


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    sample_path: str | Path,
    device: str,
    sample_rate: int,
    duration_seconds: int = 10,
    repeats: int = 3,
) -> float:
    noisy, sr = load_mono_audio(sample_path, sample_rate)
    noisy = loop_to_length(noisy, duration_seconds * sr).unsqueeze(0).to(device)
    timings: list[float] = []
    autocast_enabled = device.startswith("cuda")

    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=autocast_enabled):
            _ = model.denoise_single(noisy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    return float(mean(timings))


EvalAudioRow = tuple[ManifestRow, torch.Tensor, torch.Tensor, int]
_EVAL_AUDIO_CACHE: dict[tuple[str, int, int | None], list[EvalAudioRow]] = {}


def _load_eval_audio_rows(
    manifest_path: str,
    *,
    sample_rate: int,
    use_cache: bool,
    max_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[EvalAudioRow]:
    resolved = Path(manifest_path).resolve().as_posix()
    cache_key = (resolved, sample_rate, max_files)
    if use_cache and cache_key in _EVAL_AUDIO_CACHE:
        if progress_callback is not None:
            progress_callback(f"eval cache hit for {Path(manifest_path).name}: {len(_EVAL_AUDIO_CACHE[cache_key])} rows")
        return _EVAL_AUDIO_CACHE[cache_key]

    manifest_rows = read_pair_manifest(manifest_path)
    if max_files is not None:
        manifest_rows = manifest_rows[:max_files]
    if progress_callback is not None:
        progress_callback(f"loading eval audio from {Path(manifest_path).name}: {len(manifest_rows)} rows")

    loaded: list[EvalAudioRow] = []
    total_rows = len(manifest_rows)
    for row_index, row in enumerate(manifest_rows, start=1):
        noisy, sr = load_mono_audio(row.noisy, sample_rate)
        clean, _ = load_mono_audio(row.clean, sample_rate)
        loaded.append((row, noisy.contiguous(), clean.contiguous(), sr))
        if progress_callback is not None and (row_index == total_rows or row_index == 1 or row_index % 100 == 0):
            progress_callback(f"loaded eval audio {row_index}/{total_rows} from {Path(manifest_path).name}")

    if use_cache:
        _EVAL_AUDIO_CACHE[cache_key] = loaded
    return loaded


@torch.inference_mode()
def evaluate_manifest(
    model: nn.Module,
    manifest_path: str,
    device: str,
    *,
    sample_rate: int,
    bandwidth: str | None = None,
    compute_dnsmos: bool,
    compute_composite: bool = True,
    sample_dir: str | Path | None = None,
    sample_count: int = 0,
    max_files: int | None = None,
    batch_size: int = 1,
    cache_audio: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    profile = resolve_bandwidth(bandwidth, sample_rate=sample_rate)
    if compute_dnsmos and sample_rate != 16000:
        raise ValueError("DNSMOS evaluation requires 16000 Hz audio.")
    rows = _load_eval_audio_rows(
        manifest_path,
        sample_rate=sample_rate,
        use_cache=cache_audio,
        max_files=max_files,
        progress_callback=progress_callback,
    )

    pesq_values: list[float] = []
    stoi_values: list[float] = []
    sisdr_values: list[float] = []
    delta_snr_values: list[float] = []
    csig_values: list[float] = []
    cbak_values: list[float] = []
    covl_values: list[float] = []
    dnsmos_sig: list[float] = []
    dnsmos_bak: list[float] = []
    dnsmos_ovr: list[float] = []
    saved_samples: list[str] = []

    out_dir = Path(sample_dir) if sample_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    temp_dnsmos_dir = Path(tempfile.mkdtemp(prefix="sebench_dnsmos_")) if compute_dnsmos else None
    model_was_training = model.training
    model.eval()
    eval_batch_size = max(1, int(batch_size))
    autocast_enabled = device.startswith("cuda")

    try:
        row_index = 0
        total_rows = len(rows)
        while row_index < len(rows):
            batch_rows = rows[row_index:row_index + eval_batch_size]
            noisy_batch = pad_sequence([item[1] for item in batch_rows], batch_first=True)
            try:
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=autocast_enabled):
                    enhanced_batch = model.denoise_single(
                        noisy_batch.to(device, non_blocking=device.startswith("cuda"))
                    ).cpu()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.startswith("cuda") and eval_batch_size > 1:
                    torch.cuda.empty_cache()
                    eval_batch_size = max(1, eval_batch_size // 2)
                    continue
                raise
            if progress_callback is not None:
                completed = min(row_index + len(batch_rows), total_rows)
                if completed == total_rows or completed == len(batch_rows) or completed % 128 == 0:
                    progress_callback(
                        f"evaluated {completed}/{total_rows} files from {Path(manifest_path).name} "
                        f"(batch_size={eval_batch_size})"
                    )

            for batch_offset, (row, noisy_full, clean_full, sr) in enumerate(batch_rows):
                enhanced = enhanced_batch[batch_offset]
                aligned = min(noisy_full.numel(), clean_full.numel(), enhanced.numel())
                noisy = noisy_full[:aligned]
                clean = clean_full[:aligned]
                enhanced = enhanced[:aligned]

                clean_np = tensor_to_numpy_mono(clean)
                noisy_np = tensor_to_numpy_mono(noisy)
                enhanced_np = tensor_to_numpy_mono(enhanced)

                pesq = pesq_score(
                    clean_np,
                    enhanced_np,
                    sr,
                    bandwidth=profile.name,
                )
                if np.isfinite(pesq):
                    pesq_values.append(pesq)
                stoi_value = stoi_score(clean_np, enhanced_np, sr, extended=False)
                if np.isfinite(stoi_value):
                    stoi_values.append(stoi_value)
                sisdr_value = sisdr(clean_np, enhanced_np)
                if np.isfinite(sisdr_value):
                    sisdr_values.append(sisdr_value)
                delta_snr_value = delta_snr(clean_np, noisy_np, enhanced_np)
                if np.isfinite(delta_snr_value):
                    delta_snr_values.append(delta_snr_value)
                if compute_composite and np.isfinite(pesq):
                    try:
                        composite = composite_scores(clean_np, enhanced_np, sr, pesq_value=pesq)
                    except ValueError:
                        composite = None
                    if composite is not None:
                        csig_values.append(composite["csig"])
                        cbak_values.append(composite["cbak"])
                        covl_values.append(composite["covl"])

                save_sample_triplet = out_dir is not None and row_index + batch_offset < sample_count
                enhanced_path: Path | None = None
                raw_enhanced_path: Path | None = None
                if save_sample_triplet:
                    enhanced_path = out_dir / f"{Path(row.noisy).stem}_enh.wav"
                    save_mono_audio(enhanced_path, enhanced, sr)
                    if getattr(model, "postfilter_active", False) and hasattr(model, "denoise_raw"):
                        raw_enhanced = model.denoise_raw(
                            noisy.unsqueeze(0).to(device, non_blocking=device.startswith("cuda"))
                        ).squeeze(0).cpu()
                        raw_enhanced = raw_enhanced[:aligned]
                        raw_enhanced_path = out_dir / f"{Path(row.noisy).stem}_raw_enh.wav"
                        save_mono_audio(raw_enhanced_path, raw_enhanced, sr)

                if compute_dnsmos:
                    dnsmos_path = enhanced_path
                    if dnsmos_path is None and temp_dnsmos_dir is not None:
                        dnsmos_path = temp_dnsmos_dir / f"{row_index + batch_offset:05d}_enh.wav"
                        save_mono_audio(dnsmos_path, enhanced, sr)
                    if dnsmos_path is not None:
                        dns = dnsmos_wav(dnsmos_path.as_posix())
                        mos_sig = float(dns["mos_sig"])
                        mos_bak = float(dns["mos_bak"])
                        mos_ovr = float(dns["mos_ovr"])
                        if np.isfinite(mos_sig):
                            dnsmos_sig.append(mos_sig)
                        if np.isfinite(mos_bak):
                            dnsmos_bak.append(mos_bak)
                        if np.isfinite(mos_ovr):
                            dnsmos_ovr.append(mos_ovr)
                        if temp_dnsmos_dir is not None and dnsmos_path.parent == temp_dnsmos_dir:
                            dnsmos_path.unlink(missing_ok=True)

                if save_sample_triplet and enhanced_path is not None:
                    noisy_path = out_dir / f"{Path(row.noisy).stem}_noisy.wav"
                    clean_path = out_dir / f"{Path(row.clean).stem}_clean.wav"
                    save_mono_audio(noisy_path, noisy, sr)
                    save_mono_audio(clean_path, clean, sr)
                    saved_samples.append(noisy_path.as_posix())
                    saved_samples.append(clean_path.as_posix())
                    if raw_enhanced_path is not None:
                        saved_samples.append(raw_enhanced_path.as_posix())
                    saved_samples.append(enhanced_path.as_posix())

            row_index += len(batch_rows)
    finally:
        if temp_dnsmos_dir is not None:
            shutil.rmtree(temp_dnsmos_dir, ignore_errors=True)
        if model_was_training:
            model.train()

    metrics: dict[str, Any] = {
        "bandwidth": profile.name,
        "reference_bandwidth": profile.name,
        "sample_rate": profile.sample_rate,
        "pesq_mode": profile.pesq_mode,
        "count": len(rows),
        "pesq_count": len(pesq_values),
        "stoi_count": len(stoi_values),
        "sisdr_count": len(sisdr_values),
        "delta_snr_count": len(delta_snr_values),
        "composite_count": len(csig_values),
        "dnsmos_count": len(dnsmos_ovr),
        "pesq_mean": float(mean(pesq_values)) if pesq_values else float("nan"),
        "stoi_mean": float(mean(stoi_values)) if stoi_values else float("nan"),
        "sisdr_mean": float(mean(sisdr_values)) if sisdr_values else float("nan"),
        "delta_snr_mean": float(mean(delta_snr_values)) if delta_snr_values else float("nan"),
        "sample_paths": saved_samples,
    }
    if csig_values:
        metrics["csig_mean"] = float(mean(csig_values))
        metrics["cbak_mean"] = float(mean(cbak_values))
        metrics["covl_mean"] = float(mean(covl_values))
    if dnsmos_sig:
        metrics["dnsmos_sig_mean"] = float(mean(dnsmos_sig))
        metrics["dnsmos_bak_mean"] = float(mean(dnsmos_bak))
        metrics["dnsmos_ovr_mean"] = float(mean(dnsmos_ovr))
    return metrics


def _start_run(config: ExperimentConfig) -> tuple[Any, str]:
    experiment_id = configure_mlflow(
        tracking_uri=config.mlflow_uri,
        experiment_name=config.experiment_name,
        artifact_root=config.mlflow_artifact_root,
    )
    nested = mlflow.active_run() is not None or bool(config.parent_run_id)
    terminate_matching_runs(
        tracking_uri=config.mlflow_uri,
        experiment_name=config.experiment_name,
        run_name=config.run_name,
        phase=config.phase,
    )
    tags = {
        "model_family": config.model_family,
        "variant": config.variant,
        "phase": config.phase or "",
        "loss_recipe": config.loss_recipe,
    }
    if config.parent_run_id and mlflow.active_run() is None:
        tags["mlflow.parentRunId"] = config.parent_run_id
    run = mlflow.start_run(
        run_name=config.run_name,
        experiment_id=experiment_id,
        nested=nested,
        tags=tags,
        log_system_metrics=config.log_system_metrics,
    )
    return run, experiment_id


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.device, config.gpu_ids = _normalize_runtime_devices(config.device, config.gpu_ids)
    set_seed(config.seed)
    apply_runtime_profile(config)
    _validate_manifest_integrity(config)
    rank_eval_manifests, select_eval_manifests, test_eval_manifests = _resolve_eval_manifest_groups(config)
    rank_eval_every, select_eval_every = _resolve_eval_frequency(config)

    if config.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(config.device))
        if config.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = not config.deterministic
        torch.backends.cudnn.deterministic = bool(config.deterministic)
        torch.use_deterministic_algorithms(bool(config.deterministic))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    run, _ = _start_run(config)
    previous_handlers = install_termination_handlers()
    checkpoint_path = Path(config.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.final{checkpoint_path.suffix}")
    if config.training_state_out:
        training_state_path = Path(config.training_state_out)
    else:
        training_state_path = checkpoint_path.parent / f"{checkpoint_path.stem}.latest_state.pt"
    training_state_path.parent.mkdir(parents=True, exist_ok=True)
    training_state_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_states"
    training_state_dir.mkdir(parents=True, exist_ok=True)
    history_json_path = checkpoint_path.parent / "training_history.json"
    history_csv_path = checkpoint_path.parent / "training_history.csv"
    history_plot_path = checkpoint_path.parent / "training_history.png"
    progress_json_path = Path(config.progress_json_out) if config.progress_json_out else None
    history_plot_every_epochs = max(int(config.history_plot_every_epochs or 1), 1)
    history_plot_final_only = bool(config.history_plot_final_only)

    model = build_enhancer(
        config.model_family,
        config.variant,
        spectral_native_gate=config.spectral_native_gate,
        postfilter_mode=config.postfilter_mode,
        postfilter_preset=config.postfilter_preset,
        train_postfilter=config.train_postfilter,
        erb_bands=config.erb_bands,
        context_frames=config.context_frames,
        guidance_classic=config.guidance_classic,
        qat=config.qat,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        initialize_from_official=not bool(config.init_checkpoint),
    ).to(config.device)
    if config.init_checkpoint:
        init_package = load_checkpoint_package(config.init_checkpoint, map_location="cpu")
        target = _unwrap_runtime_model(model)
        target_state = target.state_dict()
        compatible_state: dict[str, torch.Tensor] = {}
        skipped_keys: list[str] = []
        for key, value in init_package["state_dict"].items():
            target_value = target_state.get(key)
            if target_value is None or target_value.shape != value.shape:
                skipped_keys.append(key)
                continue
            compatible_state[key] = value.to(dtype=target_value.dtype)
        if not compatible_state:
            raise RuntimeError(
                f"Init checkpoint `{config.init_checkpoint}` has no shape-compatible parameters for "
                f"{config.model_family}/{config.variant} "
                f"(source={init_package.get('model_family')}/{init_package.get('variant')})."
            )
        target.load_state_dict(compatible_state, strict=False)
        print(
            f"[init_checkpoint] loaded compatible params from {Path(config.init_checkpoint).name}: "
            f"source={init_package.get('model_family')}/{init_package.get('variant')} "
            f"loaded={len(compatible_state)} skipped={len(skipped_keys)}",
            file=sys.stderr,
            flush=True,
        )

    model = _wrap_model_for_runtime(model, config.device, config.gpu_ids)

    train_model: nn.Module = model
    compile_active = False
    if config.enable_torch_compile and hasattr(torch, "compile") and len(list(config.gpu_ids or [])) <= 1:
        try:
            train_model = torch.compile(model, mode=config.torch_compile_mode)
            compile_active = True
        except Exception as exc:
            mlflow.log_params({"torch_compile_enabled": "false", "torch_compile_error": str(exc)})
    mlflow.log_params(
        {
            "torch_compile_requested": str(bool(config.enable_torch_compile)).lower(),
            "torch_compile_active": str(bool(compile_active)).lower(),
            "torch_compile_mode": str(config.torch_compile_mode),
        }
    )

    alternating_discriminator = config.metric_discriminator_mode == "alternating"
    if config.metric_discriminator_mode not in {"frozen", "alternating"}:
        raise ValueError(
            "metric_discriminator_mode must be `frozen` or `alternating`."
        )
    pesq_proxy_model = None
    if config.pesq_proxy_checkpoint:
        pesq_proxy_model = load_pesq_proxy_checkpoint(
            config.pesq_proxy_checkpoint,
            device=config.device,
            freeze=not alternating_discriminator,
        )
    discriminator_optimizer: torch.optim.Optimizer | None = None
    discriminator_checkpoint_path: Path | None = None
    discriminator_refresh_history: list[dict[str, Any]] = []
    if alternating_discriminator:
        if config.loss_recipe.upper() != "T0_PESQ":
            raise ValueError(
                "Alternating MetricGAN discriminator is restricted to T0_PESQ."
            )
        if not isinstance(pesq_proxy_model, SpeechBrainMetricDiscriminator):
            raise ValueError(
                "Alternating mode requires a SpeechBrainMetricDiscriminator checkpoint."
            )
        if not config.metric_discriminator_replay_root:
            raise ValueError(
                "Alternating mode requires a Desktop-local discriminator replay root."
            )
        discriminator_optimizer = torch.optim.Adam(
            pesq_proxy_model.parameters(),
            lr=float(config.metric_discriminator_lr),
        )
        discriminator_checkpoint_path = (
            checkpoint_path.parent / "metric_discriminator.pt"
        )

    loss_fn = CompositeEnhancementLoss(
        config.loss_recipe,
        sample_rate=config.sample_rate,
        erb_bands=config.erb_bands,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        pesq_proxy=pesq_proxy_model,
        metric_proxy_weight=config.metric_proxy_weight,
        teacher_anchor_weight=config.teacher_anchor_weight,
    )
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError(f"Model family `{config.model_family}` has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.02)
    scheduler = None
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr,
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    autocast_scaler = None
    if config.amp and config.device.startswith("cuda"):
        autocast_scaler = torch.amp.GradScaler("cuda", enabled=True)

    def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temp_path)
        temp_path.replace(destination)

    def _snapshot_basename(epoch: int, global_step: int, reason: str) -> str:
        return f"epoch{epoch:03d}_step{global_step:08d}_{reason}.pt"

    primary_rank_label = next(iter(rank_eval_manifests), None)
    primary_select_label = next(iter(select_eval_manifests), None)
    primary_test_label = next(iter(test_eval_manifests), None)

    best_score = float("-inf")
    best_epoch = 0
    best_select_metrics: dict[str, float] = {}
    best_rank_metrics: dict[str, float] = {}
    best_rank_metrics_by_split: dict[str, dict[str, Any]] = {}
    best_select_metrics_by_split: dict[str, dict[str, Any]] = {}
    epochs_without_improve = 0
    global_step = 0
    start_epoch = 1
    last_completed_epoch = 0
    history_rows: list[dict[str, Any]] = []
    early_stopped = False
    stop_reason = "completed_max_epochs"
    stop_epoch = max(start_epoch - 1, 0)

    resume_state_path: Path | None = None
    resume_loader_generator_states: dict[str, torch.Tensor] = {}
    if config.resume_training_state:
        setting = str(config.resume_training_state).strip()
        if setting.lower() == "auto_if_exists":
            if training_state_path.exists():
                resume_state_path = training_state_path
        elif setting:
            candidate = Path(setting)
            if candidate.exists():
                resume_state_path = candidate

    if resume_state_path is not None:
        state_payload = torch.load(
            resume_state_path,
            map_location="cpu",
            weights_only=True,
        )
        base_model = _unwrap_runtime_model(model)
        base_model.load_state_dict(state_payload["model_state"], strict=False)
        if "optimizer_state" in state_payload and state_payload["optimizer_state"]:
            optimizer.load_state_dict(state_payload["optimizer_state"])
        if scheduler is not None and state_payload.get("scheduler_state"):
            scheduler.load_state_dict(state_payload["scheduler_state"])
        if autocast_scaler is not None and state_payload.get("scaler_state"):
            autocast_scaler.load_state_dict(state_payload["scaler_state"])
        if config.optimizer_lr_override_after_resume is not None:
            override_lr = float(config.optimizer_lr_override_after_resume)
            if override_lr < 0.0:
                raise ValueError(
                    "optimizer_lr_override_after_resume cannot be negative."
                )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = override_lr
            if scheduler is not None and hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [
                    override_lr for _ in optimizer.param_groups
                ]
        _restore_rng_state(state_payload.get("rng_state"))
        resume_loader_generator_states = dict(
            state_payload.get("train_loader_generator_states") or {}
        )
        if (
            discriminator_optimizer is not None
            and state_payload.get("metric_discriminator_state")
            and isinstance(pesq_proxy_model, SpeechBrainMetricDiscriminator)
        ):
            discriminator_refresh_history = _restore_metric_discriminator_state(
                state_payload,
                discriminator=pesq_proxy_model,
                optimizer=discriminator_optimizer,
                replay_root=str(config.metric_discriminator_replay_root),
            )

        best_score = float(state_payload.get("best_score", best_score))
        best_epoch = int(state_payload.get("best_epoch", best_epoch))
        best_select_metrics = dict(state_payload.get("best_select_metrics", {}))
        best_rank_metrics = dict(state_payload.get("best_rank_metrics", {}))
        best_rank_metrics_by_split = dict(state_payload.get("best_rank_metrics_by_split", {}))
        best_select_metrics_by_split = dict(state_payload.get("best_select_metrics_by_split", {}))
        epochs_without_improve = int(state_payload.get("epochs_without_improve", epochs_without_improve))
        global_step = int(state_payload.get("global_step", global_step))
        resume_start_epoch, resume_completed_epoch = _resume_epoch_position(
            state_payload
        )
        start_epoch = resume_start_epoch
        last_completed_epoch = max(
            last_completed_epoch,
            resume_completed_epoch,
        )
        loaded_history = state_payload.get("history_rows")
        if isinstance(loaded_history, list):
            history_rows = [row for row in loaded_history if isinstance(row, dict)]
        mlflow.log_params({"resumed_from_training_state": resume_state_path.as_posix()})

    params = flatten_params(asdict(config))
    params["train_manifest_hash"] = manifest_hash(config.train_csv)
    if config.train_csv_schedule:
        params["train_csv_schedule_hashes"] = {path: manifest_hash(path) for path in config.train_csv_schedule}
    if rank_eval_manifests:
        params["rank_eval_manifest_hashes"] = {label: manifest_hash(path) for label, path in rank_eval_manifests.items()}
    if select_eval_manifests:
        params["select_eval_manifest_hashes"] = {label: manifest_hash(path) for label, path in select_eval_manifests.items()}
    if test_eval_manifests:
        params["test_eval_manifest_hashes"] = {label: manifest_hash(path) for label, path in test_eval_manifests.items()}
    if config.teacher_cache_manifest:
        params["teacher_cache_manifest_hash"] = manifest_hash(config.teacher_cache_manifest)
    if config.teacher_cache_schedule:
        params["teacher_cache_schedule_hashes"] = {path: manifest_hash(path) for path in config.teacher_cache_schedule}
    mlflow.log_params(params)

    sample_manifest_candidates = list(rank_eval_manifests.values()) or list(select_eval_manifests.values()) or [config.train_csv]
    sample_path = read_pair_manifest(sample_manifest_candidates[0])[0].noisy

    summary: dict[str, Any] = {
        "run_id": run.info.run_id,
        "run_name": config.run_name,
        "model_family": config.model_family,
        "variant": config.variant,
        "loss_recipe": config.loss_recipe,
        "seed": config.seed,
        "postfilter_mode": config.postfilter_mode,
        "postfilter_preset": config.postfilter_preset,
        "train_postfilter": config.train_postfilter,
        "spectral_native_gate": config.spectral_native_gate,
        "teacher_source_run_id": config.teacher_source_run_id,
        "teacher_variant": config.teacher_variant,
        "audit_only": config.audit_only,
        "teacher_cache_manifest": config.teacher_cache_manifest,
        "guidance_classic": config.guidance_classic,
        "erb_bands": config.erb_bands,
        "context_frames": config.context_frames,
        "qat": config.qat,
        "quantize_dynamic": config.quantize_dynamic,
        "selection_metric": config.selection_metric,
        "selection_guardrail_metric": config.selection_guardrail_metric,
        "selection_guardrail_min": config.selection_guardrail_min,
        "metric_discriminator_mode": config.metric_discriminator_mode,
    }

    checkpoint_every_steps = max(int(config.checkpoint_every_steps or 0), 0)
    checkpoint_every_seconds = max(float(config.checkpoint_every_minutes or 0.0), 0.0) * 60.0
    checkpoint_snapshot_every_periods = max(int(config.checkpoint_snapshot_every_periods or 0), 0)
    checkpoint_keep_last = max(int(config.checkpoint_keep_last or 0), 0)
    history_persist_every_periods = max(int(config.history_persist_every_periods or 1), 1)
    last_state_save_step = global_step
    last_state_save_ts = time.monotonic()
    periodic_save_count = 0

    def _build_training_state(epoch: int, step: int, reason: str) -> dict[str, Any]:
        base_model = _unwrap_runtime_model(model)
        payload = {
            "epoch": int(epoch),
            "global_step": int(step),
            "reason": reason,
            "config": asdict(config),
            "model_state": base_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": autocast_scaler.state_dict() if autocast_scaler is not None else None,
            "rng_state": _capture_rng_state(),
            "train_loader_generator_states": {
                json.dumps(key): loader.generator.get_state()
                for key, loader in train_loader_cache.items()
                if loader.generator is not None
            },
            **_training_control_state_fields(
                scheduler=scheduler,
                best_score=best_score,
                best_epoch=best_epoch,
                best_rank_metrics=best_rank_metrics,
                best_select_metrics=best_select_metrics,
                best_rank_metrics_by_split=best_rank_metrics_by_split,
                best_select_metrics_by_split=best_select_metrics_by_split,
                epochs_without_improve=epochs_without_improve,
                history_rows=history_rows,
            ),
        }
        if (
            isinstance(pesq_proxy_model, SpeechBrainMetricDiscriminator)
            and discriminator_optimizer is not None
        ):
            payload.update(
                _metric_discriminator_state_fields(
                    discriminator=pesq_proxy_model,
                    optimizer=discriminator_optimizer,
                    refresh_history=discriminator_refresh_history,
                    replay_root=str(config.metric_discriminator_replay_root),
                )
            )
        return payload

    def _save_training_state(epoch: int, step: int, reason: str, *, snapshot: bool) -> None:
        payload = _build_training_state(epoch=epoch, step=step, reason=reason)
        _atomic_torch_save(payload, training_state_path)
        if snapshot:
            snapshot_path = training_state_dir / _snapshot_basename(epoch=epoch, global_step=step, reason=reason)
            _atomic_torch_save(payload, snapshot_path)
            if checkpoint_keep_last > 0:
                existing = sorted(training_state_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                for stale in existing[checkpoint_keep_last:]:
                    stale.unlink(missing_ok=True)

    last_plotted_epoch = -1

    def _persist_training_history(current_epoch: int, *, force_plot: bool = False, allow_epoch_plot: bool = True) -> None:
        nonlocal last_plotted_epoch
        write_plot = force_plot
        if not write_plot and not history_plot_final_only and allow_epoch_plot:
            if current_epoch % history_plot_every_epochs == 0 and current_epoch != last_plotted_epoch:
                write_plot = True
        if write_plot:
            last_plotted_epoch = current_epoch
        _save_training_history_artifacts(
            history_rows,
            history_json_path=history_json_path,
            history_csv_path=history_csv_path,
            history_plot_path=history_plot_path,
            write_plot=write_plot,
        )

    if history_rows:
        _persist_training_history(max(start_epoch - 1, 0), force_plot=not history_plot_final_only)

    run_status = "FINISHED"
    current_progress_epoch = max(start_epoch - 1, 0)

    def _write_progress(state: str, **extra: Any) -> None:
        if progress_json_path is None:
            return
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": config.run_name,
            "phase": config.phase,
            "model_family": config.model_family,
            "variant": config.variant,
            "state": state,
            "epoch": int(extra.pop("epoch", current_progress_epoch)),
            "global_step": int(extra.pop("global_step", global_step)),
            "selection_metric": config.selection_metric,
            "target_floor": config.target_floor,
            **extra,
        }
        if math.isfinite(best_score):
            payload["best_selection_score"] = float(best_score)
            payload["best_epoch"] = int(best_epoch)
            if config.target_floor is not None:
                payload["best_target_gap"] = float(config.target_floor - float(best_score))
        _atomic_write_json(payload, progress_json_path)

    def _eval_progress(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[eval] {timestamp} {message}", file=sys.stderr, flush=True)
        _write_progress("evaluating", message=message)

    train_loader_cache: dict[tuple[str, str | None], DataLoader] = {}

    def _loader_for_epoch(epoch: int) -> tuple[DataLoader, str, str | None]:
        schedule = list(config.train_csv_schedule or [])
        train_manifest = schedule[(epoch - 1) % len(schedule)] if schedule else config.train_csv
        teacher_schedule = list(config.teacher_cache_schedule or [])
        teacher_cache_manifest = teacher_schedule[(epoch - 1) % len(teacher_schedule)] if teacher_schedule else config.teacher_cache_manifest
        key = (train_manifest, teacher_cache_manifest)
        loader = train_loader_cache.get(key)
        if loader is None:
            loader_config = replace(config, teacher_cache_manifest=teacher_cache_manifest)
            loader = build_dataloader(train_manifest, loader_config, shuffle=True)
            saved_generator_state = resume_loader_generator_states.get(
                json.dumps(key)
            )
            if (
                saved_generator_state is not None
                and loader.generator is not None
            ):
                loader.generator.set_state(saved_generator_state)
            train_loader_cache[key] = loader
        return loader, train_manifest, teacher_cache_manifest

    def _evaluate_group(
        model_for_eval: nn.Module,
        split_name: str,
        manifest_map: dict[str, str],
        *,
        compute_dnsmos: bool,
        compute_composite: bool,
        sample_root: Path | None,
        sample_count: int,
        max_files: int | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
        results: dict[str, dict[str, Any]] = {}
        flat: dict[str, float] = {}
        primary_label = next(iter(manifest_map), split_name)
        for label, manifest_path in manifest_map.items():
            start_ts = time.monotonic()
            _eval_progress(f"{split_name}: starting {label} evaluation (manifest={Path(manifest_path).name})")
            metrics = evaluate_manifest(
                model_for_eval,
                manifest_path,
                config.device,
                sample_rate=config.sample_rate,
                bandwidth=config.bandwidth,
                compute_dnsmos=compute_dnsmos,
                compute_composite=compute_composite,
                sample_dir=(sample_root / label) if sample_root is not None else None,
                sample_count=sample_count,
                max_files=max_files,
                batch_size=config.eval_batch_size,
                cache_audio=config.cache_eval_audio,
                progress_callback=_eval_progress,
            )
            results[label] = metrics
            _eval_progress(
                f"{split_name}: finished {label} evaluation in {time.monotonic() - start_ts:.1f}s "
                f"(pesq={metrics['pesq_mean']:.4f}, stoi={metrics['stoi_mean']:.4f})"
            )
        flat.update(_flatten_split_metrics(primary_label, results))
        return results, flat

    def _select_primary_metrics(payload: dict[str, dict[str, Any]], primary_label: str | None) -> dict[str, Any]:
        if primary_label and primary_label in payload:
            return dict(payload[primary_label])
        if payload:
            return dict(next(iter(payload.values())))
        return {}

    def _seed_best_from_current_model() -> None:
        nonlocal best_score, best_epoch, best_rank_metrics, best_select_metrics, best_rank_metrics_by_split, best_select_metrics_by_split
        if not config.evaluate_init_checkpoint:
            return
        init_flat: dict[str, float] = {}
        init_rank_results: dict[str, dict[str, Any]] = {}
        init_select_results: dict[str, dict[str, Any]] = {}
        if rank_eval_manifests:
            init_rank_results, rank_flat = _evaluate_group(
                model,
                "init_val_rank",
                rank_eval_manifests,
                compute_dnsmos=False,
                compute_composite=config.rank_compute_composite,
                sample_root=None,
                sample_count=0,
                max_files=config.rank_max_eval_files if config.rank_max_eval_files is not None else config.max_eval_files,
            )
            init_flat.update(rank_flat)
        if select_eval_manifests:
            init_select_results, select_flat = _evaluate_group(
                model,
                "init_val_select",
                select_eval_manifests,
                compute_dnsmos=False,
                compute_composite=config.select_compute_composite,
                sample_root=None,
                sample_count=0,
                max_files=config.final_max_eval_files if config.final_max_eval_files is not None else config.max_eval_files,
            )
            init_flat.update(select_flat)
        init_score = _metric_value_from_flat(init_flat, config.selection_metric)
        guardrail_value = _metric_value_from_flat(init_flat, config.selection_guardrail_metric)
        guardrail_ok = True
        if config.selection_guardrail_metric and config.selection_guardrail_min is not None:
            guardrail_ok = guardrail_value is not None and float(guardrail_value) >= float(config.selection_guardrail_min)
        if init_score is None or not guardrail_ok:
            return
        best_score = float(init_score)
        best_epoch = 0
        best_rank_metrics_by_split = init_rank_results
        best_select_metrics_by_split = init_select_results
        best_rank_metrics = _select_primary_metrics(init_rank_results, primary_rank_label)
        best_select_metrics = _select_primary_metrics(init_select_results, primary_select_label)
        mlflow.log_metrics({f"init/{key}": value for key, value in init_flat.items()}, step=0)
        save_checkpoint_package(
            checkpoint_path,
            model=model,
            model_family=config.model_family,
            variant=config.variant,
            extra={
                "epoch": 0,
                "loss_recipe": config.loss_recipe,
                "seed": config.seed,
                "selection_metric": config.selection_metric,
                "best_score": float(best_score),
            },
        )
        history_rows.append(
            {
                "row_type": "init",
                "epoch": 0,
                "global_step": 0,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "eval_performed": 1,
                "selection_score": float(best_score),
                "target_gap": float(config.target_floor - best_score) if config.target_floor is not None else None,
                "improved": 1,
                "epochs_without_improve": 0,
                "early_stop_triggered": 0,
                **{key: float(value) for key, value in init_flat.items()},
            }
        )
        _save_training_state(epoch=0, step=0, reason="init_eval", snapshot=False)
        _persist_training_history(0, force_plot=not history_plot_final_only)
        _write_progress(
            "init_eval_complete",
            epoch=0,
            global_step=0,
            selection_score=float(best_score),
            target_gap=float(config.target_floor - best_score) if config.target_floor is not None else None,
            threshold_met=bool(config.target_floor is not None and best_score >= config.target_floor),
        )

    _seed_best_from_current_model()

    try:
        _write_progress("starting", epoch=max(start_epoch - 1, 0), global_step=global_step)
        for epoch in range(start_epoch, config.epochs + 1):
            stop_epoch = epoch
            current_progress_epoch = epoch
            epoch_start_ts = time.monotonic()
            train_loader, current_train_manifest, current_teacher_cache_manifest = _loader_for_epoch(epoch)
            _write_progress(
                "training_epoch",
                epoch=epoch,
                global_step=global_step,
                train_manifest=current_train_manifest,
                teacher_cache_manifest=current_teacher_cache_manifest or "",
            )
            discriminator_metrics: dict[str, float] = {}
            generator_update_accepted = True
            if alternating_discriminator:
                if (
                    not isinstance(
                        pesq_proxy_model, SpeechBrainMetricDiscriminator
                    )
                    or discriminator_optimizer is None
                    or discriminator_checkpoint_path is None
                ):
                    raise RuntimeError(
                        "Alternating discriminator was not initialized."
                    )
                _write_progress(
                    "metric_discriminator_refresh",
                    epoch=epoch,
                    global_step=global_step,
                    train_manifest=current_train_manifest,
                )
                refresh = refresh_metricgan_discriminator(
                    discriminator=pesq_proxy_model,
                    optimizer=discriminator_optimizer,
                    generator=_unwrap_runtime_model(model),
                    train_manifest=current_train_manifest,
                    replay_root=str(config.metric_discriminator_replay_root),
                    checkpoint_out=discriminator_checkpoint_path,
                    epoch=epoch,
                    device=config.device,
                    max_rows=int(config.metric_discriminator_rows),
                    calibration_rows=int(
                        config.metric_discriminator_calibration_rows
                    ),
                    calibration_gate={
                        "min_records": int(
                            config.metric_discriminator_min_calibration_records
                        ),
                        "max_normalized_mae": float(
                            config.metric_discriminator_max_normalized_mae
                        ),
                        "min_pearson": float(
                            config.metric_discriminator_min_pearson
                        ),
                        "min_spearman": float(
                            config.metric_discriminator_min_spearman
                        ),
                        "min_prediction_std": float(
                            config.metric_discriminator_min_prediction_std
                        ),
                        "range_tolerance_raw": float(
                            config.metric_discriminator_range_tolerance_raw
                        ),
                    },
                    history_portion=float(
                        config.metric_discriminator_history_portion
                    ),
                    seed=int(config.seed),
                    grad_clip=float(config.grad_clip),
                    progress_callback=_eval_progress,
                )
                discriminator_refresh_history.append(refresh)
                generator_update_accepted = bool(
                    refresh["calibration_gate"]["passed"]
                ) and not bool(config.metric_discriminator_calibration_only)
                discriminator_metrics = {
                    "discriminator_current_first_mse": float(
                        refresh["current_first_mse"]
                    ),
                    "discriminator_historical_mse": float(
                        refresh["historical_mse"]
                    ),
                    "discriminator_current_second_mse": float(
                        refresh["current_second_mse"]
                    ),
                    "discriminator_current_mae_pesq": float(
                        refresh["calibration"]["mae"]
                    ),
                    "discriminator_current_pearson": float(
                        refresh["calibration"]["pearson"]
                    ),
                    "discriminator_current_spearman": float(
                        refresh["calibration"]["spearman"]
                    ),
                    "discriminator_current_normalized_mae": float(
                        refresh["calibration"]["normalized_mae"]
                    ),
                    "discriminator_calibration_gate_passed": float(
                        bool(refresh["calibration_gate"]["passed"])
                    ),
                    "generator_update_accepted": float(
                        generator_update_accepted
                    ),
                }

            def _step_callback(_step_in_epoch: int, running_train: dict[str, float]) -> None:
                nonlocal global_step, last_state_save_step, last_state_save_ts, periodic_save_count
                global_step += 1
                due_steps = checkpoint_every_steps > 0 and (global_step - last_state_save_step) >= checkpoint_every_steps
                due_time = checkpoint_every_seconds > 0 and (time.monotonic() - last_state_save_ts) >= checkpoint_every_seconds
                if due_steps or due_time:
                    periodic_save_count += 1
                    snapshot_now = (
                        checkpoint_snapshot_every_periods > 0
                        and (periodic_save_count % checkpoint_snapshot_every_periods) == 0
                    )
                    _save_training_state(epoch=epoch, step=global_step, reason="periodic", snapshot=snapshot_now)
                    if config.record_step_history:
                        history_rows.append(
                            {
                                "row_type": "step",
                                "epoch": int(epoch),
                                "step_in_epoch": int(_step_in_epoch),
                                "global_step": int(global_step),
                                "lr": float(optimizer.param_groups[0]["lr"]),
                                "eval_performed": 0,
                                "selection_score": None,
                                "improved": None,
                                "epochs_without_improve": int(epochs_without_improve),
                                "early_stop_triggered": 0,
                                "train_manifest": current_train_manifest,
                                "teacher_cache_manifest": current_teacher_cache_manifest or "",
                                **{key: float(value) for key, value in running_train.items()},
                            }
                        )
                        if periodic_save_count % history_persist_every_periods == 0:
                            _persist_training_history(epoch, force_plot=False, allow_epoch_plot=False)
                    _write_progress(
                        "training_epoch",
                        epoch=epoch,
                        global_step=global_step,
                        train_manifest=current_train_manifest,
                        teacher_cache_manifest=current_teacher_cache_manifest or "",
                        train_loss=(
                            float(running_train["loss"])
                            if math.isfinite(float(running_train.get("loss", float("nan"))))
                            else None
                        ),
                    )
                    last_state_save_step = global_step
                    last_state_save_ts = time.monotonic()

            if generator_update_accepted:
                train_metrics = run_epoch(
                    train_model,
                    train_loader,
                    optimizer,
                    loss_fn,
                    config,
                    epoch,
                    autocast_scaler,
                    step_callback=_step_callback,
                )
            else:
                train_metrics = {
                    "loss": 0.0,
                    "generator_update_skipped": 1.0,
                }
            train_metrics.update(discriminator_metrics)
            epoch_seconds = time.monotonic() - epoch_start_ts
            _write_progress(
                "epoch_train_complete",
                epoch=epoch,
                global_step=global_step,
                epoch_seconds=float(epoch_seconds),
                train_loss=float(train_metrics["loss"]),
            )
            mlflow.log_metrics({f"train/{key}": value for key, value in train_metrics.items()}, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            history_row: dict[str, Any] = {
                "row_type": "epoch",
                "epoch": int(epoch),
                "global_step": int(global_step),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": float(epoch_seconds),
                "eval_performed": 0,
                "selection_score": None,
                "improved": None,
                "epochs_without_improve": int(epochs_without_improve),
                "early_stop_triggered": 0,
                "train_manifest": current_train_manifest,
                "teacher_cache_manifest": current_teacher_cache_manifest or "",
            }
            for key, value in train_metrics.items():
                history_row[f"train/{key}"] = float(value)

            _save_training_state(epoch=epoch, step=global_step, reason="epoch", snapshot=False)
            last_completed_epoch = epoch
            last_state_save_step = global_step
            last_state_save_ts = time.monotonic()

            if scheduler is not None and config.scheduler == "cosine":
                scheduler.step(epoch - 1)

            eval_flat: dict[str, float] = {}
            rank_results: dict[str, dict[str, Any]] = {}
            select_results: dict[str, dict[str, Any]] = {}

            should_rank_eval = bool(rank_eval_manifests) and rank_eval_every > 0 and epoch % rank_eval_every == 0
            if should_rank_eval:
                _write_progress("rank_eval_start", epoch=epoch, global_step=global_step)
                rank_results, rank_flat = _evaluate_group(
                    model,
                    "val_rank",
                    rank_eval_manifests,
                    compute_dnsmos=False,
                    compute_composite=config.rank_compute_composite,
                    sample_root=None,
                    sample_count=0,
                    max_files=config.rank_max_eval_files if config.rank_max_eval_files is not None else config.max_eval_files,
                )
                mlflow.log_metrics(rank_flat, step=epoch)
                eval_flat.update(rank_flat)
                history_row["eval_performed"] = 1
                for key, value in rank_flat.items():
                    history_row[key] = float(value)

            should_select_eval = bool(select_eval_manifests) and select_eval_every > 0 and epoch % select_eval_every == 0
            if should_select_eval:
                _write_progress("select_eval_start", epoch=epoch, global_step=global_step)
                select_results, select_flat = _evaluate_group(
                    model,
                    "val_select",
                    select_eval_manifests,
                    compute_dnsmos=False,
                    compute_composite=config.select_compute_composite,
                    sample_root=None,
                    sample_count=0,
                    max_files=config.final_max_eval_files if config.final_max_eval_files is not None else config.max_eval_files,
                )
                mlflow.log_metrics(select_flat, step=epoch)
                eval_flat.update(select_flat)
                history_row["eval_performed"] = 1
                for key, value in select_flat.items():
                    history_row[key] = float(value)

            score = _metric_value_from_flat(eval_flat, config.selection_metric)
            if score is None and not select_eval_manifests and should_rank_eval:
                score = _metric_value_from_flat(eval_flat, config.selection_metric) or _selection_score(_select_primary_metrics(rank_results, primary_rank_label))

            stop_after_evaluation = False
            if score is not None:
                history_row["selection_score"] = float(score)
                target_gap = None
                if config.target_floor is not None:
                    target_gap = float(config.target_floor - float(score))
                    history_row["target_gap"] = target_gap
                    mlflow.log_metric("monitor/target_gap", target_gap, step=epoch)
                    mlflow.log_metric("monitor/threshold_met", 1.0 if float(score) >= float(config.target_floor) else 0.0, step=epoch)
                guardrail_value = _metric_value_from_flat(eval_flat, config.selection_guardrail_metric)
                guardrail_ok = True
                if config.selection_guardrail_metric and config.selection_guardrail_min is not None:
                    guardrail_ok = guardrail_value is not None and float(guardrail_value) >= float(config.selection_guardrail_min)
                    history_row["selection_guardrail_value"] = float(guardrail_value) if guardrail_value is not None else None
                    history_row["selection_guardrail_passed"] = 1 if guardrail_ok else 0
                selection_eligible = bool(
                    not alternating_discriminator
                    or generator_update_accepted
                )
                history_row["selection_eligible"] = int(selection_eligible)
                if (
                    selection_eligible
                    and scheduler is not None
                    and config.scheduler == "plateau"
                ):
                    scheduler.step(float(score))
                _write_progress(
                    "selection_scored",
                    epoch=epoch,
                    global_step=global_step,
                    selection_score=float(score),
                    target_gap=target_gap,
                    threshold_met=bool(config.target_floor is not None and float(score) >= float(config.target_floor)),
                    guardrail_value=float(guardrail_value) if guardrail_value is not None else None,
                    guardrail_passed=guardrail_ok,
                )

                if (
                    selection_eligible
                    and guardrail_ok
                    and float(score) > best_score
                ):
                    best_score = float(score)
                    best_epoch = epoch
                    epochs_without_improve = 0
                    if rank_results:
                        best_rank_metrics_by_split = rank_results
                        best_rank_metrics = _select_primary_metrics(rank_results, primary_rank_label)
                    if select_results:
                        best_select_metrics_by_split = select_results
                        best_select_metrics = _select_primary_metrics(select_results, primary_select_label)
                    history_row["improved"] = 1
                    mlflow.log_metric("best/selection_score", float(score), step=epoch)
                    save_checkpoint_package(
                        checkpoint_path,
                        model=model,
                        model_family=config.model_family,
                        variant=config.variant,
                        extra={
                            "epoch": epoch,
                            "loss_recipe": config.loss_recipe,
                            "seed": config.seed,
                            "selection_metric": config.selection_metric,
                            "best_score": float(score),
                        },
                    )
                    _write_progress(
                        "best_updated",
                        epoch=epoch,
                        global_step=global_step,
                        selection_score=float(score),
                        target_gap=float(config.target_floor - float(score)) if config.target_floor is not None else None,
                    )
                elif selection_eligible:
                    epochs_without_improve += 1
                    history_row["improved"] = 0
                else:
                    history_row["improved"] = None
                    _write_progress(
                        "generator_update_skipped",
                        epoch=epoch,
                        global_step=global_step,
                        selection_score=float(score),
                        calibration_gate_passed=False,
                    )

                history_row["epochs_without_improve"] = int(epochs_without_improve)

                if selection_eligible and epoch >= config.min_epochs and config.early_stop_patience > 0 and epochs_without_improve >= config.early_stop_patience:
                    early_stopped = True
                    stop_reason = "early_stopping"
                    history_row["early_stop_triggered"] = 1
                    stop_after_evaluation = True
                    _write_progress(
                        "early_stopping",
                        epoch=epoch,
                        global_step=global_step,
                        selection_score=float(score),
                        epochs_without_improve=int(epochs_without_improve),
                    )
            history_row["lr_after_eval"] = float(
                optimizer.param_groups[0]["lr"]
            )
            history_rows.append(history_row)
            _persist_training_history(
                epoch,
                force_plot=stop_after_evaluation,
            )
            if should_rank_eval or should_select_eval:
                _save_training_state(
                    epoch=epoch,
                    step=global_step,
                    reason="evaluation",
                    snapshot=False,
                )
                last_state_save_step = global_step
                last_state_save_ts = time.monotonic()
                if (
                    config.interrupt_after_evaluation_epoch is not None
                    and epoch == config.interrupt_after_evaluation_epoch
                ):
                    raise PlannedTrainingInterruption(
                        f"planned interruption after evaluation epoch {epoch}"
                    )
            if stop_after_evaluation:
                break

        if not checkpoint_path.exists():
            save_checkpoint_package(
                checkpoint_path,
                model=model,
                model_family=config.model_family,
                variant=config.variant,
                extra={
                    "epoch": int(stop_epoch),
                    "loss_recipe": config.loss_recipe,
                    "seed": config.seed,
                    "selection_metric": config.selection_metric,
                    "best_score": float(best_score),
                },
            )
        _save_training_state(epoch=last_completed_epoch, step=global_step, reason="final", snapshot=False)
        _persist_training_history(stop_epoch, force_plot=True)
        _write_progress("training_complete", epoch=last_completed_epoch, global_step=global_step)

        save_checkpoint_package(
            final_checkpoint_path,
            model=model,
            model_family=config.model_family,
            variant=config.variant,
            extra={
                "epoch": int(stop_epoch),
                "loss_recipe": config.loss_recipe,
                "seed": config.seed,
                "final_checkpoint": True,
            },
        )

        mlflow.log_artifact(checkpoint_path.as_posix(), artifact_path="checkpoints")
        mlflow.log_artifact(final_checkpoint_path.as_posix(), artifact_path="checkpoints")
        mlflow.log_artifact(training_state_path.as_posix(), artifact_path="checkpoints")
        if (
            discriminator_checkpoint_path is not None
            and discriminator_checkpoint_path.is_file()
        ):
            mlflow.log_artifact(
                discriminator_checkpoint_path.as_posix(),
                artifact_path="checkpoints",
            )
        if history_json_path.exists():
            mlflow.log_artifact(history_json_path.as_posix(), artifact_path="reports")
        if history_csv_path.exists():
            mlflow.log_artifact(history_csv_path.as_posix(), artifact_path="reports")
        if history_plot_path.exists():
            mlflow.log_artifact(history_plot_path.as_posix(), artifact_path="reports")

        final_model, _ = load_model_from_checkpoint(
            checkpoint_path,
            device=config.device,
            model_family=config.model_family,
            variant=config.variant,
        )
        final_model = _wrap_model_for_runtime(final_model, config.device, config.gpu_ids)

        rank_metrics_by_split: dict[str, dict[str, Any]] = {}
        select_metrics_by_split: dict[str, dict[str, Any]] = {}
        test_metrics_by_split: dict[str, dict[str, Any]] = {}

        if rank_eval_manifests:
            rank_metrics_by_split, rank_flat = _evaluate_group(
                final_model,
                "final_val_rank",
                rank_eval_manifests,
                compute_dnsmos=False,
                compute_composite=config.rank_compute_composite,
                sample_root=None,
                sample_count=0,
                max_files=config.rank_max_eval_files if config.rank_max_eval_files is not None else config.max_eval_files,
            )
            mlflow.log_metrics({f"final/{key}": value for key, value in rank_flat.items()})
            log_dict_artifact(rank_metrics_by_split, "reports/final_val_rank_metrics_by_split.json")

        if select_eval_manifests:
            sample_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_samples"
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
            select_metrics_by_split, select_flat = _evaluate_group(
                final_model,
                "final_val_select",
                select_eval_manifests,
                compute_dnsmos=config.eval_dnsmos,
                compute_composite=config.select_compute_composite,
                sample_root=sample_dir,
                sample_count=config.sample_count,
                max_files=config.final_max_eval_files if config.final_max_eval_files is not None else config.max_eval_files,
            )
            if sample_dir.exists():
                mlflow.log_artifacts(sample_dir.as_posix(), artifact_path="samples")
            mlflow.log_metrics({f"best/{key}": value for key, value in select_flat.items()})
            log_dict_artifact(select_metrics_by_split, "reports/best_val_select_metrics_by_split.json")

        if test_eval_manifests:
            test_sample_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_test_samples"
            if test_sample_dir.exists():
                shutil.rmtree(test_sample_dir, ignore_errors=True)
            test_metrics_by_split, test_flat = _evaluate_group(
                final_model,
                "final_test",
                test_eval_manifests,
                compute_dnsmos=config.eval_dnsmos,
                compute_composite=config.select_compute_composite,
                sample_root=test_sample_dir,
                sample_count=config.sample_count,
                max_files=config.final_max_eval_files if config.final_max_eval_files is not None else config.max_eval_files,
            )
            if test_sample_dir.exists():
                mlflow.log_artifacts(test_sample_dir.as_posix(), artifact_path="test_samples")
            mlflow.log_metrics({f"test/{key}": value for key, value in test_flat.items()})
            log_dict_artifact(test_metrics_by_split, "reports/test_metrics_by_split.json")

        primary_rank_metrics = _select_primary_metrics(rank_metrics_by_split, primary_rank_label)
        primary_select_metrics = _select_primary_metrics(select_metrics_by_split, primary_select_label)
        primary_test_metrics = _select_primary_metrics(test_metrics_by_split, primary_test_label)
        benchmark_latency = benchmark_inference(
            final_model,
            sample_path=sample_path,
            device=config.device,
            sample_rate=config.sample_rate,
            duration_seconds=config.benchmark_seconds,
            repeats=config.benchmark_repeats,
        )
        mlflow.log_metric("best/inference_seconds_10s", benchmark_latency)

        if primary_select_metrics:
            best_select_metrics = primary_select_metrics
        if primary_rank_metrics:
            best_rank_metrics = primary_rank_metrics

        summary.update(
            {
                "best_epoch": best_epoch,
                "stop_epoch": stop_epoch,
                "early_stopped": early_stopped,
                "stop_reason": stop_reason,
                "global_step": global_step,
                "best_score": best_score,
                "best_val_rank_pesq": primary_rank_metrics.get("pesq_mean") or best_rank_metrics.get("pesq_mean"),
                "best_val_select_pesq": primary_select_metrics.get("pesq_mean") or best_select_metrics.get("pesq_mean"),
                "best_val_select_dnsmos_ovr": primary_select_metrics.get("dnsmos_ovr_mean") or best_select_metrics.get("dnsmos_ovr_mean"),
                "inference_seconds_10s": benchmark_latency,
                "checkpoint_out": checkpoint_path.as_posix(),
                "checkpoint_out_final": final_checkpoint_path.as_posix(),
                "training_state_out": training_state_path.as_posix(),
                "history_json": history_json_path.as_posix(),
                "history_csv": history_csv_path.as_posix(),
                "history_plot": history_plot_path.as_posix(),
                "metric_discriminator_checkpoint": (
                    discriminator_checkpoint_path.as_posix()
                    if discriminator_checkpoint_path is not None
                    else None
                ),
                "metric_discriminator_refresh_history": list(
                    discriminator_refresh_history
                ),
                "metric_discriminator_accepted_update_count": sum(
                    int(
                        bool(item.get("calibration_gate", {}).get("passed"))
                        and not config.metric_discriminator_calibration_only
                    )
                    for item in discriminator_refresh_history
                ),
                "val_rank_metrics": primary_rank_metrics,
                "val_select_metrics": primary_select_metrics,
                "test_metrics": primary_test_metrics,
                "val_rank_metrics_by_split": rank_metrics_by_split,
                "val_select_metrics_by_split": select_metrics_by_split,
                "test_metrics_by_split": test_metrics_by_split,
                "target_floor": config.target_floor,
                "threshold_met": bool(config.target_floor is not None and (primary_select_metrics.get("pesq_mean") or best_select_metrics.get("pesq_mean") or float("-inf")) >= config.target_floor),
            }
        )
        _write_progress(
            "final_evaluation_complete",
            epoch=stop_epoch,
            global_step=global_step,
            best_val_select_pesq=summary.get("best_val_select_pesq"),
            target_gap=(
                float(config.target_floor - float(summary.get("best_val_select_pesq")))
                if config.target_floor is not None and summary.get("best_val_select_pesq") is not None
                else None
            ),
            checkpoint_out=checkpoint_path.as_posix(),
        )
        if config.log_torch_model:
            mlflow.pytorch.log_model(final_model, artifact_path="model")
        log_dict_artifact(summary, "reports/run_summary.json")
        return summary
    except PlannedTrainingInterruption:
        run_status = "KILLED"
        stop_reason = "planned_interruption"
        _write_progress(
            "planned_interruption",
            epoch=last_completed_epoch,
            global_step=global_step,
        )
        raise
    except KeyboardInterrupt:
        run_status = "KILLED"
        stop_reason = "interrupted"
        try:
            if history_rows:
                _persist_training_history(stop_epoch, force_plot=True)
        except Exception:
            pass
        try:
            _save_training_state(epoch=last_completed_epoch, step=global_step, reason="interrupted", snapshot=False)
        except Exception:
            pass
        _write_progress("interrupted", epoch=last_completed_epoch, global_step=global_step)
        raise
    except BaseException:
        run_status = "FAILED"
        stop_reason = "failed"
        try:
            if history_rows:
                _persist_training_history(stop_epoch, force_plot=True)
        except Exception:
            pass
        try:
            _save_training_state(epoch=last_completed_epoch, step=global_step, reason="failed", snapshot=False)
        except Exception:
            pass
        _write_progress("failed", epoch=last_completed_epoch, global_step=global_step)
        raise
    finally:
        restore_termination_handlers(previous_handlers)
        mlflow.end_run(status=run_status)

def summary_from_existing(existing: dict[str, Any]) -> dict[str, Any]:
    metrics = existing.get("metrics", {})
    params = existing.get("params", {})
    return {
        "run_id": existing["run_id"],
        "run_name": existing["tags"].get("mlflow.runName"),
        "model_family": params.get("model_family"),
        "variant": params.get("variant"),
        "loss_recipe": params.get("loss_recipe"),
        "seed": int(params["seed"]) if "seed" in params else None,
        "postfilter_mode": params.get("postfilter_mode"),
        "postfilter_preset": params.get("postfilter_preset"),
        "train_postfilter": params.get("train_postfilter"),
        "spectral_native_gate": params.get("spectral_native_gate"),
        "teacher_source_run_id": params.get("teacher_source_run_id"),
        "teacher_variant": params.get("teacher_variant"),
        "audit_only": params.get("audit_only"),
        "teacher_cache_manifest": params.get("teacher_cache_manifest"),
        "guidance_classic": params.get("guidance_classic"),
        "erb_bands": int(params["erb_bands"]) if "erb_bands" in params and params["erb_bands"] not in {"null", ""} else None,
        "context_frames": int(params["context_frames"]) if "context_frames" in params and params["context_frames"] not in {"null", ""} else None,
        "qat": params.get("qat"),
        "quantize_dynamic": params.get("quantize_dynamic"),
        "best_val_select_pesq": metrics.get("best/val_select_pesq_mean"),
        "best_val_select_stoi": metrics.get("best/val_select_stoi_mean"),
        "best_val_select_dnsmos_ovr": metrics.get("best/val_select_dnsmos_ovr_mean"),
        "best_val_rank_pesq": metrics.get("best/val_rank_pesq_mean") or metrics.get("val_rank/pesq_mean"),
        "inference_seconds_10s": metrics.get("best/inference_seconds_10s"),
        "teacher_accuracy_drop_pesq": metrics.get("teacher_accuracy_drop_pesq"),
        "teacher_accuracy_drop_stoi": metrics.get("teacher_accuracy_drop_stoi"),
        "teacher_accuracy_drop_sisdr": metrics.get("teacher_accuracy_drop_sisdr"),
        "test_metrics": {
            "pesq_mean": metrics.get("test/pesq_mean"),
            "stoi_mean": metrics.get("test/stoi_mean"),
            "sisdr_mean": metrics.get("test/sisdr_mean"),
            "delta_snr_mean": metrics.get("test/delta_snr_mean"),
            "csig_mean": metrics.get("test/csig_mean"),
            "cbak_mean": metrics.get("test/cbak_mean"),
            "covl_mean": metrics.get("test/covl_mean"),
            "dnsmos_sig_mean": metrics.get("test/dnsmos_sig_mean"),
            "dnsmos_bak_mean": metrics.get("test/dnsmos_bak_mean"),
            "dnsmos_ovr_mean": metrics.get("test/dnsmos_ovr_mean"),
        },
    }


def default_experiment_config(**overrides: Any) -> ExperimentConfig:
    family = overrides.get("model_family", "metricgan_plus_teacher_wb")
    variant = overrides.get("variant", "base")
    segment_len = int(overrides.get("segment_len", 32000))
    profile = suggest_runtime_profile(family, variant, segment_len)
    batch_size = overrides.get("batch_size", profile["batch_size"])
    grad_accum = overrides.get("grad_accum", profile["grad_accum"])
    num_workers = overrides.get("num_workers", profile["num_workers"])
    eval_batch_size = overrides.get("eval_batch_size", profile["eval_batch_size"])
    config = ExperimentConfig(
        batch_size=batch_size,
        grad_accum=grad_accum,
        num_workers=num_workers,
        eval_batch_size=eval_batch_size,
        **overrides,
    )
    return config
