#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from sebench import build_voicebank_campaign_splits
from sebench.audio import load_mono_audio, loop_to_length, manifest_hash
from sebench.checkpoints import load_model_from_checkpoint
from sebench.data import partition_manifest_rows, read_pair_manifest, unique_manifest_rows, write_pair_manifest
from sebench.mlflow_utils import DEFAULT_EXPERIMENT_NAME
from sebench.losses import PESQProxyRegressor, save_pesq_proxy_checkpoint
from sebench.models import build_enhancer
from sebench.postfilters import resolve_postfilter_config, spectral_gate_waveform
from sebench.reporting import generate_report, read_json, write_csv, write_json
from sebench.runtime import require_cuda_device
from sebench.staging import stage_dataset_manifests
from sebench.stm32sim import simulate_model_across_profiles
from sebench.teacher_cache import build_teacher_cache, filter_teacher_cache_manifest
from metrics.pesq import pesq_score
from sebench.training import (
    ExperimentConfig,
    autotune_loader_profile,
    benchmark_inference,
    evaluate_manifest,
    run_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SPEAKER_RE = re.compile(r"^(p\d+)_", re.IGNORECASE)
ACADEMIC_EXPLICIT_SUMMARY_VERSION = 2
VOICEBANK_OFFICIAL_TEST_COUNT = 824


def _expand_tree(payload: Any, context: dict[str, str]) -> Any:
    if isinstance(payload, str):
        rendered = payload
        for key, value in context.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered
    if isinstance(payload, list):
        return [_expand_tree(item, context) for item in payload]
    if isinstance(payload, dict):
        return {key: _expand_tree(value, context) for key, value in payload.items()}
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    context = {"project_root": PROJECT_ROOT.as_posix()}
    resolved = _expand_tree(raw, context)
    for _ in range(3):
        paths = resolved.get("paths", {})
        context.update({key: str(value) for key, value in paths.items() if isinstance(value, (str, Path))})
        resolved = _expand_tree(resolved, context)
    return resolved


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _io_staging_cfg(config: dict[str, Any]) -> dict[str, Any]:
    io_cfg = dict(config.get("io_staging") or {})
    io_cfg.setdefault("enabled", False)
    io_cfg.setdefault("roots", [])
    io_cfg.setdefault("stage_subdir", "ULP_STAGE_AUDIO")
    io_cfg.setdefault("verify_mode", "size_only")
    io_cfg.setdefault("verify_hash_every_n", 2000)
    io_cfg.setdefault("copy_workers", 12)
    io_cfg.setdefault("stage_val_test", True)
    io_cfg.setdefault(
        "staged_manifest_dir",
        (Path(config["paths"]["output_root"]) / "combined" / "staged_manifests").as_posix(),
    )
    io_cfg.setdefault("reserve_gb_overrides", {})
    return io_cfg


def _resolve_staged_manifest_path(config: dict[str, Any], source_manifest: str | Path) -> str:
    source_path = Path(source_manifest)
    io_cfg = _io_staging_cfg(config)
    if not bool(io_cfg.get("enabled", False)):
        return source_path.as_posix()
    staged_dir = Path(str(io_cfg["staged_manifest_dir"]))
    candidate = staged_dir / f"{source_path.stem}_staged.csv"
    if candidate.exists():
        return candidate.as_posix()
    return source_path.as_posix()


def _loader_autotune_cache_path(
    config: dict[str, Any],
    *,
    train_csv: str,
    device: str,
) -> Path:
    training_cfg = config.get("training", {})
    token_payload = {
        "train_manifest_hash": manifest_hash(train_csv),
        "device": device,
        "batch_size": training_cfg.get("batch_size"),
        "segment_len": training_cfg.get("segment_len"),
        "sample_rate": training_cfg.get("sample_rate"),
        "n_fft": training_cfg.get("n_fft"),
        "hop_length": training_cfg.get("hop_length"),
    }
    token = hashlib.sha1(json.dumps(token_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    out_dir = Path(config["paths"]["output_root"]) / "autotune"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"loader_tuning_{token}.json"


def _resolve_loader_overrides(
    config: dict[str, Any],
    *,
    train_csv: str,
    device: str,
) -> dict[str, int] | None:
    training_cfg = config.get("training", {})
    auto_cfg = dict(training_cfg.get("autotune_loader") or {})
    if not bool(auto_cfg.get("enabled", False)):
        return None

    cache_path = _loader_autotune_cache_path(config, train_csv=train_csv, device=device)
    if cache_path.exists() and not bool(auto_cfg.get("force", False)):
        payload = read_json(cache_path)
        winner = dict(payload.get("winner") or {})
        if "num_workers" in winner and "prefetch_factor" in winner:
            return {
                "num_workers": int(winner["num_workers"]),
                "prefetch_factor": int(winner["prefetch_factor"]),
            }

    candidates_workers = [int(v) for v in auto_cfg.get("candidates_num_workers", [4, 6, 8])]
    candidates_prefetch = [int(v) for v in auto_cfg.get("candidates_prefetch_factor", [2, 4])]
    benchmark_steps = int(auto_cfg.get("benchmark_steps", 120))
    warmup_steps = int(auto_cfg.get("warmup_steps", 30))

    bench_cfg = ExperimentConfig(
        train_csv=train_csv,
        checkpoint_out=(Path(config["paths"]["output_root"]) / "autotune" / "noop.pt").as_posix(),
        segment_len=int(training_cfg.get("segment_len", 32000)),
        sample_rate=int(training_cfg.get("sample_rate", 16000)),
        n_fft=int(training_cfg.get("n_fft", 512)),
        hop_length=int(training_cfg.get("hop_length", 160)),
        win_length=int(training_cfg.get("win_length", 320)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        grad_accum=int(training_cfg.get("grad_accum", 1)),
        num_workers=int(training_cfg.get("num_workers", 4)),
        prefetch_factor=int(training_cfg.get("prefetch_factor", 2)),
        persistent_workers=bool(training_cfg.get("persistent_workers", True)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        device=device,
        amp=False,
    )
    payload = autotune_loader_profile(
        bench_cfg,
        candidates_num_workers=candidates_workers,
        candidates_prefetch_factor=candidates_prefetch,
        max_steps=benchmark_steps,
        warmup_steps=warmup_steps,
    )
    write_json(cache_path, payload)
    winner = dict(payload["winner"])
    return {
        "num_workers": int(winner["num_workers"]),
        "prefetch_factor": int(winner["prefetch_factor"]),
    }


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _unique_path(base: Path, *, is_dir: bool) -> Path:
    if not base.exists():
        if is_dir:
            base.mkdir(parents=True, exist_ok=False)
        return base
    for idx in range(1, 1000):
        if is_dir:
            candidate = base.with_name(f"{base.name}_{idx:02d}")
        else:
            candidate = base.with_name(f"{base.stem}_{idx:02d}{base.suffix}")
        if candidate.exists():
            continue
        if is_dir:
            candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    raise RuntimeError(f"Could not allocate unique path for {base}")


def _create_run_dir(group_root: Path, run_name: str) -> Path:
    run_root = group_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    candidate = run_root / f"run_{_timestamp_token()}"
    return _unique_path(candidate, is_dir=True)


def _iter_run_dirs(group_root: Path, run_name: str) -> list[Path]:
    run_root = group_root / run_name
    if not run_root.exists():
        return []
    dirs = [item for item in run_root.iterdir() if item.is_dir()]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return dirs


def _find_latest_run_artifact(
    group_root: Path,
    run_name: str,
    artifact_name: str,
    *,
    exclude_dir: Path | None = None,
) -> str | None:
    excluded = exclude_dir.resolve() if exclude_dir is not None else None
    for run_dir in _iter_run_dirs(group_root, run_name):
        if excluded is not None and run_dir.resolve() == excluded:
            continue
        candidate = run_dir / artifact_name
        if candidate.exists():
            return candidate.as_posix()
    return None


def _phase_resume_state(
    group_root: Path,
    run_name: str,
    *,
    min_epoch: int = 1,
) -> dict[str, Any]:
    accepted_reasons = {"epoch", "final", "interrupted", "failed", "periodic", "best"}
    for run_dir in _iter_run_dirs(group_root, run_name):
        state_path = run_dir / "latest_state.pt"
        if not state_path.exists():
            continue
        try:
            state = torch.load(state_path, map_location="cpu")
        except Exception:
            continue

        reason = str(state.get("reason") or "").strip().lower()
        epoch = int(state.get("epoch", 0) or 0)
        if epoch < int(min_epoch):
            continue
        if reason not in accepted_reasons:
            continue

        checkpoint_path = run_dir / "model.pt"
        payload = {
            "resume_training_state": state_path.as_posix(),
            "resume_epoch": epoch,
            "resume_reason": reason,
        }
        if checkpoint_path.exists():
            payload["init_checkpoint"] = checkpoint_path.as_posix()
        return payload
    return {}


def _summary_group_dir(output_root: Path, group: str) -> Path:
    return output_root / "summaries" / group


def _write_group_summary(output_root: Path, group: str, payload: dict[str, Any]) -> Path:
    summary_dir = _summary_group_dir(output_root, group)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _unique_path(summary_dir / f"summary_{_timestamp_token()}.json", is_dir=False)
    write_json(summary_path, payload)
    return summary_path


def _checkpoint_group_summary(output_root: Path, group: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary_path = _write_group_summary(output_root, group, payload)
    enriched = dict(payload)
    enriched["summary_path"] = summary_path.as_posix()
    return enriched


def _latest_group_summary_path(output_root: Path, group: str) -> Path | None:
    summary_dir = _summary_group_dir(output_root, group)
    if not summary_dir.exists():
        return None
    candidates = sorted(summary_dir.glob("summary_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return None


def _load_latest_group_summary(
    output_root: Path,
    group: str,
    *,
    legacy_filename: str | None = None,
) -> dict[str, Any] | None:
    latest = _latest_group_summary_path(output_root, group)
    if latest is not None:
        return read_json(latest)
    if legacy_filename:
        legacy_path = output_root / legacy_filename
        if legacy_path.exists():
            return read_json(legacy_path)
    return None


def _copy_manifest(src: str | Path, dst: str | Path) -> str:
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)
    return dst_path.as_posix()


def materialize_test_manifest_8k(config: dict[str, Any], *, force: bool = False) -> str:
    dst = Path(config["dataset"]["test_csv_8k"])
    if dst.exists() and not force:
        return dst.as_posix()
    return _copy_manifest(config["dataset"]["test_csv_16k"], dst)


def _concat_csvs(output_csv: str | Path, input_csvs: list[str | Path], force: bool = False) -> str:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path.as_posix()

    headers = None
    rows: list[list[str]] = []
    seen: set[str] = set()
    for in_csv in input_csvs:
        with Path(in_csv).open(newline="") as f:
            reader = csv.reader(f)
            file_headers = next(reader, None)
            if file_headers is None:
                continue
            if headers is None:
                headers = file_headers
            elif headers != file_headers:
                raise ValueError(f"CSV header mismatch: {in_csv}")
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2 and headers[:2] == ["noisy", "clean"]:
                    key = f"{_normalize_path(row[0])}|{_normalize_path(row[1])}"
                else:
                    key = "\x1f".join(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    if headers is None:
        raise ValueError(f"No rows to concat for {output_csv}")

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return output_path.as_posix()


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _pair_key(noisy_path: str, clean_path: str) -> str:
    return f"{_normalize_path(noisy_path)}|{_normalize_path(clean_path)}"


def _clean_key(clean_path: str) -> str:
    value = _normalize_path(clean_path)
    for marker in ("/clean_train/", "/clean_val/", "/clean_test/", "/clean_sources/"):
        if marker in value:
            return value.split(marker, 1)[1].lstrip("/")
    return value


def _speaker_key(clean_path: str) -> str | None:
    match = SPEAKER_RE.match(Path(clean_path).name)
    if not match:
        return None
    return match.group(1).lower()


def _write_manifest_rows(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["noisy", "clean"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"noisy": row.noisy.as_posix(), "clean": row.clean.as_posix()})


def _manifest_row_count(path: str | Path) -> int:
    return len(read_pair_manifest(path))


def _manifest_exists(path: str | Path | None) -> bool:
    return bool(str(path or "").strip()) and Path(str(path)).exists()


def _cached_explicit_summary_is_valid(summary: dict[str, Any]) -> bool:
    if int(summary.get("policy_version", 0) or 0) != ACADEMIC_EXPLICIT_SUMMARY_VERSION:
        return False
    required_paths: list[str] = []
    combined = dict(summary.get("combined_manifests") or {})
    required_paths.extend(
        [
            str(combined.get("train_fit") or ""),
            str(combined.get("val_rank") or ""),
            str(combined.get("val_select") or ""),
            str(combined.get("test") or ""),
        ]
    )
    per_domain = dict(summary.get("per_domain_manifests") or {})
    for domain in ("voicebank", "dns5"):
        payload = dict(per_domain.get(domain) or {})
        required_paths.extend(
            [
                str(payload.get("train_fit") or ""),
                str(payload.get("val_rank") or ""),
                str(payload.get("val_select") or ""),
            ]
        )
        optional_test = str(payload.get("test") or "")
        if optional_test and not Path(optional_test).exists():
            return False
    return all(path and Path(path).exists() for path in required_paths)


def _materialize_explicit_runtime_tests(
    config: dict[str, Any],
    *,
    dataset_cfg: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    out_root = Path(config["paths"]["output_root"]) / "combined" / "explicit_runtime_tests"
    out_root.mkdir(parents=True, exist_ok=True)

    combined_test_csv = str(dataset_cfg.get("combined_test_csv") or "").strip()
    combined_rows = read_pair_manifest(combined_test_csv) if _manifest_exists(combined_test_csv) else []
    voicebank_expected = int(dataset_cfg.get("voicebank_test_expected_count", VOICEBANK_OFFICIAL_TEST_COUNT))
    if voicebank_expected <= 0:
        raise ValueError("dataset.voicebank_test_expected_count must be > 0 for academic runtime reconstruction.")

    voicebank_test_csv = str(dataset_cfg.get("voicebank_test_csv") or "").strip()
    dns5_test_csv = str(dataset_cfg.get("dns5_test_csv_runtime") or dataset_cfg.get("dns5_test_csv") or "").strip()
    notes: list[str] = []

    voicebank_test = ""
    voicebank_source = "missing"
    if _manifest_exists(voicebank_test_csv):
        voicebank_rows = _manifest_row_count(voicebank_test_csv)
        if voicebank_rows == voicebank_expected:
            voicebank_test = voicebank_test_csv
            voicebank_source = "explicit_runtime_manifest"
        else:
            notes.append(
                f"Ignored voicebank_test_csv={voicebank_test_csv} because rows={voicebank_rows} "
                f"!= expected_official_rows={voicebank_expected}."
            )

    if not voicebank_test:
        if len(combined_rows) < voicebank_expected:
            raise ValueError(
                "Cannot reconstruct VoiceBank official test from combined_test_csv: "
                f"rows={len(combined_rows)} expected_at_least={voicebank_expected}"
            )
        voicebank_runtime_test = out_root / "voicebank_test_runtime.csv"
        if force or not voicebank_runtime_test.exists():
            _write_manifest_rows(voicebank_runtime_test, combined_rows[:voicebank_expected])
        voicebank_test = voicebank_runtime_test.as_posix()
        voicebank_source = "reconstructed_from_combined_prefix"
        notes.append(
            "VoiceBank test was reconstructed from the ordered prefix of combined_test_csv because "
            "the raw official test manifest was unavailable in the runtime bundle."
        )

    dns5_test = ""
    dns5_source = "missing"
    if _manifest_exists(dns5_test_csv):
        dns5_test = dns5_test_csv
        dns5_source = "explicit_runtime_manifest"

    if not dns5_test and combined_rows:
        dns5_rows = combined_rows[voicebank_expected:]
        if dns5_rows:
            dns5_runtime_test = out_root / "dns5_test_runtime.csv"
            if force or not dns5_runtime_test.exists():
                _write_manifest_rows(dns5_runtime_test, dns5_rows)
            dns5_test = dns5_runtime_test.as_posix()
            dns5_source = "reconstructed_from_combined_suffix"
            notes.append(
                "DNS5 test was reconstructed from the ordered suffix of combined_test_csv because "
                "no explicit DNS5 runtime test manifest was provided."
            )

    combined_test_runtime = ""
    combined_test_inputs = [path for path in (voicebank_test, dns5_test) if str(path).strip()]
    if combined_test_inputs:
        combined_test_runtime = _concat_csvs(
            out_root / "combined_test_runtime.csv",
            combined_test_inputs,
            force=True,
        )

    return {
        "voicebank_test_csv": voicebank_test,
        "voicebank_test_source": voicebank_source,
        "voicebank_test_rows": _manifest_row_count(voicebank_test) if voicebank_test else 0,
        "voicebank_test_expected_rows": voicebank_expected,
        "dns5_test_csv": dns5_test,
        "dns5_test_source": dns5_source,
        "dns5_test_rows": _manifest_row_count(dns5_test) if dns5_test else 0,
        "dns5_test_included_in_combined": bool(dns5_test),
        "combined_test_csv": combined_test_runtime or combined_test_csv,
        "notes": notes,
    }


def _manifest_keysets(manifest_path: str | Path) -> dict[str, Any]:
    rows = read_pair_manifest(manifest_path)
    pair_set: set[str] = set()
    clean_set: set[str] = set()
    for row in rows:
        pair_set.add(_pair_key(row.noisy.as_posix(), row.clean.as_posix()))
        clean_set.add(_clean_key(row.clean.as_posix()))
    return {
        "rows": len(rows),
        "pair_set": pair_set,
        "clean_set": clean_set,
        "duplicate_pairs": len(rows) - len(pair_set),
        "duplicate_clean_keys": len(rows) - len(clean_set),
    }


def _audit_manifest_bundle(
    manifests: dict[str, str | Path],
    *,
    strict: bool,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    for label, path in manifests.items():
        loaded[label] = _manifest_keysets(path)

    per_manifest = {
        label: {
            "rows": payload["rows"],
            "duplicate_pairs": payload["duplicate_pairs"],
            "duplicate_clean_keys": payload["duplicate_clean_keys"],
            "unique_pairs": len(payload["pair_set"]),
            "unique_clean_keys": len(payload["clean_set"]),
        }
        for label, payload in loaded.items()
    }

    categories: dict[str, dict[str, set[str]]] = {
        "train": {"pair": set(), "clean": set()},
        "val": {"pair": set(), "clean": set()},
        "test": {"pair": set(), "clean": set()},
    }

    for label, payload in loaded.items():
        lower = label.lower()
        if lower.startswith("train"):
            bucket = "train"
        elif lower.startswith("val"):
            bucket = "val"
        elif lower.startswith("test"):
            bucket = "test"
        else:
            continue
        categories[bucket]["pair"].update(payload["pair_set"])
        categories[bucket]["clean"].update(payload["clean_set"])

    boundaries = {
        "train_vs_val": {
            "pair_overlap": len(categories["train"]["pair"] & categories["val"]["pair"]),
            "clean_overlap": len(categories["train"]["clean"] & categories["val"]["clean"]),
        },
        "train_vs_test": {
            "pair_overlap": len(categories["train"]["pair"] & categories["test"]["pair"]),
            "clean_overlap": len(categories["train"]["clean"] & categories["test"]["clean"]),
        },
        "val_vs_test": {
            "pair_overlap": len(categories["val"]["pair"] & categories["test"]["pair"]),
            "clean_overlap": len(categories["val"]["clean"] & categories["test"]["clean"]),
        },
    }

    summary = {
        "manifests": {label: Path(path).as_posix() for label, path in manifests.items()},
        "per_manifest": per_manifest,
        "boundaries": boundaries,
    }

    if out_path:
        write_json(Path(out_path), summary)

    if strict:
        dup_issues = [
            label
            for label, payload in per_manifest.items()
            if payload["duplicate_pairs"] > 0 or payload["duplicate_clean_keys"] > 0
        ]
        if dup_issues:
            raise ValueError(f"Duplicate rows/clean keys detected in manifests: {dup_issues}")
        for boundary, payload in boundaries.items():
            if payload["pair_overlap"] > 0 or payload["clean_overlap"] > 0:
                raise ValueError(f"Data leakage detected on {boundary}: {payload}")

    return summary


def _prepare_voicebank_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    manifests = build_voicebank_campaign_splits(
        train_csv=config["dataset"]["train_csv_16k"],
        output_dir=config["dataset"]["campaign_dir_8k"],
        val_speakers=tuple(config["dataset"]["val_speakers"]),
        rank_count=int(config["dataset"]["rank_count"]),
    )
    manifests["test_8k"] = materialize_test_manifest_8k(config, force=force)
    return {
        "campaign_manifests": manifests,
        "val_speakers": list(config["dataset"]["val_speakers"]),
        "rank_count": int(config["dataset"]["rank_count"]),
    }


def _prepare_dns5_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    train_csv = config["dataset"].get("train_csv_16k")
    test_csv = config["dataset"].get("test_csv_16k")
    val_rank_csv = config["dataset"].get("val_rank_csv")
    val_select_csv = config["dataset"].get("val_select_csv")
    if not train_csv or not test_csv:
        raise ValueError("DNS5 dataset requires train_csv_16k and test_csv_16k")
    return {
        "campaign_manifests": {
            "train_fit": train_csv,
            "val_rank": val_rank_csv or "",
            "val_select": val_select_csv or "",
            "test": test_csv,
        },
        "val_speakers": config["dataset"].get("val_speakers", []),
        "rank_count": int(config["dataset"].get("rank_count", 0)),
    }


def _prepare_combined_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    combined_dir = Path(config["paths"]["output_root"]) / "combined_manifest"
    combined_dir.mkdir(parents=True, exist_ok=True)
    vbd_train = config["dataset"]["vbd_train_csv_16k"]
    dns_train = config["dataset"]["dns5_train_csv_16k"]
    combined_train = combined_dir / "train_combined.csv"
    _concat_csvs(combined_train, [vbd_train, dns_train], force=force)

    vbd_test = config["dataset"]["vbd_test_csv_16k"]
    dns_test = config["dataset"]["dns5_test_csv_16k"]
    combined_test = combined_dir / "test_combined.csv"
    _concat_csvs(combined_test, [vbd_test, dns_test], force=force)

    return {
        "campaign_manifests": {
            "train_fit": combined_train.as_posix(),
            "test": combined_test.as_posix(),
        },
        "val_speakers": config["dataset"].get("val_speakers", []),
        "rank_count": int(config["dataset"].get("rank_count", 0)),
    }


def _split_manifest_rank_select(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    rank_count: int,
    force: bool = False,
    prefix: str = "split",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_csv = output_dir / f"{prefix}_rank.csv"
    select_csv = output_dir / f"{prefix}_select.csv"
    if rank_csv.exists() and select_csv.exists() and not force:
        return {"rank": rank_csv.as_posix(), "select": select_csv.as_posix()}

    rows = read_pair_manifest(manifest_path)
    dedup: dict[str, Any] = {}
    for row in rows:
        key = _pair_key(row.noisy.as_posix(), row.clean.as_posix())
        if key not in dedup:
            dedup[key] = row
    ordered = sorted(dedup.values(), key=lambda row: _pair_key(row.noisy.as_posix(), row.clean.as_posix()))

    rank_size = max(0, min(int(rank_count), len(ordered)))
    rank_rows = ordered[:rank_size]
    select_rows = ordered[rank_size:]

    _write_manifest_rows(rank_csv, rank_rows)
    _write_manifest_rows(select_csv, select_rows)
    return {"rank": rank_csv.as_posix(), "select": select_csv.as_posix()}


def _split_manifest_train_test(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    test_fraction: float,
    seed: int = 42,
    force: bool = False,
    prefix: str = "split",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / f"{prefix}_train.csv"
    test_csv = output_dir / f"{prefix}_test.csv"
    if train_csv.exists() and test_csv.exists() and not force:
        return {"train": train_csv.as_posix(), "test": test_csv.as_posix(), "reused": True}

    rows = read_pair_manifest(manifest_path)
    dedup: dict[str, Any] = {}
    for row in rows:
        key = _pair_key(row.noisy.as_posix(), row.clean.as_posix())
        if key not in dedup:
            dedup[key] = row
    unique_rows = list(dedup.values())

    target_fraction = max(0.0, min(0.5, float(test_fraction)))
    if not unique_rows:
        _write_manifest_rows(train_csv, [])
        _write_manifest_rows(test_csv, [])
        return {"train": train_csv.as_posix(), "test": test_csv.as_posix(), "reused": False, "rows": 0}

    groups: dict[str, list[Any]] = defaultdict(list)
    for row in unique_rows:
        groups[_clean_key(row.clean.as_posix())].append(row)
    group_keys = sorted(groups.keys())
    rng = random.Random(int(seed))
    rng.shuffle(group_keys)

    total = len(unique_rows)
    target_test = int(round(total * target_fraction))
    target_test = max(1, min(target_test, total - 1)) if total > 1 and target_fraction > 0 else 0

    train_rows: list[Any] = []
    test_rows: list[Any] = []
    for key in group_keys:
        bucket = test_rows if len(test_rows) < target_test else train_rows
        bucket.extend(groups[key])

    # Safety: never leave train empty when we have enough rows.
    if not train_rows and len(test_rows) > 1:
        train_rows.append(test_rows.pop())

    _write_manifest_rows(train_csv, train_rows)
    _write_manifest_rows(test_csv, test_rows)
    return {
        "train": train_csv.as_posix(),
        "test": test_csv.as_posix(),
        "reused": False,
        "rows_total": len(rows),
        "rows_unique": len(unique_rows),
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
        "duplicates_removed": len(rows) - len(unique_rows),
        "target_test_fraction": target_fraction,
    }


def _prepare_academic_combined_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    explicit_voicebank_keys = ("voicebank_train_fit_csv", "voicebank_val_rank_csv", "voicebank_val_select_csv")
    explicit_dns5_keys = ("dns5_train_fit_csv", "dns5_val_rank_csv", "dns5_val_select_csv")
    if all(str(dataset_cfg.get(key) or "").strip() for key in (*explicit_voicebank_keys, *explicit_dns5_keys)):
        summary_path = Path(config["paths"]["output_root"]) / "prepare_data" / "academic_combined_explicit_summary.json"
        if summary_path.exists() and not force:
            existing = read_json(summary_path)
            if _cached_explicit_summary_is_valid(existing):
                return existing
        runtime_tests = _materialize_explicit_runtime_tests(config, dataset_cfg=dataset_cfg, force=force)
        combined_manifests = {
            "train_fit": str(dataset_cfg["combined_train_csv"]),
            "val_rank": str(dataset_cfg["combined_val_rank_csv"]),
            "val_select": str(dataset_cfg["combined_val_select_csv"]),
            "test": str(runtime_tests.get("combined_test_csv") or dataset_cfg.get("combined_test_csv") or ""),
        }
        integrity_manifests = {key: value for key, value in combined_manifests.items() if str(value).strip()}
        integrity = _audit_manifest_bundle(
            integrity_manifests,
            strict=True,
            out_path=Path(config["paths"]["output_root"]) / "combined" / "combined_integrity_summary.json",
        )
        payload = {
            "voicebank": {
                "train_csv": str(dataset_cfg.get("voicebank_train_fit_csv") or ""),
                "test_csv": str(runtime_tests.get("voicebank_test_csv") or ""),
                "campaign_manifests": {
                    "train_fit": str(dataset_cfg["voicebank_train_fit_csv"]),
                    "val_rank": str(dataset_cfg["voicebank_val_rank_csv"]),
                    "val_select": str(dataset_cfg["voicebank_val_select_csv"]),
                },
                "val_speakers": list(dataset_cfg.get("val_speakers") or []),
                "rank_count": int(dataset_cfg.get("rank_count", 0)),
                "test_source": str(runtime_tests.get("voicebank_test_source") or "missing"),
                "test_expected_rows": int(runtime_tests.get("voicebank_test_expected_rows", VOICEBANK_OFFICIAL_TEST_COUNT)),
            },
            "dns5": {
                "train_csv": str(dataset_cfg.get("dns5_train_fit_csv") or ""),
                "train_fit_csv": str(dataset_cfg.get("dns5_train_fit_csv") or ""),
                "val_csv": str(dataset_cfg.get("dns5_val_select_csv") or ""),
                "val_split": {
                    "rank": str(dataset_cfg["dns5_val_rank_csv"]),
                    "select": str(dataset_cfg["dns5_val_select_csv"]),
                },
                "val_rank_count": int(dataset_cfg.get("dns5_rank_count", 0)),
                "test_csv": str(runtime_tests.get("dns5_test_csv") or ""),
                "test_source": str(runtime_tests.get("dns5_test_source") or "missing"),
                "test_included_in_combined": bool(runtime_tests.get("dns5_test_included_in_combined", False)),
                "train_test_split": {},
            },
            "per_domain_manifests": {
                "voicebank": {
                    "train_fit": str(dataset_cfg["voicebank_train_fit_csv"]),
                    "val_rank": str(dataset_cfg["voicebank_val_rank_csv"]),
                    "val_select": str(dataset_cfg["voicebank_val_select_csv"]),
                    "test": str(runtime_tests.get("voicebank_test_csv") or ""),
                },
                "dns5": {
                    "train_fit": str(dataset_cfg["dns5_train_fit_csv"]),
                    "val_rank": str(dataset_cfg["dns5_val_rank_csv"]),
                    "val_select": str(dataset_cfg["dns5_val_select_csv"]),
                    "test": str(runtime_tests.get("dns5_test_csv") or ""),
                },
            },
            "combined_manifests": combined_manifests,
            "integrity": integrity,
            "notes": [
                "Academic combined dataset loaded from explicit prebuilt per-domain manifests.",
                *list(runtime_tests.get("notes") or []),
            ],
            "policy_version": ACADEMIC_EXPLICIT_SUMMARY_VERSION,
        }
        write_json(summary_path, payload)
        return payload

    voicebank_root = Path(dataset_cfg["voicebank_root"])
    dns5_root = Path(dataset_cfg["dns5_root"])

    # VoiceBank+DEMAND uses official train/test; validation is built from official train.
    vbd_train = (voicebank_root / "16k" / "train.csv").as_posix()
    vbd_test = (voicebank_root / "16k" / "test.csv").as_posix()
    campaign_dir = dataset_cfg.get("voicebank_campaign_dir") or (voicebank_root / "16k" / "campaign").as_posix()
    val_speakers = tuple(dataset_cfg.get("val_speakers") or ("p239", "p286", "p244", "p270"))
    rank_count = int(dataset_cfg.get("rank_count", 128))
    vbd_campaign = build_voicebank_campaign_splits(
        train_csv=vbd_train,
        output_dir=campaign_dir,
        val_speakers=val_speakers,
        rank_count=rank_count,
    )

    # DNS5 has train+val locally. Split val into disjoint rank/select for stable training signals.
    dns5_train = dataset_cfg.get("dns5_train_csv") or (dns5_root / "train.csv").as_posix()
    dns5_val = dataset_cfg.get("dns5_val_csv") or (dns5_root / "val.csv").as_posix()
    dns5_rank_count = int(dataset_cfg.get("dns5_rank_count", 4096))
    dns5_test_from_train_fraction = float(dataset_cfg.get("dns5_test_from_train_fraction", 0.10))
    split_seed = int(dataset_cfg.get("split_seed", 42))
    combined_dir = Path(config["paths"]["output_root"]) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    dns5_split = _split_manifest_rank_select(
        dns5_val,
        combined_dir / "dns5_val_split",
        rank_count=dns5_rank_count,
        force=force,
        prefix="dns5_val",
    )

    dns5_test_candidate = str(dataset_cfg.get("dns5_test_csv") or "").strip()
    dns5_train_for_fit = dns5_train
    dns5_test_for_combined: str | None = None
    dns5_test_source = "none"
    dns5_train_test_split: dict[str, Any] | None = None

    if dns5_test_candidate and Path(dns5_test_candidate).exists():
        dns5_test_for_combined = dns5_test_candidate
        dns5_test_source = "external_dns5_test_csv"
    elif dns5_test_from_train_fraction > 0.0:
        dns5_train_test_split = _split_manifest_train_test(
            dns5_train,
            combined_dir / "dns5_train_test_split",
            test_fraction=dns5_test_from_train_fraction,
            seed=split_seed,
            force=force,
            prefix="dns5",
        )
        dns5_train_for_fit = dns5_train_test_split["train"]
        dns5_test_for_combined = dns5_train_test_split["test"]
        dns5_test_source = "derived_from_dns5_train"

    # Build combined manifests.
    combined_train = _concat_csvs(
        dataset_cfg["combined_train_csv"],
        [vbd_campaign["train_fit"], dns5_train_for_fit],
        force=force,
    )
    combined_val_rank = _concat_csvs(
        dataset_cfg["combined_val_rank_csv"],
        [vbd_campaign["val_rank"], dns5_split["rank"]],
        force=force,
    )
    combined_val_select = _concat_csvs(
        dataset_cfg["combined_val_select_csv"],
        [vbd_campaign["val_select"], dns5_split["select"]],
        force=force,
    )

    test_inputs = [vbd_test]
    dns5_test_included = False
    if dns5_test_for_combined and Path(dns5_test_for_combined).exists():
        test_inputs.append(dns5_test_for_combined)
        dns5_test_included = True
    combined_test = _concat_csvs(
        dataset_cfg["combined_test_csv"],
        test_inputs,
        force=force,
    )

    combined_manifests = {
        "train_fit": combined_train,
        "val_rank": combined_val_rank,
        "val_select": combined_val_select,
        "test": combined_test,
    }
    integrity = _audit_manifest_bundle(
        combined_manifests,
        strict=True,
        out_path=combined_dir / "combined_integrity_summary.json",
    )

    return {
        "voicebank": {
            "train_csv": vbd_train,
            "test_csv": vbd_test,
            "campaign_manifests": vbd_campaign,
            "val_speakers": list(val_speakers),
            "rank_count": rank_count,
        },
        "dns5": {
            "train_csv": dns5_train,
            "train_fit_csv": dns5_train_for_fit,
            "val_csv": dns5_val,
            "val_split": dns5_split,
            "val_rank_count": dns5_rank_count,
            "test_csv": dns5_test_for_combined or "",
            "test_source": dns5_test_source,
            "test_included_in_combined": dns5_test_included,
            "train_test_split": dns5_train_test_split or {},
        },
        "per_domain_manifests": {
            "voicebank": {
                "train_fit": vbd_campaign["train_fit"],
                "val_rank": vbd_campaign["val_rank"],
                "val_select": vbd_campaign["val_select"],
                "test": vbd_test,
            },
            "dns5": {
                "train_fit": dns5_train_for_fit,
                "val_rank": dns5_split["rank"],
                "val_select": dns5_split["select"],
                "test": dns5_test_for_combined or "",
            },
        },
        "combined_manifests": combined_manifests,
        "integrity": integrity,
        "notes": [
            "VoiceBank test uses official test.csv.",
            "VoiceBank validation is speaker-holdout from official train.csv.",
            "DNS5 val is split into disjoint val_rank/val_select.",
            "DNS5 test comes from explicit dns5_test_csv or (fallback) deterministic split from DNS5 train.",
        ],
    }


def _academic_dataset_bundle(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    dataset_type = config["dataset"].get("dataset_type", "voicebank")
    if dataset_type != "academic_combined":
        raise ValueError(f"This workflow expects dataset_type=academic_combined, got: {dataset_type}")
    return _prepare_academic_combined_dataset(config, force=force)


def _per_domain_manifests(config: dict[str, Any], *, force: bool = False) -> dict[str, dict[str, str]]:
    return _academic_dataset_bundle(config, force=force)["per_domain_manifests"]


def _resolved_per_domain_manifests(config: dict[str, Any], *, force: bool = False) -> dict[str, dict[str, str]]:
    resolved: dict[str, dict[str, str]] = {}
    for domain_name, manifest_map in _per_domain_manifests(config, force=force).items():
        resolved[domain_name] = {
            split_name: (_resolve_staged_manifest_path(config, path) if str(path).strip() else "")
            for split_name, path in manifest_map.items()
        }
    return resolved


def _pair_set_from_manifest(manifest_path: str | Path) -> set[tuple[str, str]]:
    return {
        (row.noisy.as_posix().lower(), row.clean.as_posix().lower())
        for row in read_pair_manifest(manifest_path)
    }


def _sample_manifest_rows(manifest_path: str | Path, *, max_rows: int, seed: int) -> list[Any]:
    rows = unique_manifest_rows(read_pair_manifest(manifest_path))
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def _build_replay_schedule_manifests(
    voicebank_train_csv: str | Path,
    dns5_train_csv: str | Path,
    *,
    out_dir: str | Path,
    seed: int,
    dns_fraction: float = 1.0,
    prefix: str,
    force: bool = False,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f"{prefix}_summary.json"
    if summary_path.exists() and not force:
        existing = read_json(summary_path)
        existing_mixed = [str(path) for path in existing.get("mixed_manifests", [])]
        if (
            existing.get("voicebank_train_csv") == str(voicebank_train_csv)
            and existing.get("dns5_train_csv") == str(dns5_train_csv)
            and float(existing.get("dns_fraction", dns_fraction)) == float(dns_fraction)
            and all(Path(path).exists() for path in existing_mixed)
        ):
            return existing

    voice_rows = unique_manifest_rows(read_pair_manifest(voicebank_train_csv))
    dns_rows = unique_manifest_rows(read_pair_manifest(dns5_train_csv))
    shard_size = max(1, int(round(len(voice_rows) * float(dns_fraction))))
    dns_shards = partition_manifest_rows(dns_rows, shard_size=shard_size, seed=seed)

    dns_manifests: list[str] = []
    mixed_manifests: list[str] = []
    for index, shard_rows in enumerate(dns_shards, start=1):
        dns_path = out_root / f"{prefix}_dns_shard_{index:03d}.csv"
        mixed_path = out_root / f"{prefix}_mixed_{index:03d}.csv"
        write_pair_manifest(dns_path, shard_rows)
        write_pair_manifest(mixed_path, unique_manifest_rows([*voice_rows, *shard_rows]))
        dns_manifests.append(dns_path.as_posix())
        mixed_manifests.append(mixed_path.as_posix())

    summary = {
        "voicebank_train_csv": str(voicebank_train_csv),
        "dns5_train_csv": str(dns5_train_csv),
        "voicebank_rows": len(voice_rows),
        "dns5_rows": len(dns_rows),
        "dns_fraction": float(dns_fraction),
        "shard_size": int(shard_size),
        "dns_manifests": dns_manifests,
        "mixed_manifests": mixed_manifests,
    }
    write_json(summary_path, summary)
    return summary


def _build_teacher_cache_schedule(
    teacher_cache_manifest: str | Path,
    train_manifests: list[str],
    *,
    out_dir: str | Path,
    prefix: str,
    force: bool = False,
) -> list[str]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for index, train_manifest in enumerate(train_manifests, start=1):
        out_path = out_root / f"{prefix}_{index:03d}.csv"
        allowed_pairs = _pair_set_from_manifest(train_manifest)
        outputs.append(
            filter_teacher_cache_manifest(
                teacher_cache_manifest,
                allowed_pairs=allowed_pairs,
                out_path=out_path,
            )
        )
    return outputs


def _teacher_cache_summary_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]) / "teacher_cache" / "summary.json"


def _load_teacher_cache_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = _teacher_cache_summary_path(config)
    if summary_path.exists():
        return read_json(summary_path)
    return None


def _resolve_teacher_cache_manifest(config: dict[str, Any]) -> str:
    summary = _load_teacher_cache_summary(config)
    if summary and summary.get("manifest"):
        return str(summary["manifest"])
    return str(config["teacher_cache"]["manifest"])


def _teacher_eval_overrides(
    config: dict[str, Any],
    per_domain: dict[str, dict[str, str]],
    *,
    guardrail_floor: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "val_rank_csv": _resolve_staged_manifest_path(config, per_domain["voicebank"]["val_rank"]),
        "val_select_csv": _resolve_staged_manifest_path(config, per_domain["voicebank"]["val_select"]),
        "rank_eval_manifests": {"dns5_val_rank": _resolve_staged_manifest_path(config, per_domain["dns5"]["val_rank"])},
        "select_eval_manifests": {"dns5_val_select": _resolve_staged_manifest_path(config, per_domain["dns5"]["val_select"])},
        "selection_metric": "val_select/pesq_mean",
        "rank_eval_every": 1,
        "select_eval_every": 2,
    }
    if guardrail_floor is not None:
        payload["selection_guardrail_metric"] = "dns5_val_select/pesq_mean"
        payload["selection_guardrail_min"] = float(guardrail_floor)
    return payload


def _resolve_teacher_start_checkpoint(config: dict[str, Any]) -> str:
    output_root = Path(config["paths"]["output_root"])
    teacher_summary = _load_latest_group_summary(output_root, "teacher_training", legacy_filename="teacher_training_results.json")
    if teacher_summary and teacher_summary.get("winner", {}).get("checkpoint_out"):
        return str(teacher_summary["winner"]["checkpoint_out"])
    teacher_group_root = output_root / "checkpoints" / "teacher"
    base_name = str(config.get("reference", {}).get("teacher_run_name") or "metricgan_plus_native8k-small-teacher-training-seed0")
    for run_name in (base_name, f"{base_name}-phase_c", f"{base_name}-phase_b", f"{base_name}-phase_a"):
        candidate = _find_latest_run_artifact(teacher_group_root, run_name, "model.pt")
        if candidate:
            return candidate
    resume_checkpoint = str(config.get("teacher_training", {}).get("resume_checkpoint") or "").strip()
    if resume_checkpoint and Path(resume_checkpoint).exists():
        return resume_checkpoint
    return str(config["paths"]["teacher_source_checkpoint"])


def _resolve_teacher_baseline_checkpoint(config: dict[str, Any]) -> str:
    reference_checkpoint = str(config["paths"].get("teacher_source_checkpoint") or "").strip()
    if reference_checkpoint:
        return reference_checkpoint
    return _resolve_teacher_start_checkpoint(config)


def _sg_baseline(noisy: torch.Tensor) -> torch.Tensor:
    config = resolve_postfilter_config("sg_input_floor", "medium")
    return spectral_gate_waveform(noisy.unsqueeze(0), noisy.unsqueeze(0), config).squeeze(0)


def _build_proxy_records(
    config: dict[str, Any],
    per_domain: dict[str, dict[str, str]],
    *,
    phase_a_checkpoint: str,
    phase_b_checkpoint: str,
    device: str,
    force: bool = False,
) -> tuple[list[dict[str, Any]], Path]:
    proxy_cfg = dict(config.get("teacher_training", {}).get("pesq_proxy") or {})
    out_root = Path(config["paths"]["output_root"]) / "teacher_pesq_proxy"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "proxy_records.json"
    enhanced_dir = out_root / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    max_rows = int(proxy_cfg.get("max_samples_per_domain", 512))
    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        meta = existing.get("metadata", {})
        if (
            meta.get("phase_a_checkpoint") == str(phase_a_checkpoint)
            and meta.get("phase_b_checkpoint") == str(phase_b_checkpoint)
            and int(meta.get("max_samples_per_domain", max_rows)) == max_rows
        ):
            return existing.get("records", []), out_root

    sample_rate = int(config["training"]["sample_rate"])
    seed = int(proxy_cfg.get("seed", 42))

    domain_rows: list[tuple[str, Any]] = []
    domain_rows.extend(("voicebank", row) for row in _sample_manifest_rows(per_domain["voicebank"]["train_fit"], max_rows=max_rows, seed=seed))
    domain_rows.extend(("dns5", row) for row in _sample_manifest_rows(per_domain["dns5"]["train_fit"], max_rows=max_rows, seed=seed + 1))

    sources: list[tuple[str, torch.nn.Module | None]] = []
    sources.append(("noisy", None))
    sources.append(("spectral_gating", None))
    try:
        raw_model = build_enhancer("metricgan_plus", "small").to(device)
        raw_model.eval()
        sources.append(("metricgan_raw", raw_model))
    except Exception:
        pass
    for label, checkpoint in (("phase_a", phase_a_checkpoint), ("phase_b", phase_b_checkpoint)):
        if checkpoint and Path(checkpoint).exists():
            model, _ = load_model_from_checkpoint(checkpoint, device=device)
            model.eval()
            sources.append((label, model))

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for source_name, source_model in sources:
            for domain_name, row in domain_rows:
                noisy, _ = load_mono_audio(row.noisy, sample_rate)
                clean, _ = load_mono_audio(row.clean, sample_rate)
                if source_name == "noisy":
                    enhanced = noisy.clone()
                elif source_name == "spectral_gating":
                    enhanced = _sg_baseline(noisy)
                else:
                    enhanced = source_model.denoise_single(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()
                aligned = min(noisy.numel(), clean.numel(), enhanced.numel())
                noisy = noisy[:aligned]
                clean = clean[:aligned]
                enhanced = enhanced[:aligned]
                token = hashlib.sha1(f"{source_name}|{domain_name}|{row.noisy}|{row.clean}".encode("utf-8")).hexdigest()[:16]
                enhanced_path = enhanced_dir / f"{token}.pt"
                torch.save(enhanced, enhanced_path)
                pesq = pesq_score(clean.cpu().numpy(), enhanced.cpu().numpy(), sample_rate)
                records.append(
                    {
                        "domain": domain_name,
                        "source": source_name,
                        "noisy": row.noisy.as_posix(),
                        "clean": row.clean.as_posix(),
                        "enhanced": enhanced_path.as_posix(),
                        "pesq": float(pesq),
                    }
                )

    write_json(
        manifest_path,
        {
            "records": records,
            "count": len(records),
            "metadata": {
                "phase_a_checkpoint": str(phase_a_checkpoint),
                "phase_b_checkpoint": str(phase_b_checkpoint),
                "max_samples_per_domain": int(max_rows),
                "sample_rate": int(sample_rate),
                "seed": int(seed),
                "voicebank_train_manifest": str(per_domain["voicebank"]["train_fit"]),
                "dns5_train_manifest": str(per_domain["dns5"]["train_fit"]),
            },
        },
    )
    return records, out_root


class _ProxyRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], *, sample_rate: int) -> None:
        self.records = list(records)
        self.sample_rate = int(sample_rate)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        noisy, _ = load_mono_audio(record["noisy"], self.sample_rate)
        clean, _ = load_mono_audio(record["clean"], self.sample_rate)
        enhanced = torch.load(record["enhanced"], map_location="cpu").float()
        aligned = min(noisy.numel(), clean.numel(), enhanced.numel())
        return {
            "noisy": noisy[:aligned],
            "clean": clean[:aligned],
            "enhanced": enhanced[:aligned],
            "target": torch.tensor(float(record["pesq"]), dtype=torch.float32),
        }


def _proxy_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "noisy": pad_sequence([item["noisy"] for item in batch], batch_first=True),
        "clean": pad_sequence([item["clean"] for item in batch], batch_first=True),
        "enhanced": pad_sequence([item["enhanced"] for item in batch], batch_first=True),
        "target": torch.stack([item["target"] for item in batch]),
    }


def _train_teacher_pesq_proxy(
    config: dict[str, Any],
    per_domain: dict[str, dict[str, str]],
    *,
    phase_a_checkpoint: str,
    phase_b_checkpoint: str,
    device: str,
    force: bool = False,
) -> str:
    proxy_cfg = dict(config.get("teacher_training", {}).get("pesq_proxy") or {})
    out_root = Path(config["paths"]["output_root"]) / "teacher_pesq_proxy"
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_root / "pesq_proxy.pt"
    summary_path = out_root / "summary.json"
    if checkpoint_path.exists() and summary_path.exists() and not force:
        existing = read_json(summary_path)
        if (
            existing.get("phase_a_checkpoint") == str(phase_a_checkpoint)
            and existing.get("phase_b_checkpoint") == str(phase_b_checkpoint)
        ):
            return checkpoint_path.as_posix()

    records, records_root = _build_proxy_records(
        config,
        per_domain,
        phase_a_checkpoint=phase_a_checkpoint,
        phase_b_checkpoint=phase_b_checkpoint,
        device=device,
        force=force,
    )
    rng = random.Random(int(proxy_cfg.get("seed", 42)))
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(0.1 * len(shuffled)))) if len(shuffled) > 1 else 0
    val_records = shuffled[:val_count] if val_count else shuffled
    train_records = shuffled[val_count:] if val_count else shuffled
    if not train_records:
        train_records = list(val_records)

    sample_rate = int(config["training"]["sample_rate"])
    train_dataset = _ProxyRecordDataset(train_records, sample_rate=sample_rate)
    val_dataset = _ProxyRecordDataset(val_records, sample_rate=sample_rate)
    batch_size = int(proxy_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_proxy_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_proxy_collate)

    model = PESQProxyRegressor(
        sample_rate=sample_rate,
        n_fft=int(config["training"]["n_fft"]),
        hop_length=int(config["training"]["hop_length"]),
        win_length=int(config["training"]["win_length"]),
        hidden_channels=int(proxy_cfg.get("hidden_channels", 32)),
        projection_dim=int(proxy_cfg.get("projection_dim", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(proxy_cfg.get("lr", 1e-3)), weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, int(proxy_cfg.get("epochs", 8)) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            enhanced = batch["enhanced"].to(device)
            target = batch["target"].to(device)
            prediction = model(noisy, enhanced, clean)
            loss = loss_fn(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * noisy.size(0)
            seen += noisy.size(0)

        model.eval()
        val_running = 0.0
        val_seen = 0
        with torch.inference_mode():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                enhanced = batch["enhanced"].to(device)
                target = batch["target"].to(device)
                prediction = model(noisy, enhanced, clean)
                loss = loss_fn(prediction, target)
                val_running += loss.item() * noisy.size(0)
                val_seen += noisy.size(0)
        train_loss = running / max(seen, 1)
        val_loss = val_running / max(val_seen, 1)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})
        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    save_pesq_proxy_checkpoint(checkpoint_path, model.cpu())
    write_json(
        summary_path,
        {
            "checkpoint": checkpoint_path.as_posix(),
            "records_root": records_root.as_posix(),
            "records": len(records),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "best_val_loss": float(best_val),
            "history": history,
            "phase_a_checkpoint": str(phase_a_checkpoint),
            "phase_b_checkpoint": str(phase_b_checkpoint),
        },
    )
    return checkpoint_path.as_posix()


def _create_reference_splits(manifest_path: str | Path, output_dir: str | Path, *, force: bool = False) -> dict[str, str]:
    """Create deterministic, leakage-safe 80/10/10 splits for any dataset."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train_80.csv"
    val_csv = output_dir / "val_10.csv"
    test_csv = output_dir / "test_10.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists() and not force:
        return {
            "train": train_csv.as_posix(),
            "val": val_csv.as_posix(),
            "test": test_csv.as_posix(),
        }

    # Read + dedupe exact pairs.
    rows = read_pair_manifest(manifest_path)
    seen_pairs: set[str] = set()
    unique_rows = []
    for row in rows:
        key = _pair_key(row.noisy.as_posix(), row.clean.as_posix())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_rows.append(row)
    total_samples = len(unique_rows)

    # Group rows so the same clean content never crosses split boundaries.
    speaker_keys = [_speaker_key(row.clean.as_posix()) for row in unique_rows]
    speaker_hits = [value for value in speaker_keys if value]
    speaker_ratio = (len(speaker_hits) / total_samples) if total_samples else 0.0
    speaker_count = len(set(speaker_hits))
    use_speaker_groups = speaker_ratio >= 0.98 and speaker_count >= 6

    groups: dict[str, list[Any]] = defaultdict(list)
    for row in unique_rows:
        if use_speaker_groups:
            group_key = _speaker_key(row.clean.as_posix()) or _clean_key(row.clean.as_posix())
        else:
            group_key = _clean_key(row.clean.as_posix())
        groups[group_key].append(row)

    # Deterministic assignment by groups.
    group_keys = sorted(groups.keys())
    rng = random.Random(42)
    rng.shuffle(group_keys)

    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    targets = {"train": train_size, "val": val_size, "test": test_size}
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for group_key in group_keys:
        group_rows = groups[group_key]
        deficits = {name: targets[name] - counts[name] for name in ("train", "val", "test")}
        if max(deficits.values()) <= 0:
            split_name = min(counts, key=counts.get)
        else:
            split_name = max(deficits, key=deficits.get)
        assigned[split_name].extend(group_rows)
        counts[split_name] += len(group_rows)

    train_rows = assigned["train"]
    val_rows = assigned["val"]
    test_rows = assigned["test"]

    _write_manifest_rows(train_csv, train_rows)
    _write_manifest_rows(val_csv, val_rows)
    _write_manifest_rows(test_csv, test_rows)

    integrity = _audit_manifest_bundle(
        {"train": train_csv, "val": val_csv, "test": test_csv},
        strict=True,
        out_path=output_dir / "split_integrity_summary.json",
    )

    summary = {
        "original_manifest": str(Path(manifest_path).resolve()),
        "total_samples": total_samples,
        "deduplicated_rows_removed": len(rows) - len(unique_rows),
        "grouping_strategy": "speaker" if use_speaker_groups else "clean_key",
        "splits": {
            "train_80": len(train_rows),
            "val_10": len(val_rows),
            "test_10": len(test_rows),
        },
        "manifests": {
            "train": train_csv.as_posix(),
            "val": val_csv.as_posix(),
            "test": test_csv.as_posix(),
        },
        "integrity": integrity,
    }

    write_json(output_dir / "reference_split_summary.json", summary)
    return {
        "train": train_csv.as_posix(),
        "val": val_csv.as_posix(),
        "test": test_csv.as_posix(),
    }


def _prepare_reference_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Prepare reference 80/10/10 splits for VoiceBank and DNS5, and combined datasets."""

    # Create reference splits for VoiceBank
    voicebank_splits = _create_reference_splits(
        config["dataset"]["voicebank_root"] + "/16k/train.csv",
        config["dataset"]["voicebank_root"] + "/reference_splits",
        force=force
    )

    # Create reference splits for DNS5
    dns5_splits = _create_reference_splits(
        config["dataset"]["dns5_root"] + "/train.csv",
        config["dataset"]["dns5_root"] + "/reference_splits",
        force=force
    )

    # Create combined datasets
    combined_dir = Path(config["paths"]["output_root"]) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Combined training set
    _concat_csvs(
        config["dataset"]["combined_train_csv"],
        [voicebank_splits["train"], dns5_splits["train"]],
        force=force
    )

    # Combined validation set
    _concat_csvs(
        config["dataset"]["combined_val_csv"],
        [voicebank_splits["val"], dns5_splits["val"]],
        force=force
    )

    # Combined test set
    _concat_csvs(
        config["dataset"]["combined_test_csv"],
        [voicebank_splits["test"], dns5_splits["test"]],
        force=force
    )

    combined_manifests = {
        "train_fit": config["dataset"]["combined_train_csv"],
        "val_rank": config["dataset"]["combined_val_csv"],
        "val_select": config["dataset"]["combined_val_csv"],  # Use same as val_rank for simplicity
        "test": config["dataset"]["combined_test_csv"],
    }
    integrity = _audit_manifest_bundle(
        combined_manifests,
        strict=True,
        out_path=combined_dir / "combined_integrity_summary.json",
    )

    return {
        "voicebank_splits": voicebank_splits,
        "dns5_splits": dns5_splits,
        "combined_manifests": combined_manifests,
        "integrity": integrity,
        "val_speakers": [],
        "rank_count": 0,
    }


def command_prepare_data(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    dataset_type = config["dataset"].get("dataset_type", "voicebank")
    if dataset_type == "voicebank":
        summary = _prepare_voicebank_dataset(config, force=force)
    elif dataset_type == "dns5":
        summary = _prepare_dns5_dataset(config, force=force)
    elif dataset_type == "combined":
        summary = _prepare_combined_dataset(config, force=force)
    elif dataset_type == "academic_combined":
        summary = _prepare_academic_combined_dataset(config, force=force)
    elif dataset_type == "reference_combined":
        summary = _prepare_reference_dataset(config, force=force)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    io_cfg = _io_staging_cfg(config)
    if bool(io_cfg.get("enabled", False)):
        summary["io_staging"] = command_prepare_stage_distributed(config, force=force)

    write_json(Path(config["paths"]["output_root"]) / "prepare_data" / "summary.json", summary)
    return summary


def command_prepare_stage_distributed(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    io_cfg = _io_staging_cfg(config)
    if not bool(io_cfg.get("enabled", False)):
        return {"enabled": False, "staged_manifests": {}}

    roots = [str(item) for item in io_cfg.get("roots", []) if str(item).strip()]
    if not roots:
        raise ValueError("io_staging.enabled=true requires io_staging.roots.")

    dataset_cfg = config.get("dataset", {})
    manifests: dict[str, str] = {}
    train_manifest = str(dataset_cfg.get("train_fit_csv") or dataset_cfg.get("combined_train_csv") or "")
    if train_manifest and Path(train_manifest).exists():
        manifests["train_fit"] = train_manifest
    val_rank = str(dataset_cfg.get("val_rank_csv") or dataset_cfg.get("combined_val_rank_csv") or "")
    val_select = str(dataset_cfg.get("val_select_csv") or dataset_cfg.get("combined_val_select_csv") or "")
    test_manifest = str(dataset_cfg.get("test_csv") or dataset_cfg.get("combined_test_csv") or "")
    if bool(io_cfg.get("stage_val_test", True)):
        if val_rank and Path(val_rank).exists():
            manifests["val_rank"] = val_rank
        if val_select and Path(val_select).exists():
            manifests["val_select"] = val_select
        if test_manifest and Path(test_manifest).exists():
            manifests["test"] = test_manifest

    if not manifests:
        raise ValueError("No manifests found for distributed staging.")

    staged_dir = Path(str(io_cfg["staged_manifest_dir"]))
    staged_dir.mkdir(parents=True, exist_ok=True)

    def _progress(message: str) -> None:
        print(f"[io_staging] {message}", flush=True)

    summary = stage_dataset_manifests(
        manifests,
        staged_manifest_dir=staged_dir,
        roots=roots,
        stage_subdir=str(io_cfg.get("stage_subdir", "ULP_STAGE_AUDIO")),
        reserve_overrides_gb=dict(io_cfg.get("reserve_gb_overrides") or {}),
        verify_mode=str(io_cfg.get("verify_mode", "size_and_sample_hash")),
        verify_hash_every_n=int(io_cfg.get("verify_hash_every_n", 2000)),
        copy_workers=int(io_cfg.get("copy_workers", 12)),
        progress_callback=_progress,
    )
    summary["enabled"] = True
    summary["roots"] = roots
    summary["force"] = bool(force)
    summary["staged_manifest_dir"] = staged_dir.as_posix()
    summary_path = Path(config["paths"]["output_root"]) / "prepare_data" / "staging_summary.json"
    write_json(summary_path, summary)
    summary["prepare_data_summary_path"] = summary_path.as_posix()
    return summary


def command_build_teacher_cache(config: dict[str, Any], *, device: str, force: bool = False) -> dict[str, Any]:
    print("[teacher_cache] resolving teacher checkpoint and cache manifest", flush=True)
    summary_path = _teacher_cache_summary_path(config)
    dataset_summary = _academic_dataset_bundle(config, force=False)
    train_manifest = _resolve_staged_manifest_path(config, dataset_summary["combined_manifests"]["train_fit"])
    output_root = Path(config["paths"]["output_root"])
    teacher_results = _load_latest_group_summary(
        output_root,
        "teacher_training",
        legacy_filename="teacher_training_results.json",
    )
    teacher_cfg = dict(config.get("teacher_training") or {})
    default_family = str((teacher_cfg.get("families") or ["metricgan_plus_native8k"])[0])
    default_variant = str((teacher_cfg.get("variants") or ["small"])[0])
    winner = dict(teacher_results.get("winner") or {}) if teacher_results is not None else {}
    teacher_family = str(winner.get("model_family") or default_family)
    teacher_variant = str(winner.get("variant") or default_variant)
    teacher_checkpoint = str(winner.get("checkpoint_out") or "").strip()
    if not teacher_checkpoint:
        fallback_checkpoint = _teacher_family_start_checkpoint(config, teacher_family)
        teacher_checkpoint = str(fallback_checkpoint or "").strip()
    teacher_source_mode = "checkpoint" if teacher_checkpoint else "builder"

    if summary_path.exists() and not force:
        existing = read_json(summary_path)
        existing_manifest = str(existing.get("manifest") or "")
        if (
            existing.get("teacher_checkpoint") == teacher_checkpoint
            and existing.get("teacher_model_family") == teacher_family
            and existing.get("teacher_variant") == teacher_variant
            and existing.get("train_manifest") == train_manifest
            and existing_manifest
            and Path(existing_manifest).exists()
        ):
            print(f"[teacher_cache] reusing existing cache: {existing_manifest}", flush=True)
            return existing

    cache_cfg = dict(config.get("teacher_cache") or {})
    teacher_model = _build_runtime_teacher_model(
        config,
        device=device,
        model_family=teacher_family,
        variant=teacher_variant,
        checkpoint=teacher_checkpoint or None,
    )
    manifest = build_teacher_cache(
        train_manifest,
        teacher_model,
        out_dir=cache_cfg["out_dir"],
        device=device,
        target_sample_rate=int(config["training"]["sample_rate"]),
        teacher_sample_rate=int(config["training"]["sample_rate"]),
        erb_bands=int(config["training"]["erb_bands"]),
        guidance_classic=str(config["training"]["guidance_classic"]),
        batch_size=int(cache_cfg.get("batch_size", 64 if str(device).startswith("cuda") else 8)),
        num_workers=int(cache_cfg.get("num_workers", config["training"].get("num_workers", 8))),
        pin_memory=bool(cache_cfg.get("pin_memory", True)),
        persistent_workers=bool(cache_cfg.get("persistent_workers", True)),
        prefetch_factor=int(cache_cfg.get("prefetch_factor", config["training"].get("prefetch_factor", 2))),
    )
    expected_manifest = Path(cache_cfg["manifest"])
    if Path(manifest).resolve() != expected_manifest.resolve():
        expected_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, expected_manifest)
        manifest = expected_manifest.as_posix()
    summary = {
        "manifest": manifest,
        "out_dir": cache_cfg["out_dir"],
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_model_family": teacher_family,
        "teacher_variant": teacher_variant,
        "teacher_source_mode": teacher_source_mode,
        "quantized_teacher": False,
        "train_manifest": train_manifest,
        "device": device,
        "batch_size": int(cache_cfg.get("batch_size", 64 if str(device).startswith("cuda") else 8)),
        "num_workers": int(cache_cfg.get("num_workers", config["training"].get("num_workers", 8))),
    }
    write_json(summary_path, summary)
    print(f"[teacher_cache] built fp32 teacher cache: {manifest}", flush=True)
    return summary


def _phase_run_name(base_name: str, suffix: str) -> str:
    return f"{base_name}-{suffix}"


def _phase_settings(config: dict[str, Any], key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    payload = dict(defaults)
    payload.update(dict(config.get("teacher_training", {}).get(key) or {}))
    return payload


def _student_phase_settings(config: dict[str, Any], key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    payload = dict(defaults)
    payload.update(dict(config.get("stage1", {}).get(key) or {}))
    return payload


def _base_experiment_config(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    train_csv_default = str(config["dataset"].get("train_fit_csv") or config["dataset"].get("train_csv_16k"))
    val_rank_default = str(config["dataset"].get("val_rank_csv") or "")
    val_select_default = str(config["dataset"].get("val_select_csv") or "")
    test_csv_default = str(config["dataset"].get("test_csv") or config["dataset"].get("test_csv_16k") or "")
    train_csv = _resolve_staged_manifest_path(config, train_csv_default)
    val_rank_csv = _resolve_staged_manifest_path(config, val_rank_default) if val_rank_default else ""
    val_select_csv = _resolve_staged_manifest_path(config, val_select_default) if val_select_default else ""
    test_csv = _resolve_staged_manifest_path(config, test_csv_default) if test_csv_default else ""

    batch_size = training_cfg.get("batch_size")
    grad_accum = training_cfg.get("grad_accum")
    num_workers = training_cfg.get("num_workers")
    eval_batch_size = evaluation_cfg.get("eval_batch_size", training_cfg.get("eval_batch_size"))
    prefetch_factor = training_cfg.get("prefetch_factor")
    persistent_workers = training_cfg.get("persistent_workers")
    pin_memory = training_cfg.get("pin_memory")
    checkpoint_every_steps = training_cfg.get("checkpoint_every_steps")
    checkpoint_every_minutes = training_cfg.get("checkpoint_every_minutes")
    checkpoint_snapshot_every_periods = training_cfg.get("checkpoint_snapshot_every_periods")
    checkpoint_keep_last = training_cfg.get("checkpoint_keep_last")
    history_plot_every_epochs = training_cfg.get("history_plot_every_epochs")
    history_plot_final_only = bool(training_cfg.get("history_plot_final_only", False))
    history_persist_every_periods = training_cfg.get("history_persist_every_periods")
    record_step_history = bool(training_cfg.get("record_step_history", True))
    enable_torch_compile = bool(training_cfg.get("enable_torch_compile", False))
    torch_compile_mode = str(training_cfg.get("torch_compile_mode", "reduce-overhead"))
    max_eval_files = training_cfg.get("max_eval_files")
    rank_max_eval_files = training_cfg.get("rank_max_eval_files")
    final_max_eval_files = training_cfg.get("final_max_eval_files")
    cache_eval_audio = training_cfg.get("cache_eval_audio")
    rank_compute_composite = training_cfg.get("rank_compute_composite")
    select_compute_composite = training_cfg.get("select_compute_composite")
    if cache_eval_audio is None:
        cache_eval_audio = True
    if rank_compute_composite is None:
        rank_compute_composite = False
    if select_compute_composite is None:
        select_compute_composite = True

    loader_overrides = _resolve_loader_overrides(config, train_csv=train_csv, device=device)
    if loader_overrides is not None:
        num_workers = loader_overrides["num_workers"]
        prefetch_factor = loader_overrides["prefetch_factor"]

    return {
        "train_csv": train_csv,
        "val_rank_csv": val_rank_csv or None,
        "val_select_csv": val_select_csv or None,
        # Test sets are reserved for the explicit `evaluate` command only.
        "test_csv": None,
        "variant": "small",
        "segment_len": int(config["training"]["segment_len"]),
        "device": device,
        "scheduler": str(config["training"]["scheduler"]),
        "lr_factor": float(config["training"]["lr_factor"]),
        "lr_patience": int(config["training"]["lr_patience"]),
        "min_lr": float(config["training"]["min_lr"]),
        "eval_every": int(config["training"]["eval_every"]),
        "rank_eval_every": int(training_cfg.get("rank_eval_every", 1)),
        "select_eval_every": int(training_cfg.get("select_eval_every", config["training"]["eval_every"])),
        "grad_clip": float(config["training"]["grad_clip"]),
        "amp": bool(config["training"]["amp"]),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "grad_accum": int(grad_accum) if grad_accum is not None else None,
        "num_workers": int(num_workers) if num_workers is not None else None,
        "eval_batch_size": int(eval_batch_size) if eval_batch_size is not None else None,
        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
        "persistent_workers": bool(persistent_workers) if persistent_workers is not None else None,
        "pin_memory": bool(pin_memory) if pin_memory is not None else None,
        "checkpoint_every_steps": int(checkpoint_every_steps) if checkpoint_every_steps is not None else 500,
        "checkpoint_every_minutes": float(checkpoint_every_minutes) if checkpoint_every_minutes is not None else 5.0,
        "checkpoint_snapshot_every_periods": int(checkpoint_snapshot_every_periods) if checkpoint_snapshot_every_periods is not None else 0,
        "checkpoint_keep_last": int(checkpoint_keep_last) if checkpoint_keep_last is not None else 2,
        "history_plot_every_epochs": int(history_plot_every_epochs) if history_plot_every_epochs is not None else 1,
        "history_plot_final_only": bool(history_plot_final_only),
        "history_persist_every_periods": int(history_persist_every_periods) if history_persist_every_periods is not None else 4,
        "record_step_history": bool(record_step_history),
        "enable_torch_compile": bool(enable_torch_compile),
        "torch_compile_mode": str(torch_compile_mode),
        "selection_metric": "val_select/pesq_mean",
        "mlflow_uri": config["paths"]["tracking_root"],
        "mlflow_artifact_root": str(Path(config["paths"]["tracking_root"]) / "artifacts"),
        "experiment_name": str(config["tracking"]["experiment_name"]),
        "log_system_metrics": False,
        "log_torch_model": False,
        "sample_count": int(config["training"]["sample_count"]),
        "benchmark_seconds": int(config["training"]["benchmark_seconds"]),
        "benchmark_repeats": int(config["training"]["benchmark_repeats"]),
        "max_eval_files": int(max_eval_files) if max_eval_files is not None else None,
        "rank_max_eval_files": int(rank_max_eval_files) if rank_max_eval_files is not None else None,
        "final_max_eval_files": int(final_max_eval_files) if final_max_eval_files is not None else None,
        "cache_eval_audio": bool(cache_eval_audio),
        "rank_compute_composite": bool(rank_compute_composite),
        "select_compute_composite": bool(select_compute_composite),
        "postfilter_mode": "none",
        "postfilter_preset": "medium",
        "train_postfilter": False,
        "spectral_native_gate": False,
        "teacher_source_run_id": None,
        "teacher_variant": None,
        "audit_only": False,
        "teacher_cache_manifest": _resolve_teacher_cache_manifest(config),
        "guidance_classic": str(config["training"]["guidance_classic"]),
        "erb_bands": int(config["training"]["erb_bands"]),
        "context_frames": int(config["training"]["context_frames"]),
        "mcu_profile": str(config["mcu"]["profile"]),
        "sample_rate": int(config["training"]["sample_rate"]),
        "n_fft": int(config["training"]["n_fft"]),
        "hop_length": int(config["training"]["hop_length"]),
        "win_length": int(config["training"]["win_length"]),
        "eval_dnsmos": False,
    }


def _run_named_experiment(base_payload: dict[str, Any], *, group_root: Path, run_name: str, phase_tag: str, **overrides: Any) -> dict[str, Any]:
    run_dir = _create_run_dir(group_root, run_name)
    payload = {
        **base_payload,
        "checkpoint_out": (run_dir / "model.pt").as_posix(),
        "training_state_out": (run_dir / "latest_state.pt").as_posix(),
        "progress_json_out": (run_dir / "progress.json").as_posix(),
        "run_name": run_name,
        "phase": phase_tag,
        **overrides,
    }
    return run_experiment(ExperimentConfig(**payload))


def _select_teacher_winner(teacher_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not teacher_results:
        raise ValueError("No teacher training results available.")
    return max(teacher_results, key=lambda item: float(item.get("best_val_select_pesq") or float("-inf")))


def _phase_a_promotion_floor(config: dict[str, Any]) -> float | None:
    teacher_cfg = dict(config.get("teacher_training") or {})
    raw = teacher_cfg.get("phase_a_min_absolute_pesq")
    if raw is None:
        raw = teacher_cfg.get("phase_a_promotion_floor")
    if raw is None:
        return None
    return float(raw)


def _phase_a_min_gain_over_init(config: dict[str, Any]) -> float | None:
    teacher_cfg = dict(config.get("teacher_training") or {})
    raw = teacher_cfg.get("phase_a_min_gain_over_init")
    if raw is None:
        return None
    return float(raw)


def _phase_b_promotion_floor(config: dict[str, Any]) -> float | None:
    teacher_cfg = dict(config.get("teacher_training") or {})
    raw = teacher_cfg.get("phase_b_promotion_floor")
    if raw is None:
        return None
    return float(raw)


def _teacher_training_strategy(config: dict[str, Any]) -> str:
    teacher_cfg = dict(config.get("teacher_training") or {})
    raw = str(teacher_cfg.get("strategy") or "legacy_surrogate").strip().lower()
    return raw or "legacy_surrogate"


def _val_select_split_pesq(run_summary: dict[str, Any], split_name: str) -> float | None:
    split_metrics = dict(run_summary.get("val_select_metrics_by_split") or {})
    metrics = dict(split_metrics.get(split_name) or {})
    value = metrics.get("pesq_mean")
    if value is None:
        return None
    return float(value)


def _dns_voicebank_thresholds(config: dict[str, Any]) -> tuple[float, float]:
    teacher_cfg = dict(config.get("teacher_training") or {})
    dns_floor = float(teacher_cfg.get("dns_target_pesq_floor", 2.0))
    voicebank_floor = float(teacher_cfg.get("voicebank_target_pesq_floor", 2.7))
    return dns_floor, voicebank_floor


def _run_dns_first_fullsubnet_balanced_strategy(
    config: dict[str, Any],
    *,
    device: str,
    dataset_summary: dict[str, Any],
    per_domain: dict[str, dict[str, str]],
) -> dict[str, Any]:
    del dataset_summary

    teacher_cfg = dict(config.get("teacher_training") or {})
    family = str((teacher_cfg.get("families") or ["fullsubnet_plus"])[0])
    variant = str((teacher_cfg.get("variants") or ["small"])[0])
    seed = int((teacher_cfg.get("seeds") or [0])[0])
    if family != "fullsubnet_plus":
        raise ValueError(
            "teacher_training.strategy=dns_first_fullsubnet_balanced currently requires "
            "teacher_training.families=[fullsubnet_plus]."
        )

    dns_target_floor, voicebank_target_floor = _dns_voicebank_thresholds(config)
    spectral_native_gate = bool(teacher_cfg.get("spectral_native_gate", True))
    base_name = str(
        config.get("reference", {}).get("teacher_run_name")
        or f"{family}-{variant}-teacher-training-seed{seed}"
    )
    output_root = Path(config["paths"]["output_root"])
    group_root = output_root / "checkpoints" / "teacher"
    shared = {
        **_base_experiment_config(config, device=device),
        **_teacher_eval_overrides(config, per_domain),
        "model_family": family,
        "variant": variant,
        "seed": seed,
        "target_floor": None,
        "teacher_cache_manifest": None,
        "teacher_cache_schedule": None,
        "resume_training_state": None,
        "evaluate_init_checkpoint": bool(teacher_cfg.get("evaluate_init_checkpoint", False)),
        "spectral_native_gate": spectral_native_gate,
    }

    start_checkpoint = _teacher_family_start_checkpoint(config, family)
    baseline_checkpoint = _teacher_family_baseline_checkpoint(config, family)
    start_probe = _probe_teacher_start_checkpoint(
        config,
        per_domain,
        checkpoint=baseline_checkpoint,
        device=device,
        model_family=family,
        variant=variant,
    )
    print(
        "[teacher] strategy=dns_first_fullsubnet_balanced "
        f"family={family} variant={variant} seed={seed} spectral_native_gate={spectral_native_gate} "
        f"from={start_checkpoint or 'built-in initialization'}",
        flush=True,
    )

    phase_dns_cfg = _phase_settings(
        config,
        "phase_dns",
        {
            "loss_recipe": "T0",
            "lr": 3e-4,
            "epochs": 10,
            "early_stop_patience": 4,
            "min_epochs": 4,
            "rank_eval_every": 1,
            "select_eval_every": 1,
        },
    )
    print(f"[teacher] phase_dns config={phase_dns_cfg}", flush=True)
    phase_dns_name = _phase_run_name(base_name, "phase_dns")
    phase_dns_resume = _phase_resume_state(group_root, phase_dns_name)
    if phase_dns_resume:
        print(
            f"[teacher] phase_dns resuming from epoch={phase_dns_resume['resume_epoch']} "
            f"reason={phase_dns_resume['resume_reason']}",
            flush=True,
        )
    phase_dns = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_dns_name,
        phase_tag="teacher_training_phase_dns",
        train_csv=_resolve_staged_manifest_path(config, per_domain["dns5"]["train_fit"]),
        loss_recipe=str(phase_dns_cfg.get("loss_recipe", "T0")),
        lr=float(phase_dns_cfg["lr"]),
        epochs=int(phase_dns_cfg["epochs"]),
        early_stop_patience=int(phase_dns_cfg["early_stop_patience"]),
        min_epochs=int(phase_dns_cfg["min_epochs"]),
        rank_eval_every=int(phase_dns_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_dns_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_dns_resume.get("init_checkpoint") or start_checkpoint or "").strip() or None,
        resume_training_state=phase_dns_resume.get("resume_training_state"),
        selection_metric="dns5_val_select/pesq_mean",
        target_floor=float(phase_dns_cfg.get("dns_target_floor", dns_target_floor)),
        spectral_native_gate=bool(phase_dns_cfg.get("spectral_native_gate", spectral_native_gate)),
    )
    phase_dns_dns = float(phase_dns.get("best_score") or float("-inf"))
    phase_dns_voicebank = _val_select_split_pesq(phase_dns, "val_select")
    print(
        f"[teacher] phase_dns done dns5_val_select_pesq={phase_dns_dns:.4f} "
        f"voicebank_val_select_pesq={phase_dns_voicebank}",
        flush=True,
    )
    phase_dns_summary = _checkpoint_group_summary(
        output_root,
        "teacher_training",
        {
            "baseline": start_probe,
            "strategy": _teacher_training_strategy(config),
            "runs": [phase_dns],
            "phases": {"phase_dns": phase_dns},
            "winner": phase_dns,
            "dns_target_pesq_floor": dns_target_floor,
            "voicebank_target_pesq_floor": voicebank_target_floor,
            "dns_threshold_met": phase_dns_dns >= dns_target_floor,
            "voicebank_threshold_met": (
                phase_dns_voicebank is not None and phase_dns_voicebank >= voicebank_target_floor
            ),
            "phase_dns_completed": True,
            "phase_mix_started": False,
            "stop_before_student": True,
            "reason": "phase_dns_complete_pending_phase_mix",
            "threshold_met": False,
        },
    )
    print(
        f"[teacher] checkpointed phase_dns summary: {phase_dns_summary['summary_path']}",
        flush=True,
    )

    replay_seed = int(teacher_cfg.get("replay_seed", 42))
    phase_mix_cfg = _phase_settings(
        config,
        "phase_mix",
        {
            "loss_recipe": "T0",
            "lr": 1.5e-4,
            "epochs": 10,
            "early_stop_patience": 4,
            "min_epochs": 4,
            "dns_fraction": 1.0,
            "rank_eval_every": 1,
            "select_eval_every": 1,
            "dns_promotion_floor": dns_target_floor,
        },
    )
    phase_mix_schedule = _build_replay_schedule_manifests(
        per_domain["voicebank"]["train_fit"],
        per_domain["dns5"]["train_fit"],
        out_dir=output_root / "replay_manifests" / "teacher_phase_mix",
        seed=replay_seed,
        dns_fraction=float(phase_mix_cfg.get("dns_fraction", 1.0)),
        prefix="teacher_phase_mix",
    )
    print(f"[teacher] phase_mix config={phase_mix_cfg}", flush=True)
    phase_mix_name = _phase_run_name(base_name, "phase_mix")
    phase_mix_resume = _phase_resume_state(group_root, phase_mix_name)
    if phase_mix_resume:
        print(
            f"[teacher] phase_mix resuming from epoch={phase_mix_resume['resume_epoch']} "
            f"reason={phase_mix_resume['resume_reason']}",
            flush=True,
        )
    phase_mix = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_mix_name,
        phase_tag="teacher_training_phase_mix",
        train_csv=phase_mix_schedule["mixed_manifests"][0],
        train_csv_schedule=phase_mix_schedule["mixed_manifests"],
        loss_recipe=str(phase_mix_cfg.get("loss_recipe", "T0")),
        lr=float(phase_mix_cfg["lr"]),
        epochs=int(phase_mix_cfg["epochs"]),
        early_stop_patience=int(phase_mix_cfg["early_stop_patience"]),
        min_epochs=int(phase_mix_cfg["min_epochs"]),
        rank_eval_every=int(phase_mix_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_mix_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_mix_resume.get("init_checkpoint") or phase_dns["checkpoint_out"]),
        resume_training_state=phase_mix_resume.get("resume_training_state"),
        selection_metric="dns5_val_select/pesq_mean",
        target_floor=float(phase_mix_cfg.get("dns_target_floor", dns_target_floor)),
        spectral_native_gate=bool(phase_mix_cfg.get("spectral_native_gate", spectral_native_gate)),
    )
    phase_mix_dns = float(phase_mix.get("best_score") or float("-inf"))
    phase_mix_voicebank = _val_select_split_pesq(phase_mix, "val_select")
    print(
        f"[teacher] phase_mix done dns5_val_select_pesq={phase_mix_dns:.4f} "
        f"voicebank_val_select_pesq={phase_mix_voicebank}",
        flush=True,
    )
    phase_mix_summary = _checkpoint_group_summary(
        output_root,
        "teacher_training",
        {
            "baseline": start_probe,
            "strategy": _teacher_training_strategy(config),
            "runs": [phase_dns, phase_mix],
            "phases": {"phase_dns": phase_dns, "phase_mix": phase_mix},
            "winner": phase_mix,
            "dns_target_pesq_floor": dns_target_floor,
            "voicebank_target_pesq_floor": voicebank_target_floor,
            "dns_threshold_met": phase_mix_dns >= dns_target_floor,
            "voicebank_threshold_met": (
                phase_mix_voicebank is not None and phase_mix_voicebank >= voicebank_target_floor
            ),
            "phase_dns_completed": True,
            "phase_mix_completed": True,
            "phase_vb_started": False,
            "stop_before_student": True,
            "reason": "phase_mix_complete_pending_phase_vb",
            "threshold_met": False,
        },
    )
    print(
        f"[teacher] checkpointed phase_mix summary: {phase_mix_summary['summary_path']}",
        flush=True,
    )

    phase_mix_promotion_floor = float(phase_mix_cfg.get("dns_promotion_floor", dns_target_floor))
    if phase_mix_dns < phase_mix_promotion_floor:
        summary = {
            **phase_mix_summary,
            "phase_mix_stop_reasons": [
                f"phase_mix_dns5_val_select_pesq={phase_mix_dns:.4f} < phase_mix_dns_promotion_floor={phase_mix_promotion_floor:.4f}"
            ],
            "stopped_after_phase_mix": True,
            "stop_before_student": True,
            "reason": "phase_mix_below_dns_promotion_floor",
            "threshold_met": False,
        }
        summary_path = _write_group_summary(output_root, "teacher_training", summary)
        summary["summary_path"] = summary_path.as_posix()
        print(
            f"[teacher] stopping after phase_mix because dns5_val_select_pesq={phase_mix_dns:.4f} "
            f"is below phase_mix_dns_promotion_floor={phase_mix_promotion_floor:.4f}",
            flush=True,
        )
        return summary

    print("[teacher] building PESQ proxy for phase_vb", flush=True)
    pesq_proxy_checkpoint = _train_teacher_pesq_proxy(
        config,
        per_domain,
        phase_a_checkpoint=phase_dns["checkpoint_out"],
        phase_b_checkpoint=phase_mix["checkpoint_out"],
        device=device,
        force=False,
    )
    phase_vb_cfg = _phase_settings(
        config,
        "phase_vb",
        {
            "loss_recipe": "T0_PESQ",
            "lr": 7.5e-5,
            "epochs": 6,
            "early_stop_patience": 3,
            "min_epochs": 2,
            "rank_eval_every": 1,
            "select_eval_every": 1,
            "dns_guardrail_min": dns_target_floor,
            "voicebank_target_floor": voicebank_target_floor,
        },
    )
    print(f"[teacher] phase_vb config={phase_vb_cfg}", flush=True)
    phase_vb_name = _phase_run_name(base_name, "phase_vb")
    phase_vb_resume = _phase_resume_state(group_root, phase_vb_name)
    if phase_vb_resume:
        print(
            f"[teacher] phase_vb resuming from epoch={phase_vb_resume['resume_epoch']} "
            f"reason={phase_vb_resume['resume_reason']}",
            flush=True,
        )
    phase_vb = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_vb_name,
        phase_tag="teacher_training_phase_vb",
        train_csv=_resolve_staged_manifest_path(config, per_domain["voicebank"]["train_fit"]),
        loss_recipe=str(phase_vb_cfg.get("loss_recipe", "T0_PESQ")),
        pesq_proxy_checkpoint=pesq_proxy_checkpoint,
        lr=float(phase_vb_cfg["lr"]),
        epochs=int(phase_vb_cfg["epochs"]),
        early_stop_patience=int(phase_vb_cfg["early_stop_patience"]),
        min_epochs=int(phase_vb_cfg["min_epochs"]),
        rank_eval_every=int(phase_vb_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_vb_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_vb_resume.get("init_checkpoint") or phase_mix["checkpoint_out"]),
        resume_training_state=phase_vb_resume.get("resume_training_state"),
        selection_metric="val_select/pesq_mean",
        selection_guardrail_metric="dns5_val_select/pesq_mean",
        selection_guardrail_min=float(phase_vb_cfg.get("dns_guardrail_min", dns_target_floor)),
        target_floor=float(phase_vb_cfg.get("voicebank_target_floor", voicebank_target_floor)),
        spectral_native_gate=bool(phase_vb_cfg.get("spectral_native_gate", spectral_native_gate)),
    )
    phase_vb_dns = _val_select_split_pesq(phase_vb, "dns5_val_select")
    phase_vb_voicebank = _val_select_split_pesq(phase_vb, "val_select")
    dns_threshold_met = phase_vb_dns is not None and phase_vb_dns >= dns_target_floor
    voicebank_threshold_met = (
        phase_vb_voicebank is not None and phase_vb_voicebank >= voicebank_target_floor
    )
    print(
        f"[teacher] phase_vb done dns5_val_select_pesq={phase_vb_dns} "
        f"voicebank_val_select_pesq={phase_vb_voicebank}",
        flush=True,
    )

    summary = {
        "baseline": start_probe,
        "strategy": _teacher_training_strategy(config),
        "runs": [phase_dns, phase_mix, phase_vb],
        "phases": {"phase_dns": phase_dns, "phase_mix": phase_mix, "phase_vb": phase_vb},
        "winner": phase_vb,
        "dns_target_pesq_floor": dns_target_floor,
        "voicebank_target_pesq_floor": voicebank_target_floor,
        "dns_threshold_met": dns_threshold_met,
        "voicebank_threshold_met": voicebank_threshold_met,
        "threshold_met": bool(dns_threshold_met and voicebank_threshold_met),
        "pesq_proxy_checkpoint": pesq_proxy_checkpoint,
    }
    if not summary["threshold_met"]:
        summary.update(
            {
                "stop_before_student": True,
                "reason": "dual_domain_threshold_not_met",
            }
        )
    summary_path = _write_group_summary(output_root, "teacher_training", summary)
    summary["summary_path"] = summary_path.as_posix()
    print(
        f"[teacher] final summary dns_threshold_met={dns_threshold_met} "
        f"voicebank_threshold_met={voicebank_threshold_met} summary={summary_path}",
        flush=True,
    )
    return summary


def _run_dns_first_cmgan_pesq_guarded_strategy(
    config: dict[str, Any],
    *,
    device: str,
    dataset_summary: dict[str, Any],
    per_domain: dict[str, dict[str, str]],
) -> dict[str, Any]:
    del dataset_summary

    teacher_cfg = dict(config.get("teacher_training") or {})
    family = str((teacher_cfg.get("families") or ["cmgan_small"])[0])
    variant = str((teacher_cfg.get("variants") or ["base"])[0])
    seed = int((teacher_cfg.get("seeds") or [0])[0])
    if family != "cmgan_small":
        raise ValueError(
            "teacher_training.strategy=dns_first_cmgan_pesq_guarded currently requires "
            "teacher_training.families=[cmgan_small]."
        )

    dns_target_floor, voicebank_target_floor = _dns_voicebank_thresholds(config)
    spectral_native_gate = bool(teacher_cfg.get("spectral_native_gate", True))
    base_name = str(
        config.get("reference", {}).get("teacher_run_name")
        or f"{family}-{variant}-teacher-training-seed{seed}"
    )
    output_root = Path(config["paths"]["output_root"])
    group_root = output_root / "checkpoints" / "teacher"
    shared = {
        **_base_experiment_config(config, device=device),
        **_teacher_eval_overrides(config, per_domain),
        "model_family": family,
        "variant": variant,
        "seed": seed,
        "target_floor": None,
        "teacher_cache_manifest": None,
        "teacher_cache_schedule": None,
        "resume_training_state": None,
        "evaluate_init_checkpoint": bool(teacher_cfg.get("evaluate_init_checkpoint", False)),
        "spectral_native_gate": spectral_native_gate,
    }

    start_checkpoint = _teacher_family_start_checkpoint(config, family)
    baseline_checkpoint = _teacher_family_baseline_checkpoint(config, family)
    start_probe = _probe_teacher_start_checkpoint(
        config,
        per_domain,
        checkpoint=baseline_checkpoint,
        device=device,
        model_family=family,
        variant=variant,
    )
    print(
        "[teacher] strategy=dns_first_cmgan_pesq_guarded "
        f"family={family} variant={variant} seed={seed} spectral_native_gate={spectral_native_gate} "
        f"from={start_checkpoint or 'built-in initialization'}",
        flush=True,
    )

    phase_dns_boot_cfg = _phase_settings(
        config,
        "phase_dns_boot",
        {
            "loss_recipe": "T0",
            "lr": 3e-4,
            "epochs": 3,
            "early_stop_patience": 2,
            "min_epochs": 2,
            "rank_eval_every": 1,
            "select_eval_every": 1,
            "dns_target_floor": 1.8,
        },
    )
    print(f"[teacher] phase_dns_boot config={phase_dns_boot_cfg}", flush=True)
    phase_dns_boot_name = _phase_run_name(base_name, "phase_dns_boot")
    phase_dns_boot_resume = _phase_resume_state(group_root, phase_dns_boot_name)
    if phase_dns_boot_resume:
        print(
            f"[teacher] phase_dns_boot resuming from epoch={phase_dns_boot_resume['resume_epoch']} "
            f"reason={phase_dns_boot_resume['resume_reason']}",
            flush=True,
        )
    phase_dns_boot = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_dns_boot_name,
        phase_tag="teacher_training_phase_dns_boot",
        train_csv=_resolve_staged_manifest_path(config, per_domain["dns5"]["train_fit"]),
        loss_recipe=str(phase_dns_boot_cfg.get("loss_recipe", "T0")),
        lr=float(phase_dns_boot_cfg["lr"]),
        epochs=int(phase_dns_boot_cfg["epochs"]),
        early_stop_patience=int(phase_dns_boot_cfg["early_stop_patience"]),
        min_epochs=int(phase_dns_boot_cfg["min_epochs"]),
        rank_eval_every=int(phase_dns_boot_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_dns_boot_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_dns_boot_resume.get("init_checkpoint") or start_checkpoint or "").strip() or None,
        resume_training_state=phase_dns_boot_resume.get("resume_training_state"),
        selection_metric="dns5_val_select/pesq_mean",
        target_floor=float(phase_dns_boot_cfg.get("dns_target_floor", 1.8)),
        spectral_native_gate=bool(phase_dns_boot_cfg.get("spectral_native_gate", spectral_native_gate)),
    )
    phase_dns_boot_dns = float(phase_dns_boot.get("best_score") or float("-inf"))
    phase_dns_boot_voicebank = _val_select_split_pesq(phase_dns_boot, "val_select")
    print(
        f"[teacher] phase_dns_boot done dns5_val_select_pesq={phase_dns_boot_dns:.4f} "
        f"voicebank_val_select_pesq={phase_dns_boot_voicebank}",
        flush=True,
    )
    phase_dns_boot_summary = _checkpoint_group_summary(
        output_root,
        "teacher_training",
        {
            "baseline": start_probe,
            "strategy": _teacher_training_strategy(config),
            "runs": [phase_dns_boot],
            "phases": {"phase_dns_boot": phase_dns_boot},
            "winner": phase_dns_boot,
            "dns_target_pesq_floor": dns_target_floor,
            "voicebank_target_pesq_floor": voicebank_target_floor,
            "dns_threshold_met": phase_dns_boot_dns >= dns_target_floor,
            "voicebank_threshold_met": (
                phase_dns_boot_voicebank is not None and phase_dns_boot_voicebank >= voicebank_target_floor
            ),
            "phase_dns_boot_completed": True,
            "phase_dns_pesq_started": False,
            "stop_before_student": True,
            "reason": "phase_dns_boot_complete_pending_phase_dns_pesq",
            "threshold_met": False,
        },
    )
    print(
        f"[teacher] checkpointed phase_dns_boot summary: {phase_dns_boot_summary['summary_path']}",
        flush=True,
    )

    print("[teacher] building PESQ proxy for phase_dns_pesq", flush=True)
    pesq_proxy_checkpoint = _train_teacher_pesq_proxy(
        config,
        per_domain,
        phase_a_checkpoint=phase_dns_boot["checkpoint_out"],
        phase_b_checkpoint=phase_dns_boot["checkpoint_out"],
        device=device,
        force=False,
    )
    phase_dns_pesq_cfg = _phase_settings(
        config,
        "phase_dns_pesq",
        {
            "loss_recipe": "T0_PESQ",
            "lr": 2.0e-4,
            "epochs": 8,
            "early_stop_patience": 3,
            "min_epochs": 3,
            "rank_eval_every": 1,
            "select_eval_every": 1,
            "dns_promotion_floor": 1.9,
        },
    )
    print(f"[teacher] phase_dns_pesq config={phase_dns_pesq_cfg}", flush=True)
    phase_dns_pesq_name = _phase_run_name(base_name, "phase_dns_pesq")
    phase_dns_pesq_resume = _phase_resume_state(group_root, phase_dns_pesq_name)
    if phase_dns_pesq_resume:
        print(
            f"[teacher] phase_dns_pesq resuming from epoch={phase_dns_pesq_resume['resume_epoch']} "
            f"reason={phase_dns_pesq_resume['resume_reason']}",
            flush=True,
        )
    phase_dns_pesq = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_dns_pesq_name,
        phase_tag="teacher_training_phase_dns_pesq",
        train_csv=_resolve_staged_manifest_path(config, per_domain["dns5"]["train_fit"]),
        loss_recipe=str(phase_dns_pesq_cfg.get("loss_recipe", "T0_PESQ")),
        pesq_proxy_checkpoint=pesq_proxy_checkpoint,
        lr=float(phase_dns_pesq_cfg["lr"]),
        epochs=int(phase_dns_pesq_cfg["epochs"]),
        early_stop_patience=int(phase_dns_pesq_cfg["early_stop_patience"]),
        min_epochs=int(phase_dns_pesq_cfg["min_epochs"]),
        rank_eval_every=int(phase_dns_pesq_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_dns_pesq_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_dns_pesq_resume.get("init_checkpoint") or phase_dns_boot["checkpoint_out"]),
        resume_training_state=phase_dns_pesq_resume.get("resume_training_state"),
        selection_metric="dns5_val_select/pesq_mean",
        target_floor=float(phase_dns_pesq_cfg.get("dns_target_floor", dns_target_floor)),
        spectral_native_gate=bool(phase_dns_pesq_cfg.get("spectral_native_gate", spectral_native_gate)),
    )
    phase_dns_pesq_dns = float(phase_dns_pesq.get("best_score") or float("-inf"))
    phase_dns_pesq_voicebank = _val_select_split_pesq(phase_dns_pesq, "val_select")
    print(
        f"[teacher] phase_dns_pesq done dns5_val_select_pesq={phase_dns_pesq_dns:.4f} "
        f"voicebank_val_select_pesq={phase_dns_pesq_voicebank}",
        flush=True,
    )
    phase_dns_pesq_summary = _checkpoint_group_summary(
        output_root,
        "teacher_training",
        {
            "baseline": start_probe,
            "strategy": _teacher_training_strategy(config),
            "runs": [phase_dns_boot, phase_dns_pesq],
            "phases": {"phase_dns_boot": phase_dns_boot, "phase_dns_pesq": phase_dns_pesq},
            "winner": phase_dns_pesq,
            "dns_target_pesq_floor": dns_target_floor,
            "voicebank_target_pesq_floor": voicebank_target_floor,
            "dns_threshold_met": phase_dns_pesq_dns >= dns_target_floor,
            "voicebank_threshold_met": (
                phase_dns_pesq_voicebank is not None and phase_dns_pesq_voicebank >= voicebank_target_floor
            ),
            "phase_dns_boot_completed": True,
            "phase_dns_pesq_completed": True,
            "phase_mix_guarded_started": False,
            "stop_before_student": True,
            "reason": "phase_dns_pesq_complete_pending_phase_mix_guarded",
            "threshold_met": False,
            "pesq_proxy_checkpoint": pesq_proxy_checkpoint,
        },
    )
    print(
        f"[teacher] checkpointed phase_dns_pesq summary: {phase_dns_pesq_summary['summary_path']}",
        flush=True,
    )

    phase_dns_pesq_promotion_floor = float(phase_dns_pesq_cfg.get("dns_promotion_floor", 1.9))
    if phase_dns_pesq_dns < phase_dns_pesq_promotion_floor:
        summary = {
            **phase_dns_pesq_summary,
            "phase_dns_pesq_stop_reasons": [
                f"phase_dns_pesq_dns5_val_select_pesq={phase_dns_pesq_dns:.4f} < phase_dns_pesq_dns_promotion_floor={phase_dns_pesq_promotion_floor:.4f}"
            ],
            "stopped_after_phase_dns_pesq": True,
            "stop_before_student": True,
            "reason": "phase_dns_pesq_below_dns_promotion_floor",
            "threshold_met": False,
        }
        summary_path = _write_group_summary(output_root, "teacher_training", summary)
        summary["summary_path"] = summary_path.as_posix()
        print(
            f"[teacher] stopping after phase_dns_pesq because dns5_val_select_pesq={phase_dns_pesq_dns:.4f} "
            f"is below phase_dns_pesq_dns_promotion_floor={phase_dns_pesq_promotion_floor:.4f}",
            flush=True,
        )
        return summary

    replay_seed = int(teacher_cfg.get("replay_seed", 42))
    phase_mix_guarded_cfg = _phase_settings(
        config,
        "phase_mix_guarded",
        {
            "loss_recipe": "T0_PESQ",
            "lr": 1.25e-4,
            "epochs": 8,
            "early_stop_patience": 3,
            "min_epochs": 3,
            "dns_fraction": 1.0,
            "rank_eval_every": 1,
            "select_eval_every": 1,
            "dns_promotion_floor": dns_target_floor,
            "voicebank_guardrail_min": 1.95,
        },
    )
    phase_mix_guarded_schedule = _build_replay_schedule_manifests(
        per_domain["voicebank"]["train_fit"],
        per_domain["dns5"]["train_fit"],
        out_dir=output_root / "replay_manifests" / "teacher_phase_mix_guarded",
        seed=replay_seed,
        dns_fraction=float(phase_mix_guarded_cfg.get("dns_fraction", 1.0)),
        prefix="teacher_phase_mix_guarded",
    )
    print(f"[teacher] phase_mix_guarded config={phase_mix_guarded_cfg}", flush=True)
    phase_mix_guarded_name = _phase_run_name(base_name, "phase_mix_guarded")
    phase_mix_guarded_resume = _phase_resume_state(group_root, phase_mix_guarded_name)
    if phase_mix_guarded_resume:
        print(
            f"[teacher] phase_mix_guarded resuming from epoch={phase_mix_guarded_resume['resume_epoch']} "
            f"reason={phase_mix_guarded_resume['resume_reason']}",
            flush=True,
        )
    phase_mix_guarded = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_mix_guarded_name,
        phase_tag="teacher_training_phase_mix_guarded",
        train_csv=phase_mix_guarded_schedule["mixed_manifests"][0],
        train_csv_schedule=phase_mix_guarded_schedule["mixed_manifests"],
        loss_recipe=str(phase_mix_guarded_cfg.get("loss_recipe", "T0_PESQ")),
        pesq_proxy_checkpoint=pesq_proxy_checkpoint,
        lr=float(phase_mix_guarded_cfg["lr"]),
        epochs=int(phase_mix_guarded_cfg["epochs"]),
        early_stop_patience=int(phase_mix_guarded_cfg["early_stop_patience"]),
        min_epochs=int(phase_mix_guarded_cfg["min_epochs"]),
        rank_eval_every=int(phase_mix_guarded_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_mix_guarded_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_mix_guarded_resume.get("init_checkpoint") or phase_dns_pesq["checkpoint_out"]),
        resume_training_state=phase_mix_guarded_resume.get("resume_training_state"),
        selection_metric="dns5_val_select/pesq_mean",
        selection_guardrail_metric="val_select/pesq_mean",
        selection_guardrail_min=float(phase_mix_guarded_cfg.get("voicebank_guardrail_min", 1.95)),
        target_floor=float(phase_mix_guarded_cfg.get("dns_target_floor", dns_target_floor)),
        spectral_native_gate=bool(phase_mix_guarded_cfg.get("spectral_native_gate", spectral_native_gate)),
    )
    phase_mix_guarded_dns = float(phase_mix_guarded.get("best_score") or float("-inf"))
    phase_mix_guarded_voicebank = _val_select_split_pesq(phase_mix_guarded, "val_select")
    print(
        f"[teacher] phase_mix_guarded done dns5_val_select_pesq={phase_mix_guarded_dns:.4f} "
        f"voicebank_val_select_pesq={phase_mix_guarded_voicebank}",
        flush=True,
    )
    phase_mix_guarded_summary = _checkpoint_group_summary(
        output_root,
        "teacher_training",
        {
            "baseline": start_probe,
            "strategy": _teacher_training_strategy(config),
            "runs": [phase_dns_boot, phase_dns_pesq, phase_mix_guarded],
            "phases": {
                "phase_dns_boot": phase_dns_boot,
                "phase_dns_pesq": phase_dns_pesq,
                "phase_mix_guarded": phase_mix_guarded,
            },
            "winner": phase_mix_guarded,
            "dns_target_pesq_floor": dns_target_floor,
            "voicebank_target_pesq_floor": voicebank_target_floor,
            "dns_threshold_met": phase_mix_guarded_dns >= dns_target_floor,
            "voicebank_threshold_met": (
                phase_mix_guarded_voicebank is not None and phase_mix_guarded_voicebank >= voicebank_target_floor
            ),
            "phase_dns_boot_completed": True,
            "phase_dns_pesq_completed": True,
            "phase_mix_guarded_completed": True,
            "phase_vb_started": False,
            "stop_before_student": True,
            "reason": "phase_mix_guarded_complete_pending_phase_vb",
            "threshold_met": False,
            "pesq_proxy_checkpoint": pesq_proxy_checkpoint,
        },
    )
    print(
        f"[teacher] checkpointed phase_mix_guarded summary: {phase_mix_guarded_summary['summary_path']}",
        flush=True,
    )

    phase_mix_guarded_promotion_floor = float(phase_mix_guarded_cfg.get("dns_promotion_floor", dns_target_floor))
    if phase_mix_guarded_dns < phase_mix_guarded_promotion_floor:
        summary = {
            **phase_mix_guarded_summary,
            "phase_mix_guarded_stop_reasons": [
                f"phase_mix_guarded_dns5_val_select_pesq={phase_mix_guarded_dns:.4f} < phase_mix_guarded_dns_promotion_floor={phase_mix_guarded_promotion_floor:.4f}"
            ],
            "stopped_after_phase_mix_guarded": True,
            "stop_before_student": True,
            "reason": "phase_mix_guarded_below_dns_promotion_floor",
            "threshold_met": False,
        }
        summary_path = _write_group_summary(output_root, "teacher_training", summary)
        summary["summary_path"] = summary_path.as_posix()
        print(
            f"[teacher] stopping after phase_mix_guarded because dns5_val_select_pesq={phase_mix_guarded_dns:.4f} "
            f"is below phase_mix_guarded_dns_promotion_floor={phase_mix_guarded_promotion_floor:.4f}",
            flush=True,
        )
        return summary

    phase_vb_cfg = _phase_settings(
        config,
        "phase_vb",
        {
            "loss_recipe": "T0_PESQ",
            "lr": 7.5e-5,
            "epochs": 6,
            "early_stop_patience": 3,
            "min_epochs": 2,
            "rank_eval_every": 1,
            "select_eval_every": 1,
            "dns_guardrail_min": dns_target_floor,
            "voicebank_target_floor": voicebank_target_floor,
        },
    )
    print(f"[teacher] phase_vb config={phase_vb_cfg}", flush=True)
    phase_vb_name = _phase_run_name(base_name, "phase_vb")
    phase_vb_resume = _phase_resume_state(group_root, phase_vb_name)
    if phase_vb_resume:
        print(
            f"[teacher] phase_vb resuming from epoch={phase_vb_resume['resume_epoch']} "
            f"reason={phase_vb_resume['resume_reason']}",
            flush=True,
        )
    phase_vb = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_vb_name,
        phase_tag="teacher_training_phase_vb",
        train_csv=_resolve_staged_manifest_path(config, per_domain["voicebank"]["train_fit"]),
        loss_recipe=str(phase_vb_cfg.get("loss_recipe", "T0_PESQ")),
        pesq_proxy_checkpoint=pesq_proxy_checkpoint,
        lr=float(phase_vb_cfg["lr"]),
        epochs=int(phase_vb_cfg["epochs"]),
        early_stop_patience=int(phase_vb_cfg["early_stop_patience"]),
        min_epochs=int(phase_vb_cfg["min_epochs"]),
        rank_eval_every=int(phase_vb_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_vb_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_vb_resume.get("init_checkpoint") or phase_mix_guarded["checkpoint_out"]),
        resume_training_state=phase_vb_resume.get("resume_training_state"),
        selection_metric="val_select/pesq_mean",
        selection_guardrail_metric="dns5_val_select/pesq_mean",
        selection_guardrail_min=float(phase_vb_cfg.get("dns_guardrail_min", dns_target_floor)),
        target_floor=float(phase_vb_cfg.get("voicebank_target_floor", voicebank_target_floor)),
        spectral_native_gate=bool(phase_vb_cfg.get("spectral_native_gate", spectral_native_gate)),
    )
    phase_vb_dns = _val_select_split_pesq(phase_vb, "dns5_val_select")
    phase_vb_voicebank = _val_select_split_pesq(phase_vb, "val_select")
    dns_threshold_met = phase_vb_dns is not None and phase_vb_dns >= dns_target_floor
    voicebank_threshold_met = (
        phase_vb_voicebank is not None and phase_vb_voicebank >= voicebank_target_floor
    )
    print(
        f"[teacher] phase_vb done dns5_val_select_pesq={phase_vb_dns} "
        f"voicebank_val_select_pesq={phase_vb_voicebank}",
        flush=True,
    )

    summary = {
        "baseline": start_probe,
        "strategy": _teacher_training_strategy(config),
        "runs": [phase_dns_boot, phase_dns_pesq, phase_mix_guarded, phase_vb],
        "phases": {
            "phase_dns_boot": phase_dns_boot,
            "phase_dns_pesq": phase_dns_pesq,
            "phase_mix_guarded": phase_mix_guarded,
            "phase_vb": phase_vb,
        },
        "winner": phase_vb,
        "dns_target_pesq_floor": dns_target_floor,
        "voicebank_target_pesq_floor": voicebank_target_floor,
        "dns_threshold_met": dns_threshold_met,
        "voicebank_threshold_met": voicebank_threshold_met,
        "threshold_met": bool(dns_threshold_met and voicebank_threshold_met),
        "pesq_proxy_checkpoint": pesq_proxy_checkpoint,
    }
    if not summary["threshold_met"]:
        summary.update(
            {
                "stop_before_student": True,
                "reason": "dual_domain_threshold_not_met",
            }
        )
    summary_path = _write_group_summary(output_root, "teacher_training", summary)
    summary["summary_path"] = summary_path.as_posix()
    print(
        f"[teacher] final summary dns_threshold_met={dns_threshold_met} "
        f"voicebank_threshold_met={voicebank_threshold_met} summary={summary_path}",
        flush=True,
    )
    return summary


def _run_teacher_preserving_refiner_strategy(
    config: dict[str, Any],
    *,
    device: str,
    dataset_summary: dict[str, Any],
    per_domain: dict[str, dict[str, str]],
) -> dict[str, Any]:
    combined = dataset_summary["combined_manifests"]
    teacher_cfg = dict(config.get("teacher_training") or {})
    family = str((teacher_cfg.get("families") or ["metricgan_plus_refiner"])[0])
    variant = str((teacher_cfg.get("variants") or ["small"])[0])
    seed = int((teacher_cfg.get("seeds") or [0])[0])
    if family != "metricgan_plus_refiner":
        raise ValueError(
            "teacher_training.strategy=teacher_preserving_voicebank_refiner currently requires "
            "teacher_training.families=[metricgan_plus_refiner]."
        )

    base_name = str(config.get("reference", {}).get("teacher_run_name") or f"{family}-{variant}-teacher-training-seed{seed}")
    group_root = Path(config["paths"]["output_root"]) / "checkpoints" / "teacher"
    shared = {
        **_base_experiment_config(config, device=device),
        **_teacher_eval_overrides(config, per_domain),
        "model_family": family,
        "variant": variant,
        "seed": seed,
        "target_floor": float(teacher_cfg.get("target_pesq_floor", 3.1)),
        "teacher_cache_manifest": None,
        "teacher_cache_schedule": None,
        "resume_training_state": None,
        "evaluate_init_checkpoint": bool(teacher_cfg.get("evaluate_init_checkpoint", False)),
    }

    start_checkpoint = _teacher_family_start_checkpoint(config, family)
    baseline_checkpoint = _teacher_family_baseline_checkpoint(config, family)
    start_probe = _probe_teacher_start_checkpoint(
        config,
        per_domain,
        checkpoint=baseline_checkpoint,
        device=device,
        model_family=family,
        variant=variant,
    )
    start_probe_vbd = start_probe.get("voicebank_val_select_pesq")
    dns5_baseline = start_probe.get("dns5_val_select_pesq")
    guardrail_floor = None
    if dns5_baseline is not None:
        guardrail_floor = float(dns5_baseline) - 0.05

    print(
        "[teacher] strategy=teacher_preserving_voicebank_refiner "
        f"family={family} variant={variant} seed={seed} from={start_checkpoint or 'built-in initialization'}",
        flush=True,
    )

    cache_summary = command_build_teacher_cache(config, device=device, force=False)
    voicebank_train_manifest = _resolve_staged_manifest_path(config, per_domain["voicebank"]["train_fit"])
    phase_a_cache_manifest = _build_teacher_cache_schedule(
        cache_summary["manifest"],
        [voicebank_train_manifest],
        out_dir=Path(config["paths"]["output_root"]) / "teacher_cache" / "teacher_phase_a_voicebank",
        prefix="teacher_phase_a_voicebank",
    )[0]

    phase_a_cfg = _phase_settings(
        config,
        "phase_a",
        {
            "loss_recipe": "D1",
            "lr": 1e-5,
            "epochs": 4,
            "early_stop_patience": 2,
            "min_epochs": 2,
            "rank_eval_every": 1,
            "select_eval_every": 1,
        },
    )
    print(f"[teacher] phase_a config={phase_a_cfg}", flush=True)
    phase_a_name = _phase_run_name(base_name, "phase_a")
    phase_a_resume = _phase_resume_state(group_root, phase_a_name)
    if phase_a_resume:
        print(
            f"[teacher] phase_a resuming from epoch={phase_a_resume['resume_epoch']} "
            f"reason={phase_a_resume['resume_reason']}",
            flush=True,
        )
    phase_a = _run_named_experiment(
        {**shared, **_teacher_eval_overrides(config, per_domain, guardrail_floor=guardrail_floor)},
        group_root=group_root,
        run_name=phase_a_name,
        phase_tag="teacher_training_phase_a",
        train_csv=voicebank_train_manifest,
        teacher_cache_manifest=phase_a_cache_manifest,
        loss_recipe=str(phase_a_cfg.get("loss_recipe", "D1")),
        lr=float(phase_a_cfg["lr"]),
        epochs=int(phase_a_cfg["epochs"]),
        early_stop_patience=int(phase_a_cfg["early_stop_patience"]),
        min_epochs=int(phase_a_cfg["min_epochs"]),
        rank_eval_every=int(phase_a_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_a_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_a_resume.get("init_checkpoint") or start_checkpoint or "").strip() or None,
        resume_training_state=phase_a_resume.get("resume_training_state"),
    )
    print(
        f"[teacher] phase_a done best_val_select_pesq={phase_a.get('best_val_select_pesq')} "
        f"dns5_guardrail_floor={guardrail_floor}",
        flush=True,
    )

    output_root = Path(config["paths"]["output_root"])
    phase_a_floor = _phase_a_promotion_floor(config)
    phase_a_min_gain = _phase_a_min_gain_over_init(config)
    phase_a_best = float(phase_a.get("best_val_select_pesq") or float("-inf"))
    phase_a_gain = None
    if start_probe_vbd is not None:
        phase_a_gain = float(phase_a_best - float(start_probe_vbd))

    teacher_floor = float(teacher_cfg.get("target_pesq_floor", 3.1))
    phase_a_summary = _checkpoint_group_summary(
        output_root,
        "teacher_training",
        {
            "baseline": start_probe,
            "strategy": _teacher_training_strategy(config),
            "teacher_cache": cache_summary,
            "runs": [phase_a],
            "phases": {"phase_a": phase_a},
            "winner": phase_a,
            "target_pesq_floor": teacher_floor,
            "phase_a_min_absolute_pesq": phase_a_floor,
            "phase_a_min_gain_over_init": phase_a_min_gain,
            "phase_a_gain_over_init": phase_a_gain,
            "phase_b_promotion_floor": _phase_b_promotion_floor(config),
            "phase_a_completed": True,
            "phase_b_started": False,
            "stop_before_student": True,
            "reason": "phase_a_complete_pending_phase_b",
            "threshold_met": phase_a_best >= teacher_floor,
        },
    )
    print(
        f"[teacher] checkpointed phase_a summary: {phase_a_summary['summary_path']}",
        flush=True,
    )

    phase_a_stop_reasons: list[str] = []
    if phase_a_floor is not None and phase_a_best < phase_a_floor:
        phase_a_stop_reasons.append(
            f"phase_a_best_val_select_pesq={phase_a_best:.4f} < phase_a_min_absolute_pesq={phase_a_floor:.4f}"
        )
    if phase_a_min_gain is not None:
        if phase_a_gain is None or phase_a_gain < phase_a_min_gain:
            observed_gain = "n/a" if phase_a_gain is None else f"{phase_a_gain:.4f}"
            phase_a_stop_reasons.append(
                f"phase_a_gain_over_init={observed_gain} < phase_a_min_gain_over_init={phase_a_min_gain:.4f}"
            )
    if phase_a_stop_reasons:
        summary = {
            **phase_a_summary,
            "phase_a_stop_reasons": phase_a_stop_reasons,
            "stopped_after_phase_a": True,
            "stop_before_student": True,
            "reason": "phase_a_below_promotion_floor",
            "threshold_met": phase_a_best >= teacher_floor,
        }
        summary_path = _write_group_summary(output_root, "teacher_training", summary)
        summary["summary_path"] = summary_path.as_posix()
        print(
            f"[teacher] stopping after phase_a because {'; '.join(phase_a_stop_reasons)}",
            flush=True,
        )
        return summary

    replay_seed = int(teacher_cfg.get("replay_seed", 42))
    phase_b_cfg = _phase_settings(
        config,
        "phase_b",
        {
            "loss_recipe": "D1",
            "lr": 5e-6,
            "epochs": 6,
            "early_stop_patience": 3,
            "min_epochs": 2,
            "dns_fraction": 0.1,
            "rank_eval_every": 1,
            "select_eval_every": 1,
        },
    )
    phase_b_schedule = _build_replay_schedule_manifests(
        per_domain["voicebank"]["train_fit"],
        per_domain["dns5"]["train_fit"],
        out_dir=Path(config["paths"]["output_root"]) / "replay_manifests" / "teacher_phase_b",
        seed=replay_seed,
        dns_fraction=float(phase_b_cfg.get("dns_fraction", 0.1)),
        prefix="teacher_phase_b",
    )
    phase_b_cache_schedule = _build_teacher_cache_schedule(
        cache_summary["manifest"],
        phase_b_schedule["mixed_manifests"],
        out_dir=Path(config["paths"]["output_root"]) / "teacher_cache" / "teacher_phase_b",
        prefix="teacher_phase_b",
    )
    print(f"[teacher] phase_b config={phase_b_cfg}", flush=True)
    phase_b_name = _phase_run_name(base_name, "phase_b")
    phase_b_resume = _phase_resume_state(group_root, phase_b_name)
    if phase_b_resume:
        print(
            f"[teacher] phase_b resuming from epoch={phase_b_resume['resume_epoch']} "
            f"reason={phase_b_resume['resume_reason']}",
            flush=True,
        )
    phase_b = _run_named_experiment(
        {**shared, **_teacher_eval_overrides(config, per_domain, guardrail_floor=guardrail_floor)},
        group_root=group_root,
        run_name=phase_b_name,
        phase_tag="teacher_training_phase_b",
        train_csv=phase_b_schedule["mixed_manifests"][0],
        train_csv_schedule=phase_b_schedule["mixed_manifests"],
        teacher_cache_manifest=phase_b_cache_schedule[0],
        teacher_cache_schedule=phase_b_cache_schedule,
        loss_recipe=str(phase_b_cfg.get("loss_recipe", "D1")),
        lr=float(phase_b_cfg["lr"]),
        epochs=int(phase_b_cfg["epochs"]),
        early_stop_patience=int(phase_b_cfg["early_stop_patience"]),
        min_epochs=int(phase_b_cfg["min_epochs"]),
        rank_eval_every=int(phase_b_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_b_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_b_resume.get("init_checkpoint") or phase_a["checkpoint_out"]),
        resume_training_state=phase_b_resume.get("resume_training_state"),
    )
    print(f"[teacher] phase_b done best_val_select_pesq={phase_b.get('best_val_select_pesq')}", flush=True)

    phase_b_floor = _phase_b_promotion_floor(config)
    phase_b_best = float(phase_b.get("best_val_select_pesq") or float("-inf"))
    teacher_floor = float(teacher_cfg.get("target_pesq_floor", 3.1))
    winner = _select_teacher_winner([phase_a, phase_b])
    summary = {
        "baseline": start_probe,
        "strategy": _teacher_training_strategy(config),
        "teacher_cache": cache_summary,
        "runs": [phase_a, phase_b],
        "phases": {"phase_a": phase_a, "phase_b": phase_b},
        "winner": winner,
        "target_pesq_floor": teacher_floor,
        "phase_a_min_absolute_pesq": phase_a_floor,
        "phase_a_min_gain_over_init": phase_a_min_gain,
        "phase_a_gain_over_init": phase_a_gain,
        "phase_b_promotion_floor": phase_b_floor,
        "threshold_met": float(winner.get("best_val_select_pesq") or float("-inf")) >= teacher_floor,
    }
    if phase_b_floor is not None and phase_b_best < phase_b_floor:
        summary.update(
            {
                "stopped_after_phase_b": True,
                "stop_before_student": True,
                "reason": "phase_b_below_promotion_floor",
            }
        )
        print(
            f"[teacher] stopping after phase_b because best_val_select_pesq={phase_b_best:.4f} "
            f"is below phase_b_promotion_floor={phase_b_floor:.4f}",
            flush=True,
        )

    output_root = Path(config["paths"]["output_root"])
    summary_path = _write_group_summary(output_root, "teacher_training", summary)
    summary["summary_path"] = summary_path.as_posix()
    print(
        f"[teacher] winner checkpoint={winner.get('checkpoint_out')} "
        f"best_val_select_pesq={winner.get('best_val_select_pesq')} "
        f"threshold_met={summary['threshold_met']}",
        flush=True,
    )
    return summary


def _teacher_init_probe_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]) / "teacher_baseline" / "summary.json"


def _teacher_family_start_checkpoint(config: dict[str, Any], family: str) -> str | None:
    teacher_cfg = dict(config.get("teacher_training") or {})
    explicit = str(teacher_cfg.get("resume_checkpoint") or "").strip()
    if explicit:
        return explicit
    if family == "metricgan_plus_native8k":
        return _resolve_teacher_start_checkpoint(config)
    return None


def _teacher_family_baseline_checkpoint(config: dict[str, Any], family: str) -> str | None:
    teacher_cfg = dict(config.get("teacher_training") or {})
    explicit = str(teacher_cfg.get("baseline_checkpoint") or "").strip()
    if explicit:
        return explicit
    if family == "metricgan_plus_native8k":
        return _resolve_teacher_baseline_checkpoint(config)
    return None


def _build_runtime_teacher_model(
    config: dict[str, Any],
    *,
    device: str,
    model_family: str,
    variant: str,
    checkpoint: str | None,
) -> torch.nn.Module:
    if checkpoint:
        model, _ = load_model_from_checkpoint(checkpoint, device=device)
        return model
    model = build_enhancer(
        model_family,
        variant,
        spectral_native_gate=False,
        postfilter_mode="none",
        postfilter_preset="medium",
        train_postfilter=False,
        erb_bands=int(config["training"]["erb_bands"]),
        context_frames=int(config["training"]["context_frames"]),
        guidance_classic=str(config["training"]["guidance_classic"]),
        qat=False,
        sample_rate=int(config["training"]["sample_rate"]),
        n_fft=int(config["training"]["n_fft"]),
        hop_length=int(config["training"]["hop_length"]),
        win_length=int(config["training"]["win_length"]),
    )
    model.to(device)
    model.eval()
    return model


def _probe_teacher_start_checkpoint(
    config: dict[str, Any],
    per_domain: dict[str, dict[str, str]],
    *,
    checkpoint: str | None,
    device: str,
    model_family: str,
    variant: str,
    force: bool = False,
) -> dict[str, Any]:
    def _probe_progress(message: str) -> None:
        print(f"[teacher-probe] {message}", flush=True)

    summary_path = _teacher_init_probe_path(config)
    voicebank_manifest = _resolve_staged_manifest_path(config, per_domain["voicebank"]["val_select"])
    dns5_manifest = _resolve_staged_manifest_path(config, per_domain["dns5"]["val_select"])
    sample_rate = int(config["training"]["sample_rate"])
    eval_batch_size = int(config.get("evaluation", {}).get("eval_batch_size", 8) or 8)
    dns5_cap = int(config.get("training", {}).get("final_max_eval_files", 0) or 0)
    cache_key = {
        "checkpoint": str(checkpoint or ""),
        "source_mode": "checkpoint" if checkpoint else "builder",
        "voicebank_manifest": str(voicebank_manifest),
        "dns5_manifest": str(dns5_manifest),
        "sample_rate": sample_rate,
        "eval_batch_size": eval_batch_size,
        "dns5_cap": dns5_cap,
        "model_family": model_family,
        "variant": variant,
    }
    if summary_path.exists() and not force:
        existing = read_json(summary_path)
        if all(existing.get(key) == value for key, value in cache_key.items()):
            return existing

    if checkpoint:
        print(f"[teacher] probing init checkpoint on val_select: {checkpoint}", flush=True)
    else:
        print(
            f"[teacher] probing built-in teacher on val_select family={model_family} variant={variant}",
            flush=True,
        )
    model = _build_runtime_teacher_model(
        config,
        device=device,
        model_family=model_family,
        variant=variant,
        checkpoint=checkpoint,
    )
    voicebank_metrics = evaluate_manifest(
        model,
        voicebank_manifest,
        device,
        sample_rate=sample_rate,
        compute_dnsmos=False,
        compute_composite=False,
        batch_size=eval_batch_size,
        cache_audio=True,
        progress_callback=_probe_progress,
    )
    dns5_metrics = evaluate_manifest(
        model,
        dns5_manifest,
        device,
        sample_rate=sample_rate,
        compute_dnsmos=False,
        compute_composite=False,
        batch_size=eval_batch_size,
        cache_audio=True,
        max_files=dns5_cap if dns5_cap > 0 else None,
        progress_callback=_probe_progress,
    )
    summary = {
        **cache_key,
        "voicebank_val_select_metrics": voicebank_metrics,
        "dns5_val_select_metrics": dns5_metrics,
        "voicebank_val_select_pesq": voicebank_metrics.get("pesq_mean"),
        "dns5_val_select_pesq": dns5_metrics.get("pesq_mean"),
    }
    write_json(summary_path, summary)
    return summary


def _direct_frozen_teacher_summary(
    config: dict[str, Any],
    *,
    device: str,
    model_family: str,
    variant: str,
    seed: int,
    run_name: str,
    per_domain: dict[str, dict[str, str]],
) -> dict[str, Any]:
    probe = _probe_teacher_start_checkpoint(
        config,
        per_domain,
        checkpoint=None,
        device=device,
        model_family=model_family,
        variant=variant,
    )
    teacher_floor = float((config.get("teacher_training") or {}).get("target_pesq_floor", 3.1))
    best_score = float(probe.get("voicebank_val_select_pesq") or float("-inf"))
    winner = {
        "audit_only": False,
        "best_epoch": 0,
        "best_score": best_score,
        "best_val_select_pesq": best_score,
        "checkpoint_out": "",
        "context_frames": int(config["training"]["context_frames"]),
        "early_stopped": False,
        "erb_bands": int(config["training"]["erb_bands"]),
        "guidance_classic": str(config["training"]["guidance_classic"]),
        "loss_recipe": "FROZEN_TEACHER_EVAL",
        "model_family": model_family,
        "run_name": run_name,
        "seed": int(seed),
        "selection_metric": "val_select/pesq_mean",
        "target_floor": teacher_floor,
        "teacher_source_mode": "builder",
        "threshold_met": best_score >= teacher_floor,
        "val_rank_metrics": {},
        "val_rank_metrics_by_split": {},
        "val_select_metrics": dict(probe["voicebank_val_select_metrics"]),
        "val_select_metrics_by_split": {
            "val_select": dict(probe["voicebank_val_select_metrics"]),
            "dns5_val_select": dict(probe["dns5_val_select_metrics"]),
        },
        "variant": variant,
    }
    summary = {
        "baseline": probe,
        "frozen_teacher": True,
        "phases": {},
        "runs": [],
        "target_pesq_floor": teacher_floor,
        "threshold_met": best_score >= teacher_floor,
        "winner": winner,
    }
    output_root = Path(config["paths"]["output_root"])
    summary_path = _write_group_summary(output_root, "teacher_training", summary)
    summary["summary_path"] = summary_path.as_posix()
    return summary


def command_train_teacher(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    dataset_summary = _academic_dataset_bundle(config, force=False)
    per_domain = _resolved_per_domain_manifests(config, force=False)
    strategy = _teacher_training_strategy(config)
    if strategy == "dns_first_fullsubnet_balanced":
        return _run_dns_first_fullsubnet_balanced_strategy(
            config,
            device=device,
            dataset_summary=dataset_summary,
            per_domain=per_domain,
        )
    if strategy == "dns_first_cmgan_pesq_guarded":
        return _run_dns_first_cmgan_pesq_guarded_strategy(
            config,
            device=device,
            dataset_summary=dataset_summary,
            per_domain=per_domain,
        )
    if strategy == "teacher_preserving_voicebank_refiner":
        return _run_teacher_preserving_refiner_strategy(
            config,
            device=device,
            dataset_summary=dataset_summary,
            per_domain=per_domain,
        )
    combined = dataset_summary["combined_manifests"]
    teacher_cfg = dict(config.get("teacher_training") or {})
    family = str((teacher_cfg.get("families") or ["metricgan_plus_native8k"])[0])
    variant = str((teacher_cfg.get("variants") or ["small"])[0])
    seed = int((teacher_cfg.get("seeds") or [0])[0])
    base_name = str(config.get("reference", {}).get("teacher_run_name") or f"{family}-{variant}-teacher-training-seed{seed}")
    group_root = Path(config["paths"]["output_root"]) / "checkpoints" / "teacher"
    shared = {
        **_base_experiment_config(config, device=device),
        **_teacher_eval_overrides(config, per_domain),
        "model_family": family,
        "variant": variant,
        "seed": seed,
        "target_floor": float(teacher_cfg.get("target_pesq_floor", 3.1)),
        "teacher_cache_manifest": None,
        "teacher_cache_schedule": None,
        "resume_training_state": None,
        "evaluate_init_checkpoint": bool(teacher_cfg.get("evaluate_init_checkpoint", False)),
    }
    if family == "metricgan_plus":
        print(
            f"[teacher] using frozen direct teacher family={family} variant={variant} seed={seed}",
            flush=True,
        )
        return _direct_frozen_teacher_summary(
            config,
            device=device,
            model_family=family,
            variant=variant,
            seed=seed,
            run_name=base_name,
            per_domain=per_domain,
        )

    start_checkpoint = _teacher_family_start_checkpoint(config, family)
    baseline_checkpoint = _teacher_family_baseline_checkpoint(config, family)
    start_probe = _probe_teacher_start_checkpoint(
        config,
        per_domain,
        checkpoint=baseline_checkpoint,
        device=device,
        model_family=family,
        variant=variant,
    )
    start_probe_vbd = start_probe.get("voicebank_val_select_pesq")
    print(
        f"[teacher] starting lineage family={family} variant={variant} seed={seed} "
        f"from={start_checkpoint or 'built-in initialization'}",
        flush=True,
    )
    phase_a_cfg = _phase_settings(
        config,
        "phase_a",
        {"lr": 5e-5, "epochs": 6, "early_stop_patience": 3, "min_epochs": 3, "rank_eval_every": 1, "select_eval_every": 2},
    )
    print(f"[teacher] phase_a config={phase_a_cfg}", flush=True)
    phase_a_name = _phase_run_name(base_name, "phase_a")
    phase_a_resume = _phase_resume_state(group_root, phase_a_name)
    if phase_a_resume:
        print(
            f"[teacher] phase_a resuming from epoch={phase_a_resume['resume_epoch']} "
            f"reason={phase_a_resume['resume_reason']}",
            flush=True,
        )
    phase_a_init_checkpoint = str(phase_a_resume.get("init_checkpoint") or start_checkpoint or "").strip() or None
    phase_a = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=phase_a_name,
        phase_tag="teacher_training_phase_a",
        train_csv=_resolve_staged_manifest_path(config, combined["train_fit"]),
        loss_recipe="T0",
        lr=float(phase_a_cfg["lr"]),
        epochs=int(phase_a_cfg["epochs"]),
        early_stop_patience=int(phase_a_cfg["early_stop_patience"]),
        min_epochs=int(phase_a_cfg["min_epochs"]),
        rank_eval_every=int(phase_a_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_a_cfg.get("select_eval_every", 2)),
        init_checkpoint=phase_a_init_checkpoint,
        resume_training_state=phase_a_resume.get("resume_training_state"),
    )
    dns5_phase_a = phase_a.get("val_select_metrics_by_split", {}).get("dns5_val_select", {})
    guardrail_floor = None
    if dns5_phase_a.get("pesq_mean") is not None:
        guardrail_floor = float(dns5_phase_a["pesq_mean"]) - 0.05
    print(
        f"[teacher] phase_a done best_val_select_pesq={phase_a.get('best_val_select_pesq')} "
        f"dns5_guardrail_floor={guardrail_floor}",
        flush=True,
    )
    phase_a_floor = _phase_a_promotion_floor(config)
    phase_a_min_gain = _phase_a_min_gain_over_init(config)
    phase_a_best = float(phase_a.get("best_val_select_pesq") or float("-inf"))
    phase_a_gain = None
    if start_probe_vbd is not None:
        phase_a_gain = float(phase_a_best - float(start_probe_vbd))
    phase_a_stop_reasons: list[str] = []
    if phase_a_floor is not None and phase_a_best < phase_a_floor:
        phase_a_stop_reasons.append(
            f"phase_a_best_val_select_pesq={phase_a_best:.4f} < phase_a_min_absolute_pesq={phase_a_floor:.4f}"
        )
    if phase_a_min_gain is not None:
        if phase_a_gain is None or phase_a_gain < phase_a_min_gain:
            observed_gain = "n/a" if phase_a_gain is None else f"{phase_a_gain:.4f}"
            phase_a_stop_reasons.append(
                f"phase_a_gain_over_init={observed_gain} < phase_a_min_gain_over_init={phase_a_min_gain:.4f}"
            )
    if phase_a_stop_reasons:
        teacher_floor = float(teacher_cfg.get("target_pesq_floor", 3.1))
        summary = {
            "baseline": start_probe,
            "runs": [phase_a],
            "phases": {"phase_a": phase_a},
            "winner": phase_a,
            "target_pesq_floor": teacher_floor,
            "phase_a_min_absolute_pesq": phase_a_floor,
            "phase_a_min_gain_over_init": phase_a_min_gain,
            "phase_a_gain_over_init": phase_a_gain,
            "phase_a_stop_reasons": phase_a_stop_reasons,
            "stopped_after_phase_a": True,
            "stop_before_student": True,
            "reason": "phase_a_below_promotion_floor",
            "threshold_met": phase_a_best >= teacher_floor,
        }
        output_root = Path(config["paths"]["output_root"])
        summary_path = _write_group_summary(output_root, "teacher_training", summary)
        summary["summary_path"] = summary_path.as_posix()
        print(
            f"[teacher] stopping after phase_a because {'; '.join(phase_a_stop_reasons)}",
            flush=True,
        )
        return summary

    replay_seed = int(teacher_cfg.get("replay_seed", 42))
    phase_b_cfg = _phase_settings(
        config,
        "phase_b",
        {"lr": 1.5e-5, "epochs": 12, "early_stop_patience": 5, "min_epochs": 4, "dns_fraction": 0.25, "rank_eval_every": 1, "select_eval_every": 1},
    )
    phase_b_schedule = _build_replay_schedule_manifests(
        per_domain["voicebank"]["train_fit"],
        per_domain["dns5"]["train_fit"],
        out_dir=Path(config["paths"]["output_root"]) / "replay_manifests" / "teacher_phase_b",
        seed=replay_seed,
        dns_fraction=float(phase_b_cfg.get("dns_fraction", 0.25)),
        prefix="teacher_phase_b",
    )
    print(f"[teacher] phase_b config={phase_b_cfg}", flush=True)
    phase_b_name = _phase_run_name(base_name, "phase_b")
    phase_b_resume = _phase_resume_state(group_root, phase_b_name)
    if phase_b_resume:
        print(
            f"[teacher] phase_b resuming from epoch={phase_b_resume['resume_epoch']} "
            f"reason={phase_b_resume['resume_reason']}",
            flush=True,
        )
    phase_b = _run_named_experiment(
        {**shared, **_teacher_eval_overrides(config, per_domain, guardrail_floor=guardrail_floor)},
        group_root=group_root,
        run_name=phase_b_name,
        phase_tag="teacher_training_phase_b",
        train_csv=phase_b_schedule["mixed_manifests"][0],
        train_csv_schedule=phase_b_schedule["mixed_manifests"],
        loss_recipe="T0",
        lr=float(phase_b_cfg["lr"]),
        epochs=int(phase_b_cfg["epochs"]),
        early_stop_patience=int(phase_b_cfg["early_stop_patience"]),
        min_epochs=int(phase_b_cfg["min_epochs"]),
        rank_eval_every=int(phase_b_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_b_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_b_resume.get("init_checkpoint") or phase_a["checkpoint_out"]),
        resume_training_state=phase_b_resume.get("resume_training_state"),
    )
    print(f"[teacher] phase_b done best_val_select_pesq={phase_b.get('best_val_select_pesq')}", flush=True)
    phase_b_floor = _phase_b_promotion_floor(config)
    phase_b_best = float(phase_b.get("best_val_select_pesq") or float("-inf"))
    if phase_b_floor is not None and phase_b_best < phase_b_floor:
        teacher_floor = float(teacher_cfg.get("target_pesq_floor", 3.1))
        winner = _select_teacher_winner([phase_a, phase_b])
        summary = {
            "baseline": start_probe,
            "runs": [phase_a, phase_b],
            "phases": {"phase_a": phase_a, "phase_b": phase_b},
            "winner": winner,
            "target_pesq_floor": teacher_floor,
            "phase_a_min_absolute_pesq": phase_a_floor,
            "phase_a_min_gain_over_init": phase_a_min_gain,
            "phase_a_gain_over_init": phase_a_gain,
            "phase_b_promotion_floor": phase_b_floor,
            "stopped_after_phase_b": True,
            "stop_before_student": True,
            "reason": "phase_b_below_promotion_floor",
            "threshold_met": float(winner.get("best_val_select_pesq") or float("-inf")) >= teacher_floor,
        }
        output_root = Path(config["paths"]["output_root"])
        summary_path = _write_group_summary(output_root, "teacher_training", summary)
        summary["summary_path"] = summary_path.as_posix()
        print(
            f"[teacher] stopping after phase_b because best_val_select_pesq={phase_b_best:.4f} "
            f"is below phase_b_promotion_floor={phase_b_floor:.4f}",
            flush=True,
        )
        return summary

    print("[teacher] building PESQ proxy for phase_c", flush=True)
    pesq_proxy_checkpoint = _train_teacher_pesq_proxy(
        config,
        per_domain,
        phase_a_checkpoint=phase_a["checkpoint_out"],
        phase_b_checkpoint=phase_b["checkpoint_out"],
        device=device,
        force=False,
    )
    phase_c_cfg = _phase_settings(
        config,
        "phase_c",
        {"lr": 5e-6, "epochs": 6, "early_stop_patience": 3, "min_epochs": 2, "dns_fraction": 0.1, "rank_eval_every": 1, "select_eval_every": 1},
    )
    print(f"[teacher] phase_c config={phase_c_cfg}", flush=True)
    phase_c_schedule = _build_replay_schedule_manifests(
        per_domain["voicebank"]["train_fit"],
        per_domain["dns5"]["train_fit"],
        out_dir=Path(config["paths"]["output_root"]) / "replay_manifests" / "teacher_phase_c",
        seed=replay_seed,
        dns_fraction=float(phase_c_cfg.get("dns_fraction", 0.5)),
        prefix="teacher_phase_c",
    )
    phase_c_name = _phase_run_name(base_name, "phase_c")
    phase_c_resume = _phase_resume_state(group_root, phase_c_name)
    if phase_c_resume:
        print(
            f"[teacher] phase_c resuming from epoch={phase_c_resume['resume_epoch']} "
            f"reason={phase_c_resume['resume_reason']}",
            flush=True,
        )
    phase_c = _run_named_experiment(
        {**shared, **_teacher_eval_overrides(config, per_domain, guardrail_floor=guardrail_floor)},
        group_root=group_root,
        run_name=phase_c_name,
        phase_tag="teacher_training_phase_c",
        train_csv=phase_c_schedule["mixed_manifests"][0],
        train_csv_schedule=phase_c_schedule["mixed_manifests"],
        loss_recipe="T0_PESQ",
        pesq_proxy_checkpoint=pesq_proxy_checkpoint,
        lr=float(phase_c_cfg["lr"]),
        epochs=int(phase_c_cfg["epochs"]),
        early_stop_patience=int(phase_c_cfg["early_stop_patience"]),
        min_epochs=int(phase_c_cfg["min_epochs"]),
        rank_eval_every=int(phase_c_cfg.get("rank_eval_every", 1)),
        select_eval_every=int(phase_c_cfg.get("select_eval_every", 1)),
        init_checkpoint=str(phase_c_resume.get("init_checkpoint") or phase_b["checkpoint_out"]),
        resume_training_state=phase_c_resume.get("resume_training_state"),
    )
    print(f"[teacher] phase_c done best_val_select_pesq={phase_c.get('best_val_select_pesq')}", flush=True)

    runs = [phase_a, phase_b, phase_c]
    winner = _select_teacher_winner(runs)
    teacher_floor = float(teacher_cfg.get("target_pesq_floor", 3.1))
    summary = {
        "baseline": start_probe,
        "runs": runs,
        "phases": {"phase_a": phase_a, "phase_b": phase_b, "phase_c": phase_c},
        "winner": winner,
        "target_pesq_floor": teacher_floor,
        "phase_a_min_absolute_pesq": phase_a_floor,
        "phase_a_min_gain_over_init": phase_a_min_gain,
        "phase_a_gain_over_init": phase_a_gain,
        "phase_b_promotion_floor": phase_b_floor,
        "threshold_met": float(winner.get("best_val_select_pesq") or float("-inf")) >= teacher_floor,
        "pesq_proxy_checkpoint": pesq_proxy_checkpoint,
    }
    output_root = Path(config["paths"]["output_root"])
    summary_path = _write_group_summary(output_root, "teacher_training", summary)
    summary["summary_path"] = summary_path.as_posix()
    print(
        f"[teacher] winner checkpoint={winner.get('checkpoint_out')} "
        f"best_val_select_pesq={winner.get('best_val_select_pesq')} "
        f"threshold_met={summary['threshold_met']}",
        flush=True,
    )
    return summary


def _stage1_run_name(family: str, config: dict[str, Any], seed: int) -> str:
    return f"{family}-seed{seed}"


def _run_student_candidate(
    config: dict[str, Any],
    *,
    device: str,
    family: str,
    seed: int,
    teacher_summary: dict[str, Any],
) -> dict[str, Any]:
    dataset_summary = _academic_dataset_bundle(config, force=False)
    per_domain = _resolved_per_domain_manifests(config, force=False)
    combined = dataset_summary["combined_manifests"]
    teacher_cache_manifest = _resolve_teacher_cache_manifest(config)
    stage_cfg = dict(config.get("stage1") or {})
    base_name = _stage1_run_name(family, config, seed)
    group_root = Path(config["paths"]["output_root"]) / "checkpoints" / "stage1"
    shared = {
        **_base_experiment_config(config, device=device),
        **_teacher_eval_overrides(config, per_domain),
        "model_family": family,
        "variant": "small",
        "seed": int(seed),
        "target_floor": float(stage_cfg.get("target_pesq_floor", 2.8605)),
        "teacher_cache_manifest": teacher_cache_manifest,
        "teacher_variant": "fp32",
        "evaluate_init_checkpoint": bool(stage_cfg.get("evaluate_init_checkpoint", False)),
        "resume_training_state": None,
    }
    print(f"[stage1] starting candidate family={family} seed={seed}", flush=True)
    phase_s1_cfg = _student_phase_settings(config, "phase_s1", {"lr": 5e-4, "epochs": 24, "early_stop_patience": 6, "min_epochs": 8})
    print(f"[stage1] {family} phase_s1 config={phase_s1_cfg}", flush=True)
    phase_s1 = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=_phase_run_name(base_name, "phase_s1"),
        phase_tag="student_training_phase_s1",
        train_csv=_resolve_staged_manifest_path(config, combined["train_fit"]),
        loss_recipe=str(config["stage1"].get("loss_recipe", "D1")),
        lr=float(phase_s1_cfg["lr"]),
        epochs=int(phase_s1_cfg["epochs"]),
        early_stop_patience=int(phase_s1_cfg["early_stop_patience"]),
        min_epochs=int(phase_s1_cfg["min_epochs"]),
        init_checkpoint=None,
    )
    print(f"[stage1] {family} phase_s1 done best_val_select_pesq={phase_s1.get('best_val_select_pesq')}", flush=True)

    replay_seed = int(config.get("stage1", {}).get("replay_seed", 42))
    phase_s2_schedule = _build_replay_schedule_manifests(
        per_domain["voicebank"]["train_fit"],
        per_domain["dns5"]["train_fit"],
        out_dir=Path(config["paths"]["output_root"]) / "replay_manifests" / f"{family}_phase_s2",
        seed=replay_seed,
        dns_fraction=1.0,
        prefix=f"{family}_phase_s2",
    )
    cache_schedule = _build_teacher_cache_schedule(
        teacher_cache_manifest,
        phase_s2_schedule["mixed_manifests"],
        out_dir=Path(config["paths"]["output_root"]) / "teacher_cache" / f"{family}_phase_s2",
        prefix=f"{family}_phase_s2",
    )
    phase_s2_cfg = _student_phase_settings(config, "phase_s2", {"lr": 2e-4, "epochs": 12, "early_stop_patience": 4, "min_epochs": 4})
    print(f"[stage1] {family} phase_s2 config={phase_s2_cfg}", flush=True)
    phase_s2 = _run_named_experiment(
        shared,
        group_root=group_root,
        run_name=_phase_run_name(base_name, "phase_s2"),
        phase_tag="student_training_phase_s2",
        train_csv=phase_s2_schedule["mixed_manifests"][0],
        train_csv_schedule=phase_s2_schedule["mixed_manifests"],
        teacher_cache_manifest=cache_schedule[0],
        teacher_cache_schedule=cache_schedule,
        loss_recipe=str(config["stage1"].get("loss_recipe", "D1")),
        lr=float(phase_s2_cfg["lr"]),
        epochs=int(phase_s2_cfg["epochs"]),
        early_stop_patience=int(phase_s2_cfg["early_stop_patience"]),
        min_epochs=int(phase_s2_cfg["min_epochs"]),
        init_checkpoint=phase_s1["checkpoint_out"],
    )
    print(f"[stage1] {family} phase_s2 done best_val_select_pesq={phase_s2.get('best_val_select_pesq')}", flush=True)
    winner = _select_teacher_winner([phase_s1, phase_s2])
    return {
        "family": family,
        "seed": int(seed),
        "runs": [phase_s1, phase_s2],
        "winner": winner,
    }


def _select_stage1_winner(candidates: list[dict[str, Any]], *, floor: float) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No stage1 results available.")
    by_family = {candidate["family"]: candidate for candidate in candidates}
    primary = next(iter(candidates))
    primary_score = float(primary["winner"].get("best_val_select_pesq") or float("-inf"))
    fallback = candidates[1] if len(candidates) > 1 else None
    if fallback is not None:
        fallback_score = float(fallback["winner"].get("best_val_select_pesq") or float("-inf"))
        if primary_score < floor <= fallback_score:
            return fallback["winner"]
        if primary_score >= floor and fallback_score >= floor and abs(primary_score - fallback_score) <= 0.02:
            return primary["winner"]
        return max([primary["winner"], fallback["winner"]], key=lambda item: float(item.get("best_val_select_pesq") or float("-inf")))
    return primary["winner"]


def _teacher_threshold_required_for_stage1(config: dict[str, Any]) -> bool:
    return bool((config.get("stage1") or {}).get("require_teacher_threshold", True))


def command_train_stage1(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    teacher_summary = _load_latest_group_summary(Path(config["paths"]["output_root"]), "teacher_training", legacy_filename="teacher_training_results.json")
    if teacher_summary is None:
        teacher_summary = command_train_teacher(config, device=device)
    stage_cfg = dict(config.get("stage1") or {})
    primary_family = str(stage_cfg.get("primary_family") or (stage_cfg.get("families") or ["metricgan_plus_native8k_causal_s"])[0])
    fallback_family = str(stage_cfg.get("fallback_family") or ((stage_cfg.get("families") or [None, "metricgan_plus_native8k_causal_n6"])[1] if len(stage_cfg.get("families") or []) > 1 else "metricgan_plus_native8k_causal_n6"))
    seed = int((stage_cfg.get("seeds") or [0])[0])
    student_floor = float(stage_cfg.get("target_pesq_floor", 2.8605))
    teacher_threshold_met = bool(teacher_summary.get("threshold_met", False))
    teacher_best = float(teacher_summary.get("winner", {}).get("best_val_select_pesq") or float("-inf"))
    teacher_floor = float(teacher_summary.get("target_pesq_floor", (config.get("teacher_training") or {}).get("target_pesq_floor", 3.1)))

    if _teacher_threshold_required_for_stage1(config) and not teacher_threshold_met:
        output_root = Path(config["paths"]["output_root"])
        summary = {
            "skipped": True,
            "reason": "teacher_threshold_unmet",
            "teacher_best_val_select_pesq": teacher_best,
            "teacher_target_pesq_floor": teacher_floor,
            "teacher_summary_path": teacher_summary.get("summary_path"),
            "student_floor": student_floor,
            "threshold_met": False,
            "winner": teacher_summary.get("winner", {}),
        }
        summary_path = _write_group_summary(output_root, "stage1_training", summary)
        summary["summary_path"] = summary_path.as_posix()
        print(
            f"[stage1] skipped because teacher threshold unmet "
            f"({teacher_best:.4f} < {teacher_floor:.4f})",
            flush=True,
        )
        return summary

    cache_summary = command_build_teacher_cache(config, device=device, force=False)
    print(
        f"[stage1] using teacher cache manifest={cache_summary.get('manifest')} "
        f"family={cache_summary.get('teacher_model_family')} variant={cache_summary.get('teacher_variant')}",
        flush=True,
    )

    candidates = [_run_student_candidate(config, device=device, family=primary_family, seed=seed, teacher_summary=teacher_summary)]
    primary_score = float(candidates[0]["winner"].get("best_val_select_pesq") or float("-inf"))
    if primary_score < student_floor and fallback_family:
        print(
            f"[stage1] primary family below floor ({primary_score:.4f} < {student_floor:.4f}), "
            f"launching fallback family={fallback_family}",
            flush=True,
        )
        candidates.append(_run_student_candidate(config, device=device, family=fallback_family, seed=seed, teacher_summary=teacher_summary))
    winner = _select_stage1_winner(candidates, floor=student_floor)
    summary = {
        "candidates": candidates,
        "winner": winner,
        "student_floor": student_floor,
        "teacher_cache": cache_summary,
        "threshold_met": float(winner.get("best_val_select_pesq") or float("-inf")) >= student_floor,
    }
    output_root = Path(config["paths"]["output_root"])
    summary_path = _write_group_summary(output_root, "stage1_training", summary)
    summary["summary_path"] = summary_path.as_posix()
    print(
        f"[stage1] winner checkpoint={winner.get('checkpoint_out')} "
        f"model_family={winner.get('model_family')} "
        f"best_val_select_pesq={winner.get('best_val_select_pesq')} "
        f"threshold_met={summary['threshold_met']}",
        flush=True,
    )
    return summary


def command_train_qat(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    stage1_summary = _load_latest_group_summary(output_root, "stage1_training", legacy_filename="stage1_results.json")
    if stage1_summary is None:
        stage1_summary = command_train_stage1(config, device=device)
    if stage1_summary.get("skipped"):
        result = {
            "skipped": True,
            "reason": stage1_summary.get("reason", "stage1_skipped"),
            "stage1_summary_path": stage1_summary.get("summary_path"),
        }
        summary_path = _write_group_summary(output_root, "qat_training", result)
        result["summary_path"] = summary_path.as_posix()
        return result
    winner = stage1_summary["winner"]
    student_floor = float(stage1_summary.get("student_floor", config.get("stage1", {}).get("target_pesq_floor", 2.8605)))
    if float(winner.get("best_val_select_pesq") or float("-inf")) < student_floor:
        result = {"skipped": True, "reason": "student_below_floor", "winner": winner, "student_floor": student_floor}
        summary_path = _write_group_summary(output_root, "qat_training", result)
        result["summary_path"] = summary_path.as_posix()
        return result

    dataset_summary = _academic_dataset_bundle(config, force=False)
    per_domain = _resolved_per_domain_manifests(config, force=False)
    teacher_cache_manifest = _resolve_teacher_cache_manifest(config)
    replay_seed = int(config.get("qat", {}).get("replay_seed", config.get("stage1", {}).get("replay_seed", 42)))
    replay_schedule = _build_replay_schedule_manifests(
        per_domain["voicebank"]["train_fit"],
        per_domain["dns5"]["train_fit"],
        out_dir=Path(config["paths"]["output_root"]) / "replay_manifests" / "qat",
        seed=replay_seed,
        dns_fraction=float(config.get("qat", {}).get("dns_fraction", 1.0)),
        prefix="qat",
    )
    cache_schedule = _build_teacher_cache_schedule(
        teacher_cache_manifest,
        replay_schedule["mixed_manifests"],
        out_dir=Path(config["paths"]["output_root"]) / "teacher_cache" / "qat",
        prefix="qat",
    )
    run_name = str(config["reference"]["final_qat_run_name"])
    qat_cfg = dict(config.get("qat") or {})
    print(f"[qat] starting run={run_name}", flush=True)
    result = _run_named_experiment(
        {
            **_base_experiment_config(config, device=device),
            **_teacher_eval_overrides(config, per_domain),
            "model_family": winner["model_family"],
            "variant": "small",
            "seed": int(winner["seed"]),
            "target_floor": float(student_floor),
            "teacher_cache_manifest": cache_schedule[0],
            "teacher_cache_schedule": cache_schedule,
            "evaluate_init_checkpoint": bool(qat_cfg.get("evaluate_init_checkpoint", False)),
            "resume_training_state": None,
        },
        group_root=output_root / "checkpoints" / "final",
        run_name=run_name,
        phase_tag="student_qat",
        train_csv=replay_schedule["mixed_manifests"][0],
        train_csv_schedule=replay_schedule["mixed_manifests"],
        loss_recipe=str(config["qat"]["loss_recipe"]),
        lr=float(config["qat"]["lr"]),
        epochs=int(config["qat"]["epochs"]),
        early_stop_patience=int(config["qat"]["early_stop_patience"]),
        min_epochs=int(config["qat"]["min_epochs"]),
        qat=True,
        init_checkpoint=winner["checkpoint_out"],
    )
    result["student_floor"] = student_floor
    result["threshold_met"] = float(result.get("best_val_select_pesq") or float("-inf")) >= student_floor
    summary_path = _write_group_summary(output_root, "qat_training", result)
    result["summary_path"] = summary_path.as_posix()
    print(
        f"[qat] done checkpoint={result.get('checkpoint_out')} "
        f"best_val_select_pesq={result.get('best_val_select_pesq')} "
        f"threshold_met={result['threshold_met']}",
        flush=True,
    )
    return result

def _resolve_checkpoint_for_evaluation(config: dict[str, Any], checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    output_root = Path(config["paths"]["output_root"])
    candidates: list[tuple[float, str]] = []
    for group, legacy_filename in (
        ("qat_training", "qat_result.json"),
        ("stage1_training", "stage1_results.json"),
        ("teacher_training", "teacher_training_results.json"),
    ):
        summary_path = _latest_group_summary_path(output_root, group)
        if summary_path is not None and summary_path.exists():
            summary = read_json(summary_path)
            checkpoint_out = str(summary.get("checkpoint_out") or summary.get("winner", {}).get("checkpoint_out") or "").strip()
            if checkpoint_out:
                candidates.append((summary_path.stat().st_mtime, checkpoint_out))
            continue
        if legacy_filename:
            legacy_path = output_root / legacy_filename
            if legacy_path.exists():
                summary = read_json(legacy_path)
                checkpoint_out = str(summary.get("checkpoint_out") or summary.get("winner", {}).get("checkpoint_out") or "").strip()
                if checkpoint_out:
                    candidates.append((legacy_path.stat().st_mtime, checkpoint_out))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return str(config["paths"].get("reference_final_checkpoint") or config["paths"]["teacher_source_checkpoint"])


def _evaluation_output_dir(config: dict[str, Any], label: str | None, checkpoint: str) -> Path:
    if label:
        name = label
    else:
        name = Path(checkpoint).stem
    out_dir = Path(config["paths"]["output_root"]) / "evaluations" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_canonical_eval_csv(out_dir: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    by_split_keys = ["val_rank_metrics_by_split", "val_select_metrics_by_split", "test_metrics_by_split"]
    for scope_name in by_split_keys:
        split_payload = summary.get(scope_name, {})
        scope_label = scope_name.removesuffix("_metrics_by_split")
        for split_name, metrics in split_payload.items():
            for key, value in metrics.items():
                if key == "sample_paths":
                    continue
                rows.append({"scope": scope_label, "split": split_name, "metric": key, "value": value})
    if not rows:
        for split_name in ["val_rank_metrics", "val_select_metrics", "test_metrics"]:
            scope_label = split_name.removesuffix("_metrics")
            for key, value in summary.get(split_name, {}).items():
                if key == "sample_paths":
                    continue
                rows.append({"scope": scope_label, "split": "primary", "metric": key, "value": value})
    if summary.get("benchmark_latency_10s") is not None:
        rows.append({"scope": "benchmark", "split": "primary", "metric": "benchmark_latency_10s", "value": summary.get("benchmark_latency_10s")})
    rollup = summary.get("mcu_rollup", {})
    rows.extend(
        [
            {"scope": "mcu_rollup", "split": "primary", "metric": "best_profile_name", "value": rollup.get("best_profile_name")},
            {"scope": "mcu_rollup", "split": "primary", "metric": "best_power_profile_name", "value": rollup.get("best_power_profile_name")},
            {"scope": "mcu_rollup", "split": "primary", "metric": "best_power_profile_avg_power_mw", "value": rollup.get("best_power_profile_avg_power_mw")},
        ]
    )
    write_csv(out_dir / "canonical_metrics.csv", rows, ["scope", "split", "metric", "value"])


def command_evaluate(
    config: dict[str, Any],
    *,
    device: str,
    checkpoint: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_for_evaluation(config, checkpoint)
    print(f"[evaluate] checkpoint={checkpoint_path}", flush=True)
    out_dir = _evaluation_output_dir(config, label, checkpoint_path)
    model, package = load_model_from_checkpoint(checkpoint_path, device=device)

    if config["dataset"].get("dataset_type") == "academic_combined":
        per_domain = _resolved_per_domain_manifests(config, force=False)
        eval_sets = {
            "val_rank_metrics_by_split": {
                "voicebank_val_rank": per_domain["voicebank"]["val_rank"],
                "dns5_val_rank": per_domain["dns5"]["val_rank"],
            },
            "val_select_metrics_by_split": {
                "voicebank_val_select": per_domain["voicebank"]["val_select"],
                "dns5_val_select": per_domain["dns5"]["val_select"],
            },
            "test_metrics_by_split": {},
        }
        voicebank_test = str(per_domain["voicebank"].get("test") or "").strip()
        if voicebank_test:
            eval_sets["test_metrics_by_split"]["voicebank_test"] = voicebank_test
        dns_test = str(per_domain["dns5"].get("test") or "").strip()
        if dns_test:
            eval_sets["test_metrics_by_split"]["dns5_test"] = dns_test
    else:
        val_rank_manifest = config["dataset"]["val_rank_csv"]
        val_select_manifest = config["dataset"]["val_select_csv"]
        test_manifest = materialize_test_manifest_8k(config)
        eval_sets = {
            "val_rank_metrics_by_split": {"val_rank": val_rank_manifest},
            "val_select_metrics_by_split": {"val_select": val_select_manifest},
            "test_metrics_by_split": {"test": test_manifest},
        }

    computed: dict[str, dict[str, Any]] = {}
    for scope_name, manifests in eval_sets.items():
        scope_payload: dict[str, Any] = {}
        for split_name, manifest_path in manifests.items():
            metrics = evaluate_manifest(
                model,
                manifest_path,
                device,
                sample_rate=int(config["training"]["sample_rate"]),
                compute_dnsmos=False,
                compute_composite=True,
                sample_dir=out_dir / "samples" / split_name,
                sample_count=int(config["training"]["sample_count"]),
                batch_size=int(config["evaluation"].get("eval_batch_size", config["training"].get("eval_batch_size", 8) or 8)),
                cache_audio=True,
            )
            scope_payload[split_name] = metrics
            write_json(out_dir / f"{split_name}.json", metrics)
        computed[scope_name] = scope_payload

    primary_val_rank = next(iter(computed["val_rank_metrics_by_split"].values()))
    primary_val_select = next(iter(computed["val_select_metrics_by_split"].values()))
    primary_test = next(iter(computed["test_metrics_by_split"].values())) if computed["test_metrics_by_split"] else {}
    benchmark_source = next(iter(eval_sets["val_rank_metrics_by_split"].values()))
    sample_path = read_pair_manifest(benchmark_source)[0].noisy
    benchmark_seconds = benchmark_inference(
        model,
        sample_path=sample_path,
        device=device,
        sample_rate=int(config["training"]["sample_rate"]),
        duration_seconds=int(config["training"]["benchmark_seconds"]),
        repeats=int(config["training"]["benchmark_repeats"]),
    )
    mcu_rollup = simulate_model_across_profiles(model)
    write_json(out_dir / "mcu_rollup.json", mcu_rollup)

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_package": {
            "model_family": package.get("model_family"),
            "variant": package.get("variant"),
            "model_config": package.get("model_config", {}),
        },
        "val_rank_metrics": primary_val_rank,
        "val_select_metrics": primary_val_select,
        "test_metrics": primary_test,
        "val_rank_metrics_by_split": computed["val_rank_metrics_by_split"],
        "val_select_metrics_by_split": computed["val_select_metrics_by_split"],
        "test_metrics_by_split": computed["test_metrics_by_split"],
        "benchmark_latency_10s": benchmark_seconds,
        "mcu_rollup": mcu_rollup,
    }
    write_json(out_dir / "summary.json", summary)
    _write_canonical_eval_csv(out_dir, summary)
    print(f"[evaluate] wrote summary={out_dir / 'summary.json'}", flush=True)
    return summary

def command_report(
    config: dict[str, Any],
    *,
    evaluation_dir: str | None = None,
    report_dir: str | None = None,
) -> dict[str, Any]:
    reference_export_json = str(config.get("paths", {}).get("reference_export_json") or "").strip()
    if not reference_export_json:
        return {"skipped": True, "reason": "reference_export_json_missing"}
    reference_export_path = Path(reference_export_json)
    if not reference_export_path.exists():
        return {
            "skipped": True,
            "reason": "reference_export_missing",
            "reference_export_json": reference_export_path.as_posix(),
        }
    reference_export = read_json(reference_export_path)
    if evaluation_dir is None:
        default_eval = Path(config["paths"]["output_root"]) / "evaluations" / Path(config["paths"]["reference_final_checkpoint"]).stem
        if default_eval.exists():
            evaluation_dir = default_eval.as_posix()
        else:
            evaluation_dir = (Path(config["paths"]["output_root"]) / "evaluations" / "metricgan_plus_native8k_causal_s_qat").as_posix()
    target_report_dir = Path(report_dir) if report_dir else Path(config["paths"]["output_root"]) / "reports" / "reference_qat"
    return generate_report(
        report_dir=target_report_dir,
        config=config,
        reference_export=reference_export,
        evaluation_dir=evaluation_dir,
    )


def command_run_all(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    prepare = command_prepare_data(config)
    teacher = command_train_teacher(config, device=device)
    if bool(teacher.get("stop_before_student") or teacher.get("stopped_after_phase_a") or teacher.get("stopped_after_phase_b")):
        skip_reason = str(teacher.get("reason") or "teacher_stopped_after_phase_a")
        skipped = {"skipped": True, "reason": skip_reason}
        return {
            "prepare_data": prepare,
            "teacher": teacher,
            "teacher_cache": dict(skipped),
            "stage1": dict(skipped),
            "qat": dict(skipped),
            "evaluation": dict(skipped),
            "report": dict(skipped),
        }
    stage1 = command_train_stage1(config, device=device)
    teacher_cache = stage1.get("teacher_cache") if isinstance(stage1, dict) else None
    if not teacher_cache:
        teacher_cache = {"skipped": True, "reason": "stage1_skipped_or_cache_embedded"}
    qat_cfg = dict(config.get("qat") or {})
    if bool(qat_cfg.get("auto_run", False)):
        qat = command_train_qat(config, device=device)
    else:
        qat = {"skipped": True, "reason": "auto_run_disabled"}
    eval_checkpoint = str(
        qat.get("checkpoint_out")
        or stage1.get("winner", {}).get("checkpoint_out")
        or teacher.get("winner", {}).get("checkpoint_out")
        or ""
    ).strip() or None
    evaluation = command_evaluate(config, device=device, checkpoint=eval_checkpoint)
    report_cfg = dict(config.get("report") or {})
    report = command_report(config) if bool(report_cfg.get("enabled", False)) else {"skipped": True, "reason": "report_disabled"}
    return {
        "prepare_data": prepare,
        "teacher": teacher,
        "teacher_cache": teacher_cache,
        "stage1": stage1,
        "qat": qat,
        "evaluation": evaluation,
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone repro pipeline for metricgan_plus_native8k_causal_s.")
    parser.add_argument("--config", default=(PROJECT_ROOT / "configs" / "default.yaml").as_posix())
    parser.add_argument("--device", default="auto")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare_data")
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--device", dest="subcommand_device", default=None)

    prepare_dataset = subparsers.add_parser("prepare_dataset")
    prepare_dataset.add_argument("--force", action="store_true")
    prepare_dataset.add_argument("--device", dest="subcommand_device", default=None)

    prepare_stage = subparsers.add_parser("prepare_stage_distributed")
    prepare_stage.add_argument("--force", action="store_true")
    prepare_stage.add_argument("--device", dest="subcommand_device", default=None)

    teacher_cache = subparsers.add_parser("build_teacher_cache")
    teacher_cache.add_argument("--force", action="store_true")
    teacher_cache.add_argument("--device", dest="subcommand_device", default=None)

    train_teacher = subparsers.add_parser("train_teacher")
    train_teacher.add_argument("--device", dest="subcommand_device", default=None)

    train_stage1 = subparsers.add_parser("train_stage1")
    train_stage1.add_argument("--device", dest="subcommand_device", default=None)

    train_qat = subparsers.add_parser("train_qat")
    train_qat.add_argument("--device", dest="subcommand_device", default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--label", default=None)
    evaluate.add_argument("--device", dest="subcommand_device", default=None)

    report = subparsers.add_parser("report")
    report.add_argument("--evaluation-dir", default=None)
    report.add_argument("--report-dir", default=None)
    report.add_argument("--device", dest="subcommand_device", default=None)

    run_all = subparsers.add_parser("run_all")
    run_all.add_argument("--device", dest="subcommand_device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("tracking", {})
    config["tracking"].setdefault("experiment_name", DEFAULT_EXPERIMENT_NAME)
    device = require_cuda_device(getattr(args, "subcommand_device", None) or args.device)

    if args.command in ["prepare_data", "prepare_dataset"]:
        payload = command_prepare_data(config, force=bool(args.force))
    elif args.command == "prepare_stage_distributed":
        payload = command_prepare_stage_distributed(config, force=bool(args.force))
    elif args.command == "build_teacher_cache":
        payload = command_build_teacher_cache(config, device=device, force=bool(args.force))
    elif args.command == "train_teacher":
        payload = command_train_teacher(config, device=device)
    elif args.command == "train_stage1":
        payload = command_train_stage1(config, device=device)
    elif args.command == "train_qat":
        payload = command_train_qat(config, device=device)
    elif args.command == "evaluate":
        payload = command_evaluate(config, device=device, checkpoint=args.checkpoint, label=args.label)
    elif args.command == "report":
        payload = command_report(config, evaluation_dir=args.evaluation_dir, report_dir=args.report_dir)
    elif args.command == "run_all":
        payload = command_run_all(config, device=device)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
