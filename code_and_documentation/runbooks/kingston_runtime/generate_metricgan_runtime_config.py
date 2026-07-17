#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tracking-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-rank-manifest", required=True)
    parser.add_argument("--val-select-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--voicebank-train-fit-manifest", default="")
    parser.add_argument("--voicebank-val-rank-manifest", default="")
    parser.add_argument("--voicebank-val-select-manifest", default="")
    parser.add_argument("--voicebank-test-manifest", default="")
    parser.add_argument("--voicebank-test-expected-count", type=int, default=824)
    parser.add_argument("--dns5-train-fit-manifest", default="")
    parser.add_argument("--dns5-val-rank-manifest", default="")
    parser.add_argument("--dns5-val-select-manifest", default="")
    parser.add_argument("--dns5-test-manifest", default="")
    parser.add_argument("--teacher-resume-model", default="")
    parser.add_argument("--teacher-resume-state", default="")
    parser.add_argument("--experiment-name", default="metricgan_combined_datasets_kingston")
    args = parser.parse_args()

    base_path = Path(args.base_config).resolve()
    out_path = Path(args.output_config).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg.setdefault("paths", {})
    cfg.setdefault("tracking", {})
    cfg.setdefault("dataset", {})
    cfg.setdefault("teacher_cache", {})
    cfg.setdefault("io_staging", {})
    cfg.setdefault("teacher_training", {})

    cfg["paths"]["output_root"] = Path(args.output_root).resolve().as_posix()
    cfg["paths"]["tracking_root"] = Path(args.tracking_root).resolve().as_posix()
    cfg["tracking"]["experiment_name"] = str(args.experiment_name)

    train_manifest = Path(args.train_manifest).resolve().as_posix()
    val_rank_manifest = Path(args.val_rank_manifest).resolve().as_posix()
    val_select_manifest = Path(args.val_select_manifest).resolve().as_posix()
    test_manifest = Path(args.test_manifest).resolve().as_posix()

    cfg["dataset"]["combined_train_csv"] = train_manifest
    cfg["dataset"]["combined_val_rank_csv"] = val_rank_manifest
    cfg["dataset"]["combined_val_select_csv"] = val_select_manifest
    cfg["dataset"]["combined_test_csv"] = test_manifest
    cfg["dataset"]["train_fit_csv"] = train_manifest
    cfg["dataset"]["val_rank_csv"] = val_rank_manifest
    cfg["dataset"]["val_select_csv"] = val_select_manifest
    cfg["dataset"]["test_csv"] = test_manifest
    cfg["dataset"]["voicebank_test_expected_count"] = int(args.voicebank_test_expected_count)

    optional_manifests = {
        "voicebank_train_fit_csv": args.voicebank_train_fit_manifest,
        "voicebank_val_rank_csv": args.voicebank_val_rank_manifest,
        "voicebank_val_select_csv": args.voicebank_val_select_manifest,
        "voicebank_test_csv": args.voicebank_test_manifest,
        "dns5_train_fit_csv": args.dns5_train_fit_manifest,
        "dns5_val_rank_csv": args.dns5_val_rank_manifest,
        "dns5_val_select_csv": args.dns5_val_select_manifest,
        "dns5_test_csv_runtime": args.dns5_test_manifest,
    }
    for key, value in optional_manifests.items():
        rendered = Path(value).resolve().as_posix() if value.strip() else ""
        cfg["dataset"][key] = rendered

    cfg["teacher_cache"]["out_dir"] = (Path(args.output_root) / "teacher_cache").resolve().as_posix()
    cfg["teacher_cache"]["manifest"] = (
        Path(args.output_root) / "teacher_cache" / "teacher_cache.csv"
    ).resolve().as_posix()

    cfg["io_staging"]["enabled"] = False
    cfg["io_staging"]["roots"] = []

    resume_model = args.teacher_resume_model.strip()
    resume_state = args.teacher_resume_state.strip()
    cfg["teacher_training"]["resume_checkpoint"] = resume_model if resume_model else "auto_if_exists"
    cfg["teacher_training"]["resume_training_state"] = resume_state if resume_state else "auto_if_exists"

    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(out_path.as_posix())


if __name__ == "__main__":
    main()
