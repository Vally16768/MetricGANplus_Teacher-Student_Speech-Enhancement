#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rel_from_stage(path_value: str, stage_token: str) -> Path:
    norm = Path(path_value).as_posix()
    marker = f"/{stage_token}/"
    if marker in norm:
        tail = norm.split(marker, 1)[1]
        return Path(tail)
    if norm.startswith(f"{stage_token}/"):
        return Path(norm[len(stage_token) + 1 :])
    raise ValueError(f"Path does not contain '{stage_token}': {path_value}")


def _rewrite_one_manifest(
    src_manifest: Path,
    dst_manifest: Path,
    stage_root: Path,
    stage_token: str,
    strict_exists: bool,
) -> dict:
    stage_root = stage_root.resolve()
    dst_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_manifest.with_suffix(dst_manifest.suffix + ".tmp")

    rows = 0
    missing = 0
    with src_manifest.open("r", newline="", encoding="utf-8") as in_f, tmp_path.open(
        "w", newline="", encoding="utf-8"
    ) as out_f:
        reader = csv.DictReader(in_f)
        if "noisy" not in (reader.fieldnames or ()) or "clean" not in (reader.fieldnames or ()):
            raise ValueError(f"Manifest must contain noisy,clean columns: {src_manifest}")
        writer = csv.DictWriter(out_f, fieldnames=["noisy", "clean"])
        writer.writeheader()

        for row in reader:
            noisy_rel = _rel_from_stage(row["noisy"], stage_token)
            clean_rel = _rel_from_stage(row["clean"], stage_token)
            noisy_dst = (stage_root / noisy_rel).resolve()
            clean_dst = (stage_root / clean_rel).resolve()
            if strict_exists:
                if not noisy_dst.exists():
                    missing += 1
                    raise FileNotFoundError(f"Missing staged noisy file: {noisy_dst}")
                if not clean_dst.exists():
                    missing += 1
                    raise FileNotFoundError(f"Missing staged clean file: {clean_dst}")
            writer.writerow({"noisy": noisy_dst.as_posix(), "clean": clean_dst.as_posix()})
            rows += 1

    tmp_path.replace(dst_manifest)
    return {
        "source_manifest": src_manifest.as_posix(),
        "output_manifest": dst_manifest.as_posix(),
        "rows": rows,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="Input staged manifests.")
    parser.add_argument("--output-dir", required=True, help="Output directory for rewritten manifests.")
    parser.add_argument("--stage-root", required=True, help="KINGSTON staged audio root.")
    parser.add_argument("--stage-token", default="ULP_STAGE_AUDIO")
    parser.add_argument("--strict-exists", action="store_true")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_root = Path(args.stage_root).resolve()

    reports = []
    for item in args.manifests:
        src = Path(item).resolve()
        dst = out_dir / src.name
        report = _rewrite_one_manifest(
            src_manifest=src,
            dst_manifest=dst,
            stage_root=stage_root,
            stage_token=args.stage_token,
            strict_exists=bool(args.strict_exists),
        )
        reports.append(report)

    summary = {
        "stage_root": stage_root.as_posix(),
        "manifest_count": len(reports),
        "reports": reports,
    }
    if args.summary_json:
        Path(args.summary_json).resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
