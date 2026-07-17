#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _collect_tasks(manifests: list[Path], dest_stage_root: Path, stage_token: str) -> list[tuple[Path, Path, int]]:
    tasks: dict[tuple[str, str], tuple[Path, Path, int]] = {}
    for manifest in manifests:
        row_idx = 0
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "noisy" not in (reader.fieldnames or ()) or "clean" not in (reader.fieldnames or ()):
                raise ValueError(f"Manifest must contain noisy,clean columns: {manifest}")
            for row in reader:
                row_idx += 1
                for key in ("noisy", "clean"):
                    src = Path(row[key]).resolve()
                    rel = _rel_from_stage(src.as_posix(), stage_token)
                    dst = (dest_stage_root / rel).resolve()
                    task_key = (src.as_posix(), dst.as_posix())
                    if task_key in tasks:
                        continue
                    if not src.exists():
                        raise FileNotFoundError(f"Missing source file from manifest {manifest}: {src}")
                    size = int(src.stat().st_size)
                    tasks[task_key] = (src, dst, size)
                if row_idx == 1 or row_idx % 50000 == 0:
                    print(
                        f"[sync_staged_audio] planning manifest={manifest.name} rows={row_idx} "
                        f"unique_tasks={len(tasks)}",
                        flush=True,
                    )
    return list(tasks.values())


def _copy_one(src: Path, dst: Path, expected_size: int) -> tuple[int, int]:
    if dst.exists() and int(dst.stat().st_size) == expected_size:
        return (0, 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmpcopy")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    real_size = int(dst.stat().st_size)
    if real_size != expected_size:
        raise RuntimeError(f"Size mismatch after copy: {src} -> {dst} ({real_size} != {expected_size})")
    return (1, expected_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="Input staged manifests.")
    parser.add_argument("--dest-stage-root", required=True)
    parser.add_argument("--stage-token", default="ULP_STAGE_AUDIO")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    manifests = [Path(item).resolve() for item in args.manifests]
    dest_stage_root = Path(args.dest_stage_root).resolve()
    dest_stage_root.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(manifests, dest_stage_root, args.stage_token)
    copied_files = 0
    copied_bytes = 0
    workers = max(int(args.workers), 1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_copy_one, src, dst, size): (src, dst, size) for src, dst, size in tasks}
        for idx, future in enumerate(as_completed(future_map), start=1):
            inc_files, inc_bytes = future.result()
            copied_files += inc_files
            copied_bytes += inc_bytes
            if idx == 1 or idx % 5000 == 0:
                print(
                    f"[sync_staged_audio] tasks={idx}/{len(tasks)} copied_files={copied_files} "
                    f"copied_gb={copied_bytes / (1024**3):.2f}",
                    flush=True,
                )

    summary = {
        "manifest_count": len(manifests),
        "tasks": len(tasks),
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "dest_stage_root": dest_stage_root.as_posix(),
    }
    if args.summary_json:
        Path(args.summary_json).resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
