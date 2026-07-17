from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


GB = 1024 * 1024 * 1024


@dataclass
class _RootState:
    root: Path
    reserve_bytes: int
    initial_free_bytes: int
    usable_bytes: int
    remaining_bytes: int
    planned_bytes: int = 0
    copied_bytes: int = 0


def _normalize_path(value: str | Path) -> str:
    return Path(value).as_posix().replace("\\", "/")


def _stable_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _staged_rel_path(src_path: str | Path, kind: str) -> Path:
    src_norm = _normalize_path(src_path).lower()
    digest = _stable_digest(src_norm)
    suffix = Path(src_path).suffix or ".bin"
    return Path(kind) / digest[:2] / f"{digest}{suffix}"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _default_reserve_bytes(root: Path, free_bytes: int) -> int:
    # Keep stricter reserve on system ext4; NTFS staging roots get dynamic reserve.
    root_norm = root.as_posix()
    if root_norm.startswith("/media/"):
        return int(max(10 * GB, 0.10 * free_bytes))
    return int(12 * GB)


def _resolve_reserve_bytes(
    root: Path,
    free_bytes: int,
    reserve_overrides_gb: dict[str, float] | None,
) -> int:
    if reserve_overrides_gb:
        direct = reserve_overrides_gb.get(root.as_posix())
        if direct is None:
            direct = reserve_overrides_gb.get(str(root))
        if direct is not None:
            return int(max(float(direct), 0.0) * GB)
    return _default_reserve_bytes(root, free_bytes)


def _build_root_states(
    roots: list[str | Path],
    *,
    reserve_overrides_gb: dict[str, float] | None,
) -> list[_RootState]:
    states: list[_RootState] = []
    for item in roots:
        root = Path(item).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root)
        reserve = _resolve_reserve_bytes(root, int(usage.free), reserve_overrides_gb)
        usable = max(int(usage.free) - reserve, 0)
        states.append(
            _RootState(
                root=root,
                reserve_bytes=reserve,
                initial_free_bytes=int(usage.free),
                usable_bytes=usable,
                remaining_bytes=usable,
            )
        )
    return states


def _path_ok(path: Path, expected_size: int) -> bool:
    try:
        return path.exists() and path.stat().st_size == expected_size
    except OSError:
        return False


def _choose_root(
    root_states: list[_RootState],
    *,
    noisy_rel: Path,
    clean_rel: Path,
    noisy_size: int,
    clean_size: int,
) -> tuple[_RootState, Path, Path, int]:
    candidates: list[tuple[int, int, int, _RootState, Path, Path, int]] = []
    # rank_priority: 0 => both already present, 1 => one present, 2 => none present.
    for state in root_states:
        noisy_dst = state.root / noisy_rel
        clean_dst = state.root / clean_rel
        noisy_ok = _path_ok(noisy_dst, noisy_size)
        clean_ok = _path_ok(clean_dst, clean_size)
        missing_bytes = (0 if noisy_ok else noisy_size) + (0 if clean_ok else clean_size)
        if state.remaining_bytes < missing_bytes:
            continue
        if noisy_ok and clean_ok:
            rank_priority = 0
        elif noisy_ok or clean_ok:
            rank_priority = 1
        else:
            rank_priority = 2
        candidates.append((rank_priority, missing_bytes, -state.remaining_bytes, state, noisy_dst, clean_dst, missing_bytes))

    if not candidates:
        raise RuntimeError("No staging root has enough remaining space for current pair.")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, chosen_state, noisy_dst, clean_dst, missing = candidates[0]
    return chosen_state, noisy_dst, clean_dst, missing


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmpcopy")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def stage_manifest_distributed(
    source_manifest: str | Path,
    *,
    staged_manifest_out: str | Path,
    roots: list[str | Path],
    stage_subdir: str = "ULP_STAGE_AUDIO",
    reserve_overrides_gb: dict[str, float] | None = None,
    verify_mode: str = "size_and_sample_hash",
    verify_hash_every_n: int = 2000,
    copy_workers: int = 8,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not roots:
        raise ValueError("`roots` must contain at least one staging root.")

    src_manifest = Path(source_manifest).resolve()
    out_manifest = Path(staged_manifest_out).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    staged_tmp = out_manifest.with_suffix(out_manifest.suffix + ".tmp")
    copy_plan = out_manifest.with_suffix(out_manifest.suffix + ".copy_plan.csv")
    copy_plan_tmp = copy_plan.with_suffix(copy_plan.suffix + ".tmp")

    root_states = _build_root_states(roots, reserve_overrides_gb=reserve_overrides_gb)
    total_usable = sum(state.usable_bytes for state in root_states)
    if total_usable <= 0:
        raise RuntimeError("No usable free space after reserves on staging roots.")

    stage_prefix = Path(stage_subdir)
    planned_rows = 0
    planned_missing_bytes = 0
    planned_missing_files = 0
    plan_errors: list[str] = []

    with src_manifest.open(newline="", encoding="utf-8") as in_handle, staged_tmp.open(
        "w", newline="", encoding="utf-8"
    ) as staged_handle, copy_plan_tmp.open("w", newline="", encoding="utf-8") as copy_handle:
        reader = csv.DictReader(in_handle)
        if "noisy" not in (reader.fieldnames or ()) or "clean" not in (reader.fieldnames or ()):
            raise ValueError(f"Manifest {src_manifest} must contain columns noisy, clean.")
        staged_writer = csv.DictWriter(staged_handle, fieldnames=["noisy", "clean"])
        staged_writer.writeheader()
        copy_writer = csv.DictWriter(copy_handle, fieldnames=["src", "dst", "size"])
        copy_writer.writeheader()

        for row_idx, row in enumerate(reader, start=1):
            noisy_src = Path(row["noisy"]).resolve()
            clean_src = Path(row["clean"]).resolve()
            try:
                noisy_size = int(noisy_src.stat().st_size)
                clean_size = int(clean_src.stat().st_size)
            except OSError as exc:
                plan_errors.append(f"row {row_idx}: {exc}")
                continue

            noisy_rel = stage_prefix / _staged_rel_path(noisy_src, "noisy")
            clean_rel = stage_prefix / _staged_rel_path(clean_src, "clean")
            try:
                chosen_state, noisy_dst, clean_dst, missing = _choose_root(
                    root_states,
                    noisy_rel=noisy_rel,
                    clean_rel=clean_rel,
                    noisy_size=noisy_size,
                    clean_size=clean_size,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc} row={row_idx} noisy={noisy_src} clean={clean_src} "
                    f"required_bytes={noisy_size + clean_size}"
                ) from exc

            chosen_state.remaining_bytes -= missing
            chosen_state.planned_bytes += missing
            planned_missing_bytes += missing
            planned_rows += 1

            staged_writer.writerow({"noisy": noisy_dst.as_posix(), "clean": clean_dst.as_posix()})

            if not _path_ok(noisy_dst, noisy_size):
                copy_writer.writerow({"src": noisy_src.as_posix(), "dst": noisy_dst.as_posix(), "size": noisy_size})
                planned_missing_files += 1
            if not _path_ok(clean_dst, clean_size):
                copy_writer.writerow({"src": clean_src.as_posix(), "dst": clean_dst.as_posix(), "size": clean_size})
                planned_missing_files += 1

            if progress_callback and (row_idx == 1 or row_idx % 5000 == 0):
                progress_callback(
                    f"planning {src_manifest.name}: rows={row_idx} "
                    f"planned_missing_gb={planned_missing_bytes / GB:.2f}"
                )

    if plan_errors:
        raise RuntimeError(f"Staging plan failed with missing files (first): {plan_errors[0]}")

    copy_tasks: list[tuple[Path, Path, int]] = []
    with copy_plan_tmp.open(newline="", encoding="utf-8") as copy_handle:
        reader = csv.DictReader(copy_handle)
        for row in reader:
            copy_tasks.append((Path(row["src"]), Path(row["dst"]), int(row["size"])))

    copied_files = 0
    copied_bytes = 0
    verified_hash_files = 0
    verify_stride = max(int(verify_hash_every_n), 1)
    workers = max(int(copy_workers), 1)

    def _copy_one(task: tuple[Path, Path, int]) -> tuple[int, int]:
        src, dst, expected_size = task
        if _path_ok(dst, expected_size):
            return (0, 0)
        _atomic_copy_file(src, dst)
        if not _path_ok(dst, expected_size):
            raise RuntimeError(f"Copy verification failed (size mismatch): {src} -> {dst}")
        return (1, expected_size)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(_copy_one, task): task for task in copy_tasks}
        for task_idx, fut in enumerate(as_completed(future_to_task), start=1):
            copied_inc, bytes_inc = fut.result()
            if copied_inc:
                copied_files += copied_inc
                copied_bytes += bytes_inc
                if verify_mode == "size_and_sample_hash":
                    do_hash = copied_files <= 20 or (copied_files % verify_stride == 0)
                    if do_hash:
                        src, dst, _ = future_to_task[fut]
                        src_hash = _hash_file(src)
                        dst_hash = _hash_file(dst)
                        if src_hash != dst_hash:
                            raise RuntimeError(f"Hash mismatch after copy: {src} -> {dst}")
                        verified_hash_files += 1

            if progress_callback and (task_idx == 1 or task_idx % 5000 == 0):
                progress_callback(
                    f"copying {src_manifest.name}: tasks={task_idx}/{len(copy_tasks)} "
                    f"copied_files={copied_files} copied_gb={copied_bytes / GB:.2f}"
                )

    # Commit outputs atomically only after planning + copy pass succeeds.
    copy_plan_tmp.replace(copy_plan)
    staged_tmp.replace(out_manifest)

    root_reports: list[dict[str, Any]] = []
    for state in root_states:
        root_reports.append(
            {
                "root": state.root.as_posix(),
                "reserve_bytes": state.reserve_bytes,
                "initial_free_bytes": state.initial_free_bytes,
                "usable_bytes": state.usable_bytes,
                "planned_bytes": state.planned_bytes,
                "remaining_bytes_after_plan": state.remaining_bytes,
            }
        )

    return {
        "source_manifest": src_manifest.as_posix(),
        "staged_manifest": out_manifest.as_posix(),
        "copy_plan": copy_plan.as_posix(),
        "rows": planned_rows,
        "planned_missing_files": planned_missing_files,
        "planned_missing_bytes": planned_missing_bytes,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "verified_hash_files": verified_hash_files,
        "verify_mode": verify_mode,
        "roots": root_reports,
    }


def stage_dataset_manifests(
    manifests: dict[str, str | Path],
    *,
    staged_manifest_dir: str | Path,
    roots: list[str | Path],
    stage_subdir: str = "ULP_STAGE_AUDIO",
    reserve_overrides_gb: dict[str, float] | None = None,
    verify_mode: str = "size_and_sample_hash",
    verify_hash_every_n: int = 2000,
    copy_workers: int = 8,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    out_dir = Path(staged_manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: dict[str, str] = {}
    manifest_reports: dict[str, Any] = {}
    for label, path in manifests.items():
        source = Path(path)
        staged_out = out_dir / f"{source.stem}_staged.csv"
        report = stage_manifest_distributed(
            source,
            staged_manifest_out=staged_out,
            roots=roots,
            stage_subdir=stage_subdir,
            reserve_overrides_gb=reserve_overrides_gb,
            verify_mode=verify_mode,
            verify_hash_every_n=verify_hash_every_n,
            copy_workers=copy_workers,
            progress_callback=progress_callback,
        )
        staged_paths[label] = staged_out.as_posix()
        manifest_reports[label] = report
        if progress_callback:
            progress_callback(
                f"staged {label}: rows={report['rows']} copied_files={report['copied_files']} "
                f"copied_gb={report['copied_bytes'] / GB:.2f}"
            )

    summary = {
        "staged_manifests": staged_paths,
        "reports": manifest_reports,
    }
    summary_path = out_dir / "staging_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = summary_path.as_posix()
    return summary
