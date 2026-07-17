#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _latest_file(root: Path, pattern: str) -> Path | None:
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _age_seconds(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _find_latest_progress(output_root: Path, group: str) -> tuple[Path | None, dict[str, Any] | None]:
    checkpoints_root = output_root / "checkpoints" / group
    progress_path = _latest_file(checkpoints_root, "**/progress.json")
    if progress_path is None:
        return None, None
    return progress_path, _read_json(progress_path)


def _find_latest_summary(output_root: Path, group: str) -> tuple[Path | None, dict[str, Any] | None]:
    summary_root = output_root / "summaries" / group
    summary_path = _latest_file(summary_root, "summary_*.json")
    if summary_path is None:
        return None, None
    return summary_path, _read_json(summary_path)


def _matching_processes(pattern: str | None) -> list[dict[str, str]]:
    if not pattern:
        return []
    cmd = ["ps", "-eo", "pid=,etime=,pcpu=,pmem=,state=,comm=,args="]
    output = subprocess.check_output(cmd, text=True)
    matches: list[dict[str, str]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or pattern not in line:
            continue
        parts = line.split(None, 6)
        if len(parts) != 7:
            continue
        pid, etime, pcpu, pmem, state, comm, proc_cmd = parts
        if comm == "pt_data_worker":
            continue
        if "experiment_status.py" in proc_cmd:
            continue
        matches.append(
            {
                "pid": pid,
                "etime": etime,
                "pcpu": pcpu,
                "pmem": pmem,
                "state": state,
                "comm": comm,
                "cmd": proc_cmd,
            }
        )
    return matches


def _build_section(
    output_root: Path,
    group: str,
    *,
    target_floor: float | None,
    stale_minutes: float,
) -> dict[str, Any]:
    progress_path, progress = _find_latest_progress(output_root, group)
    summary_path, summary = _find_latest_summary(output_root, f"{group}_training")
    checkpoints_root = output_root / "checkpoints" / group
    latest_state_path = _latest_file(checkpoints_root, "**/latest_state.pt")
    latest_activity_path = None
    latest_activity_mtime = -1.0
    for candidate in (progress_path, latest_state_path, summary_path):
        if candidate is None or not candidate.exists():
            continue
        candidate_mtime = candidate.stat().st_mtime
        if candidate_mtime > latest_activity_mtime:
            latest_activity_mtime = candidate_mtime
            latest_activity_path = candidate
    age_seconds = _age_seconds(latest_activity_path)
    stale = age_seconds is not None and age_seconds > stale_minutes * 60.0
    section: dict[str, Any] = {
        "group": group,
        "progress_path": progress_path.as_posix() if progress_path else None,
        "latest_state_path": latest_state_path.as_posix() if latest_state_path else None,
        "latest_activity_path": latest_activity_path.as_posix() if latest_activity_path else None,
        "summary_path": summary_path.as_posix() if summary_path else None,
        "age_seconds": age_seconds,
        "stale": stale,
        "target_floor": target_floor,
    }
    if progress:
        section["progress"] = progress
    if summary:
        section["summary"] = summary

    best_score = None
    threshold_met = None
    if summary and isinstance(summary.get("winner"), dict):
        best_score = summary["winner"].get("best_val_select_pesq")
        threshold_met = summary.get("threshold_met")
    elif progress:
        best_score = progress.get("best_selection_score") or progress.get("selection_score")
        if target_floor is not None and best_score is not None:
            threshold_met = float(best_score) >= float(target_floor)
    section["best_val_select_pesq"] = best_score
    section["threshold_met"] = threshold_met
    return section


def _recommendation(teacher: dict[str, Any], stage1: dict[str, Any]) -> str:
    teacher_summary = teacher.get("summary") or {}
    teacher_progress = teacher.get("progress") or {}
    stage1_summary = stage1.get("summary") or {}
    teacher_state = str(teacher_progress.get("state") or "")
    stage1_state = str((stage1.get("progress") or {}).get("state") or "")
    has_live_process = bool(teacher.get("live_processes"))
    if teacher_state in {"interrupted", "failed"} and not has_live_process:
        return f"teacher_{teacher_state}"
    if teacher.get("stale") and not teacher_summary:
        return "teacher_stale_investigate"
    if teacher_summary:
        if not teacher_summary.get("threshold_met", False):
            return "teacher_below_target_stop_before_student"
        if stage1_summary.get("skipped"):
            return "student_skipped_check_teacher_gate"
    if stage1_state in {"interrupted", "failed"} and not has_live_process:
        return f"student_{stage1_state}"
    if stage1.get("stale") and not stage1_summary:
        return "student_stale_investigate"
    if stage1_summary:
        if stage1_summary.get("threshold_met", False):
            return "student_on_target"
        if stage1_summary.get("skipped"):
            return str(stage1_summary.get("reason") or "student_skipped")
        return "student_below_target"
    if teacher_progress:
        state = str(teacher_progress.get("state") or "")
        if state in {"selection_scored", "best_updated"}:
            return "teacher_has_select_signal"
        return "teacher_running"
    return "no_active_progress_found"


def _print_human(status: dict[str, Any]) -> None:
    print(f"run_root: {status['run_root']}")
    if status["processes"]:
        print("processes:")
        for proc in status["processes"]:
            print(
                f"  pid={proc['pid']} etime={proc['etime']} cpu={proc['pcpu']}% "
                f"mem={proc['pmem']}% state={proc['state']}"
            )
    else:
        print("processes: none matched")

    for name in ("teacher", "stage1"):
        section = status[name]
        progress = section.get("progress") or {}
        summary = section.get("summary") or {}
        print(f"{name}:")
        print(
            f"  stale={section['stale']} age={_format_age(section['age_seconds'])} "
            f"best_val_select_pesq={section.get('best_val_select_pesq')}"
        )
        if progress:
            print(
                f"  progress state={progress.get('state')} epoch={progress.get('epoch')} "
                f"global_step={progress.get('global_step')} best={progress.get('best_selection_score')} "
                f"gap={progress.get('best_target_gap')}"
            )
            if progress.get("message"):
                print(f"  progress_message={progress['message']}")
        if summary:
            print(
                f"  summary threshold_met={summary.get('threshold_met')} "
                f"winner_best={summary.get('winner', {}).get('best_val_select_pesq')}"
            )
            if summary.get("reason"):
                print(f"  summary_reason={summary['reason']}")
    print(f"recommendation: {status['recommendation']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Show current experiment control status.")
    parser.add_argument("--run-root", required=True, help="Experiment run root containing outputs/ and tracking/.")
    parser.add_argument("--process-match", default=None, help="Optional substring used to find active training processes.")
    parser.add_argument("--teacher-floor", type=float, default=3.1)
    parser.add_argument("--student-floor", type=float, default=2.8605)
    parser.add_argument("--stale-minutes", type=float, default=45.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = run_root / "outputs"
    status = {
        "run_root": run_root.as_posix(),
        "processes": _matching_processes(args.process_match),
        "teacher": _build_section(output_root, "teacher", target_floor=args.teacher_floor, stale_minutes=args.stale_minutes),
        "stage1": _build_section(output_root, "stage1", target_floor=args.student_floor, stale_minutes=args.stale_minutes),
    }
    status["teacher"]["live_processes"] = list(status["processes"])
    status["stage1"]["live_processes"] = list(status["processes"])
    status["recommendation"] = _recommendation(status["teacher"], status["stage1"])

    if args.as_json:
        json.dump(status, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_human(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
