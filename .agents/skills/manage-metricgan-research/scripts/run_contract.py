#!/usr/bin/env python3
"""Create and validate immutable MetricGAN+ experiment run contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,95}$")
MODEL_SUFFIXES = {".ckpt", ".onnx", ".pt", ".pth", ".tflite"}


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo.as_posix(), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create(args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit("Invalid run ID; use lowercase letters, digits and hyphens.")
    if not config.is_file():
        raise SystemExit(f"Missing config: {config}")
    if run_git(repo, "status", "--porcelain"):
        raise SystemExit("Refusing to create a run from a dirty worktree.")
    commit = run_git(repo, "rev-parse", "HEAD")
    run_dir = repo / "local" / "runs" / args.run_id
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")

    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(parents=True)
    for name in ("logs", "metrics", "models", "reports"):
        (run_dir / name).mkdir()
    config_out = provenance_dir / "config_resolved.yaml"
    shutil.copy2(config, config_out)
    (provenance_dir / "command.txt").write_text(args.command.rstrip() + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "planned",
        "purpose": args.purpose,
        "git_commit": commit,
        "git_dirty": False,
        "config_sha256": sha256(config_out),
        "seed": args.seed,
        "command": args.command,
        "manifest_sha256": {},
        "initialization": {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (provenance_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "planned"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(run_dir)


def validate(args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = repo / run_dir
    required = [
        run_dir / "provenance" / "provenance.json",
        run_dir / "provenance" / "config_resolved.yaml",
        run_dir / "provenance" / "command.txt",
        run_dir / "status.json",
    ]
    missing = [path.as_posix() for path in required if not path.is_file()]
    errors = [f"missing: {path}" for path in missing]
    if not missing:
        provenance = json.loads(required[0].read_text(encoding="utf-8"))
        config_hash = sha256(required[1])
        if provenance.get("config_sha256") != config_hash:
            errors.append("config SHA-256 mismatch")
        status = json.loads(required[3].read_text(encoding="utf-8")).get("status")
        if args.stage == "canonical" and status != "valid":
            errors.append(f"canonical status must be valid, found {status!r}")
    if args.stage == "canonical":
        if not (run_dir / "metrics" / "summary.json").is_file():
            errors.append("missing metrics/summary.json")
        if not any(path.suffix.lower() in MODEL_SUFFIXES for path in (run_dir / "models").glob("*")):
            errors.append("missing selected model")
        if not any((run_dir / "logs").glob("*")):
            errors.append("missing log")
        if not any((run_dir / "reports").glob("*")):
            errors.append("missing report/figure")
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"run_contract: PASS ({args.stage}) {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--repo", default=".")
    create_parser.add_argument("--run-id", required=True)
    create_parser.add_argument("--config", required=True)
    create_parser.add_argument("--command", required=True)
    create_parser.add_argument("--seed", required=True, type=int)
    create_parser.add_argument("--purpose", required=True)
    create_parser.set_defaults(func=create)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--repo", default=".")
    validate_parser.add_argument("--run-dir", required=True)
    validate_parser.add_argument("--stage", choices=("planned", "canonical"), default="planned")
    validate_parser.set_defaults(func=validate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
