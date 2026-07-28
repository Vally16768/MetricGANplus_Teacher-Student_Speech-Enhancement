#!/usr/bin/env python3
"""Validate project scope, privacy, documentation and architecture freshness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


TEXT_SUFFIXES = {
    ".cfg", ".ini", ".json", ".log", ".md", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}
SKIP_DIRS = {".git", ".venv", "__pycache__", "historical", "local", "venv"}
SCOPE_MARKERS = ("mpsenet", "mp_senet", "mp-senet", "fullsubnet", "cmgan")
TRASH_SUFFIXES = (".bak", ".orig", ".pid", ".pyc", ".swp", ".tmp", "~")
ARCHITECTURE_SOURCES = (
    "campaign.py",
    "code_and_documentation/sebench/bandwidth.py",
    "code_and_documentation/sebench/checkpoints.py",
    "code_and_documentation/sebench/contracts.py",
    "code_and_documentation/sebench/erb.py",
    "code_and_documentation/sebench/losses.py",
    "code_and_documentation/sebench/metricgan_alternating.py",
    "code_and_documentation/sebench/metricgan_d2.py",
    "code_and_documentation/sebench/metric_proxy_training.py",
    "code_and_documentation/sebench/models.py",
    "code_and_documentation/sebench/research_plan.py",
    "code_and_documentation/sebench/runtime.py",
    "code_and_documentation/sebench/t3_perceptual.py",
    "code_and_documentation/sebench/t3_support.py",
    "code_and_documentation/sebench/t3_training.py",
    "code_and_documentation/sebench/t4_calibration.py",
    "code_and_documentation/sebench/t4_microstep.py",
    "code_and_documentation/sebench/t5_zeroth_order.py",
    "code_and_documentation/sebench/t6_affine.py",
    "code_and_documentation/sebench/t7_confidence.py",
    "code_and_documentation/sebench/t8_router.py",
    "code_and_documentation/sebench/t9_multi_router.py",
    "code_and_documentation/sebench/t10_risk_router.py",
    "code_and_documentation/sebench/t11_penalty_router.py",
    "code_and_documentation/sebench/t12_rank_router.py",
    "code_and_documentation/sebench/t13_multiobjective_router.py",
    "code_and_documentation/sebench/t14_quadratic_router.py",
    "code_and_documentation/sebench/t15_oof_calibration.py",
    "code_and_documentation/sebench/t16_fine_action_router.py",
    "code_and_documentation/sebench/teacher_cache.py",
    "code_and_documentation/sebench/training.py",
    "code_and_documentation/configs/research_plan_voicebank_wb_nb.yaml",
    "code_and_documentation/reference/torch_pesq_0.1.2.json",
    "configs/voicebank_campaign.yaml",
    "scripts/bind_voicebank_manifests.py",
)
CONTENT_RULES = {
    "personal_home": re.compile(r"/(?:home|Users)/(?!USER(?:/|\b)|<user>(?:/|\b))[^/\s\"']+"),
    "external_mount": re.compile(r"/(?:media|mnt)/[^\s\"']+"),
    "github_token": re.compile(r"(?:github_pat_|ghp_)[A-Za-z0-9_]+"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "assigned_secret": re.compile(
        r"(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[\"']?[^\s\"']+"
    ),
}
OUT_OF_SCOPE_CODE = re.compile(
    r"(?:class\s+(?:MPSENet|FullSubNetPlus|CMGANSmall)\b|"
    r"model_family\s*==\s*[\"'](?:mp_senet|fullsubnet_plus|cmgan_small)[\"'])"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(repo: Path):
    for current, dirnames, filenames in os.walk(repo, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS)
        for filename in sorted(filenames):
            path = Path(current) / filename
            relative = path.relative_to(repo)
            if relative.parts[:2] == ("docs", "audits") and filename.endswith("-inventory.json"):
                continue
            yield path


def add(issues: list[dict[str, str]], rule: str, path: Path, detail: str) -> None:
    issues.append({"rule": rule, "path": path.as_posix(), "detail": detail})


def check_architecture(repo: Path, issues: list[dict[str, str]]) -> None:
    baseline_path = repo / ".agents" / "state" / "architecture_sources.sha256"
    if not baseline_path.is_file():
        add(issues, "architecture_baseline", baseline_path, "missing")
        return
    expected = {}
    for line in baseline_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        expected[relative] = digest
    for relative in ARCHITECTURE_SOURCES:
        path = repo / relative
        if not path.is_file():
            add(issues, "architecture_source", path, "missing")
        elif expected.get(relative) != sha256(path):
            add(issues, "architecture_stale", path, "source hash changed")


def check_docs(repo: Path, issues: list[dict[str, str]]) -> None:
    index_path = repo / ".agents" / "DOCUMENTATION_INDEX.md"
    if not index_path.is_file():
        add(issues, "documentation_index", index_path, "missing")
        return
    index_text = index_path.read_text(encoding="utf-8")
    for path in files_under(repo):
        relative = path.relative_to(repo)
        if path.suffix.lower() != ".md":
            continue
        if relative.parts[:2] == ("experiments", "historical"):
            continue
        if relative.parts[:2] == (".agents", "skills"):
            continue
        if relative.as_posix() not in index_text:
            add(issues, "documentation_unindexed", relative, "missing from documentation index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    issues: list[dict[str, str]] = []
    for path in files_under(repo):
        relative = path.relative_to(repo)
        lowered = relative.as_posix().lower()
        if any(marker in lowered for marker in SCOPE_MARKERS):
            add(issues, "scope_leakage", relative, "non-MetricGAN family in path")
        if any(relative.name.endswith(suffix) for suffix in TRASH_SUFFIXES):
            add(issues, "trash_file", relative, "temporary/backup artifact")
        if relative.as_posix().startswith("code_and_documentation/runbooks/kingston_runtime/"):
            add(issues, "machine_orchestration", relative, "external-drive/server runbook")
        try:
            size = path.stat().st_size
        except OSError as exc:
            add(issues, "unreadable", relative, str(exc))
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or size > 8 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            add(issues, "unreadable", relative, str(exc))
            continue
        for rule, pattern in CONTENT_RULES.items():
            match = pattern.search(text)
            if match:
                add(issues, rule, relative, match.group(0)[:160])
        if relative.as_posix().startswith("code_and_documentation/"):
            match = OUT_OF_SCOPE_CODE.search(text)
            if match:
                add(issues, "out_of_scope_code", relative, match.group(0)[:160])

    check_architecture(repo, issues)
    check_docs(repo, issues)
    payload = {"repo": repo.as_posix(), "ok": not issues, "issue_count": len(issues), "issues": issues}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"project_guard: {'PASS' if not issues else 'FAIL'} ({len(issues)} issues)")
        for issue in issues[:200]:
            print(f"{issue['rule']}: {issue['path']}: {issue['detail']}")
        if len(issues) > 200:
            print(f"... {len(issues) - 200} additional issues omitted")
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
