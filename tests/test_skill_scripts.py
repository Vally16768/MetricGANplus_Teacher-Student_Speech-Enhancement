from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / ".agents" / "skills" / "manage-metricgan-research" / "scripts"
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
    "code_and_documentation/sebench/teacher_cache.py",
    "code_and_documentation/sebench/training.py",
    "code_and_documentation/configs/research_plan_voicebank_wb_nb.yaml",
    "code_and_documentation/reference/torch_pesq_0.1.2.json",
    "configs/voicebank_campaign.yaml",
    "scripts/bind_voicebank_manifests.py",
)


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


class SkillScriptTests(unittest.TestCase):
    def make_guard_fixture(self, repo: Path) -> None:
        for relative in ARCHITECTURE_SOURCES:
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# safe fixture\n", encoding="utf-8")
        docs = repo / ".agents" / "DOCUMENTATION_INDEX.md"
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text("`.agents/DOCUMENTATION_INDEX.md`\n", encoding="utf-8")
        baseline = repo / ".agents" / "state" / "architecture_sources.sha256"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for relative in ARCHITECTURE_SOURCES:
            digest = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            lines.append(f"{digest}  {relative}")
        baseline.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_project_guard_passes_safe_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            self.make_guard_fixture(repo)
            result = run(
                sys.executable,
                (SCRIPTS / "project_guard.py").as_posix(),
                "--repo",
                repo.as_posix(),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_project_guard_finds_private_path_and_scope_code(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            self.make_guard_fixture(repo)
            private = repo / "config.yaml"
            private.write_text(
                "dataset: /" + "home" + "/alice/private-data\n",
                encoding="utf-8",
            )
            models = repo / "code_and_documentation" / "sebench" / "models.py"
            models.write_text("class MPSENet: pass\n", encoding="utf-8")
            result = run(
                sys.executable,
                (SCRIPTS / "project_guard.py").as_posix(),
                "--repo",
                repo.as_posix(),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("personal_home", result.stdout)
            self.assertIn("out_of_scope_code", result.stdout)

    def test_run_contract_create_and_validate_planned(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            run("git", "init", "-q", repo.as_posix())
            run("git", "-C", repo.as_posix(), "config", "user.email", "test@example.invalid")
            run("git", "-C", repo.as_posix(), "config", "user.name", "Test")
            config = repo / "config.yaml"
            config.write_text("seed: 7\n", encoding="utf-8")
            run("git", "-C", repo.as_posix(), "add", "config.yaml")
            run("git", "-C", repo.as_posix(), "commit", "-qm", "fixture")
            create = run(
                sys.executable,
                (SCRIPTS / "run_contract.py").as_posix(),
                "create",
                "--repo",
                repo.as_posix(),
                "--run-id",
                "20260726-student-s-s7-test",
                "--config",
                config.as_posix(),
                "--command",
                "python repro.py train_stage1",
                "--seed",
                "7",
                "--purpose",
                "contract test",
            )
            self.assertEqual(create.returncode, 0, create.stderr)
            validate = run(
                sys.executable,
                (SCRIPTS / "run_contract.py").as_posix(),
                "validate",
                "--repo",
                repo.as_posix(),
                "--run-dir",
                "local/runs/20260726-student-s-s7-test",
                "--stage",
                "planned",
            )
            self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)

    def test_run_contract_refuses_dirty_repo(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            run("git", "init", "-q", repo.as_posix())
            run("git", "-C", repo.as_posix(), "config", "user.email", "test@example.invalid")
            run("git", "-C", repo.as_posix(), "config", "user.name", "Test")
            config = repo / "config.yaml"
            config.write_text("seed: 1\n", encoding="utf-8")
            run("git", "-C", repo.as_posix(), "add", "config.yaml")
            run("git", "-C", repo.as_posix(), "commit", "-qm", "fixture")
            (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            result = run(
                sys.executable,
                (SCRIPTS / "run_contract.py").as_posix(),
                "create",
                "--repo",
                repo.as_posix(),
                "--run-id",
                "20260726-dirty-s1-test",
                "--config",
                config.as_posix(),
                "--command",
                "python repro.py train_stage1",
                "--seed",
                "1",
                "--purpose",
                "dirty test",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dirty worktree", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
