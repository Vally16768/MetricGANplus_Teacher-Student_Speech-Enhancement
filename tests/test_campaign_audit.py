from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from campaign import (
    BASELINE_CELL_ORDER,
    BASELINE_SCOPE,
    CELL_ORDER,
    _best_teacher,
    _effective_training,
    _teacher_cache_identity,
    audit_campaign_run,
    monitor_campaign_run,
    sha256,
)


class CampaignAuditTests(unittest.TestCase):
    def make_run(self, root: Path, *, baseline: bool = False) -> None:
        (root / "metrics").mkdir(parents=True)
        (root / "models").mkdir()
        (root / "provenance").mkdir()
        (root / "reports").mkdir()
        rows: list[dict[str, object]] = []
        cells = {}
        inventory = {}
        cell_order = BASELINE_CELL_ORDER if baseline else CELL_ORDER
        for cell in cell_order:
            bandwidth = "nb" if "-NB" in cell else "wb"
            sample_rate = 8000 if bandwidth == "nb" else 16000
            sample = root / "samples" / f"{cell}.wav"
            sample.parent.mkdir(exist_ok=True)
            sample.write_bytes(b"RIFF-fixture")
            split_payloads = {}
            for split_key, split_name in (
                ("val_rank_metrics", "val_rank"),
                ("val_select_metrics", "val_select"),
                ("test_metrics", "test"),
            ):
                payload = {
                    "bandwidth": bandwidth,
                    "reference_bandwidth": bandwidth,
                    "sample_rate": sample_rate,
                    "pesq_mode": bandwidth,
                    "count": 1,
                    "pesq_mean": 2.5,
                    "sample_paths": [sample.as_posix()] if split_name == "test" else [],
                }
                split_payloads[split_key] = payload
                for metric, value in payload.items():
                    if metric != "sample_paths":
                        rows.append(
                            {
                                "cell": cell,
                                "bandwidth": bandwidth,
                                "split": split_name,
                                "metric": metric,
                                "value": value,
                            }
                        )
            cells[cell] = split_payloads
            model = root / "models" / f"{cell}.pt"
            model.write_bytes(cell.encode("utf-8"))
            inventory[cell] = {
                "path": model.as_posix(),
                "bytes": model.stat().st_size,
                "sha256": sha256(model),
            }

        metrics_csv = root / "metrics" / "canonical_metrics.csv"
        with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["cell", "bandwidth", "split", "metric", "value"],
            )
            writer.writeheader()
            writer.writerows(rows)
        report = root / "reports" / "report.md"
        report.write_text("# fixture\n", encoding="utf-8")
        plot = root / "reports" / "plot.png"
        plot.write_bytes(b"PNG-fixture")
        summary = {
            "campaign_scope": BASELINE_SCOPE if baseline else "teacher_improvement_two_stage",
            "expected_cells": list(cell_order),
            "verification_only": True,
            "selected_teacher": "T0-WB-OFFICIAL",
            "cells": cells,
            "model_inventory": inventory,
            "canonical_metrics_csv": metrics_csv.as_posix(),
            "test_pesq_plot": plot.as_posix(),
            "report": report.as_posix(),
        }
        if baseline:
            summary["baseline_contract"] = {
                "passed": True,
                "teacher": "T0-WB-OFFICIAL",
                "students": ["S0-WB", "S0-NB"],
                "teacher_checkpoint_sha256": inventory[
                    "T0-WB-OFFICIAL"
                ]["sha256"],
            }
        else:
            summary["teacher_promotion_gate"] = {
                "passed": False,
                "verification_override": True,
            }
        (root / "metrics" / "campaign_summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        (root / "status.json").write_text(
            json.dumps({"status": "smoke-passed", "valid_for_promotion": False}),
            encoding="utf-8",
        )
        (root / "provenance" / "provenance.json").write_text(
            json.dumps({"run_id": "fixture", "verification_only": True}),
            encoding="utf-8",
        )

    def test_complete_run_reconciles(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_run(root)
            result = audit_campaign_run(root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertEqual(result["cell_count"], 7)
        self.assertEqual(result["model_count"], 7)

    def test_official_baseline_run_reconciles_three_cells(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_run(root, baseline=True)
            result = audit_campaign_run(root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertEqual(result["campaign_scope"], BASELINE_SCOPE)
        self.assertEqual(result["cell_count"], 3)
        self.assertEqual(result["model_count"], 3)

    def test_model_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_run(root)
            (root / "models" / "T1-WB-BASE.pt").write_bytes(b"tampered")
            result = audit_campaign_run(root)
        self.assertFalse(result["valid"])
        self.assertIn("model size mismatch: T1-WB-BASE", result["issues"])
        self.assertIn("model hash mismatch: T1-WB-BASE", result["issues"])

    def test_pilot_overrides_full_training_without_changing_full(self) -> None:
        config = {
            "training": {"batch_size": 8, "student_epochs": 20},
            "pilot": {"batch_size": 4, "student_epochs": 3},
        }
        self.assertEqual(_effective_training(config, "pilot")["student_epochs"], 3)
        self.assertEqual(_effective_training(config, "full")["student_epochs"], 20)

    def test_failed_verification_gate_keeps_official_teacher_downstream(self) -> None:
        metrics = {
            "pesq_mean": 3.0,
            "stoi_mean": 0.93,
            "sisdr_mean": 9.0,
            "delta_snr_mean": 0.0,
        }
        official = {
            "checkpoint_out": "official.pt",
            "val_select_metrics": metrics,
        }
        candidate = {
            "checkpoint_out": "candidate.pt",
            "val_select_metrics": dict(metrics),
        }
        config = {
            "training": {
                "teacher_min_pesq_gain": 0.01,
                "teacher_max_stoi_drop": 0.002,
                "teacher_max_sisdr_drop": 0.25,
            }
        }
        name, summary, gate = _best_teacher(
            official,
            candidate,
            candidate,
            config=config,
            verification_only=True,
        )
        self.assertEqual(name, "T0-WB-OFFICIAL")
        self.assertIs(summary, official)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["selected_candidate"], "T1-WB-BASE")
        self.assertEqual(gate["downstream_teacher"], "T0-WB-OFFICIAL")

    def test_teacher_cache_identity_ignores_stage_label_and_tracks_contract(self) -> None:
        key, contract = _teacher_cache_identity(
            checkpoint_hash="a" * 64,
            manifest_hash="b" * 64,
            cache_config={"cache_inputs": False, "storage_dtype": "float16"},
        )
        same_key, _ = _teacher_cache_identity(
            checkpoint_hash="a" * 64,
            manifest_hash="b" * 64,
            cache_config={"cache_inputs": False, "storage_dtype": "float16"},
        )
        changed_key, _ = _teacher_cache_identity(
            checkpoint_hash="a" * 64,
            manifest_hash="b" * 64,
            cache_config={"cache_inputs": False, "storage_dtype": "float32"},
        )
        self.assertEqual(key, same_key)
        self.assertNotEqual(key, changed_key)
        self.assertEqual(contract["targets"]["nb"]["sample_rate"], 8000)

    def test_monitor_reads_campaign_and_cell_progress(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "tracking").mkdir(parents=True)
            (root / "cells" / "T-WB-BASE").mkdir(parents=True)
            (root / "status.json").write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )
            (root / "tracking" / "campaign_progress.json").write_text(
                json.dumps(
                    {
                        "current_stage": "T-WB-BASE",
                        "completed_stage_count": 2,
                        "stages": {"T-WB-BASE": {"status": "running"}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "cells" / "T-WB-BASE" / "progress.json").write_text(
                json.dumps({"epoch": 1, "global_step": 10}),
                encoding="utf-8",
            )
            result = monitor_campaign_run(root)
        self.assertEqual(result["current_stage"], "T-WB-BASE")
        self.assertEqual(result["completed_stage_count"], 2)
        self.assertEqual(result["cells"]["T-WB-BASE"]["progress"]["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
