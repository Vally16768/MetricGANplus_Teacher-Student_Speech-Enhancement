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
    CONVERGED_BASELINE_SCOPE,
    STUDENT_CONTINUATION_CELL_ORDER,
    STUDENT_CONTINUATION_SCOPE,
    _best_teacher,
    _effective_training,
    _portable_baseline_cell,
    _student_schedule,
    _teacher_cache_identity,
    _write_portable_history,
    audit_campaign_run,
    close_converged_baseline,
    monitor_campaign_run,
    sha256,
)


class CampaignAuditTests(unittest.TestCase):
    def make_run(
        self,
        root: Path,
        *,
        baseline: bool = False,
        continuation: bool = False,
    ) -> None:
        (root / "metrics").mkdir(parents=True)
        (root / "models").mkdir()
        (root / "provenance").mkdir()
        (root / "reports").mkdir()
        rows: list[dict[str, object]] = []
        cells = {}
        inventory = {}
        cell_order = (
            STUDENT_CONTINUATION_CELL_ORDER
            if continuation
            else (BASELINE_CELL_ORDER if baseline else CELL_ORDER)
        )
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
            cells[cell]["best_epoch"] = 0 if cell.startswith("T0-") else 20
            if continuation:
                cells[cell]["best_epoch"] = 25
                cells[cell]["stop_epoch"] = 30
                cells[cell]["stop_reason"] = "early_stopping"
                history = root / "cells" / cell / "training_history.csv"
                history.parent.mkdir(parents=True, exist_ok=True)
                with history.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["epoch", "val_select_pesq"],
                    )
                    writer.writeheader()
                    writer.writerow({"epoch": 20, "val_select_pesq": 2.4})
                    writer.writerow({"epoch": 25, "val_select_pesq": 2.5})
            model = root / "models" / f"{cell}.pt"
            model.write_bytes(cell.encode("utf-8"))
            cells[cell]["checkpoint_out"] = model.as_posix()
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
            "campaign_scope": (
                STUDENT_CONTINUATION_SCOPE
                if continuation
                else (
                    BASELINE_SCOPE
                    if baseline
                    else "teacher_improvement_two_stage"
                )
            ),
            "expected_cells": list(cell_order),
            "verification_only": True,
            "selected_teacher": "T0-WB-OFFICIAL",
            "cells": cells,
            "model_inventory": inventory,
            "canonical_metrics_csv": metrics_csv.as_posix(),
            "test_pesq_plot": plot.as_posix(),
            "report": report.as_posix(),
        }
        if continuation:
            sources = {}
            for cell in STUDENT_CONTINUATION_CELL_ORDER:
                state = root / "sources" / f"{cell}.state.pt"
                source_model = root / "sources" / f"{cell}.model.pt"
                state.parent.mkdir(exist_ok=True)
                state.write_bytes(f"{cell}-state".encode("utf-8"))
                source_model.write_bytes(f"{cell}-model".encode("utf-8"))
                sources[cell] = {
                    "epoch": 20,
                    "training_state": state.as_posix(),
                    "training_state_sha256": sha256(state),
                    "model": source_model.as_posix(),
                    "model_sha256": sha256(source_model),
                }
            summary["student_continuation_contract"] = {
                "passed": True,
                "students": list(STUDENT_CONTINUATION_CELL_ORDER),
                "max_epochs": 50,
                "schedule": {
                    "scheduler": "plateau",
                    "early_stop_patience": 8,
                    "lr_factor": 0.5,
                    "lr_patience": 2,
                    "min_lr": 1e-6,
                },
                "sources": sources,
            }
        elif baseline:
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
        (root / "provenance" / "config_resolved.yaml").write_text(
            "dataset:\n  name: VoiceBank+DEMAND\n",
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

    def test_audit_resolves_portable_run_relative_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_run(root, baseline=True)
            summary_path = root / "metrics" / "campaign_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["canonical_metrics_csv"] = "metrics/canonical_metrics.csv"
            summary["test_pesq_plot"] = "reports/plot.png"
            summary["report"] = "reports/report.md"
            for cell, item in summary["model_inventory"].items():
                item["path"] = f"models/{cell}.pt"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            result = audit_campaign_run(root)
        self.assertTrue(result["valid"], result["issues"])

    def test_portable_cell_removes_local_and_sample_paths(self) -> None:
        portable = _portable_baseline_cell(
            "S0-WB",
            {
                "run_name": "S0-WB",
                "checkpoint_out": "PRIVATE_ROOT/run/model.pt",
                "training_state_out": "PRIVATE_ROOT/run/state.pt",
                "teacher_cache_manifest": "PRIVATE_MOUNT/cache.csv",
                "history_csv": "PRIVATE_ROOT/history.csv",
                "continued_from": {
                    "epoch": 20,
                    "model": "PRIVATE_ROOT/old.pt",
                    "model_sha256": "abc",
                    "training_state": "PRIVATE_ROOT/state.pt",
                    "training_state_sha256": "def",
                },
                "test_metrics": {
                    "sample_paths": ["PRIVATE_ROOT/sample.wav"],
                    "pesq_mean": 3.0,
                },
            },
        )
        rendered = json.dumps(portable)
        self.assertNotIn("PRIVATE_ROOT", rendered)
        self.assertNotIn("PRIVATE_MOUNT", rendered)
        self.assertNotIn("teacher_cache_manifest", portable)
        self.assertNotIn("training_state_out", portable)
        self.assertEqual(portable["test_metrics"]["sample_paths"], [])
        self.assertEqual(portable["checkpoint_out"], "models/S0-WB.pt")
        self.assertEqual(
            portable["continued_from"]["training_state_sha256"],
            "def",
        )

    def test_portable_history_removes_machine_bound_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_csv = root / "source.csv"
            target_csv = root / "target.csv"
            source_csv.write_text(
                "epoch,loss,teacher_cache_manifest,train_manifest\n"
                "1,0.5,PRIVATE_CACHE,PRIVATE_MANIFEST\n",
                encoding="utf-8",
            )
            _write_portable_history(source_csv, target_csv)
            source_json = root / "source.json"
            target_json = root / "target.json"
            source_json.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "epoch": 1,
                                "loss": 0.5,
                                "teacher_cache_manifest": "PRIVATE_CACHE",
                                "train_manifest": "PRIVATE_MANIFEST",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            _write_portable_history(source_json, target_json)
            csv_text = target_csv.read_text(encoding="utf-8")
            json_payload = json.loads(target_json.read_text(encoding="utf-8"))
        self.assertEqual(csv_text, "epoch,loss\n1,0.5\n")
        self.assertEqual(json_payload, {"rows": [{"epoch": 1, "loss": 0.5}]})

    def test_student_continuation_reconciles_two_cells(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_run(root, continuation=True)
            result = audit_campaign_run(root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertEqual(result["campaign_scope"], STUDENT_CONTINUATION_SCOPE)
        self.assertEqual(result["cell_count"], 2)
        self.assertEqual(result["model_count"], 2)

    def test_converged_baseline_closure_binds_sources_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runs = Path(raw)
            baseline = runs / "baseline"
            continuation = runs / "continuation"
            self.make_run(baseline, baseline=True)
            self.make_run(continuation, continuation=True)
            continuation_summary_path = (
                continuation / "metrics" / "campaign_summary.json"
            )
            continuation_summary = json.loads(
                continuation_summary_path.read_text(encoding="utf-8")
            )
            contract = continuation_summary["student_continuation_contract"]
            contract["source_run_id"] = baseline.name
            baseline_summary = json.loads(
                (baseline / "metrics" / "campaign_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            for cell in STUDENT_CONTINUATION_CELL_ORDER:
                baseline_model = Path(
                    baseline_summary["model_inventory"][cell]["path"]
                )
                source_model = Path(contract["sources"][cell]["model"])
                source_model.write_bytes(baseline_model.read_bytes())
                contract["sources"][cell]["model_sha256"] = sha256(
                    source_model
                )
            continuation_summary_path.write_text(
                json.dumps(continuation_summary),
                encoding="utf-8",
            )
            result = close_converged_baseline(
                baseline_run_dir=baseline,
                continuation_run_dir=continuation,
                run_id="closure",
            )
            closure = runs / "closure"
            audit = audit_campaign_run(closure)
            summary = json.loads(
                (closure / "metrics" / "campaign_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertTrue(result["audit"]["valid"], result["audit"]["issues"])
        self.assertTrue(audit["valid"], audit["issues"])
        self.assertEqual(
            summary["campaign_scope"],
            CONVERGED_BASELINE_SCOPE,
        )
        self.assertEqual(
            set(summary["cells"]),
            set(BASELINE_CELL_ORDER),
        )

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
            "training": {"batch_size": 8, "student_epochs": 50},
            "pilot": {"batch_size": 4, "student_epochs": 3},
        }
        self.assertEqual(_effective_training(config, "pilot")["student_epochs"], 3)
        self.assertEqual(_effective_training(config, "full")["student_epochs"], 50)

    def test_student_full_schedule_allows_lr_recovery_before_early_stop(self) -> None:
        effective = {
            "student_lr_factor": 0.5,
            "student_lr_patience": 2,
            "student_min_lr": 1e-6,
            "student_early_stop_patience": 8,
        }
        self.assertEqual(
            _student_schedule(effective, mode="full"),
            {
                "early_stop_patience": 8,
                "lr_factor": 0.5,
                "lr_patience": 2,
                "min_lr": 1e-6,
            },
        )

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
