from __future__ import annotations

import sys
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code_and_documentation"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from sebench.t3_perceptual import (  # noqa: E402
    DifferentiablePESQInspiredLoss,
    MultiResolutionSTFTLoss,
    T0LogMagnitudeAnchorLoss,
    T3TeacherObjective,
    TrueLengthSISDRLoss,
    calibrate_t3_gradient_weights,
)
from sebench.t3_support import (  # noqa: E402
    audit_t3_direction,
    audit_t3_identities,
    prepare_t3_identities,
)


def _fixture(
    *,
    samples: int = 6_000,
    seed: int = 7,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    clean = 0.03 * torch.randn(1, samples, generator=generator, device=device)
    teacher = clean + 0.005 * torch.randn(
        clean.shape,
        generator=generator,
        device=device,
    )
    candidate = teacher + 0.0005 * torch.randn(
        teacher.shape,
        generator=generator,
        device=device,
    )
    return clean, teacher, candidate


class T3PerceptualLossTests(unittest.TestCase):
    def test_pmsqe_is_explicitly_wb_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            DifferentiablePESQInspiredLoss(sample_rate=8_000)

    def test_pmsqe_short_signal_has_finite_gradient(self) -> None:
        clean, _, candidate = _fixture(samples=2_000)
        candidate.requires_grad_(True)
        loss = DifferentiablePESQInspiredLoss()(candidate, clean)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(candidate.grad)
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(float(candidate.grad.norm()), 0.0)

    def test_t3_identities_are_pair_and_clean_disjoint_from_t2(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train_manifest = root / "train.csv"
            teacher_manifest = root / "teacher.csv"
            metadata = root / "metadata.json"
            t2_support = root / "t2.json"
            train_rows = []
            teacher_rows = []
            for index in range(10):
                noisy = root / f"noisy-{index}.wav"
                clean = root / f"clean-{index}.wav"
                teacher = root / f"teacher-{index}.pt"
                noisy.write_bytes(b"fixture")
                clean.write_bytes(b"fixture")
                torch.save(torch.zeros(512, dtype=torch.float16), teacher)
                train_rows.append({"noisy": noisy.as_posix(), "clean": clean.as_posix()})
                teacher_rows.append(
                    {
                        "noisy": noisy.as_posix(),
                        "clean": clean.as_posix(),
                        "teacher_wav": teacher.as_posix(),
                    }
                )
            for path, fieldnames, rows in (
                (train_manifest, ["noisy", "clean"], train_rows),
                (
                    teacher_manifest,
                    ["noisy", "clean", "teacher_wav"],
                    teacher_rows,
                ),
            ):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            train_hash = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
            metadata.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "cache_inputs": False,
                        "storage_dtype": "float16",
                        "teacher_checkpoint_sha256": "a" * 64,
                        "train_manifest_sha256": train_hash,
                    }
                ),
                encoding="utf-8",
            )
            t2_support.write_text(
                json.dumps({"records": train_rows[:2]}),
                encoding="utf-8",
            )
            result = prepare_t3_identities(
                train_manifest=train_manifest,
                teacher_cache_manifest=teacher_manifest,
                teacher_cache_metadata=metadata,
                t2_support_paths=[t2_support],
                output_dir=root / "support",
                expected_teacher_sha256="a" * 64,
                train_rows=2,
                calibration_rows=2,
                audit_rows=2,
                seed=17,
            )
            audit = audit_t3_identities(result["identities_path"])
            self.assertTrue(audit["valid"], audit["issues"])
            self.assertEqual(result["counts"], {"train": 2, "calibration": 2, "audit": 2})
            excluded_clean = {row["clean"] for row in train_rows[:2]}
            self.assertFalse(
                excluded_clean
                & {str(row["clean"]) for row in result["records"]}
            )

    def test_t3_direction_gate_uses_only_audit_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            parents = []
            for parent_index in range(50):
                candidates = []
                for candidate_index, delta in enumerate((-0.02, -0.01, 0.01, 0.02)):
                    path = root / f"{parent_index}-{candidate_index}.pt"
                    torch.save(torch.zeros(8, dtype=torch.float16), path)
                    candidates.append(
                        {
                            "candidate": path.as_posix(),
                            "delta_pesq": delta,
                            "delta_pmsqe": -delta,
                        }
                    )
                parents.append(
                    {
                        "partition": "audit",
                        "estimated_input_snr_db": float(parent_index),
                        "candidates": candidates,
                    }
                )
            source = root / "candidates.json"
            source.write_text(
                json.dumps(
                    {
                        "status": "candidates_complete",
                        "parents": parents,
                    }
                ),
                encoding="utf-8",
            )
            weights = root / "weights.json"
            weights.write_text(
                json.dumps(
                    {
                        "status": "weights_frozen",
                        "median_gradient_norms": {
                            "supervised": 1.0,
                            "anchor": 1.0,
                            "pmsqe": 1.0,
                        },
                        "parameter_gradient_evidence": {
                            "all_finite": True,
                            "parameter_tensors_total": 21,
                            "parameter_tensors_with_gradient": 21,
                            "aggregate_l2_norm": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = audit_t3_direction(
                candidates_path=source,
                weights_path=weights,
                output_dir=root / "report",
            )
            self.assertTrue(result["valid"])
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["summaries"]["audit"]["eligible_pairs"],
                200,
            )
            self.assertEqual(
                result["summaries"]["audit"]["sign_agreement"],
                1.0,
            )

    def test_pmsqe_silent_reference_is_safe_and_neutral(self) -> None:
        clean = torch.zeros(1, 6_000)
        candidate = torch.zeros_like(clean, requires_grad=True)
        loss = DifferentiablePESQInspiredLoss()(candidate, clean)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertEqual(float(candidate.grad.norm()), 0.0)

    def test_true_lengths_make_padding_invariant(self) -> None:
        clean_a, teacher_a, candidate_a = _fixture(samples=4_000, seed=10)
        clean_b, teacher_b, candidate_b = _fixture(samples=6_000, seed=11)
        clean = torch.zeros(2, 7_000)
        teacher = torch.zeros_like(clean)
        candidate = torch.zeros_like(clean)
        clean[0, :4_000], clean[1, :6_000] = clean_a[0], clean_b[0]
        teacher[0, :4_000], teacher[1, :6_000] = teacher_a[0], teacher_b[0]
        candidate[0, :4_000], candidate[1, :6_000] = candidate_a[0], candidate_b[0]
        # Deliberately different garbage after the true end.
        candidate[0, 4_000:] = 0.75
        candidate[1, 6_000:] = -0.75
        lengths = torch.tensor([4_000, 6_000])
        losses = (
            MultiResolutionSTFTLoss(),
            TrueLengthSISDRLoss(),
            T0LogMagnitudeAnchorLoss(),
        )
        references = (clean, clean, teacher)
        individual_candidates = (candidate_a, candidate_b)
        individual_references = (
            (clean_a, clean_b),
            (clean_a, clean_b),
            (teacher_a, teacher_b),
        )
        for module, batch_reference, pair_references in zip(
            losses,
            references,
            individual_references,
            strict=True,
        ):
            batch_value = module(candidate, batch_reference, lengths=lengths)
            expected = torch.stack(
                [
                    module(individual_candidates[0], pair_references[0]),
                    module(individual_candidates[1], pair_references[1]),
                ]
            ).mean()
            self.assertTrue(
                torch.allclose(batch_value, expected, rtol=2e-5, atol=2e-6),
                (type(module).__name__, float(batch_value), float(expected)),
            )

    def test_pmsqe_is_joint_amplitude_scale_invariant(self) -> None:
        clean, _, candidate = _fixture()
        module = DifferentiablePESQInspiredLoss()
        original = module(candidate, clean)
        scaled = module(0.2 * candidate, 0.2 * clean)
        self.assertTrue(torch.allclose(original, scaled, rtol=2e-4, atol=2e-4))

    def test_e1_and_e2_are_matched_except_for_pmsqe(self) -> None:
        clean, teacher, candidate = _fixture()
        e1 = T3TeacherObjective(branch="E1-SUP", anchor_weight=0.5)
        e2 = T3TeacherObjective(
            branch="E2-PMSQE",
            anchor_weight=0.5,
            pmsqe_weight=0.01,
        )
        e1_values = e1(candidate, clean, teacher)
        e2_values = e2(candidate, clean, teacher)
        self.assertEqual(float(e1_values.pmsqe), 0.0)
        self.assertGreater(float(e2_values.pmsqe), 0.0)
        expected_delta = 0.01 * e2_values.pmsqe
        self.assertTrue(
            torch.allclose(e2_values.total - e1_values.total, expected_delta)
        )

    def test_gradient_calibration_enforces_component_norm_bounds(self) -> None:
        clean, teacher, candidate = _fixture()
        candidate.requires_grad_(True)
        result = calibrate_t3_gradient_weights(
            candidate=candidate,
            clean=clean,
            teacher_t0=teacher,
        )
        anchor_contribution = result.anchor_weight * result.anchor_norm
        pmsqe_contribution = result.pmsqe_weight * result.pmsqe_norm
        anchor_fraction = anchor_contribution / (
            result.supervised_norm + anchor_contribution
        )
        pmsqe_fraction = pmsqe_contribution / (
            result.supervised_norm + anchor_contribution + pmsqe_contribution
        )
        self.assertLessEqual(anchor_fraction, 0.50 + 1e-6)
        self.assertLessEqual(pmsqe_fraction, 0.10 + 1e-6)
        self.assertGreater(result.anchor_weight, 0.0)
        self.assertGreater(result.pmsqe_weight, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_amp_forward_backward_is_finite(self) -> None:
        clean, teacher, candidate = _fixture(device="cuda")
        candidate.requires_grad_(True)
        objective = T3TeacherObjective(
            branch="E2-PMSQE",
            anchor_weight=0.5,
            pmsqe_weight=0.01,
        ).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            values = objective(candidate, clean, teacher)
        values.total.backward()
        self.assertTrue(torch.isfinite(values.total))
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(float(candidate.grad.norm()), 0.0)
        calibration_candidate = (
            teacher + 0.0005 * torch.randn_like(teacher)
        ).requires_grad_(True)
        calibration = calibrate_t3_gradient_weights(
            candidate=calibration_candidate,
            clean=clean,
            teacher_t0=teacher,
        )
        self.assertGreater(calibration.anchor_weight, 0.0)
        self.assertGreater(calibration.pmsqe_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
