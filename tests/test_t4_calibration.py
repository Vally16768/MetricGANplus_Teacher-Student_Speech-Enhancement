from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code_and_documentation"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from sebench.models import build_enhancer  # noqa: E402
from sebench.checkpoints import (  # noqa: E402
    load_model_from_checkpoint,
    save_checkpoint_package,
)
from sebench.t3_perceptual import T3LossBreakdown  # noqa: E402
from sebench.t4_calibration import apply_uniform_mask_logit_bias  # noqa: E402
from sebench.t4_microstep import (  # noqa: E402
    interpolate_state_dict,
    t4_microstep_loss,
)
from sebench.t5_zeroth_order import (  # noqa: E402
    apply_frequency_logit_curve,
    frequency_curve_from_knots,
    prepare_t5_support_manifests,
)
from sebench.t6_affine import apply_affine_logit_calibration  # noqa: E402
from sebench.t7_confidence import (  # noqa: E402
    confidence_candidate_grid,
    configure_confidence_calibration,
)
from sebench.t8_router import (  # noqa: E402
    configure_adaptive_router,
    fit_ridge_router,
)
from sebench.t9_multi_router import (  # noqa: E402
    configure_multi_action_router,
    prepare_t9_support_manifests,
)
from sebench.t10_risk_router import prepare_t10_calibration_manifest  # noqa: E402
from sebench.t11_penalty_router import penalize_ridges  # noqa: E402
from sebench.t12_rank_router import rank_policy_grid  # noqa: E402


class T4CalibrationTests(unittest.TestCase):
    def test_folded_bias_matches_mask_logit_variant(self) -> None:
        torch.manual_seed(4)
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        model.eval()
        noisy = 0.02 * torch.randn(1, 1, 4_000)
        with torch.no_grad():
            expected = model.forward_with_mask_logit_delta(noisy, -0.04)
            apply_uniform_mask_logit_bias(model, -0.04)
            observed = model(noisy)
        self.assertTrue(torch.allclose(observed, expected, atol=2e-6, rtol=1e-5))

    def test_bias_is_bounded(self) -> None:
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        with self.assertRaisesRegex(ValueError, "bounded"):
            apply_uniform_mask_logit_bias(model, -0.11)

    def test_microstep_interpolation_is_exact_and_bounded(self) -> None:
        base = {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(2)}
        proposal = {
            "weight": torch.tensor([5.0, -1.0]),
            "counter": torch.tensor(2),
        }
        observed = interpolate_state_dict(base, proposal, 0.25)
        self.assertTrue(torch.equal(observed["weight"], torch.tensor([2.0, 2.0])))
        self.assertEqual(int(observed["counter"]), 2)
        with self.assertRaisesRegex(ValueError, "alpha"):
            interpolate_state_dict(base, proposal, 1.01)

    def test_microstep_loss_makes_pmsqe_primary_with_constraints(self) -> None:
        breakdown = T3LossBreakdown(
            total=torch.tensor(99.0),
            mrstft=torch.tensor(2.0),
            sisdr=torch.tensor(3.0),
            anchor=torch.tensor(4.0),
            pmsqe=torch.tensor(5.0),
        )
        observed = t4_microstep_loss(
            breakdown,
            anchor_weight=0.5,
            pmsqe_weight=0.2,
            constraint_scale=0.1,
        )
        expected = 0.1 * (2.0 + 0.1 * 3.0 + 0.5 * 4.0) + 0.2 * 5.0
        self.assertAlmostEqual(float(observed), expected, places=6)

    def test_t5_uniform_curve_matches_scalar_bias_and_enforces_bounds(self) -> None:
        curve = frequency_curve_from_knots([-0.1] * 8)
        self.assertEqual(tuple(curve.shape), (257,))
        self.assertTrue(torch.allclose(curve, torch.full((257,), -0.1)))
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        before = model.mask_generator.linear2.bias.detach().clone()
        observed = apply_frequency_logit_curve(model, [-0.1] * 8)
        self.assertTrue(torch.allclose(observed, torch.full((257,), -0.1)))
        self.assertTrue(
            torch.allclose(
                model.mask_generator.linear2.bias.detach() - before,
                torch.full((257,), -0.1),
                atol=3e-8,
            )
        )
        with self.assertRaisesRegex(ValueError, "bounds"):
            frequency_curve_from_knots([-0.21] + [-0.1] * 7)

    def test_t5_support_is_train_only_and_disjoint(self) -> None:
        records = []
        for index in range(5):
            records.append(
                {
                    "partition": "train" if index < 4 else "calibration",
                    "token": f"pair-{index}",
                    "clean_token": f"clean-{index}",
                    "noisy": f"/dataset/noisy-{index}.wav",
                    "clean": f"/dataset/clean-{index}.wav",
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = root / "identities.json"
            identities.write_text(
                json.dumps({"records": records}),
                encoding="utf-8",
            )
            summary = prepare_t5_support_manifests(
                identities,
                root / "support",
                fit_count=2,
                calibration_count=2,
            )
            self.assertEqual(summary["fit"]["count"], 2)
            self.assertEqual(summary["calibration"]["count"], 2)
            self.assertFalse(
                set(summary["fit"]["tokens"])
                & set(summary["calibration"]["tokens"])
            )
            self.assertEqual(summary["pair_overlap"], 0)
            self.assertEqual(summary["clean_overlap"], 0)

    def test_t6_affine_folds_exact_scale_and_curve(self) -> None:
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        weight = model.mask_generator.linear2.weight.detach().clone()
        bias = model.mask_generator.linear2.bias.detach().clone()
        curve = apply_affine_logit_calibration(
            model,
            scale=1.2,
            coefficients=[-0.1] * 8,
        )
        self.assertTrue(
            torch.allclose(model.mask_generator.linear2.weight, 1.2 * weight)
        )
        self.assertTrue(
            torch.allclose(
                model.mask_generator.linear2.bias,
                1.2 * bias + curve,
                atol=3e-8,
            )
        )

    def test_t7_confidence_formula_and_disabled_parity(self) -> None:
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        logits = torch.tensor([-10.0, -4.0, 2.0])
        self.assertTrue(torch.equal(model.calibrate_mask_logits(logits), logits))
        configure_confidence_calibration(
            model,
            enabled=True,
            low=-0.4,
            high=0.05,
            threshold=-4.0,
            temperature=1.5,
        )
        gate = torch.sigmoid((logits + 4.0) / 1.5)
        expected = logits - 0.4 + 0.45 * gate
        self.assertTrue(
            torch.allclose(model.calibrate_mask_logits(logits), expected)
        )
        self.assertEqual(len(confidence_candidate_grid()), 24)

    def test_t7_checkpoint_roundtrip_preserves_calibration(self) -> None:
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        configure_confidence_calibration(
            model,
            enabled=True,
            low=-0.3,
            high=0.05,
            threshold=-2.0,
            temperature=1.5,
        )
        noisy = 0.02 * torch.randn(1, 1, 4_000)
        with torch.no_grad():
            expected = model(noisy)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "t7.pt"
            save_checkpoint_package(
                checkpoint,
                model,
                model_family="metricgan_plus_teacher_official_wb",
                variant="small",
            )
            observed_model, package = load_model_from_checkpoint(checkpoint)
            with torch.no_grad():
                observed = observed_model(noisy)
        self.assertTrue(torch.allclose(observed, expected, atol=2e-6, rtol=1e-5))
        self.assertTrue(
            package["model_config"]["confidence_calibration_enabled"]
        )
        self.assertEqual(
            package["model_config"]["confidence_calibration_threshold"], -2.0
        )

    def test_t8_ridge_recovers_clean_free_linear_direction(self) -> None:
        rng = np.random.default_rng(8)
        features = rng.normal(size=(80, 16))
        direction = np.linspace(-0.03, 0.03, 16)
        labels = features @ direction + 0.004
        ridge = fit_ridge_router(features, labels)
        predictions = (
            (features - np.asarray(ridge["feature_mean"]))
            / np.asarray(ridge["feature_scale"])
        ) @ np.asarray(ridge["weights"]) + ridge["bias"]
        self.assertGreater(np.corrcoef(predictions, labels)[0, 1], 0.999)
        self.assertIn(ridge["selected_lambda"], (0.001, 0.01, 0.1, 1.0, 10.0))

    def test_t8_router_selects_exact_base_or_candidate_and_roundtrips(self) -> None:
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        noisy = 0.02 * torch.randn(1, 1, 4_000)
        configure_confidence_calibration(
            model,
            enabled=False,
            low=-0.3,
            high=0.0,
            threshold=0.0,
            temperature=1.5,
        )
        with torch.no_grad():
            expected_base = model(noisy)
        configure_confidence_calibration(
            model,
            enabled=True,
            low=-0.3,
            high=0.0,
            threshold=0.0,
            temperature=1.5,
        )
        with torch.no_grad():
            expected_candidate = model(noisy)
        ridge = {
            "feature_mean": [0.0] * 16,
            "feature_scale": [1.0] * 16,
            "weights": [0.0] * 16,
            "bias": 1.0,
        }
        configure_adaptive_router(model, ridge=ridge, threshold=0.0)
        with torch.no_grad():
            observed_candidate = model(noisy)
        self.assertTrue(
            torch.allclose(
                observed_candidate,
                expected_candidate,
                atol=2e-6,
                rtol=1e-5,
            )
        )
        ridge["bias"] = -1.0
        configure_adaptive_router(model, ridge=ridge, threshold=0.0)
        with torch.no_grad():
            observed_base = model(noisy)
        self.assertTrue(
            torch.allclose(observed_base, expected_base, atol=2e-6, rtol=1e-5)
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "t8.pt"
            save_checkpoint_package(
                checkpoint,
                model,
                model_family="metricgan_plus_teacher_official_wb",
                variant="small",
            )
            reloaded, package = load_model_from_checkpoint(checkpoint)
            with torch.no_grad():
                roundtrip = reloaded(noisy)
        self.assertTrue(torch.allclose(roundtrip, observed_base, atol=2e-6, rtol=1e-5))
        self.assertTrue(package["model_config"]["adaptive_router_enabled"])

    def test_t9_router_selects_exact_action_and_roundtrips(self) -> None:
        model = build_enhancer(
            "metricgan_plus_teacher_official_wb",
            "small",
            initialize_from_official=False,
            n_fft=512,
            hop_length=256,
            win_length=512,
        )
        noisy = 0.02 * torch.randn(1, 1, 4_000)
        configure_confidence_calibration(
            model,
            enabled=True,
            low=-0.6,
            high=0.0,
            threshold=0.0,
            temperature=1.5,
        )
        with torch.no_grad():
            expected = model(noisy)
        ridges = []
        for bias in (-1.0, -0.5, 1.0, 0.0):
            ridges.append(
                {
                    "feature_mean": [0.0] * 16,
                    "feature_scale": [1.0] * 16,
                    "weights": [0.0] * 16,
                    "bias": bias,
                }
            )
        configure_multi_action_router(
            model,
            ridges=ridges,
            threshold=0.0,
        )
        with torch.no_grad():
            observed = model(noisy)
        self.assertTrue(torch.allclose(observed, expected, atol=2e-6, rtol=1e-5))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "t9.pt"
            save_checkpoint_package(
                checkpoint,
                model,
                model_family="metricgan_plus_teacher_official_wb",
                variant="small",
            )
            reloaded, package = load_model_from_checkpoint(checkpoint)
            with torch.no_grad():
                roundtrip = reloaded(noisy)
        self.assertTrue(torch.allclose(roundtrip, observed, atol=2e-6, rtol=1e-5))
        self.assertTrue(package["model_config"]["multi_router_enabled"])
        self.assertEqual(
            package["model_config"]["multi_router_lows"],
            [-0.2, -0.4, -0.6, -0.8],
        )

    def test_t9_support_is_partition_and_clean_disjoint(self) -> None:
        records = []
        for partition, count, offset in (
            ("train", 840, 0),
            ("calibration", 130, 10_000),
        ):
            for index in range(count):
                token = f"{partition}-{index}"
                records.append(
                    {
                        "partition": partition,
                        "token": token,
                        "clean_token": f"clean-{offset + index}",
                        "noisy": f"/dataset/{token}.wav",
                        "clean": f"/dataset/clean-{offset + index}.wav",
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = root / "identities.json"
            identities.write_text(
                json.dumps({"records": records}),
                encoding="utf-8",
            )
            support = prepare_t9_support_manifests(
                identities,
                root / "support",
            )
        self.assertEqual(support["fit"]["count"], 256)
        self.assertEqual(support["calibration"]["count"], 128)
        self.assertEqual(support["pair_overlap"], 0)
        self.assertEqual(support["clean_overlap"], 0)

    def test_t10_support_is_fresh_audit_partition(self) -> None:
        records = []
        for partition, count, offset in (
            ("train", 4, 0),
            ("calibration", 4, 100),
            ("audit", 4, 200),
        ):
            for index in range(count):
                records.append(
                    {
                        "partition": partition,
                        "token": f"{partition}-{index}",
                        "clean_token": f"clean-{offset + index}",
                        "noisy": f"/dataset/{partition}-{index}.wav",
                        "clean": f"/dataset/clean-{offset + index}.wav",
                    }
                )
        t9 = {
            "support": {
                "fit": {
                    "tokens": ["train-0"],
                    "clean_tokens": ["clean-0"],
                },
                "calibration": {
                    "tokens": ["calibration-0"],
                    "clean_tokens": ["clean-100"],
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = root / "identities.json"
            identities.write_text(
                json.dumps({"records": records}),
                encoding="utf-8",
            )
            support = prepare_t10_calibration_manifest(
                identities,
                t9,
                root / "support",
                calibration_count=4,
            )
        self.assertEqual(support["source_partition"], "audit")
        self.assertEqual(support["count"], 4)
        self.assertEqual(support["t9_pair_overlap"], 0)
        self.assertEqual(support["t9_clean_overlap"], 0)

    def test_t11_penalty_favors_milder_actions_without_changing_weights(self) -> None:
        ridges = [
            {
                "bias": 0.02,
                "weights": [float(index)] * 16,
                "feature_mean": [0.0] * 16,
                "feature_scale": [1.0] * 16,
            }
            for index in range(4)
        ]
        adjusted = penalize_ridges(
            ridges,
            (-0.2, -0.4, -0.6, -0.8),
            0.02,
        )
        self.assertGreater(adjusted[0]["bias"], adjusted[3]["bias"])
        self.assertEqual(adjusted[2]["weights"], ridges[2]["weights"])
        self.assertEqual(ridges[3]["bias"], 0.02)

    def test_t12_rank_grid_selects_only_guardrail_safe_policy(self) -> None:
        records = []
        for _ in range(2):
            records.append(
                {
                    "base": {"pesq": 2.0, "stoi": 0.9, "sisdr": 5.0},
                    "actions": [
                        {"pesq": 2.02, "stoi": 0.899, "sisdr": 4.8},
                        {"pesq": 2.02, "stoi": 0.899, "sisdr": 4.8},
                        {"pesq": 2.025, "stoi": 0.898, "sisdr": 4.75},
                        {"pesq": 2.04, "stoi": 0.897, "sisdr": 4.5},
                    ],
                }
            )
        predictions = np.asarray([[0.02, 0.02, 0.025, 0.04]] * 2)
        candidates, selected = rank_policy_grid(
            records,
            predictions,
            lows=(-0.2, -0.4, -0.6, -0.8),
            penalties=(0.0, 0.05),
            thresholds=(0.0,),
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(selected["eligible"])
        self.assertEqual(selected["penalty"], 0.05)
        self.assertGreaterEqual(selected["deltas"]["pesq_mean"], 0.01)
        self.assertGreaterEqual(selected["deltas"]["sisdr_mean"], -0.25)


if __name__ == "__main__":
    unittest.main()
