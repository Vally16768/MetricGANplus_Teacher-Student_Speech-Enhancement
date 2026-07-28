from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code_and_documentation"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from sebench.models import build_enhancer  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
