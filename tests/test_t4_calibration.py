from __future__ import annotations

import sys
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


if __name__ == "__main__":
    unittest.main()
