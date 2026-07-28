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
from sebench.t4_calibration import apply_uniform_mask_logit_bias  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
