from __future__ import annotations

import sys
import inspect
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code_and_documentation"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from sebench.review_evidence import (  # noqa: E402
    BandwidthLimitedTeacher,
    NoisyPassthrough,
    paired_bootstrap,
)
from sebench.training import evaluate_manifest  # noqa: E402


class ReviewEvidenceTests(unittest.TestCase):
    def test_evaluate_manifest_exposes_sample_metric_output(self) -> None:
        self.assertIn(
            "sample_metrics_out",
            inspect.signature(evaluate_manifest).parameters,
        )

    def test_noisy_passthrough_is_exact(self) -> None:
        waveform = torch.randn(2, 1, 800)
        self.assertTrue(torch.equal(NoisyPassthrough()(waveform), waveform))
        self.assertTrue(
            torch.equal(
                NoisyPassthrough().denoise_single(waveform),
                waveform,
            )
        )

    def test_bandwidth_limited_teacher_preserves_8k_shape(self) -> None:
        wrapper = BandwidthLimitedTeacher(torch.nn.Identity())
        waveform = torch.randn(2, 1, 803)
        observed = wrapper(waveform)
        waveform_single = waveform.squeeze(1)
        observed_single = wrapper.denoise_single(waveform_single)
        self.assertEqual(observed.shape, waveform.shape)
        self.assertEqual(observed_single.shape, waveform_single.shape)
        self.assertTrue(torch.isfinite(observed).all())
        self.assertTrue(torch.isfinite(observed_single).all())

    def test_bandwidth_limited_teacher_rejects_non_batched_single_input(self) -> None:
        wrapper = BandwidthLimitedTeacher(torch.nn.Identity())
        with self.assertRaisesRegex(ValueError, "batch, length"):
            wrapper.denoise_single(torch.randn(803))

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        left = np.asarray([2.0, 3.0, 4.0, 5.0])
        right = np.asarray([1.0, 2.0, 3.0, 4.0])
        first = paired_bootstrap(left, right, draws=2_000)
        second = paired_bootstrap(left, right, draws=2_000)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 1.0)
        self.assertEqual(first["ci95_low"], 1.0)
        self.assertEqual(first["ci95_high"], 1.0)


if __name__ == "__main__":
    unittest.main()
