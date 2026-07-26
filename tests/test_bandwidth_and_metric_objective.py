from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from metrics import pesq as pesq_module  # noqa: E402
from sebench.bandwidth import resolve_bandwidth  # noqa: E402
from sebench.contracts import validate_bandwidth_contract  # noqa: E402
from sebench.erb import (  # noqa: E402
    frontend_defaults_for_sample_rate,
    waveform_to_erb_mask,
)
from sebench.losses import CompositeEnhancementLoss, MetricGANGeneratorObjective  # noqa: E402
from sebench.research_plan import validate_research_plan  # noqa: E402
from sebench.runtime import require_shared_venv, require_training_cuda  # noqa: E402
from sebench.training import _normalize_runtime_devices  # noqa: E402


class ConstantProxy(torch.nn.Module):
    def forward(
        self,
        source: torch.Tensor,
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        del source, reference
        return candidate.mean(dim=(-1, -2)) + 3.0


class BandwidthContractTests(unittest.TestCase):
    def test_profiles_reject_cross_band_sample_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "mismatch"):
            resolve_bandwidth("nb", sample_rate=16000)

    def test_frontend_contract_is_explicit(self) -> None:
        profile = validate_bandwidth_contract(
            {
                "training": {
                    "bandwidth": "wb",
                    "sample_rate": 16000,
                    "n_fft": 512,
                    "hop_length": 160,
                    "win_length": 320,
                }
            }
        )
        self.assertEqual(profile["pesq_mode"], "wb")

    def test_canonical_plan_has_wb_teacher_and_two_students(self) -> None:
        plan_path = (
            CODE_ROOT / "configs" / "research_plan_voicebank_wb_nb.yaml"
        )
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        result = validate_research_plan(plan)
        self.assertTrue(result["valid"])
        self.assertEqual(result["teacher"]["name"], "wb")
        self.assertEqual(set(result["students"]), {"wb", "nb"})

    def test_pesq_receives_the_declared_mode(self) -> None:
        reference = np.zeros(160, dtype=np.float32)
        degraded = np.zeros(160, dtype=np.float32)
        with mock.patch.object(pesq_module, "_pesq", return_value=3.5) as metric:
            score = pesq_module.pesq_score(
                reference,
                degraded,
                8000,
                bandwidth="nb",
            )
        self.assertEqual(score, 3.5)
        self.assertEqual(metric.call_args.args[-1], "nb")

    def test_erb_frontend_uses_bandwidth_matched_defaults(self) -> None:
        self.assertEqual(frontend_defaults_for_sample_rate(16000), (512, 160, 320))
        self.assertEqual(frontend_defaults_for_sample_rate(8000), (256, 80, 160))

    def test_erb_mask_is_finite_for_wb_and_nb(self) -> None:
        for sample_rate, length in ((16000, 1600), (8000, 800)):
            noisy = torch.randn(2, 1, length)
            enhanced = noisy * 0.75
            mask = waveform_to_erb_mask(
                noisy,
                enhanced,
                erb_bands=16,
                sample_rate=sample_rate,
            )
            self.assertEqual(mask.shape[0:2], (2, 16))
            self.assertTrue(torch.isfinite(mask).all())
            self.assertTrue(torch.all(mask >= 0.0))
            self.assertTrue(torch.all(mask <= 2.0))


class MetricObjectiveTests(unittest.TestCase):
    def test_generator_objective_backpropagates_to_candidate_only(self) -> None:
        proxy = ConstantProxy()
        objective = MetricGANGeneratorObjective(proxy)
        candidate = torch.ones(2, 1, 32, requires_grad=True)
        reference = torch.zeros_like(candidate)
        loss, prediction = objective(candidate, reference)
        loss.backward()
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertIsNotNone(candidate.grad)

    def test_student_metric_recipe_adds_proxy_gradient(self) -> None:
        loss_fn = CompositeEnhancementLoss(
            "D1_PESQ",
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
            erb_bands=8,
            pesq_proxy=ConstantProxy(),
            metric_proxy_weight=0.1,
        )
        enhanced = torch.randn(1, 1, 1024, requires_grad=True)
        clean = torch.randn_like(enhanced)
        noisy = torch.randn_like(enhanced)
        teacher_wav = torch.randn_like(enhanced)
        teacher_mask = torch.rand(1, 8, 13)
        breakdown = loss_fn(
            enhanced,
            clean,
            noisy,
            epoch=1,
            total_epochs=1,
            teacher_wav=teacher_wav,
            teacher_mask_erb=teacher_mask,
        )
        breakdown.total.backward()
        self.assertIsNotNone(enhanced.grad)
        self.assertNotEqual(float(breakdown.predicted_pesq), 0.0)


class RuntimeContractTests(unittest.TestCase):
    def test_unindexed_cuda_is_normalized_to_current_gpu(self) -> None:
        with (
            mock.patch(
                "sebench.training.require_cuda_device",
                return_value="cuda",
            ),
            mock.patch(
                "sebench.training.torch.cuda.current_device",
                return_value=0,
            ),
        ):
            device, gpu_ids = _normalize_runtime_devices("cuda", None)
        self.assertEqual(device, "cuda:0")
        self.assertEqual(gpu_ids, [0])

    def test_shared_venv_contract_uses_resolved_prefix(self) -> None:
        expected = Path(sys.prefix).resolve()
        with mock.patch.dict(
            "os.environ",
            {"METRICGAN_SHARED_VENV": expected.as_posix()},
            clear=False,
        ):
            self.assertEqual(require_shared_venv("/unused"), expected)

    def test_shared_venv_contract_rejects_different_prefix(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"METRICGAN_SHARED_VENV": "/definitely/not/the/active/venv"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "shared project"):
                require_shared_venv("/unused")

    def test_training_rejects_cpu_even_when_cpu_is_requested(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "GPU-only"):
            require_training_cuda("cpu")

    def test_training_auto_rejects_machine_without_cuda(self) -> None:
        with mock.patch("sebench.runtime.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "GPU-only"):
                require_training_cuda("auto")


if __name__ == "__main__":
    unittest.main()
