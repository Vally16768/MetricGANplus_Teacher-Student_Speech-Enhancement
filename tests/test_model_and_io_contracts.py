from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package  # noqa: E402
from sebench.contracts import validate_io_boundaries  # noqa: E402
from sebench.losses import (  # noqa: E402
    PESQProxyRegressor,
    load_pesq_proxy_checkpoint,
    save_pesq_proxy_checkpoint,
)
from sebench.models import build_enhancer, build_metricgan_causal_lite  # noqa: E402


class IOContractTests(unittest.TestCase):
    def test_output_outside_dataset_is_allowed(self) -> None:
        validate_io_boundaries(
            {
                "paths": {"dataset_root": "/data/input", "output_root": "/work/output"},
                "dataset": {},
                "teacher_cache": {"out_dir": "/work/output/teacher_cache"},
            }
        )

    def test_output_inside_dataset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_io_boundaries(
                {
                    "paths": {
                        "dataset_root": "/data/input",
                        "output_root": "/data/input/generated",
                    },
                    "dataset": {},
                }
            )


class StudentArchitectureTests(unittest.TestCase):
    def test_explicit_wb_nb_aliases_enforce_sample_rate(self) -> None:
        wb = build_enhancer(
            "metricgan_plus_student_wb",
            "small",
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            win_length=320,
        )
        nb = build_enhancer(
            "metricgan_plus_student_nb",
            "small",
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
        )
        self.assertEqual(wb.model_config["sample_rate"], 16000)
        self.assertEqual(nb.model_config["sample_rate"], 8000)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            build_enhancer(
                "metricgan_plus_student_nb",
                "small",
                sample_rate=16000,
                n_fft=512,
                hop_length=160,
                win_length=320,
            )

    def test_causal_s_forward_preserves_shape(self) -> None:
        model = build_metricgan_causal_lite(
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
            family="metricgan_plus_native8k_causal_s",
            qat=False,
        ).eval()
        waveform = torch.zeros(1, 1, 1600)
        with torch.no_grad():
            enhanced = model(waveform)
        self.assertEqual(tuple(enhanced.shape), tuple(waveform.shape))
        self.assertFalse(model.model_config["non_causal"])
        self.assertEqual(model.model_config["rnn_type"], "gru")

    def test_qat_flag_is_recorded(self) -> None:
        model = build_metricgan_causal_lite(
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
            family="metricgan_plus_native8k_causal_s",
            qat=True,
        )
        self.assertTrue(model.qat)
        self.assertTrue(model.model_config["qat"])

    def test_safe_checkpoint_round_trip(self) -> None:
        model = build_enhancer(
            "metricgan_plus_student_nb",
            "small",
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "student.pt"
            save_checkpoint_package(
                checkpoint,
                model,
                "metricgan_plus_student_nb",
                "small",
            )
            restored, package = load_model_from_checkpoint(checkpoint)
        self.assertEqual(package["model_family"], "metricgan_plus_student_nb")
        self.assertEqual(restored.model_config["sample_rate"], 8000)

    def test_safe_metric_proxy_round_trip(self) -> None:
        proxy = PESQProxyRegressor(
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
            bandwidth="nb",
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "proxy.pt"
            save_pesq_proxy_checkpoint(checkpoint, proxy)
            restored = load_pesq_proxy_checkpoint(checkpoint)
        self.assertEqual(restored.bandwidth, "nb")
        self.assertEqual(restored.sample_rate, 8000)


if __name__ == "__main__":
    unittest.main()
