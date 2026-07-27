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
    def test_causal_max_wb_nb_aliases_match_recovered_architecture(self) -> None:
        wb = build_enhancer(
            "metricgan_plus_student_wb_causal_max",
            "small",
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            win_length=320,
        )
        nb = build_enhancer(
            "metricgan_plus_student_nb_causal_max",
            "small",
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
        )
        self.assertEqual(wb.model_config["sample_rate"], 16000)
        self.assertEqual(nb.model_config["sample_rate"], 8000)
        for model in (wb, nb):
            self.assertEqual(model.model_config["hidden_size"], 160)
            self.assertEqual(model.model_config["num_layers"], 3)
            self.assertEqual(model.model_config["linear_dims"][1], 224)
            self.assertEqual(model.model_config["lookahead_ms"], 16.0)
        self.assertEqual(sum(p.numel() for p in wb.parameters()), 604_386)
        self.assertEqual(sum(p.numel() for p in nb.parameters()), 514_018)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            build_enhancer(
                "metricgan_plus_student_nb_causal_max",
                "small",
                sample_rate=16000,
                n_fft=512,
                hop_length=160,
                win_length=320,
            )

    def test_causal_max_wb_nb_forward_backward(self) -> None:
        profiles = (
            ("metricgan_plus_student_wb_causal_max", 16000, 512, 160, 320, 1600),
            ("metricgan_plus_student_nb_causal_max", 8000, 256, 80, 160, 800),
        )
        for family, sample_rate, n_fft, hop, win, length in profiles:
            with self.subTest(family=family):
                model = build_enhancer(
                    family,
                    "small",
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                )
                waveform = torch.randn(2, 1, length)
                enhanced = model(waveform)
                self.assertEqual(tuple(enhanced.shape), tuple(waveform.shape))
                loss = enhanced.square().mean()
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
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
            "metricgan_plus_student_nb_causal_max",
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
                "metricgan_plus_student_nb_causal_max",
                "small",
            )
            restored, package = load_model_from_checkpoint(checkpoint)
        self.assertEqual(
            package["model_family"],
            "metricgan_plus_student_nb_causal_max",
        )
        self.assertEqual(restored.model_config["sample_rate"], 8000)
        self.assertEqual(restored.model_config["num_layers"], 3)

    def test_legacy_student_alias_remains_checkpoint_compatible(self) -> None:
        model = build_enhancer(
            "metricgan_plus_student_nb",
            "small",
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
        )
        self.assertEqual(model.model_config["hidden_size"], 96)
        self.assertEqual(model.model_config["num_layers"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy_student.pt"
            save_checkpoint_package(
                checkpoint,
                model,
                "metricgan_plus_student_nb",
                "small",
            )
            restored, package = load_model_from_checkpoint(checkpoint)
        self.assertEqual(package["model_family"], "metricgan_plus_student_nb")
        self.assertEqual(restored.model_config["hidden_size"], 96)

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
