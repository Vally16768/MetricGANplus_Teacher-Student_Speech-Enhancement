from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.losses import (  # noqa: E402
    SpeechBrainMetricDiscriminator,
    load_pesq_proxy_checkpoint,
)
from sebench.metricgan_alternating import (  # noqa: E402
    normalize_pesq,
    refresh_metricgan_discriminator,
)


class IdentityTeacher(torch.nn.Module):
    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        return noisy * 0.8


class AlternatingMetricGANTests(unittest.TestCase):
    def test_pesq_normalization_matches_official_recipe(self) -> None:
        self.assertAlmostEqual(normalize_pesq(-0.5), 0.0)
        self.assertAlmostEqual(normalize_pesq(4.5), 1.0)
        self.assertAlmostEqual(normalize_pesq(2.0), 0.5)

    def test_discriminator_matches_official_layer_contract_and_round_trip(
        self,
    ) -> None:
        discriminator = SpeechBrainMetricDiscriminator()
        self.assertEqual(discriminator.conv1.in_channels, 2)
        self.assertEqual(discriminator.conv1.out_channels, 15)
        self.assertEqual(tuple(discriminator.conv1.kernel_size), (5, 5))
        for layer in (
            discriminator.conv1,
            discriminator.conv2,
            discriminator.conv3,
            discriminator.conv4,
            discriminator.linear1,
            discriminator.linear2,
            discriminator.linear3,
        ):
            self.assertTrue(hasattr(layer, "weight_orig"))
        waveform = torch.randn(1, 8192)
        score = discriminator.normalized_score(waveform, waveform)
        self.assertEqual(tuple(score.shape), (1,))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "discriminator.pt"
            from sebench.losses import save_pesq_proxy_checkpoint

            save_pesq_proxy_checkpoint(checkpoint, discriminator)
            restored = load_pesq_proxy_checkpoint(checkpoint, freeze=False)
        self.assertIsInstance(restored, SpeechBrainMetricDiscriminator)
        self.assertTrue(all(parameter.requires_grad for parameter in restored.parameters()))

    def test_refresh_uses_local_generated_cache_without_copying_inputs(
        self,
    ) -> None:
        discriminator = SpeechBrainMetricDiscriminator(base_channels=2)
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        teacher = IdentityTeacher()
        waveform = torch.linspace(-0.5, 0.5, 8192)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.csv"
            manifest.write_text(
                "noisy,clean\n/external/noisy.wav,/external/clean.wav\n",
                encoding="utf-8",
            )
            replay = root / "local_replay"
            checkpoint = root / "metric_discriminator.pt"
            with (
                mock.patch(
                    "sebench.metricgan_alternating._aligned_inputs",
                    return_value=(waveform, waveform * 0.9),
                ),
                mock.patch(
                    "sebench.metricgan_alternating.pesq_score",
                    return_value=2.5,
                ),
            ):
                summary = refresh_metricgan_discriminator(
                    discriminator=discriminator,
                    optimizer=optimizer,
                    generator=teacher,
                    train_manifest=manifest,
                    replay_root=replay,
                    checkpoint_out=checkpoint,
                    epoch=1,
                    device="cpu",
                    max_rows=1,
                    history_portion=0.2,
                )
            self.assertEqual(
                summary["strategy"],
                "speechbrain_current_historical_current",
            )
            self.assertEqual(summary["current_record_count"], 1)
            self.assertEqual(summary["historical_record_count"], 1)
            self.assertTrue(checkpoint.is_file())
            index = json.loads(
                (replay / "epoch_0001" / "index.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(index["cache_inputs"])
            self.assertEqual(index["storage_dtype"], "float16")
            enhanced_files = list((replay / "epoch_0001" / "enhanced").glob("*.pt"))
            self.assertEqual(len(enhanced_files), 1)
            self.assertFalse((replay / "noisy").exists())
            self.assertFalse((replay / "clean").exists())
            cached = torch.load(
                enhanced_files[0],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(cached.dtype, torch.float16)
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in discriminator.parameters()
                )
            )


if __name__ == "__main__":
    unittest.main()
