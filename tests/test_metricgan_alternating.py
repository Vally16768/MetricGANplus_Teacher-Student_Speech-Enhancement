from __future__ import annotations

import json
import sys
import tempfile
import unittest
import csv
import hashlib
from pathlib import Path
from unittest import mock

import torch
from speechbrain.lobes.models.MetricGAN import MetricDiscriminator
from speechbrain.processing.features import STFT, spectral_magnitude


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.losses import (  # noqa: E402
    SpeechBrainMetricDiscriminator,
    load_pesq_proxy_checkpoint,
)
from sebench.metricgan_alternating import (  # noqa: E402
    evaluate_calibration_gate,
    normalize_pesq,
    refresh_metricgan_discriminator,
)
from sebench.metric_proxy_training import build_proxy_records  # noqa: E402
from sebench.metricgan_d2 import audit_d2_support, prepare_d2_support  # noqa: E402


class IdentityTeacher(torch.nn.Module):
    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        return noisy * 0.8


class AlternatingMetricGANTests(unittest.TestCase):
    def test_pesq_normalization_matches_official_recipe(self) -> None:
        self.assertAlmostEqual(normalize_pesq(-0.5), 0.0)
        self.assertAlmostEqual(normalize_pesq(4.5), 1.0)
        self.assertAlmostEqual(normalize_pesq(2.0), 0.5)

    def test_clean_discriminator_label_is_exact_official_target(self) -> None:
        waveform = torch.linspace(-0.5, 0.5, 8192)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "pairs.csv"
            manifest.write_text(
                "noisy,clean\n/external/noisy.wav,/external/clean.wav\n",
                encoding="utf-8",
            )
            with (
                mock.patch(
                    "sebench.metric_proxy_training.load_mono_audio",
                    return_value=(waveform, 16_000),
                ),
                mock.patch(
                    "sebench.metric_proxy_training.pesq_score",
                    return_value=2.0,
                ) as metric,
            ):
                payload = build_proxy_records(
                    train_manifest=manifest,
                    validation_manifest=manifest,
                    output_dir=root / "records",
                    bandwidth="wb",
                    candidate_teacher_checkpoint=None,
                    max_train_rows=1,
                    max_validation_rows=1,
                    device="cpu",
                )
            clean_records = [
                record
                for record in payload["records"]
                if record["source"] == "clean"
            ]
            self.assertEqual(len(clean_records), 2)
            self.assertTrue(
                all(record["pesq"] == 4.5 for record in clean_records)
            )
            self.assertEqual(metric.call_count, 8)

    def test_current_output_calibration_gate_is_explicit(self) -> None:
        calibration = {
            "record_count": 100,
            "normalized_mae": 0.05,
            "pearson": 0.85,
            "spearman": 0.82,
            "prediction_std": 0.2,
            "prediction_min": 1.5,
            "prediction_max": 3.5,
            "target_min": 1.4,
            "target_max": 3.6,
        }
        gate = evaluate_calibration_gate(
            calibration,
            min_records=100,
            max_normalized_mae=0.06,
            min_pearson=0.8,
            min_spearman=0.8,
        )
        self.assertTrue(gate["passed"], gate)
        calibration["prediction_std"] = 0.0
        failed = evaluate_calibration_gate(
            calibration,
            min_records=100,
            max_normalized_mae=0.06,
            min_pearson=0.8,
            min_spearman=0.8,
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["prediction_variance"])

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

    def test_discriminator_frontend_matches_pinned_speechbrain_recipe(
        self,
    ) -> None:
        torch.manual_seed(17)
        candidate = torch.randn(2, 16_000)
        local = SpeechBrainMetricDiscriminator()
        official_stft = STFT(
            sample_rate=16_000,
            win_length=32,
            hop_length=16,
            n_fft=512,
            window_fn=torch.hamming_window,
        )
        official_features = torch.log1p(
            spectral_magnitude(official_stft(candidate), power=0.5)
        )
        torch.testing.assert_close(
            local._features(candidate),
            official_features,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_discriminator_output_matches_pinned_speechbrain_model(
        self,
    ) -> None:
        torch.manual_seed(23)
        official = MetricDiscriminator()
        local = SpeechBrainMetricDiscriminator()
        renamed_state = {}
        for key, value in official.state_dict().items():
            local_key = key
            for official_prefix, local_prefix in (
                ("BN.", "batch_norm."),
                ("Linear1.", "linear1."),
                ("Linear2.", "linear2."),
                ("Linear3.", "linear3."),
            ):
                if local_key.startswith(official_prefix):
                    local_key = local_prefix + local_key[len(official_prefix) :]
                    break
            renamed_state[local_key] = value
        local.load_state_dict(renamed_state, strict=True)
        official.eval()
        local.eval()

        candidate = torch.randn(1, 16_000)
        reference = torch.randn(1, 16_000)
        official_stft = STFT(
            sample_rate=16_000,
            win_length=32,
            hop_length=16,
            n_fft=512,
            window_fn=torch.hamming_window,
        )

        def features(waveform: torch.Tensor) -> torch.Tensor:
            return torch.log1p(
                spectral_magnitude(official_stft(waveform), power=0.5)
            )

        official_input = torch.stack(
            [features(candidate), features(reference)],
            dim=1,
        )
        with torch.inference_mode():
            expected = official(official_input).squeeze(-1)
            observed = local.normalized_score(candidate, reference)
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)

    def test_discriminator_eval_is_batch_invariant_at_true_length(self) -> None:
        torch.manual_seed(29)
        discriminator = SpeechBrainMetricDiscriminator().eval()
        candidates = torch.randn(2, 12_000)
        references = torch.randn(2, 12_000)
        lengths = torch.tensor([12_000, 9_000])
        candidates[1, 9_000:] = 17.0
        references[1, 9_000:] = -19.0
        with torch.inference_mode():
            true_length_scores = discriminator.normalized_score(
                candidates,
                references,
                lengths=lengths,
            )
            separate = torch.cat(
                [
                    discriminator.normalized_score(
                        candidates[index : index + 1, : int(lengths[index])],
                        references[index : index + 1, : int(lengths[index])],
                    )
                    for index in range(2)
                ]
            )
        torch.testing.assert_close(
            true_length_scores,
            separate,
            rtol=1e-5,
            atol=1e-6,
        )

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
                mock.patch(
                    "sebench.metricgan_alternating._update_discriminator",
                    wraps=__import__(
                        "sebench.metricgan_alternating",
                        fromlist=["_update_discriminator"],
                    )._update_discriminator,
                ) as update,
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
            self.assertEqual(update.call_count, 7)
            self.assertEqual(
                [float(call.args[4]) for call in update.call_args_list],
                [1.0, 0.6, 0.6, 0.6, 1.0, 0.6, 0.6],
            )
            self.assertTrue(
                all(call.args[2].dim() == 1 for call in update.call_args_list)
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

    def test_d2_support_is_disjoint_resumable_and_does_not_copy_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest = root / "train.csv"
            train_manifest.write_text(
                "noisy,clean\n"
                + "".join(
                    f"/external/noisy-{index}.wav,/external/clean-{index}.wav\n"
                    for index in range(8)
                ),
                encoding="utf-8",
            )
            cache_manifest = root / "teacher-cache.csv"
            teacher_paths = []
            for index in range(8):
                teacher_path = root / f"teacher-{index}.pt"
                torch.save(torch.ones(4096, dtype=torch.float16), teacher_path)
                teacher_paths.append(teacher_path)
            with cache_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("noisy", "clean", "teacher_wav"),
                )
                writer.writeheader()
                for index, teacher_path in enumerate(teacher_paths):
                    writer.writerow(
                        {
                            "noisy": f"/external/noisy-{index}.wav",
                            "clean": f"/external/clean-{index}.wav",
                            "teacher_wav": teacher_path.as_posix(),
                        }
                    )
            teacher_hash = "a" * 64
            metadata = root / "cache-metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "cache_inputs": False,
                        "storage_dtype": "float16",
                        "teacher_checkpoint_sha256": teacher_hash,
                        "train_manifest_sha256": hashlib.sha256(
                            train_manifest.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            waveform = torch.linspace(-0.2, 0.2, 4096)
            with (
                mock.patch(
                    "sebench.metricgan_d2._load_aligned",
                    return_value=(waveform, waveform * 0.9, waveform * 0.95),
                ),
                mock.patch(
                    "sebench.metricgan_d2.pesq_score",
                    side_effect=[
                        value
                        for index in range(8)
                        for value in (2.0 + index * 0.1, 1.5 + index * 0.1)
                    ],
                ),
            ):
                payload = prepare_d2_support(
                    train_manifest=train_manifest,
                    teacher_cache_manifest=cache_manifest,
                    teacher_cache_metadata=metadata,
                    output_dir=root / "support",
                    expected_teacher_sha256=teacher_hash,
                    train_rows=4,
                    calibration_rows=2,
                    audit_rows=2,
                    seed=7,
                )
            self.assertEqual(
                payload["counts"],
                {"train": 4, "calibration": 2, "audit": 2},
            )
            self.assertTrue(payload["utterance_disjoint"])
            self.assertFalse(payload["speaker_disjoint_verified"])
            self.assertEqual(len(payload["records"]), 8)
            self.assertFalse((root / "support" / "noisy").exists())
            self.assertFalse((root / "support" / "clean").exists())
            self.assertTrue((root / "support" / "coverage.png").is_file())
            self.assertEqual(
                payload["source_hashes_before"],
                payload["source_hashes_after"],
            )
            run_root = root / "run"
            (run_root / "support").mkdir(parents=True)
            for name in ("support.json", "coverage.json", "coverage.png"):
                source = root / "support" / name
                target = run_root / "support" / name
                if source.suffix == ".png":
                    target.write_bytes(source.read_bytes())
                else:
                    target.write_text(
                        source.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
            audit = audit_d2_support(run_root)
            self.assertTrue(audit["valid"], audit)
            self.assertEqual(audit["record_count"], 8)


if __name__ == "__main__":
    unittest.main()
