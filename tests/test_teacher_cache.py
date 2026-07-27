from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.teacher_cache import (  # noqa: E402
    TeacherCacheDataset,
    TeacherCacheTarget,
    build_multi_target_teacher_cache,
)


class HalfTeacher(torch.nn.Module):
    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        return noisy * 0.5


class TeacherCacheTests(unittest.TestCase):
    def test_local_fp16_cache_omits_dataset_copies_and_loads_as_float32(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            audio_root = root / "dataset"
            audio_root.mkdir()
            samples = torch.linspace(-0.8, 0.8, 2048).unsqueeze(0)
            noisy = audio_root / "noisy.wav"
            clean = audio_root / "clean.wav"
            torchaudio.save(noisy.as_posix(), samples, 16000)
            torchaudio.save(clean.as_posix(), samples * 0.9, 16000)
            manifest = root / "train.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("noisy", "clean"))
                writer.writeheader()
                writer.writerow(
                    {"noisy": noisy.as_posix(), "clean": clean.as_posix()}
                )

            cache_root = root / "local_cache"
            outputs = build_multi_target_teacher_cache(
                manifest,
                HalfTeacher(),
                out_dir=cache_root,
                device="cpu",
                targets=[
                    TeacherCacheTarget(
                        name="wb",
                        sample_rate=16000,
                        erb_bands=8,
                    )
                ],
                batch_size=1,
                num_workers=0,
                cache_inputs=False,
                storage_dtype="float16",
            )
            with Path(outputs["wb"]).open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["noisy_cache"], "")
            self.assertEqual(row["clean_cache"], "")
            teacher_wave = torch.load(
                row["teacher_wav"],
                map_location="cpu",
                weights_only=True,
            )
            teacher_mask = torch.load(
                row["teacher_mask_erb"],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(teacher_wave.dtype, torch.float16)
            self.assertEqual(teacher_mask.dtype, torch.float16)
            self.assertFalse(any((cache_root / "wb" / "noisy_cache").iterdir()))
            self.assertFalse(any((cache_root / "wb" / "clean_cache").iterdir()))

            dataset = TeacherCacheDataset(
                outputs["wb"],
                segment_len=2048,
                sample_rate=16000,
                n_fft=512,
                hop_length=160,
            )
            loaded = dataset[0]
            source, _ = torchaudio.load(noisy.as_posix())
            expected = source.squeeze(0) * 0.5
            self.assertEqual(loaded["teacher_wav"].dtype, torch.float32)
            self.assertLess(
                float((loaded["teacher_wav"] - expected).abs().max()),
                5e-4,
            )
            self.assertTrue(torch.isfinite(loaded["teacher_mask_erb"]).all())

            # A cache created under an older storage contract must be rebuilt,
            # not silently reused under the canonical FP16 declaration.
            torch.save(teacher_wave.float(), row["teacher_wav"])
            resumed = build_multi_target_teacher_cache(
                manifest,
                HalfTeacher(),
                out_dir=cache_root,
                device="cpu",
                targets=[
                    TeacherCacheTarget(
                        name="wb",
                        sample_rate=16000,
                        erb_bands=8,
                    )
                ],
                batch_size=1,
                num_workers=0,
                resume=True,
                cache_inputs=False,
                storage_dtype="float16",
            )
            self.assertEqual(resumed, outputs)
            rebuilt_teacher_wave = torch.load(
                row["teacher_wav"],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(rebuilt_teacher_wave.dtype, torch.float16)

    def test_cache_rejects_unknown_storage_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "storage_dtype"):
            build_multi_target_teacher_cache(
                "unused.csv",
                HalfTeacher(),
                out_dir="unused",
                device="cpu",
                targets=[TeacherCacheTarget(name="wb", sample_rate=16000)],
                storage_dtype="bfloat16",
            )


if __name__ == "__main__":
    unittest.main()
