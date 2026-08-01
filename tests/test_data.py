from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.data import ManifestRow, VoiceBankDemandDataset  # noqa: E402


class VoiceBankDemandDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_pair(
        root: Path,
        name: str,
        *,
        frames: int,
        sample_rate: int = 16000,
    ) -> ManifestRow:
        signal = torch.linspace(-0.8, 0.8, frames).unsqueeze(0)
        noisy = root / f"{name}_noisy.wav"
        clean = root / f"{name}_clean.wav"
        torchaudio.save(noisy.as_posix(), signal, sample_rate)
        torchaudio.save(clean.as_posix(), signal * 0.5, sample_rate)
        return ManifestRow(noisy=noisy, clean=clean)

    def test_nb_resampling_returns_exact_segments_for_short_and_long_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [
                self._write_pair(root, "short", frames=12001),
                self._write_pair(root, "long", frames=40001),
            ]
            dataset = VoiceBankDemandDataset(
                "unused.csv",
                rows=rows,
                segment_len=8000,
                sample_rate=8000,
            )

            short_noisy, short_clean = dataset[0]
            long_noisy, long_clean = dataset[1]

            self.assertEqual(short_noisy.shape, (8000,))
            self.assertEqual(short_clean.shape, (8000,))
            self.assertEqual(long_noisy.shape, (8000,))
            self.assertEqual(long_clean.shape, (8000,))
            self.assertTrue(torch.all(short_noisy[6001:] == 0))
            self.assertTrue(torch.all(short_clean[6001:] == 0))

    def test_nb_default_collation_has_uniform_target_rate_shape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [
                self._write_pair(root, "short", frames=12001),
                self._write_pair(root, "long", frames=40001),
            ]
            dataset = VoiceBankDemandDataset(
                "unused.csv",
                rows=rows,
                segment_len=8000,
                sample_rate=8000,
            )

            noisy, clean = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

            self.assertEqual(noisy.shape, (2, 8000))
            self.assertEqual(clean.shape, (2, 8000))

    def test_wb_native_rate_behavior_remains_fixed_length_and_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            row = self._write_pair(root, "wb", frames=20000)
            dataset = VoiceBankDemandDataset(
                "unused.csv",
                rows=[row],
                segment_len=16000,
                sample_rate=16000,
            )

            torch.manual_seed(7)
            noisy, clean = dataset[0]

            self.assertEqual(noisy.shape, (16000,))
            self.assertEqual(clean.shape, (16000,))
            self.assertLess(float((clean - noisy * 0.5).abs().max()), 5e-5)


if __name__ == "__main__":
    unittest.main()
