from __future__ import annotations

import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "bind_voicebank_manifests.py"
SPEC = importlib.util.spec_from_file_location("bind_voicebank_manifests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BINDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BINDING)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestBindingTests(unittest.TestCase):
    def test_binding_rewrites_copy_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            old_root = "/old/data"
            new_root = root / "mounted"
            noisy = new_root / "noisy.wav"
            clean = new_root / "clean.wav"
            new_root.mkdir()
            noisy.write_bytes(b"noisy")
            clean.write_bytes(b"clean")
            source = root / "source.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["noisy", "clean"])
                writer.writeheader()
                writer.writerow(
                    {
                        "noisy": f"{old_root}/noisy.wav",
                        "clean": f"{old_root}/clean.wav",
                    }
                )
            before = file_hash(source)
            output = root / "local" / "bound.csv"
            result = BINDING.bind_manifest(
                source,
                output,
                mappings=[(old_root, new_root.as_posix())],
            )
            self.assertEqual(file_hash(source), before)
            self.assertEqual(result["rows"], 1)
            rows = BINDING.read_rows(output)
            self.assertEqual(rows[0]["noisy"], noisy.as_posix())

    def test_binding_rejects_missing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.csv"
            source.write_text("noisy,clean\n/old/a.wav,/old/b.wav\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                BINDING.bind_manifest(
                    source,
                    root / "bound.csv",
                    mappings=[("/old", (root / "mounted").as_posix())],
                )


if __name__ == "__main__":
    unittest.main()
