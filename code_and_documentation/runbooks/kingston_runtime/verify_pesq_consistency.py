#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_pesq_function(project_root: Path, module_alias: str):
    pesq_path = project_root / "metrics" / "pesq.py"
    spec = importlib.util.spec_from_file_location(module_alias, pesq_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {pesq_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pesq_score


def _load_scores(metricgan_project: Path, ultra_project: Path) -> dict:
    metricgan_pesq = _load_pesq_function(metricgan_project, "metricgan_pesq_module")
    ultra_pesq = _load_pesq_function(ultra_project, "ultra_pesq_module")

    rng = np.random.default_rng(1234)
    results: dict[str, float] = {}
    for sr in (8000, 16000):
        t = np.linspace(0, 1.0, int(sr), endpoint=False, dtype=np.float32)
        clean = 0.20 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        deg = clean + 0.01 * rng.standard_normal(clean.shape[0], dtype=np.float32)

        score_m = float(metricgan_pesq(clean, deg, sr))
        score_u = float(ultra_pesq(clean, deg, sr))
        if not np.isfinite(score_m) or not np.isfinite(score_u):
            raise RuntimeError(f"Non-finite PESQ score at sr={sr}")
        if abs(score_m - score_u) > 1e-6:
            raise RuntimeError(f"PESQ mismatch between projects at sr={sr}: {score_m} vs {score_u}")
        results[f"pesq_{sr}"] = score_m

    ok_invalid_sr = False
    try:
        metricgan_pesq(np.zeros(16, dtype=np.float32), np.zeros(16, dtype=np.float32), 44100)
    except ValueError:
        ok_invalid_sr = True
    if not ok_invalid_sr:
        raise RuntimeError("Expected ValueError for invalid sample rate (44100).")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metricgan-project", required=True)
    parser.add_argument("--ultra-project", required=True)
    args = parser.parse_args()

    metricgan_project = Path(args.metricgan_project).resolve()
    ultra_project = Path(args.ultra_project).resolve()
    payload = _load_scores(metricgan_project, ultra_project)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
