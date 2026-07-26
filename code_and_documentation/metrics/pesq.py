# pesq.py — STRICT
from __future__ import annotations
import numpy as np

from sebench.bandwidth import resolve_bandwidth

try:
    from pesq import pesq as _pesq
except Exception as e:
    raise ImportError("The 'pesq' package (ITU-T P.862) is required.") from e

def pesq_score(
    ref: np.ndarray,
    deg: np.ndarray,
    sr: int,
    *,
    bandwidth: str | None = None,
) -> float:
    if ref.ndim != 1 or deg.ndim != 1:
        raise ValueError("pesq_score expects 1D mono arrays.")
    profile = resolve_bandwidth(bandwidth, sample_rate=sr)
    try:
        score = float(
            _pesq(
                sr,
                ref.astype(np.float32),
                deg.astype(np.float32),
                profile.pesq_mode,
            )
        )
    except Exception as exc:
        message = str(exc).lower()
        name = exc.__class__.__name__.lower()
        if "no utterances detected" in message or "noutterances" in name:
            return float("nan")
        raise
    if not np.isfinite(score):
        return float("nan")
    return score
