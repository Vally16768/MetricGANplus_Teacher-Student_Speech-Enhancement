# pesq.py — STRICT
from __future__ import annotations
import numpy as np

try:
    from pesq import pesq as _pesq
except Exception as e:
    raise ImportError("The 'pesq' package (ITU-T P.862) is required.") from e

def pesq_score(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    if ref.ndim != 1 or deg.ndim != 1:
        raise ValueError("pesq_score expects 1D mono arrays.")
    if sr not in (8000, 16000):
        raise ValueError("PESQ supports only 8000 or 16000 Hz.")
    try:
        score = float(_pesq(sr, ref.astype(np.float32), deg.astype(np.float32), 'wb' if sr == 16000 else 'nb'))
    except Exception as exc:
        message = str(exc).lower()
        name = exc.__class__.__name__.lower()
        if "no utterances detected" in message or "noutterances" in name:
            return float("nan")
        raise
    if not np.isfinite(score):
        return float("nan")
    return score
