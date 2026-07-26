"""Explicit narrow-band/wide-band contracts for training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BandwidthProfile:
    name: str
    sample_rate: int
    pesq_mode: str
    n_fft: int
    hop_length: int
    win_length: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


PROFILES = {
    "nb": BandwidthProfile(
        name="nb",
        sample_rate=8_000,
        pesq_mode="nb",
        n_fft=256,
        hop_length=80,
        win_length=160,
    ),
    "wb": BandwidthProfile(
        name="wb",
        sample_rate=16_000,
        pesq_mode="wb",
        n_fft=512,
        hop_length=160,
        win_length=320,
    ),
}


def infer_bandwidth(sample_rate: int) -> str:
    for profile in PROFILES.values():
        if profile.sample_rate == int(sample_rate):
            return profile.name
    raise ValueError(
        f"Cannot infer a PESQ bandwidth for sample_rate={sample_rate}. "
        "Supported contracts are NB/8000 Hz and WB/16000 Hz."
    )


def resolve_bandwidth(bandwidth: str | None, *, sample_rate: int | None = None) -> BandwidthProfile:
    name = str(bandwidth or "").strip().lower()
    if not name:
        if sample_rate is None:
            raise ValueError("An explicit bandwidth (`nb` or `wb`) is required.")
        name = infer_bandwidth(sample_rate)
    if name not in PROFILES:
        raise ValueError(f"Unsupported bandwidth {bandwidth!r}. Use `nb` or `wb`.")
    profile = PROFILES[name]
    if sample_rate is not None and int(sample_rate) != profile.sample_rate:
        raise ValueError(
            f"Bandwidth/sample-rate mismatch: {name.upper()} requires "
            f"{profile.sample_rate} Hz, got {sample_rate} Hz."
        )
    return profile


def validate_frontend(
    bandwidth: str,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> BandwidthProfile:
    profile = resolve_bandwidth(bandwidth, sample_rate=sample_rate)
    observed = (int(n_fft), int(hop_length), int(win_length))
    expected = (profile.n_fft, profile.hop_length, profile.win_length)
    if observed != expected:
        raise ValueError(
            f"{profile.name.upper()} frontend mismatch: expected "
            f"n_fft/hop/win={expected}, got {observed}."
        )
    return profile
