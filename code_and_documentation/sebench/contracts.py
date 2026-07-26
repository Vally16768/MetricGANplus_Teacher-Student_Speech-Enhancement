"""Machine-independent safety contracts for the canonical campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bandwidth import resolve_bandwidth, validate_frontend


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_io_boundaries(config: dict[str, Any]) -> None:
    """Refuse mutable outputs placed inside dataset input roots."""
    paths = dict(config.get("paths") or {})
    dataset = dict(config.get("dataset") or {})
    teacher_cache = dict(config.get("teacher_cache") or {})
    read_roots = [
        paths.get("dataset_root"),
        dataset.get("voicebank_root"),
        dataset.get("voicebank_campaign_dir"),
    ]
    write_roots = [
        paths.get("output_root"),
        paths.get("tracking_root"),
        teacher_cache.get("out_dir"),
    ]
    resolved_reads = [
        Path(str(value)).expanduser().resolve(strict=False)
        for value in read_roots
        if str(value or "").strip()
    ]
    resolved_writes = [
        Path(str(value)).expanduser().resolve(strict=False)
        for value in write_roots
        if str(value or "").strip()
    ]
    for write_root in resolved_writes:
        for read_root in resolved_reads:
            if _is_within(write_root, read_root):
                raise ValueError(
                    "Unsafe config: mutable output is inside a dataset input root: "
                    f"write={write_root} dataset={read_root}"
                )


def validate_bandwidth_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the model frontend against its WB/NB profile."""
    training = dict(config.get("training") or {})
    bandwidth = str(training.get("bandwidth") or "").strip().lower()
    if not bandwidth:
        bandwidth = resolve_bandwidth(
            None,
            sample_rate=int(training["sample_rate"]),
        ).name
    profile = validate_frontend(
        bandwidth,
        sample_rate=int(training["sample_rate"]),
        n_fft=int(training["n_fft"]),
        hop_length=int(training["hop_length"]),
        win_length=int(training["win_length"]),
    )
    return profile.as_dict()
