#!/usr/bin/env python3
"""Create verified local-only bindings for frozen VoiceBank manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


REQUIRED_SPLITS = ("train_fit", "val_rank", "val_select", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got: {value}")
    name, raw = value.split("=", 1)
    if not name.strip() or not raw.strip():
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got: {value}")
    return name.strip(), raw.strip()


def apply_mappings(value: str, mappings: list[tuple[str, str]]) -> str:
    rendered = value
    for old, new in mappings:
        if rendered == old or rendered.startswith(old.rstrip("/") + "/"):
            rendered = new.rstrip("/") + rendered[len(old.rstrip("/")) :]
            break
    return rendered


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"noisy", "clean"}.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain noisy,clean columns: {path}")
        return [
            {"noisy": str(row["noisy"]).strip(), "clean": str(row["clean"]).strip()}
            for row in reader
        ]


def write_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["noisy", "clean"])
        writer.writeheader()
        writer.writerows(rows)


def bind_manifest(
    source: Path,
    output: Path,
    *,
    mappings: list[tuple[str, str]],
    max_rows: int | None = None,
) -> dict[str, object]:
    source_before = sha256(source)
    rows = read_rows(source)
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    bound_rows = [
        {
            "noisy": apply_mappings(row["noisy"], mappings),
            "clean": apply_mappings(row["clean"], mappings),
        }
        for row in rows
    ]
    missing = [
        path
        for row in bound_rows
        for path in (row["noisy"], row["clean"])
        if not Path(path).is_file()
    ]
    if missing:
        sample = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} bound audio paths do not exist for {source}; sample:\n{sample}"
        )
    pairs = {(row["noisy"], row["clean"]) for row in bound_rows}
    clean = {row["clean"] for row in bound_rows}
    if len(pairs) != len(bound_rows) or len(clean) != len(bound_rows):
        raise ValueError(
            f"Duplicate pair/clean identity after binding {source}: "
            f"rows={len(bound_rows)} pairs={len(pairs)} clean={len(clean)}"
        )
    write_rows(output, bound_rows)
    source_after = sha256(source)
    if source_before != source_after:
        raise RuntimeError(f"Source manifest changed while binding: {source}")
    return {
        "source": source.as_posix(),
        "source_sha256": source_before,
        "bound": output.as_posix(),
        "bound_sha256": sha256(output),
        "rows": len(bound_rows),
        "missing_audio": 0,
        "duplicate_pairs": 0,
        "duplicate_clean": 0,
    }


def validate_split_isolation(bound: dict[str, dict[str, object]]) -> dict[str, int]:
    rows = {
        name: read_rows(Path(str(payload["bound"])))
        for name, payload in bound.items()
    }
    pair_sets = {
        name: {(row["noisy"], row["clean"]) for row in split_rows}
        for name, split_rows in rows.items()
    }
    clean_sets = {
        name: {row["clean"] for row in split_rows}
        for name, split_rows in rows.items()
    }
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(REQUIRED_SPLITS):
        for right in REQUIRED_SPLITS[left_index + 1 :]:
            pair_key = f"{left}__{right}_pair_overlap"
            clean_key = f"{left}__{right}_clean_overlap"
            overlaps[pair_key] = len(pair_sets[left] & pair_sets[right])
            overlaps[clean_key] = len(clean_sets[left] & clean_sets[right])
    nonzero = {key: value for key, value in overlaps.items() if value}
    if nonzero:
        raise ValueError(f"Split leakage detected after binding: {nonzero}")
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        action="append",
        type=parse_assignment,
        required=True,
        help="Repeat as train_fit=/path.csv, val_rank=..., val_select=..., test=...",
    )
    parser.add_argument(
        "--mapping",
        action="append",
        type=parse_assignment,
        required=True,
        help="Repeat for each old-prefix=new-prefix mapping.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-train", type=int, default=0)
    parser.add_argument("--smoke-val-rank", type=int, default=0)
    parser.add_argument("--smoke-val-select", type=int, default=0)
    parser.add_argument("--smoke-test", type=int, default=0)
    parser.add_argument("--pilot-train", type=int, default=0)
    parser.add_argument("--pilot-val-rank", type=int, default=0)
    parser.add_argument("--pilot-val-select", type=int, default=0)
    parser.add_argument("--pilot-test", type=int, default=0)
    args = parser.parse_args()

    split_paths = {name: Path(value).expanduser().resolve() for name, value in args.split}
    missing_splits = sorted(set(REQUIRED_SPLITS) - set(split_paths))
    extra_splits = sorted(set(split_paths) - set(REQUIRED_SPLITS))
    if missing_splits or extra_splits:
        raise ValueError(
            f"Expected splits {REQUIRED_SPLITS}; missing={missing_splits} extra={extra_splits}"
        )
    mappings = [(old, str(Path(new).expanduser().resolve())) for old, new in args.mapping]
    output_dir = Path(args.output_dir).expanduser().resolve()
    smoke_limits = {
        "train_fit": args.smoke_train,
        "val_rank": args.smoke_val_rank,
        "val_select": args.smoke_val_select,
        "test": args.smoke_test,
    }
    pilot_limits = {
        "train_fit": args.pilot_train,
        "val_rank": args.pilot_val_rank,
        "val_select": args.pilot_val_select,
        "test": args.pilot_test,
    }

    full_dir = output_dir / "full"
    bound = {
        name: bind_manifest(
            split_paths[name],
            full_dir / f"{name}.csv",
            mappings=mappings,
        )
        for name in REQUIRED_SPLITS
    }
    full_overlap = validate_split_isolation(bound)

    def bind_subset(
        profile_name: str,
        limits: dict[str, int],
    ) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
        subset_bound: dict[str, dict[str, object]] = {}
        if not any(value > 0 for value in limits.values()):
            return subset_bound, {}
        subset_dir = output_dir / profile_name
        for split_name in REQUIRED_SPLITS:
            limit = limits[split_name]
            if limit <= 0:
                raise ValueError(
                    f"All --{profile_name}-* limits must be positive when creating "
                    f"{profile_name} manifests."
                )
            subset_bound[split_name] = bind_manifest(
                split_paths[split_name],
                subset_dir / f"{split_name}.csv",
                mappings=mappings,
                max_rows=limit,
            )
        return subset_bound, validate_split_isolation(subset_bound)

    smoke_bound, smoke_overlap = bind_subset("smoke", smoke_limits)
    pilot_bound, pilot_overlap = bind_subset("pilot", pilot_limits)

    audit = {
        "schema_version": 1,
        "dataset": "VoiceBank+DEMAND",
        "source_read_only": True,
        "mappings": [{"old": old, "new": new} for old, new in mappings],
        "full": bound,
        "full_overlap": full_overlap,
        "smoke": smoke_bound,
        "smoke_overlap": smoke_overlap,
        "pilot": pilot_bound,
        "pilot_overlap": pilot_overlap,
    }
    audit_path = output_dir / "binding_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
