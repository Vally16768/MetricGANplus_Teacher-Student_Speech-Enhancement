#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_and_documentation"
sys.path.insert(0, CODE_ROOT.as_posix())

from sebench.research_plan import validate_research_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default=(
            CODE_ROOT / "configs" / "research_plan_voicebank_wb_nb.yaml"
        ).as_posix(),
    )
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8"))
    print(json.dumps(validate_research_plan(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
