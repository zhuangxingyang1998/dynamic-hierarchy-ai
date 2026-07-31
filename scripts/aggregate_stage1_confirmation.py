"""Aggregate eight revised Stage 1 training-seed results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dynamic_hierarchy.stage1_confirmation import aggregate_confirmation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.results
    ]
    aggregate = aggregate_confirmation(results, result_paths=args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(aggregate, indent=2))
    return 0 if aggregate["stage2_unblocked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
