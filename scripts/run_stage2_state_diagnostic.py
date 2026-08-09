"""Run the bounded read-only Stage 2 R5 hidden-state diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_hierarchy.stage2_ladder_runtime import atomic_write_json
from dynamic_hierarchy.stage2_state_diagnostic import run_canonical_state_diagnostic


def _validate_output_path(run_dir: Path, output: Path) -> None:
    run_resolved = run_dir.resolve()
    output_resolved = output.resolve()
    if output_resolved.is_relative_to(run_resolved):
        raise ValueError("state diagnostic output must remain outside the canonical run")
    if output.exists():
        raise FileExistsError(f"state diagnostic output already exists: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _validate_output_path(args.run_dir, args.output)
    result = run_canonical_state_diagnostic(args.run_dir)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "branches": len(result["branches"]),
                "reserve_evaluated": result["claim_boundary"]["reserve_evaluated"],
                "optimizer_updates": result["execution"]["optimizer_updates"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "diagnostic_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
