"""Create one immutable-input Stage 1 run snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_hierarchy.snapshot import create_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = create_snapshot(project_root, run_dir / "snapshot")
    print(json.dumps({"run_dir": str(run_dir), "snapshot_manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
