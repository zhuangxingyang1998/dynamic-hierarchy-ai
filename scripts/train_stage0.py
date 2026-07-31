"""Run one configuration-driven Stage 0 baseline experiment."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dynamic_hierarchy.config import load_config
from dynamic_hierarchy.provenance import runtime_provenance
from dynamic_hierarchy.reporting import fallback_observability
from dynamic_hierarchy.training import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    config = load_config(args.config)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        metrics = train(config)
    runtime_warnings = [str(warning.message) for warning in captured_warnings]
    fallback_status = fallback_observability(config.device, runtime_warnings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"stage0-{config.device}-{args.config.stem}-{stamp}.json"
    record = {
        "schema_version": 4,
        "config": config.to_dict(),
        "metrics": metrics.to_dict(),
        "runtime_warnings": runtime_warnings,
        "fallback_observability": fallback_status,
        "provenance": runtime_provenance(
            config.cpu_threads,
            {
                "backend": metrics.performance.backend,
                "device_name": metrics.performance.device_name,
                "determinism_status": metrics.performance.determinism_status,
                "synchronization_method": metrics.performance.synchronization_method,
                "fallback_observability": fallback_status,
            },
        ),
        "note": "Smoke completed successfully, but it is not a research conclusion or evidence for candidate hierarchy hypotheses.",
    }
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics.to_dict(), "runtime_warnings": runtime_warnings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
