"""Run repeated Stage 0 measurements and retain every sample."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dynamic_hierarchy.config import load_config
from dynamic_hierarchy.provenance import runtime_provenance
from dynamic_hierarchy.reporting import fallback_observability, summarize_measurements
from dynamic_hierarchy.training import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    config = load_config(args.config)
    samples: list[dict[str, object]] = []
    for repeat in range(1, args.repeats + 1):
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            metrics = train(config)
        runtime_warnings = [str(warning.message) for warning in captured_warnings]
        samples.append(
            {
                "repeat": repeat,
                "metrics": metrics.to_dict(),
                "runtime_warnings": runtime_warnings,
                "fallback_observability": fallback_observability(config.device, runtime_warnings),
            }
        )

    performance_samples = [sample["metrics"]["performance"] for sample in samples]
    summary = {
        field: summarize_measurements([float(performance[field]) for performance in performance_samples])
        for field in ("training_seconds", "steps_per_second", "examples_per_second")
    }
    combined_warnings = [
        warning
        for sample in samples
        for warning in sample["runtime_warnings"]
    ]
    fallback_status = fallback_observability(config.device, combined_warnings)
    backend_metadata = {
        "backend": performance_samples[-1]["backend"],
        "device_name": performance_samples[-1]["device_name"],
        "deterministic_requested": performance_samples[-1]["deterministic_requested"],
        "deterministic_algorithms_enabled": performance_samples[-1]["deterministic_algorithms_enabled"],
        "determinism_status": performance_samples[-1]["determinism_status"],
        "synchronization_method": performance_samples[-1]["synchronization_method"],
        "timing_barrier": performance_samples[-1]["timing_barrier"],
        "fallback_observability": fallback_status,
    }
    record = {
        "schema_version": 1,
        "kind": "repeated_performance_benchmark",
        "config": config.to_dict(),
        "repeats": args.repeats,
        "samples": samples,
        "summary": summary,
        "fallback_observability": fallback_status,
        "provenance": runtime_provenance(config.cpu_threads, backend_metadata),
        "note": (
            "Performance measurement only; accuracy is retained for diagnostics and is not a research conclusion. "
            "Warmup updates are excluded from throughput. Approximate p95 uses nearest-rank."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"benchmark-{config.device}-{args.config.stem}-{stamp}.json"
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary, "fallback_observability": fallback_status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
