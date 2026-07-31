"""Shared run-record helpers."""

from __future__ import annotations

import math
import statistics


def fallback_observability(backend: str, runtime_warnings: list[str]) -> dict[str, str]:
    if backend == "directml":
        warning_detail = (
            "no Python warnings observed"
            if not runtime_warnings
            else f"{len(runtime_warnings)} Python warning(s) observed; inspect runtime_warnings"
        )
        return {
            "status": "unknown",
            "detail": f"no public DirectML fallback counter; {warning_detail}",
        }
    return {
        "status": "not_applicable",
        "detail": "CPU backend selected; DirectML fallback observability does not apply",
    }


def summarize_measurements(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        raise ValueError("at least one measurement is required")
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": len(values),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "approx_p95": ordered[p95_index],
        "p95_method": "nearest-rank",
    }
