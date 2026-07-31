"""Fail-closed DirectML import, device, operator, and backward validation."""

from __future__ import annotations

import json
import traceback
import warnings

from dynamic_hierarchy.reporting import fallback_observability


def main() -> int:
    report: dict[str, object] = {"status": "failed", "checks": {}}
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            import torch
            import torch_directml

            report["checks"]["import"] = "passed"
            device = torch_directml.device(0)
            report["device"] = str(device)
            report["device_name"] = torch_directml.device_name(0).rstrip("\x00")
            report["checks"]["device"] = "passed"
            left = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
            right = torch.tensor([3.0, 4.0], device=device)
            addition = left + right
            report["addition"] = addition.detach().cpu().tolist()
            report["checks"]["addition"] = "passed"
            matrix = torch.arange(1.0, 5.0, device=device).reshape(2, 2)
            product = matrix @ matrix
            report["matmul"] = product.detach().cpu().tolist()
            report["checks"]["matmul"] = "passed"
            loss = addition.sum() + product.sum()
            loss.backward()
            report["gradient"] = left.grad.detach().cpu().tolist()
            report["checks"]["backward"] = "passed"
            report["synchronization_method"] = "tensor.detach().cpu() barriers"
            report["status"] = "passed"
        except Exception as error:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
            report["traceback"] = traceback.format_exc()
    runtime_warnings = [str(warning.message) for warning in captured_warnings]
    report["runtime_warnings"] = runtime_warnings
    report["fallback_observability"] = fallback_observability("directml", runtime_warnings)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
