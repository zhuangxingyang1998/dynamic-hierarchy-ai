"""Backend resolution and explicit synchronization boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Backend:
    name: str
    device: torch.device
    device_name: str
    deterministic_algorithms_enabled: bool
    determinism_status: str
    synchronization_method: str

    def synchronize(self, tensor: torch.Tensor | None = None) -> None:
        if tensor is not None:
            tensor.detach().cpu()
        elif self.name == "directml":
            torch.zeros((), device=self.device).cpu()

    def scalar(self, tensor: torch.Tensor) -> float:
        return float(tensor.detach().cpu().item())

    def metadata(self) -> dict[str, str]:
        return {
            "backend": self.name,
            "device": str(self.device),
            "device_name": self.device_name,
            "deterministic_algorithms_enabled": self.deterministic_algorithms_enabled,
            "determinism_status": self.determinism_status,
            "synchronization_method": self.synchronization_method,
        }


def resolve_backend(name: str, cpu_threads: int, deterministic: bool = True) -> Backend:
    # The fused eval-only Transformer path is unavailable on DirectML; both backends use the equivalent standard path.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(cpu_threads)
    if name == "cpu":
        torch.use_deterministic_algorithms(deterministic)
        device_name = os.environ.get("PROCESSOR_IDENTIFIER", "CPU")
        return Backend(
            name="cpu",
            device=torch.device("cpu"),
            device_name=device_name,
            deterministic_algorithms_enabled=deterministic,
            determinism_status=(
                "strict torch deterministic algorithms enabled for this CPU run"
                if deterministic
                else "torch deterministic algorithms disabled for this performance run"
            ),
            synchronization_method=(
                "synchronous CPU execution; updated model parameter passed to synchronize() at timing boundaries"
            ),
        )
    if name == "directml":
        if deterministic:
            raise ValueError(
                "deterministic=true is not supported for DirectML runs; set deterministic=false explicitly"
            )
        try:
            import torch_directml
        except ImportError as error:
            raise RuntimeError(
                "device='directml' requires the separate DirectML environment and torch-directml package"
            ) from error
        if torch_directml.device_count() < 1:
            raise RuntimeError("torch-directml imported but reported no DirectML devices")
        torch.use_deterministic_algorithms(False)
        return Backend(
            name="directml",
            device=torch_directml.device(0),
            device_name=torch_directml.device_name(0).rstrip("\x00"),
            deterministic_algorithms_enabled=False,
            determinism_status="not guaranteed by torch-directml; reproducibility is measured, not assumed",
            synchronization_method=(
                "updated model parameter detach().cpu() barrier before timing starts and after the final optimizer write"
            ),
        )
    raise ValueError(f"unsupported device backend: {name}")
