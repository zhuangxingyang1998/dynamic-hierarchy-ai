"""Hysteretic host-resource guard for cooperative background training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import psutil


@dataclass(frozen=True)
class ResourceSample:
    cpu_percent: float
    available_ram_gb: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class ResourceGuard:
    def __init__(
        self,
        cpu_pause_percent: float,
        cpu_resume_percent: float,
        ram_pause_gb: float,
        ram_resume_gb: float,
        pressure_samples: int,
        recovery_samples: int,
    ) -> None:
        self.cpu_pause_percent = cpu_pause_percent
        self.cpu_resume_percent = cpu_resume_percent
        self.ram_pause_gb = ram_pause_gb
        self.ram_resume_gb = ram_resume_gb
        self.pressure_samples = pressure_samples
        self.recovery_samples = recovery_samples
        self.paused = False
        self.pause_reason: str | None = None
        self._pressure_count = 0
        self._recovery_count = 0

    @staticmethod
    def sample() -> ResourceSample:
        return ResourceSample(
            cpu_percent=float(psutil.cpu_percent(interval=None)),
            available_ram_gb=float(psutil.virtual_memory().available / 1024**3),
        )

    def observe(self, sample: ResourceSample) -> bool:
        pressure = (
            sample.cpu_percent >= self.cpu_pause_percent
            or sample.available_ram_gb < self.ram_pause_gb
        )
        recovered = (
            sample.cpu_percent <= self.cpu_resume_percent
            and sample.available_ram_gb >= self.ram_resume_gb
        )
        if not self.paused:
            self._pressure_count = self._pressure_count + 1 if pressure else 0
            if self._pressure_count >= self.pressure_samples:
                self.paused = True
                reasons = []
                if sample.cpu_percent >= self.cpu_pause_percent:
                    reasons.append(f"system CPU {sample.cpu_percent:.1f}% >= {self.cpu_pause_percent:.1f}%")
                if sample.available_ram_gb < self.ram_pause_gb:
                    reasons.append(
                        f"available RAM {sample.available_ram_gb:.2f}GB < {self.ram_pause_gb:.2f}GB"
                    )
                self.pause_reason = "; ".join(reasons)
                self._recovery_count = 0
        else:
            self._recovery_count = self._recovery_count + 1 if recovered else 0
            if self._recovery_count >= self.recovery_samples:
                self.paused = False
                self.pause_reason = None
                self._pressure_count = 0
        return self.paused
