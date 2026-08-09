"""Frozen configuration contracts for Stage 2 R6 state congruence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .stage2_ladder_config import LadderModelSpec


R6_REVISION = "stage2-r6"
R6_PHASE = "state_congruence"
R6_PACKET = "DH-S2-R6-R7"
R6_SEED = 821601
R6_PARTITION_DIGEST = (
    "8425fa0161ac6682d4644e7350bc4d80d41fe498de03b8313a64364218f5fa52"
)
R5_CHECKPOINT_SHA256 = (
    "18327E373F937D353297811DB60C7180B9B3823FE49B4E7CDB09EE27D6EFD489"
)
R5_FROZEN_CONFIG_SHA256 = (
    "4B64023623B3DE1AC23D06E718ADA1C9BB639085CF95688A4EAC1FED03D5DCA7"
)
R5_CONFIG_DIGEST = (
    "159adaaa5bbc6854ac862f071f6709a3a722140a9f711ed5195bb1eaa17d391a"
)
R5_PARTITION_DIGEST = (
    "1701144f08fe7b7ee72b30b210c4922a14a3a4da69694ebb092db0c2cbace2d1"
)
R5_RUNG1_STATE_DIGEST = (
    "9c133c17c9dfcfe8bffbd8b71ea1a7d3ecd724dd037792be2fc6acc9e6b426ce"
)


@dataclass(frozen=True)
class Stage2CongruenceConfig:
    packet: str = R6_PACKET
    revision: str = R6_REVISION
    phase: str = R6_PHASE
    run_kind: str = "smoke"
    seed: int = R6_SEED
    device: str = "cpu"
    deterministic: bool = True
    cpu_threads: int = 4
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    steps: int = 1
    intervention_weight: float = 1.0
    max_cross_entropy: float = 0.10
    required_predicted_classes: int = 7
    checkpoint_steps: int = 1
    time_budget_minutes: float = 5.0
    yield_ms: int = 0
    model: LadderModelSpec = LadderModelSpec()
    inherited_checkpoint: str = (
        "runs/stage2-r5-ladder-directml-821501/checkpoints/"
        "r5-00000600-final.pt"
    )
    inherited_checkpoint_sha256: str = R5_CHECKPOINT_SHA256
    inherited_frozen_config_sha256: str = R5_FROZEN_CONFIG_SHA256
    inherited_config_digest: str = R5_CONFIG_DIGEST
    inherited_partition_digest: str = R5_PARTITION_DIGEST
    inherited_state_digest: str = R5_RUNG1_STATE_DIGEST
    partition_digest: str = R6_PARTITION_DIGEST
    canonical_run_dir: str = "runs/stage2-r6-congruence-directml-821601"
    cpu_pause_percent: float = 92.0
    cpu_resume_percent: float = 75.0
    ram_pause_gb: float = 1.5
    ram_resume_gb: float = 2.5
    pressure_samples: int = 3
    recovery_samples: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if self.packet != R6_PACKET:
            raise ValueError("R6 packet identity changed")
        if self.revision != R6_REVISION or self.phase != R6_PHASE:
            raise ValueError("R6 revision or phase is invalid")
        if self.run_kind not in {"smoke", "calibration_only"}:
            raise ValueError("R6 run_kind must be smoke or calibration_only")
        if self.seed != R6_SEED:
            raise ValueError("R6 seed is frozen to 821601")
        if self.device not in {"cpu", "directml"}:
            raise ValueError("R6 device must be cpu or directml")
        if self.device == "directml" and self.deterministic:
            raise ValueError("DirectML R6 runs require deterministic=false")
        if self.cpu_threads <= 0:
            raise ValueError("R6 cpu_threads must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("R6 optimizer fields are invalid")
        if self.steps <= 0 or self.intervention_weight < 0.0:
            raise ValueError("R6 training budget or intervention weight is invalid")
        if self.max_cross_entropy < 0.0:
            raise ValueError("R6 cross-entropy threshold must be nonnegative")
        if not 1 <= self.required_predicted_classes <= 7:
            raise ValueError("R6 required_predicted_classes must be in [1, 7]")
        if self.checkpoint_steps <= 0 or self.time_budget_minutes <= 0.0:
            raise ValueError("R6 checkpoint or time budget is invalid")
        if self.yield_ms < 0:
            raise ValueError("R6 yield_ms must be nonnegative")
        if not 0.0 <= self.cpu_resume_percent < self.cpu_pause_percent <= 100.0:
            raise ValueError("R6 CPU guard thresholds are invalid")
        if not 0.0 < self.ram_pause_gb < self.ram_resume_gb:
            raise ValueError("R6 RAM guard thresholds are invalid")
        if self.pressure_samples <= 0 or self.recovery_samples <= 0:
            raise ValueError("R6 resource guard samples must be positive")
        if self.inherited_checkpoint_sha256 != R5_CHECKPOINT_SHA256:
            raise ValueError("R6 inherited checkpoint hash changed")
        if self.inherited_frozen_config_sha256 != R5_FROZEN_CONFIG_SHA256:
            raise ValueError("R6 inherited frozen-config hash changed")
        if self.inherited_config_digest != R5_CONFIG_DIGEST:
            raise ValueError("R6 inherited config digest changed")
        if self.inherited_partition_digest != R5_PARTITION_DIGEST:
            raise ValueError("R6 inherited partition digest changed")
        if self.inherited_state_digest != R5_RUNG1_STATE_DIGEST:
            raise ValueError("R6 inherited state digest changed")
        if self.partition_digest != R6_PARTITION_DIGEST:
            raise ValueError("R6 partition digest changed")
        self.model.validate()
        if self.run_kind == "calibration_only":
            frozen = Stage2CongruenceConfig(
                run_kind="calibration_only",
                device="directml",
                deterministic=False,
                steps=306,
                checkpoint_steps=17,
                time_budget_minutes=30.0,
                yield_ms=1,
            )
            if self.to_dict() != frozen.to_dict():
                raise ValueError("R6 calibration_only configuration is fully frozen")


_FIELDS = set(Stage2CongruenceConfig.__dataclass_fields__)


def stage2_congruence_config_from_dict(
    raw: dict[str, object],
) -> Stage2CongruenceConfig:
    unknown = set(raw) - _FIELDS
    if unknown:
        raise ValueError(f"unknown R6 config fields: {sorted(unknown)}")
    model_raw = raw.get("model", {})
    if not isinstance(model_raw, dict):
        raise TypeError("R6 model config must be an object")
    model_unknown = set(model_raw) - set(LadderModelSpec.__dataclass_fields__)
    if model_unknown:
        raise ValueError(f"unknown R6 model fields: {sorted(model_unknown)}")
    defaults = Stage2CongruenceConfig()
    values = {
        field: raw.get(field, getattr(defaults, field))
        for field in _FIELDS
        if field != "model"
    }
    config = Stage2CongruenceConfig(
        **values,
        model=LadderModelSpec(**model_raw),
    )
    config.validate()
    return config


def load_stage2_congruence_config(
    path: str | Path,
) -> Stage2CongruenceConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("R6 config root must be an object")
    return stage2_congruence_config_from_dict(raw)
