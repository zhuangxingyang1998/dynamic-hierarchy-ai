"""Frozen configuration contracts for the Stage 2 R5 arithmetic ladder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


R5_REVISION = "stage2-r5.1"
R5_PHASE = "arithmetic_ladder"


@dataclass(frozen=True)
class LadderModelSpec:
    hidden_dim: int = 48
    feedforward_dim: int = 96
    dropout: float = 0.0

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.feedforward_dim <= 0:
            raise ValueError("ladder model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ladder dropout must be in [0, 1)")


@dataclass(frozen=True)
class Stage2LadderConfig:
    revision: str = R5_REVISION
    phase: str = R5_PHASE
    run_kind: str = "smoke"
    seed: int = 821501
    device: str = "cpu"
    deterministic: bool = True
    cpu_threads: int = 4
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    rung1_steps: int = 1
    rung2_steps: int = 1
    rung3_steps: int = 1
    auxiliary_weight: float = 1.0
    min_final_accuracy: float = 0.0
    min_paired_both_correct: float = 0.0
    atomic_max_cross_entropy: float = 100.0
    recursive_max_cross_entropy: float = 100.0
    required_predicted_classes: int = 1
    max_partial_lookup_accuracy: float = 1.0
    max_opposite_tree_accuracy: float = 1.0
    max_fixed_tree_accuracy: float = 1.0
    min_structure_accuracy_drop: float = 0.0
    bridge_max_abs_difference: float = 1.0
    checkpoint_steps: int = 1
    time_budget_minutes: float = 5.0
    yield_ms: int = 0
    model: LadderModelSpec = LadderModelSpec()
    cpu_pause_percent: float = 92.0
    cpu_resume_percent: float = 75.0
    ram_pause_gb: float = 1.5
    ram_resume_gb: float = 2.5
    pressure_samples: int = 3
    recovery_samples: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if self.revision != R5_REVISION or self.phase != R5_PHASE:
            raise ValueError("R5.1 requires revision='stage2-r5.1' and phase='arithmetic_ladder'")
        if self.run_kind not in {"smoke", "calibration_only"}:
            raise ValueError("R5 run_kind must be smoke or calibration_only")
        if self.seed < 0:
            raise ValueError("R5 seed must be nonnegative")
        if self.device not in {"cpu", "directml"}:
            raise ValueError("R5 device must be cpu or directml")
        if self.device == "directml" and self.deterministic:
            raise ValueError("DirectML R5 runs require deterministic=false")
        if self.cpu_threads <= 0:
            raise ValueError("R5 cpu_threads must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("R5 optimizer fields are invalid")
        if min(self.rung1_steps, self.rung2_steps, self.rung3_steps) <= 0:
            raise ValueError("R5 rung update budgets must be positive")
        if self.auxiliary_weight < 0.0:
            raise ValueError("R5 auxiliary_weight must be nonnegative")
        if not 0.0 <= self.min_final_accuracy <= 1.0:
            raise ValueError("R5 min_final_accuracy must be in [0, 1]")
        if not 0.0 <= self.min_paired_both_correct <= 1.0:
            raise ValueError("R5 min_paired_both_correct must be in [0, 1]")
        if self.atomic_max_cross_entropy < 0.0 or self.recursive_max_cross_entropy < 0.0:
            raise ValueError("R5 cross-entropy gates must be nonnegative")
        if not 1 <= self.required_predicted_classes <= 7:
            raise ValueError("R5 required_predicted_classes must be in [1, 7]")
        for name, value in (
            ("max_partial_lookup_accuracy", self.max_partial_lookup_accuracy),
            ("max_opposite_tree_accuracy", self.max_opposite_tree_accuracy),
            ("max_fixed_tree_accuracy", self.max_fixed_tree_accuracy),
            ("min_structure_accuracy_drop", self.min_structure_accuracy_drop),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"R5 {name} must be in [0, 1]")
        if self.bridge_max_abs_difference < 0.0:
            raise ValueError("R5 bridge tolerance must be nonnegative")
        if self.checkpoint_steps <= 0 or self.time_budget_minutes <= 0.0:
            raise ValueError("R5 checkpoint and time budgets must be positive")
        if self.yield_ms < 0:
            raise ValueError("R5 yield_ms must be nonnegative")
        if not 0.0 <= self.cpu_resume_percent < self.cpu_pause_percent <= 100.0:
            raise ValueError("R5 CPU guard thresholds are invalid")
        if not 0.0 < self.ram_pause_gb < self.ram_resume_gb:
            raise ValueError("R5 RAM guard thresholds are invalid")
        if self.pressure_samples <= 0 or self.recovery_samples <= 0:
            raise ValueError("R5 resource guard sample counts must be positive")
        self.model.validate()
        if self.run_kind == "calibration_only":
            frozen = Stage2LadderConfig(
                run_kind="calibration_only",
                device="directml",
                deterministic=False,
                rung1_steps=300,
                rung2_steps=300,
                rung3_steps=300,
                min_final_accuracy=1.0,
                min_paired_both_correct=1.0,
                atomic_max_cross_entropy=0.05,
                recursive_max_cross_entropy=0.10,
                required_predicted_classes=7,
                max_partial_lookup_accuracy=0.50,
                max_opposite_tree_accuracy=0.10,
                max_fixed_tree_accuracy=0.50,
                min_structure_accuracy_drop=0.40,
                bridge_max_abs_difference=1e-5,
                checkpoint_steps=25,
                time_budget_minutes=30.0,
                yield_ms=1,
            )
            if self.to_dict() != frozen.to_dict():
                raise ValueError("R5 calibration_only configuration is fully frozen")


_FIELDS = set(Stage2LadderConfig.__dataclass_fields__)


def stage2_ladder_config_from_dict(raw: dict[str, object]) -> Stage2LadderConfig:
    unknown = set(raw) - _FIELDS
    if unknown:
        raise ValueError(f"unknown R5 config fields: {sorted(unknown)}")
    model_raw = raw.get("model", {})
    if not isinstance(model_raw, dict):
        raise TypeError("R5 model config must be an object")
    model_unknown = set(model_raw) - set(LadderModelSpec.__dataclass_fields__)
    if model_unknown:
        raise ValueError(f"unknown R5 model fields: {sorted(model_unknown)}")
    defaults = Stage2LadderConfig()
    values = {
        field: raw.get(field, getattr(defaults, field))
        for field in _FIELDS
        if field != "model"
    }
    config = Stage2LadderConfig(
        **values,
        model=LadderModelSpec(**model_raw),
    )
    config.validate()
    return config


def load_stage2_ladder_config(path: str | Path) -> Stage2LadderConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("R5 config root must be an object")
    return stage2_ladder_config_from_dict(raw)
