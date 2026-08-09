"""Typed configuration for the Stage 2 R2 precedence-query experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


REQUIRED_STAGE2_CONTROLS = (
    "A-Q-param",
    "A-Q-flop",
    "A-recur",
    "B-query",
    "B-noQ-router",
    "B-sham",
    "F-stop",
    "F-left",
    "F-right",
    "F-add",
    "F-sub",
    "D-true",
    "D-sham",
)


@dataclass(frozen=True)
class Stage2Profile:
    name: str
    leaf_count: int
    operator_pattern: str
    category: str
    shape_partition: str = "train"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Stage 2 profile name must not be empty")
        if self.leaf_count < 3:
            raise ValueError("Stage 2 precedence profiles require at least three leaves")
        if len(self.operator_pattern) != self.leaf_count - 1:
            raise ValueError("operator_pattern length must equal leaf_count - 1")
        if set(self.operator_pattern) != {"+", "-"}:
            raise ValueError("operator_pattern must contain both '+' and '-' and no other symbols")
        if "-+" not in self.operator_pattern:
            raise ValueError(
                "operator_pattern is precedence-insensitive; an allowed pattern must contain '-+'"
            )
        if self.category not in {
            "train",
            "in_distribution",
            "length_extrapolation",
            "topology_extrapolation",
        }:
            raise ValueError(f"unsupported Stage 2 profile category: {self.category}")
        if self.shape_partition not in {"train", "heldout"}:
            raise ValueError("shape_partition must be 'train' or 'heldout'")
        if self.category == "train" and self.shape_partition != "train":
            raise ValueError("training profiles must use the train shape partition")


@dataclass(frozen=True)
class Stage2ModelSpec:
    vocab_size: int = 64
    hidden_dim: int = 64
    heads: int = 4
    layers: int = 1
    feedforward_dim: int = 128
    dropout: float = 0.0
    temperature: float = 1.0

    def validate(self) -> None:
        if self.vocab_size < 17:
            raise ValueError("Stage 2 vocab_size must include both precedence query tokens")
        if self.hidden_dim <= 0 or self.heads <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.layers <= 0 or self.feedforward_dim <= 0:
            raise ValueError("layers and feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("straight-through temperature must be positive")


def _default_train_profiles() -> tuple[Stage2Profile, ...]:
    return (
        Stage2Profile("train_n4_alternating", 4, "-+-", "train"),
        Stage2Profile("train_n5_run", 5, "--++", "train"),
        Stage2Profile("train_n6_alternating", 6, "-+-+-", "train"),
    )


def _default_evaluation_profiles() -> tuple[Stage2Profile, ...]:
    return (
        Stage2Profile("id_n5_alternating", 5, "-+-+", "in_distribution"),
        Stage2Profile("length_n8_alternating", 8, "-+-+-+-", "length_extrapolation", "heldout"),
        Stage2Profile("topology_n6_runs", 6, "--+--", "topology_extrapolation", "heldout"),
    )


def _default_a_param_model() -> Stage2ModelSpec:
    return Stage2ModelSpec(layers=3, feedforward_dim=128)


def _default_a_flop_model() -> Stage2ModelSpec:
    return Stage2ModelSpec(layers=3, feedforward_dim=80)


@dataclass(frozen=True)
class Stage2Config:
    revision: str = "stage2-r2"
    run_kind: str = "smoke"
    seed: int = 821101
    device: str = "cpu"
    deterministic: bool = True
    cpu_threads: int = 4
    optimizer_steps: int = 2
    learning_rate: float = 0.001
    families_per_stratum: int = 42
    max_generation_attempts_per_family: int = 512
    checkpoint_steps: int = 1
    evaluation_blocks: int = 1
    time_budget_minutes: float = 5.0
    yield_ms: int = 1
    cpu_pause_percent: float = 90.0
    cpu_resume_percent: float = 75.0
    ram_pause_gb: float = 4.0
    ram_resume_gb: float = 6.0
    pressure_samples: int = 3
    recovery_samples: int = 2
    controls: tuple[str, ...] = REQUIRED_STAGE2_CONTROLS
    train_profiles: tuple[Stage2Profile, ...] = field(default_factory=_default_train_profiles)
    evaluation_profiles: tuple[Stage2Profile, ...] = field(default_factory=_default_evaluation_profiles)
    model: Stage2ModelSpec = field(default_factory=Stage2ModelSpec)
    a_param_model: Stage2ModelSpec = field(default_factory=_default_a_param_model)
    a_flop_model: Stage2ModelSpec = field(default_factory=_default_a_flop_model)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if self.revision != "stage2-r2":
            raise ValueError("the active Stage 2 implementation requires revision='stage2-r2'")
        if self.run_kind not in {"smoke", "calibration_only"}:
            raise ValueError("run_kind must be 'smoke' or 'calibration_only'")
        if self.device not in {"cpu", "directml"}:
            raise ValueError("device must be 'cpu' or 'directml'")
        if self.device == "directml" and self.deterministic:
            raise ValueError("DirectML Stage 2 runs require deterministic=false")
        if self.cpu_threads <= 0 or self.optimizer_steps <= 0:
            raise ValueError("cpu_threads and optimizer_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.families_per_stratum <= 0 or self.families_per_stratum % 42:
            raise ValueError("families_per_stratum must be a positive multiple of 42")
        if self.max_generation_attempts_per_family <= 0:
            raise ValueError("max_generation_attempts_per_family must be positive")
        if self.checkpoint_steps <= 0 or self.evaluation_blocks <= 0:
            raise ValueError("checkpoint_steps and evaluation_blocks must be positive")
        if self.time_budget_minutes <= 0.0 or self.yield_ms < 0:
            raise ValueError("time_budget_minutes must be positive and yield_ms nonnegative")
        if not 0.0 <= self.cpu_resume_percent < self.cpu_pause_percent <= 100.0:
            raise ValueError("CPU resource thresholds must satisfy 0 <= resume < pause <= 100")
        if not 0.0 < self.ram_pause_gb < self.ram_resume_gb:
            raise ValueError("RAM resource thresholds must satisfy 0 < pause < resume")
        if self.pressure_samples <= 0 or self.recovery_samples <= 0:
            raise ValueError("resource hysteresis sample counts must be positive")
        if self.run_kind == "calibration_only":
            if self.seed != 821101 or self.optimizer_steps > 600:
                raise ValueError("R2 calibration requires seed 821101 and at most 600 steps")
            if self.evaluation_blocks < 10 or self.time_budget_minutes > 30.0:
                raise ValueError("R2 calibration requires >=10 evaluation blocks and <=30 minutes")
        if tuple(self.controls) != REQUIRED_STAGE2_CONTROLS:
            raise ValueError("Stage 2 R2 requires the complete ordered control matrix")
        if not self.train_profiles or not self.evaluation_profiles:
            raise ValueError("Stage 2 requires training and evaluation profiles")
        for profile in (*self.train_profiles, *self.evaluation_profiles):
            profile.validate()
        if any(profile.category != "train" for profile in self.train_profiles):
            raise ValueError("all train_profiles must have category='train'")
        if any(profile.category == "train" for profile in self.evaluation_profiles):
            raise ValueError("evaluation_profiles cannot contain training profiles")
        names = [profile.name for profile in (*self.train_profiles, *self.evaluation_profiles)]
        if len(names) != len(set(names)):
            raise ValueError("Stage 2 profile names must be unique")
        self.model.validate()
        self.a_param_model.validate()
        self.a_flop_model.validate()


def _profile(raw: Stage2Profile | dict[str, object]) -> Stage2Profile:
    if isinstance(raw, Stage2Profile):
        return raw
    return Stage2Profile(
        name=str(raw["name"]),
        leaf_count=int(raw["leaf_count"]),
        operator_pattern=str(raw["operator_pattern"]),
        category=str(raw["category"]),
        shape_partition=str(raw.get("shape_partition", "train")),
    )


def stage2_config_from_dict(raw: dict[str, object]) -> Stage2Config:
    model_raw = dict(raw.get("model", {}))
    model = Stage2ModelSpec(**model_raw)
    a_param_model = Stage2ModelSpec(**dict(raw.get("a_param_model", asdict(_default_a_param_model()))))
    a_flop_model = Stage2ModelSpec(**dict(raw.get("a_flop_model", asdict(_default_a_flop_model()))))
    config = Stage2Config(
        revision=str(raw.get("revision", "stage2-r2")),
        run_kind=str(raw.get("run_kind", "smoke")),
        seed=int(raw.get("seed", 821101)),
        device=str(raw.get("device", "cpu")),
        deterministic=bool(raw.get("deterministic", True)),
        cpu_threads=int(raw.get("cpu_threads", 4)),
        optimizer_steps=int(raw.get("optimizer_steps", 2)),
        learning_rate=float(raw.get("learning_rate", 0.001)),
        families_per_stratum=int(raw.get("families_per_stratum", 42)),
        max_generation_attempts_per_family=int(
            raw.get("max_generation_attempts_per_family", 512)
        ),
        checkpoint_steps=int(raw.get("checkpoint_steps", 1)),
        evaluation_blocks=int(raw.get("evaluation_blocks", 1)),
        time_budget_minutes=float(raw.get("time_budget_minutes", 5.0)),
        yield_ms=int(raw.get("yield_ms", 1)),
        cpu_pause_percent=float(raw.get("cpu_pause_percent", 90.0)),
        cpu_resume_percent=float(raw.get("cpu_resume_percent", 75.0)),
        ram_pause_gb=float(raw.get("ram_pause_gb", 4.0)),
        ram_resume_gb=float(raw.get("ram_resume_gb", 6.0)),
        pressure_samples=int(raw.get("pressure_samples", 3)),
        recovery_samples=int(raw.get("recovery_samples", 2)),
        controls=tuple(raw.get("controls", REQUIRED_STAGE2_CONTROLS)),
        train_profiles=tuple(
            _profile(item) for item in raw.get("train_profiles", _default_train_profiles())
        ),
        evaluation_profiles=tuple(
            _profile(item)
            for item in raw.get("evaluation_profiles", _default_evaluation_profiles())
        ),
        model=model,
        a_param_model=a_param_model,
        a_flop_model=a_flop_model,
    )
    config.validate()
    return config


def load_stage2_config(path: str | Path) -> Stage2Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 2 config root must be an object")
    return stage2_config_from_dict(raw)
