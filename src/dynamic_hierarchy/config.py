"""Configuration types for reproducible Stage 0 runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    vocab_size: int = 48
    repeat_length: int = 6
    binding_pairs: int = 4
    expression_depth: int = 3
    expression_variables: int = 6
    expression_values: int = 8

    def validate(self, max_length_scale: int) -> None:
        _positive_int("vocab_size", self.vocab_size)
        _positive_int("repeat_length", self.repeat_length)
        _positive_int("binding_pairs", self.binding_pairs)
        _positive_int("expression_depth", self.expression_depth)
        _positive_int("expression_variables", self.expression_variables)
        _positive_int("expression_values", self.expression_values)
        _positive_int("max_length_scale", max_length_scale)
        if self.repeat_length < 3:
            raise ValueError("repeat_length must be at least 3 so nonempty query prefixes can vary within a batch")
        if self.expression_depth < 2:
            raise ValueError("expression_depth must be at least 2 to require nested composition")
        if self.expression_variables < 2:
            raise ValueError("expression_variables must be at least 2")
        if self.expression_values < 2:
            raise ValueError("expression_values must be at least 2")
        expression_tokens = 8 + self.expression_values + self.expression_variables
        if expression_tokens > self.vocab_size:
            raise ValueError(
                "vocab_size is too small for expression syntax, values, and variables: "
                f"need at least {expression_tokens}, have {self.vocab_size}"
            )
        content_capacity = self.vocab_size - 4
        if content_capacity < 1:
            raise ValueError("vocab_size must reserve four special token IDs and at least one generated symbol")
        required_variables = self.binding_pairs * max_length_scale
        if required_variables > content_capacity:
            raise ValueError(
                "vocab_size is too small for unique variable bindings at the largest evaluation scale: "
                f"need {required_variables} generated symbols, have {content_capacity}"
            )


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 48
    heads: int = 4
    layers: int = 2
    feedforward_dim: int = 96
    dropout: float = 0.0

    def validate(self) -> None:
        _positive_int("embedding_dim", self.embedding_dim)
        _positive_int("heads", self.heads)
        _positive_int("layers", self.layers)
        _positive_int("feedforward_dim", self.feedforward_dim)
        if self.embedding_dim % self.heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 123
    device: str = "cpu"
    steps: int = 16
    warmup_steps: int = 0
    batch_size: int = 16
    learning_rate: float = 0.003
    eval_batches: int = 2
    eval_length_scales: tuple[int, ...] = (1, 2, 4)
    cpu_threads: int = 1
    deterministic: bool = True
    tasks: tuple[str, ...] = ("repeat_symbol", "variable_binding")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        _positive_int("steps", self.steps)
        _nonnegative_int("warmup_steps", self.warmup_steps)
        _positive_int("batch_size", self.batch_size)
        _positive_int("eval_batches", self.eval_batches)
        _positive_int("cpu_threads", self.cpu_threads)
        if self.device not in {"cpu", "directml"}:
            raise ValueError("device must be 'cpu' or 'directml'")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        if self.device == "directml" and self.deterministic:
            raise ValueError("DirectML requires deterministic=false because strict determinism is unavailable")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must not contain duplicates")
        unknown_tasks = set(self.tasks).difference({"repeat_symbol", "variable_binding", "nested_expression"})
        if unknown_tasks:
            raise ValueError(f"unknown tasks: {sorted(unknown_tasks)}")
        if not self.eval_length_scales:
            raise ValueError("eval_length_scales must not be empty")
        for scale in self.eval_length_scales:
            _positive_int("eval_length_scales entries", scale)
        if len(set(self.eval_length_scales)) != len(self.eval_length_scales):
            raise ValueError("eval_length_scales must not contain duplicates")
        self.model.validate()
        self.data.validate(max(self.eval_length_scales))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a JSON configuration while retaining small explicit defaults."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = ExperimentConfig(
        seed=raw.get("seed", 123),
        device=raw.get("device", "cpu"),
        steps=raw.get("steps", 16),
        warmup_steps=raw.get("warmup_steps", 0),
        batch_size=raw.get("batch_size", 16),
        learning_rate=raw.get("learning_rate", 0.003),
        eval_batches=raw.get("eval_batches", 2),
        eval_length_scales=tuple(raw.get("eval_length_scales", (1, 2, 4))),
        cpu_threads=raw.get("cpu_threads", 1),
        deterministic=raw.get("deterministic", True),
        tasks=tuple(raw.get("tasks", ("repeat_symbol", "variable_binding"))),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
    )
    config.validate()
    return config


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
