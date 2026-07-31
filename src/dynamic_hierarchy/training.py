"""Small training and evaluation loop shared by the CLI and tests."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn

from .backend import resolve_backend
from .config import ExperimentConfig
from .data import SyntheticTaskGenerator
from .model import SmallTransformerBaseline
from .optim import DirectMLCompatibleAdamWCore


@dataclass(frozen=True)
class RunMetrics:
    final_loss: float
    train_by_task: dict[str, "Accuracy"]
    evaluation_by_task_and_scale: dict[str, dict[str, "Accuracy"]]
    performance: "PerformanceMetrics"

    def to_dict(self) -> dict[str, object]:
        return {
            "final_loss": self.final_loss,
            "train_by_task": {task: metric.to_dict() for task, metric in self.train_by_task.items()},
            "evaluation_by_task_and_scale": {
                task: {scale: metric.to_dict() for scale, metric in scales.items()}
                for task, scales in self.evaluation_by_task_and_scale.items()
            },
            "performance": self.performance.to_dict(),
        }


@dataclass(frozen=True)
class Accuracy:
    correct: int
    total: int

    @property
    def value(self) -> float:
        return self.correct / self.total

    def to_dict(self) -> dict[str, float | int]:
        return {"correct": self.correct, "total": self.total, "accuracy": self.value}


@dataclass(frozen=True)
class PerformanceMetrics:
    backend: str
    device_name: str
    parameter_count: int
    training_seconds: float
    warmup_steps: int
    steps: int
    examples: int
    steps_per_second: float
    examples_per_second: float
    backward_completed: bool
    last_gradient_l1: float
    deterministic_requested: bool
    deterministic_algorithms_enabled: bool
    determinism_status: str
    synchronization_method: str
    timing_barrier: str
    optimizer: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return self.__dict__.copy()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def train(config: ExperimentConfig) -> RunMetrics:
    config.validate()
    backend = resolve_backend(config.device, config.cpu_threads, config.deterministic)
    set_seed(config.seed)
    model = SmallTransformerBaseline(config.data.vocab_size, config.model).to(backend.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = DirectMLCompatibleAdamWCore(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    generator = SyntheticTaskGenerator(config.data, seed=config.seed + 1)
    train_correct: dict[str, list[torch.Tensor]] = {task: [] for task in config.tasks}
    train_totals = {task: 0 for task in config.tasks}
    loss: torch.Tensor | None = None
    model.train()

    def update(step: int, measured: bool) -> torch.Tensor:
        batch = generator.batch(config.tasks[step % len(config.tasks)], config.batch_size).to(backend.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.token_ids, batch.position_features, batch.attention_mask)
        step_loss = loss_fn(logits, batch.labels)
        step_loss.backward()
        optimizer.step()
        if measured:
            train_correct[batch.task_name].append(logits.argmax(dim=-1).eq(batch.labels).sum())
            train_totals[batch.task_name] += batch.labels.numel()
        return step_loss

    for warmup_step in range(config.warmup_steps):
        loss = update(warmup_step, measured=False)

    timing_parameter = model.classifier.weight
    backend.synchronize(timing_parameter)
    training_started = time.perf_counter()
    for step in range(config.steps):
        loss = update(step, measured=True)

    assert loss is not None
    backend.synchronize(timing_parameter)
    training_seconds = time.perf_counter() - training_started
    # Scalar reads intentionally happen after the timed region and final parameter barrier.
    final_loss = backend.scalar(loss)
    final_gradient = next(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
    last_gradient_l1 = backend.scalar(final_gradient.detach().abs().sum())
    backward_completed = math.isfinite(last_gradient_l1) and last_gradient_l1 > 0.0

    model.eval()
    evaluation: dict[str, dict[str, Accuracy]] = {}
    with torch.no_grad():
        for task_index, task_name in enumerate(config.tasks):
            evaluation[task_name] = {}
            for scale in config.eval_length_scales:
                generator = SyntheticTaskGenerator(config.data, seed=config.seed + 10_000 + task_index * 100 + scale)
                correct = 0
                total = 0
                for _ in range(config.eval_batches):
                    batch = generator.batch(task_name, config.batch_size, length_scale=scale).to(backend.device)
                    logits = model(batch.token_ids, batch.position_features, batch.attention_mask)
                    correct += int(logits.argmax(dim=-1).eq(batch.labels).sum().detach().cpu().item())
                    total += batch.labels.numel()
                evaluation[task_name][str(scale)] = Accuracy(correct=correct, total=total)
    train_by_task = {
        task: Accuracy(
            correct=sum(int(value.detach().cpu().item()) for value in train_correct[task]),
            total=train_totals[task],
        )
        for task in config.tasks
    }
    examples = config.steps * config.batch_size
    performance = PerformanceMetrics(
        backend=backend.name,
        device_name=backend.device_name,
        parameter_count=parameter_count,
        training_seconds=training_seconds,
        warmup_steps=config.warmup_steps,
        steps=config.steps,
        examples=examples,
        steps_per_second=config.steps / training_seconds,
        examples_per_second=examples / training_seconds,
        backward_completed=backward_completed,
        last_gradient_l1=last_gradient_l1,
        deterministic_requested=config.deterministic,
        deterministic_algorithms_enabled=backend.deterministic_algorithms_enabled,
        determinism_status=backend.determinism_status,
        synchronization_method=backend.synchronization_method,
        timing_barrier="updated model classifier.weight synchronized before timer and after final optimizer.step",
        optimizer="DirectMLCompatibleAdamWCore (dense AdamW core subset; same implementation on CPU and DirectML)",
    )
    return RunMetrics(
        final_loss=final_loss,
        train_by_task=train_by_task,
        evaluation_by_task_and_scale=evaluation,
        performance=performance,
    )
