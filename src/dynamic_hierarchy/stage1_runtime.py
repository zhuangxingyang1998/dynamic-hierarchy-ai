"""Revised Stage 1 paired training, evaluation, gates, and atomic checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .backend import Backend, resolve_backend
from .data import SyntheticBatch
from .model import SmallTransformerBaseline, StructureDiagnostics, TrueStructureDiagnosticD
from .optim import DirectMLCompatibleAdamWCore
from .provenance import source_manifest
from .stage1_config import EvaluationSplit, Stage1Config, TrainingProfile
from .stage1_data import (
    SHAM_MAPPING_VERSION,
    RevisedStage1Generator,
    shape_catalog,
    shape_id,
    sham_structure,
)


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _hash_set_digest(values: set[str]) -> str:
    encoded = "\n".join(sorted(values)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class Stage1Trainer:
    """Train A, privileged D-true, and architecture-matched D-sham fairly."""

    MODEL_NAMES = ("A", "D_true", "D_sham")

    def __init__(
        self,
        config: Stage1Config,
        run_dir: Path,
        source_root: Path,
        snapshot_manifest_hash: str,
    ) -> None:
        config.validate()
        if not config.revised:
            raise ValueError("the active Stage1Trainer requires revision='revised-v2'")
        self.config = config
        self.run_dir = run_dir
        self.backend: Backend = resolve_backend(
            config.device,
            config.cpu_threads,
            config.deterministic,
        )
        torch.manual_seed(config.seed)
        model_a = SmallTransformerBaseline(
            config.data.vocab_size,
            config.model_a,
            output_classes=config.data.expression_values,
        ).to(self.backend.device)
        model_d_true = TrueStructureDiagnosticD(
            config.data.vocab_size,
            config.model_d,
            output_classes=config.data.expression_values,
        ).to(self.backend.device)
        model_d_sham = copy.deepcopy(model_d_true).to(self.backend.device)
        self.models: dict[str, nn.Module] = {
            "A": model_a,
            "D_true": model_d_true,
            "D_sham": model_d_sham,
        }
        self.optimizers = {
            name: DirectMLCompatibleAdamWCore(model.parameters(), lr=config.learning_rate)
            for name, model in self.models.items()
        }
        self.loss_fn = nn.CrossEntropyLoss()
        self.generator = RevisedStage1Generator(
            config.data,
            seed=config.seed + 1,
            operand_mode=config.operand_mode,
        )
        self.global_step = 0
        self.elapsed_before_resume = 0.0
        self.process_started = time.monotonic()
        self.latest_loss: dict[str, float | None] = {name: None for name in self.MODEL_NAMES}
        self.latest_evaluation: dict[str, object] = {}
        self.final_evaluation: dict[str, object] = {}
        self.gate_result: dict[str, object] = {}
        self.stage_boundary_evaluations: dict[str, dict[str, object]] = {}
        self.foundation_gate_result: dict[str, object] = {}
        self.cumulative: dict[str, dict[str, float | int]] = {
            name: {
                "optimizer_updates": 0,
                "examples": 0,
                "correct": 0,
                "loss_sum": 0.0,
                "node_count_sum": 0,
                "maximum_tree_depth_sum": 0,
                "combined_nodes": 0,
                "compose_module_calls": 0,
            }
            for name in self.MODEL_NAMES
        }
        self.curriculum_metrics: dict[str, dict[str, object]] = {
            stage.name: {
                "paired_steps": 0,
                "examples": 0,
                "generation_attempts": 0,
                "accepted_examples": 0,
                "structural_rejections": 0,
                "profile_steps": {profile.name: 0 for profile in stage.profiles},
                "label_counts": [0] * config.data.expression_values,
            }
            for stage in config.curriculum
        }
        self.training_content_hashes: set[str] = set()
        self.pre_final_evaluation_content_hashes: set[str] = set()
        self.historical_final_evaluation_content_hashes: set[str] = set()
        self.training_duplicate_contents = 0
        self.observed_training_shape_ids: set[str] = set()
        self.parameter_counts = {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in self.models.items()
        }
        if self.parameter_counts["D_true"] != self.parameter_counts["D_sham"]:
            raise RuntimeError("D-true and D-sham parameter counts must be identical")
        self.source_manifest = source_manifest(source_root)
        self.snapshot_manifest_hash = snapshot_manifest_hash
        self.last_checkpoint: str | None = None
        self.last_checkpoint_step: int | None = None

    def elapsed_seconds(self) -> float:
        return self.elapsed_before_resume + time.monotonic() - self.process_started

    def session_elapsed_seconds(self) -> float:
        return time.monotonic() - self.process_started

    def curriculum_position(self, step: int | None = None) -> dict[str, object]:
        target_step = self.global_step if step is None else step
        consumed = 0
        for stage_index, stage in enumerate(self.config.curriculum):
            if target_step < consumed + stage.steps:
                step_in_stage = target_step - consumed
                profile_index = step_in_stage % len(stage.profiles)
                profile = stage.profiles[profile_index]
                return {
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "step_in_stage": step_in_stage,
                    "stage_steps": stage.steps,
                    "profile_index": profile_index,
                    "profile": {
                        "name": profile.name,
                        "depth": profile.depth,
                        "topology": profile.topology,
                    },
                    "complete": False,
                }
            consumed += stage.steps
        final_stage = self.config.curriculum[-1]
        return {
            "stage_index": len(self.config.curriculum) - 1,
            "stage_name": final_stage.name,
            "step_in_stage": final_stage.steps,
            "stage_steps": final_stage.steps,
            "profile_index": None,
            "profile": None,
            "complete": True,
        }

    def _current_profile(self) -> tuple[str, TrainingProfile]:
        position = self.curriculum_position()
        if position["complete"]:
            raise RuntimeError("curriculum is complete")
        stage_name = str(position["stage_name"])
        stage = self.config.curriculum[int(position["stage_index"])]
        return stage_name, stage.profiles[int(position["profile_index"])]

    def _forward(
        self,
        model_name: str,
        batch: SyntheticBatch,
    ) -> tuple[torch.Tensor, StructureDiagnostics | None]:
        model = self.models[model_name]
        if model_name == "A":
            return model(batch.token_ids, batch.position_features, batch.attention_mask), None
        structure = (
            batch.structure
            if model_name == "D_true"
            else sham_structure(batch.structure, batch.token_ids)
        )
        diagnostics = model(
            batch.token_ids,
            batch.position_features,
            batch.attention_mask,
            structure,
        )
        return diagnostics.logits, diagnostics

    def _record_generation(self, stage_name: str, profile: TrainingProfile, batch: SyntheticBatch) -> None:
        generation = batch.generation
        if generation is None:
            raise RuntimeError("revised batches must include generation accounting")
        stage = self.curriculum_metrics[stage_name]
        stage["paired_steps"] = int(stage["paired_steps"]) + 1
        stage["examples"] = int(stage["examples"]) + generation.accepted
        stage["generation_attempts"] = int(stage["generation_attempts"]) + generation.attempts
        stage["accepted_examples"] = int(stage["accepted_examples"]) + generation.accepted
        stage["structural_rejections"] = (
            int(stage["structural_rejections"]) + generation.structural_rejections
        )
        profile_steps = stage["profile_steps"]
        profile_steps[profile.name] = int(profile_steps[profile.name]) + 1
        label_counts = stage["label_counts"]
        for index, count in enumerate(generation.label_counts):
            label_counts[index] = int(label_counts[index]) + count
        for content_hash in generation.content_hashes:
            if content_hash in self.training_content_hashes:
                self.training_duplicate_contents += 1
            self.training_content_hashes.add(content_hash)
        self.observed_training_shape_ids.update(generation.shape_ids)

    def train_pair(self) -> dict[str, float]:
        stage_name, profile = self._current_profile()
        effective_batch = self.generator.batch(
            self.config.effective_batch_size,
            profile.depth,
            profile.topology,
            max_structural_attempts_per_example=self.config.max_generation_attempts_per_example,
            shape_partition="train",
        )
        self._record_generation(stage_name, profile, effective_batch)
        cpu_batches = effective_batch.split(self.config.microbatch_size)
        pair_losses: dict[str, float] = {}
        for model_name in self.MODEL_NAMES:
            model = self.models[model_name]
            model.train()
            optimizer = self.optimizers[model_name]
            optimizer.zero_grad(set_to_none=True)
            detached_losses: list[torch.Tensor] = []
            detached_correct: list[torch.Tensor] = []
            node_count_sum = 0
            maximum_depth_sum = 0
            combined_nodes = 0
            compose_calls = 0
            for cpu_batch in cpu_batches:
                batch = cpu_batch.to(self.backend.device)
                logits, diagnostics = self._forward(model_name, batch)
                loss = self.loss_fn(logits, batch.labels)
                (loss / self.config.gradient_accumulation).backward()
                detached_losses.append(loss.detach())
                detached_correct.append(logits.detach().argmax(dim=-1).eq(batch.labels).sum())
                if diagnostics is not None:
                    node_count_sum += sum(diagnostics.node_counts)
                    maximum_depth_sum += sum(diagnostics.maximum_tree_depths)
                    combined_nodes += sum(diagnostics.combined_nodes)
                    compose_calls += diagnostics.compose_module_calls
            optimizer.step()
            self.backend.synchronize(next(model.parameters()))
            if self.config.yield_ms:
                time.sleep(self.config.yield_ms / 1000.0)
            average_loss = sum(self.backend.scalar(value) for value in detached_losses) / len(
                detached_losses
            )
            correct = sum(int(value.detach().cpu().item()) for value in detached_correct)
            state = self.cumulative[model_name]
            state["optimizer_updates"] = int(state["optimizer_updates"]) + 1
            state["examples"] = int(state["examples"]) + self.config.effective_batch_size
            state["correct"] = int(state["correct"]) + correct
            state["loss_sum"] = float(state["loss_sum"]) + average_loss
            state["node_count_sum"] = int(state["node_count_sum"]) + node_count_sum
            state["maximum_tree_depth_sum"] = (
                int(state["maximum_tree_depth_sum"]) + maximum_depth_sum
            )
            state["combined_nodes"] = int(state["combined_nodes"]) + combined_nodes
            state["compose_module_calls"] = int(state["compose_module_calls"]) + compose_calls
            self.latest_loss[model_name] = average_loss
            pair_losses[model_name] = average_loss
        self.global_step += 1
        return pair_losses

    def _evaluation_generator_seed(self, split: EvaluationSplit, evaluation_seed: int) -> int:
        split_component = int.from_bytes(
            hashlib.sha256(split.name.encode("utf-8")).digest()[:4],
            "little",
        )
        return evaluation_seed + split_component

    def _evaluate_split_seed(
        self,
        split: EvaluationSplit,
        evaluation_seed: int,
        examples: int,
        batch_size: int,
        *,
        enforce_content_exclusion: bool = False,
        training_content_hashes: set[str] | None = None,
        prior_evaluation_content_hashes: set[str] | None = None,
        accepted_evaluation_hashes: set[str] | None = None,
        historical_final_evaluation_content_hashes: set[str] | None = None,
    ) -> tuple[dict[str, object], set[str], set[str]]:
        if enforce_content_exclusion and accepted_evaluation_hashes is None:
            raise ValueError("active content exclusion requires a shared evaluation hash set")
        generator = RevisedStage1Generator(
            self.config.data,
            self._evaluation_generator_seed(split, evaluation_seed),
            operand_mode=self.config.operand_mode,
        )
        model_stats = {
            name: {
                "correct": 0,
                "cross_entropy_sum": 0.0,
                "prediction_counts": [0] * self.config.data.expression_values,
                "node_count_sum": 0,
                "maximum_tree_depth_sum": 0,
                "combined_nodes": 0,
                "compose_module_calls": 0,
            }
            for name in self.MODEL_NAMES
        }
        label_counts = [0] * self.config.data.expression_values
        paired = {"A_only": 0, "D_true_only": 0, "both": 0, "neither": 0}
        paired_d_sham = {
            "D_true_only": 0,
            "D_sham_only": 0,
            "both": 0,
            "neither": 0,
        }
        paired_correctness_masks: list[int] = []
        attempts = 0
        rejections = 0
        content_hashes: set[str] = set()
        duplicate_contents = 0
        exclusion_accounting: dict[str, int | float] = {
            "candidate_examples": 0,
            "accepted": 0,
            "training_content_exclusions": 0,
            "prior_evaluation_content_exclusions": 0,
            "historical_final_evaluation_content_exclusions": 0,
            "evaluation_content_exclusions": 0,
            "label_quota_rejections": 0,
            "structural_attempts": 0,
            "structural_rejections": 0,
        }
        shape_ids: set[str] = set()
        for model in self.models.values():
            model.eval()
        with torch.no_grad():
            for _ in range(examples // batch_size):
                if enforce_content_exclusion:
                    cpu_batch, batch_exclusions = generator.batch_excluding_content(
                        batch_size,
                        split.depth,
                        split.topology,
                        training_content_hashes=training_content_hashes or set(),
                        prior_evaluation_content_hashes=(
                            prior_evaluation_content_hashes or set()
                        ),
                        accepted_evaluation_hashes=accepted_evaluation_hashes,
                        max_structural_attempts_per_example=(
                            self.config.max_generation_attempts_per_example
                        ),
                        max_content_attempts_per_example=(
                            self.config.max_evaluation_generation_attempts_per_example
                        ),
                        shape_partition=split.shape_partition,
                        historical_final_evaluation_content_hashes=(
                            historical_final_evaluation_content_hashes
                        ),
                    )
                    for name, value in batch_exclusions.items():
                        if name in exclusion_accounting:
                            exclusion_accounting[name] += value
                else:
                    cpu_batch = generator.batch(
                        batch_size,
                        split.depth,
                        split.topology,
                        max_structural_attempts_per_example=(
                            self.config.max_generation_attempts_per_example
                        ),
                        shape_partition=split.shape_partition,
                    )
                generation = cpu_batch.generation
                attempts += generation.attempts
                rejections += generation.structural_rejections
                shape_ids.update(generation.shape_ids)
                for content_hash in generation.content_hashes:
                    if content_hash in content_hashes:
                        duplicate_contents += 1
                    content_hashes.add(content_hash)
                for index, count in enumerate(generation.label_counts):
                    label_counts[index] += count
                batch = cpu_batch.to(self.backend.device)
                correctness: dict[str, torch.Tensor] = {}
                for model_name in self.MODEL_NAMES:
                    logits, diagnostics = self._forward(model_name, batch)
                    predictions = logits.argmax(dim=-1)
                    correct_tensor = predictions.eq(batch.labels)
                    correctness[model_name] = correct_tensor.detach().cpu()
                    stats = model_stats[model_name]
                    stats["correct"] += int(correct_tensor.sum().detach().cpu().item())
                    loss_sum = nn.functional.cross_entropy(
                        logits,
                        batch.labels,
                        reduction="sum",
                    )
                    stats["cross_entropy_sum"] += self.backend.scalar(loss_sum)
                    prediction_counts = torch.bincount(
                        predictions.detach().cpu(),
                        minlength=self.config.data.expression_values,
                    )
                    for index, count in enumerate(prediction_counts.tolist()):
                        stats["prediction_counts"][index] += int(count)
                    if diagnostics is not None:
                        stats["node_count_sum"] += sum(diagnostics.node_counts)
                        stats["maximum_tree_depth_sum"] += sum(
                            diagnostics.maximum_tree_depths
                        )
                        stats["combined_nodes"] += sum(diagnostics.combined_nodes)
                        stats["compose_module_calls"] += diagnostics.compose_module_calls
                a_correct = correctness["A"]
                d_correct = correctness["D_true"]
                sham_correct = correctness["D_sham"]
                paired["A_only"] += int((a_correct & ~d_correct).sum().item())
                paired["D_true_only"] += int((~a_correct & d_correct).sum().item())
                paired["both"] += int((a_correct & d_correct).sum().item())
                paired["neither"] += int((~a_correct & ~d_correct).sum().item())
                paired_d_sham["D_true_only"] += int(
                    (d_correct & ~sham_correct).sum().item()
                )
                paired_d_sham["D_sham_only"] += int(
                    (~d_correct & sham_correct).sum().item()
                )
                paired_d_sham["both"] += int((d_correct & sham_correct).sum().item())
                paired_d_sham["neither"] += int(
                    (~d_correct & ~sham_correct).sum().item()
                )
                if self.config.formal_evaluation and enforce_content_exclusion:
                    masks = (
                        a_correct.to(torch.uint8)
                        + 2 * d_correct.to(torch.uint8)
                        + 4 * sham_correct.to(torch.uint8)
                    )
                    paired_correctness_masks.extend(int(value) for value in masks.tolist())

        majority = max(label_counts) / examples
        models: dict[str, object] = {}
        for model_name, stats in model_stats.items():
            item = {
                "correct": stats["correct"],
                "total": examples,
                "accuracy": stats["correct"] / examples,
                "cross_entropy": stats["cross_entropy_sum"] / examples,
                "prediction_counts": stats["prediction_counts"],
                "distinct_predicted_classes": sum(
                    count > 0 for count in stats["prediction_counts"]
                ),
            }
            if model_name != "A":
                item.update(
                    {
                        "average_node_count": stats["node_count_sum"] / examples,
                        "average_maximum_tree_depth": (
                            stats["maximum_tree_depth_sum"] / examples
                        ),
                        "average_combined_nodes": stats["combined_nodes"] / examples,
                        "compose_module_calls": stats["compose_module_calls"],
                    }
                )
            models[model_name] = item
        result = {
            "evaluation_seed": evaluation_seed,
            "models": models,
            "label_counts": label_counts,
            "majority_baseline": majority,
            "paired_outcomes": paired,
            "paired_comparisons": {
                "A_vs_D_true": paired,
                "D_true_vs_D_sham": paired_d_sham,
            },
            "generation": {
                "attempts": attempts,
                "accepted": examples,
                "acceptance_rate": examples / attempts,
                "structural_rejections": rejections,
                "duplicate_contents": duplicate_contents,
                "active_content_exclusion": enforce_content_exclusion,
                "content_exclusion": exclusion_accounting,
            },
            "shape_ids": sorted(shape_ids),
            "content_hash_count": len(content_hashes),
            "content_hash_digest": _hash_set_digest(content_hashes),
        }
        if self.config.formal_evaluation and enforce_content_exclusion:
            if (
                len(paired_correctness_masks) != examples
                or len(content_hashes) != examples
            ):
                raise RuntimeError("formal paired sample data is incomplete or duplicated")
            result["paired_sample_data"] = {
                "schema_version": 1,
                "encoding": (
                    "correctness bitmask: bit0=A, bit1=D_true, bit2=D_sham"
                ),
                "sample_count": examples,
                "content_hash_digest": _hash_set_digest(content_hashes),
                "correctness_masks": paired_correctness_masks,
            }
        return result, content_hashes, shape_ids

    @staticmethod
    def _range(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }

    def _aggregate_seed_results(self, seeds: dict[str, dict[str, object]]) -> dict[str, object]:
        models: dict[str, object] = {}
        for model_name in self.MODEL_NAMES:
            models[model_name] = {
                "accuracy": self._range(
                    [float(item["models"][model_name]["accuracy"]) for item in seeds.values()]
                ),
                "cross_entropy": self._range(
                    [
                        float(item["models"][model_name]["cross_entropy"])
                        for item in seeds.values()
                    ]
                ),
            }
        return {
            "models": models,
            "D_true_minus_A_accuracy": self._range(
                [
                    float(item["models"]["D_true"]["accuracy"])
                    - float(item["models"]["A"]["accuracy"])
                    for item in seeds.values()
                ]
            ),
            "D_true_minus_D_sham_accuracy": self._range(
                [
                    float(item["models"]["D_true"]["accuracy"])
                    - float(item["models"]["D_sham"]["accuracy"])
                    for item in seeds.values()
                ]
            ),
            "majority_baseline": self._range(
                [float(item["majority_baseline"]) for item in seeds.values()]
            ),
        }

    def evaluate_heartbeat(self) -> dict[str, object]:
        split = next(
            split
            for split in self.config.evaluation_splits
            if split.category == "in_distribution"
        )
        result, hashes, _ = self._evaluate_split_seed(
            split,
            self.config.eval_seeds[0],
            self.config.heartbeat_examples,
            self.config.heartbeat_examples,
        )
        self.pre_final_evaluation_content_hashes.update(hashes)
        self.latest_evaluation = {
            "kind": "heartbeat_candidate_only",
            "split": split.name,
            "examples": self.config.heartbeat_examples,
            "result": result,
        }
        return self.latest_evaluation

    def evaluate(self) -> dict[str, object]:
        return self.evaluate_heartbeat()

    def _completed_stage_name_at_current_step(self) -> str | None:
        consumed = 0
        for stage in self.config.curriculum:
            consumed += stage.steps
            if self.global_step == consumed:
                return stage.name
        return None

    def evaluate_pending_stage_boundary(self) -> dict[str, object] | None:
        """Evaluate fixed literal C0/C1 datasets exactly once at each saved boundary."""

        if self.config.operand_mode != "literal":
            return None
        stage_name = self._completed_stage_name_at_current_step()
        if stage_name is None or stage_name in self.stage_boundary_evaluations:
            return None
        stage_index = next(
            index for index, stage in enumerate(self.config.curriculum) if stage.name == stage_name
        )
        tasks = (
            ("C0", 0, "leaf"),
            *((("C1", 1, "skew"),) if stage_index >= 1 else ()),
        )
        evaluation: dict[str, object] = {
            "stage_name": stage_name,
            "global_step": self.global_step,
            "examples_per_task": self.config.foundation_eval_examples,
            "evaluation_seed": self.config.foundation_eval_seed,
            "fixed_balanced": True,
            "tasks": {},
        }
        for task_name, depth, topology in tasks:
            split = EvaluationSplit(
                name=f"literal_foundation_{task_name}",
                depth=depth,
                topology=topology,
                category="in_distribution",
                shape_partition="train",
                required_above_majority=True,
            )
            result, hashes, _ = self._evaluate_split_seed(
                split,
                self.config.foundation_eval_seed,
                self.config.foundation_eval_examples,
                self.config.foundation_eval_batch_size,
            )
            evaluation["tasks"][task_name] = result
            self.pre_final_evaluation_content_hashes.update(hashes)
        self.stage_boundary_evaluations[stage_name] = evaluation
        self.foundation_gate_result = self.compute_foundation_gate(evaluation)
        return evaluation

    def compute_foundation_gate(
        self,
        boundary_evaluation: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if self.config.operand_mode != "literal":
            return {
                "passed": False,
                "eligible": False,
                "reason": "bound-variable is a separate harder axis, not a structural gate",
            }
        if boundary_evaluation is None:
            final_name = self.config.curriculum[-1].name
            boundary_evaluation = self.stage_boundary_evaluations.get(final_name)
        if not boundary_evaluation or "C1" not in boundary_evaluation["tasks"]:
            return {
                "passed": False,
                "eligible": True,
                "reason": "fixed C0/C1 stage-boundary evaluation is incomplete",
            }
        examples = int(boundary_evaluation["examples_per_task"])
        tasks = boundary_evaluation["tasks"]
        conditions = {
            "minimum_fixed_examples": examples >= 700,
            "C0:A": float(tasks["C0"]["models"]["A"]["accuracy"])
            >= self.config.foundation_c0_min_accuracy,
            "C0:D_true": float(tasks["C0"]["models"]["D_true"]["accuracy"])
            >= self.config.foundation_c0_min_accuracy,
            "C1:A": float(tasks["C1"]["models"]["A"]["accuracy"])
            >= self.config.foundation_c1_min_accuracy,
            "C1:D_true": float(tasks["C1"]["models"]["D_true"]["accuracy"])
            >= self.config.foundation_c1_min_accuracy,
        }
        return {
            "passed": all(conditions.values()),
            "eligible": True,
            "stage_name": boundary_evaluation["stage_name"],
            "global_step": boundary_evaluation["global_step"],
            "examples_per_task": examples,
            "conditions": conditions,
            "thresholds": {
                "C0": self.config.foundation_c0_min_accuracy,
                "C1": self.config.foundation_c1_min_accuracy,
                "minimum_fixed_examples": 700,
            },
            "D_sham_required": False,
        }

    def _declared_training_shape_ids(self) -> set[str]:
        identifiers: set[str] = set()
        for stage in self.config.curriculum:
            for profile in stage.profiles:
                for shape in shape_catalog(profile.depth, profile.topology):
                    identifiers.add(shape_id(shape))
        return identifiers

    def evaluate_final_gate(self) -> dict[str, object]:
        self.evaluate_pending_stage_boundary()
        splits_result: dict[str, object] = {}
        datasets: dict[str, set[str]] = {}
        split_shapes: dict[str, set[str]] = {}
        accepted_evaluation_hashes: set[str] = set()
        historical_final_hashes = set(
            self.historical_final_evaluation_content_hashes
        )
        for split in self.config.evaluation_splits:
            seed_results: dict[str, dict[str, object]] = {}
            split_hashes: set[str] = set()
            shapes: set[str] = set()
            for evaluation_seed in self.config.eval_seeds:
                result, hashes, observed_shapes = self._evaluate_split_seed(
                    split,
                    evaluation_seed,
                    self.config.final_eval_examples_per_seed,
                    self.config.final_eval_batch_size,
                    enforce_content_exclusion=True,
                    training_content_hashes=self.training_content_hashes,
                    prior_evaluation_content_hashes=(
                        self.pre_final_evaluation_content_hashes
                    ),
                    accepted_evaluation_hashes=accepted_evaluation_hashes,
                    historical_final_evaluation_content_hashes=(
                        historical_final_hashes
                    ),
                )
                seed_results[str(evaluation_seed)] = result
                split_hashes.update(hashes)
                shapes.update(observed_shapes)
                datasets[f"{split.name}:{evaluation_seed}"] = hashes
            split_shapes[split.name] = shapes
            splits_result[split.name] = {
                "spec": {
                    "depth": split.depth,
                    "topology": split.topology,
                    "category": split.category,
                    "shape_partition": split.shape_partition,
                    "required_above_majority": split.required_above_majority,
                },
                "seeds": seed_results,
                "aggregate": self._aggregate_seed_results(seed_results),
            }

        pairwise_content_overlap: dict[str, int] = {}
        dataset_names = sorted(datasets)
        for left_index, left_name in enumerate(dataset_names):
            for right_name in dataset_names[left_index + 1 :]:
                overlap = len(datasets[left_name] & datasets[right_name])
                pairwise_content_overlap[f"{left_name}|{right_name}"] = overlap
        training_overlap = {
            name: len(hashes & self.training_content_hashes)
            for name, hashes in datasets.items()
        }
        pre_final_overlap = {
            name: len(hashes & self.pre_final_evaluation_content_hashes)
            for name, hashes in datasets.items()
        }
        historical_final_overlap = {
            name: len(hashes & historical_final_hashes)
            for name, hashes in datasets.items()
        }
        declared_training_shapes = self._declared_training_shape_ids()
        shape_audit: dict[str, object] = {}
        for split in self.config.evaluation_splits:
            observed = split_shapes[split.name]
            overlap = observed & declared_training_shapes
            if split.category == "in_distribution":
                valid = observed <= declared_training_shapes
            else:
                valid = not overlap
            shape_audit[split.name] = {
                "observed_shape_ids": sorted(observed),
                "overlap_with_declared_training_shapes": sorted(overlap),
                "valid_for_category": valid,
            }
        overlap_audit = {
            "training_content_hash_count": len(self.training_content_hashes),
            "training_content_hash_digest": _hash_set_digest(self.training_content_hashes),
            "training_duplicate_contents": self.training_duplicate_contents,
            "evaluation_overlap_with_training": training_overlap,
            "pre_final_evaluation_content_hash_count": len(
                self.pre_final_evaluation_content_hashes
            ),
            "pre_final_evaluation_content_hash_digest": _hash_set_digest(
                self.pre_final_evaluation_content_hashes
            ),
            "evaluation_overlap_with_pre_final": pre_final_overlap,
            "historical_final_evaluation_content_hash_count": len(
                historical_final_hashes
            ),
            "historical_final_evaluation_content_hash_digest": _hash_set_digest(
                historical_final_hashes
            ),
            "evaluation_overlap_with_historical_final": historical_final_overlap,
            "pairwise_evaluation_content_overlap": pairwise_content_overlap,
            "all_content_disjoint": (
                all(count == 0 for count in training_overlap.values())
                and all(count == 0 for count in pre_final_overlap.values())
                and all(count == 0 for count in historical_final_overlap.values())
                and all(count == 0 for count in pairwise_content_overlap.values())
            ),
            "declared_training_shape_ids": sorted(declared_training_shapes),
            "observed_training_shape_ids": sorted(self.observed_training_shape_ids),
            "shape_audit": shape_audit,
            "all_shape_rules_valid": all(
                item["valid_for_category"] is True for item in shape_audit.values()
            ),
        }
        self.final_evaluation = {
            "kind": "formal_confirmation" if self.config.formal_evaluation else "candidate_only",
            "examples_per_split_seed": self.config.final_eval_examples_per_seed,
            "evaluation_seeds": list(self.config.eval_seeds),
            "splits": splits_result,
            "overlap_audit": overlap_audit,
        }
        self.gate_result = self.compute_gate(self.final_evaluation)
        for hashes in datasets.values():
            self.historical_final_evaluation_content_hashes.update(hashes)
        return self.final_evaluation

    def compute_gate(self, evaluation: dict[str, object]) -> dict[str, object]:
        conditions: dict[str, bool] = {}
        baseline_conditions: list[bool] = []
        a_above_majority_in_distribution: list[bool] = []
        qualifying_extrapolation_splits: list[str] = []
        for split in self.config.evaluation_splits:
            seed_results = evaluation["splits"][split.name]["seeds"]
            d_over_a_threshold = (
                self.config.gate.minimum_d_advantage_in_distribution
                if split.category == "in_distribution"
                else self.config.gate.minimum_d_advantage_extrapolation
            )
            d_over_sham_threshold = (
                self.config.gate.minimum_d_over_sham_in_distribution
                if split.category == "in_distribution"
                else self.config.gate.minimum_d_over_sham_extrapolation
            )
            d_over_a = all(
                float(item["models"]["D_true"]["accuracy"])
                - float(item["models"]["A"]["accuracy"])
                >= d_over_a_threshold
                for item in seed_results.values()
            )
            d_over_sham = all(
                float(item["models"]["D_true"]["accuracy"])
                - float(item["models"]["D_sham"]["accuracy"])
                >= d_over_sham_threshold
                for item in seed_results.values()
            )
            conditions[f"{split.name}:D_true_over_A"] = d_over_a
            conditions[f"{split.name}:D_true_over_D_sham"] = d_over_sham
            if split.required_above_majority:
                a_above_majority = all(
                    float(item["models"]["A"]["accuracy"])
                    >= float(item["majority_baseline"])
                    + self.config.gate.minimum_above_majority
                    for item in seed_results.values()
                )
                d_true_above_majority = all(
                    float(item["models"]["D_true"]["accuracy"])
                    >= float(item["majority_baseline"])
                    + self.config.gate.minimum_above_majority
                    for item in seed_results.values()
                )
                if self.config.gate.baseline_policy == "joint_all_required_v1":
                    joint_above_majority = (
                        a_above_majority and d_true_above_majority
                    )
                    conditions[
                        f"{split.name}:A_and_D_true_above_majority"
                    ] = joint_above_majority
                    baseline_conditions.append(joint_above_majority)
                else:
                    a_nonconstant = all(
                        int(item["models"]["A"]["distinct_predicted_classes"]) > 1
                        for item in seed_results.values()
                    )
                    conditions[
                        f"{split.name}:D_true_above_majority"
                    ] = d_true_above_majority
                    conditions[f"{split.name}:A_nonconstant"] = a_nonconstant
                    baseline_conditions.extend(
                        (d_true_above_majority, a_nonconstant)
                    )
                    if split.category == "in_distribution":
                        a_above_majority_in_distribution.append(
                            a_above_majority
                        )
            if split.category != "in_distribution" and d_over_a and d_over_sham:
                qualifying_extrapolation_splits.append(split.name)

        if (
            self.config.gate.baseline_policy
            == "privileged_structure_posthoc_v1"
        ):
            a_sanity_pass = any(a_above_majority_in_distribution)
            conditions[
                "A_above_majority_on_at_least_one_in_distribution_split"
            ] = a_sanity_pass
            baseline_conditions.append(a_sanity_pass)

        in_distribution_conditions = [
            value
            for name, value in conditions.items()
            if any(
                name.startswith(f"{split.name}:")
                for split in self.config.evaluation_splits
                if split.category == "in_distribution"
            )
        ]
        overlap_audit = evaluation["overlap_audit"]
        conditions["all_content_disjoint"] = (
            overlap_audit["all_content_disjoint"] is True
        )
        conditions["all_shape_rules_valid"] = (
            overlap_audit["all_shape_rules_valid"] is True
        )
        conditions["at_least_one_extrapolation_split"] = (
            len(qualifying_extrapolation_splits) > 0
        )
        structural_conditions_pass = (
            all(in_distribution_conditions)
            and all(baseline_conditions)
            and conditions["all_content_disjoint"]
            and conditions["all_shape_rules_valid"]
            and conditions["at_least_one_extrapolation_split"]
        )
        structural_gate_eligible = self.config.operand_mode == "literal"
        foundation_pass = (
            self.foundation_gate_result.get("passed") is True
            if self.config.foundation_gate_required
            else True
        )
        candidate_pass = structural_conditions_pass and structural_gate_eligible and foundation_pass
        formal_scale = (
            self.config.formal_evaluation
            and self.config.final_eval_examples_per_seed >= 10_000
            and len(self.config.confirmation_training_seeds)
            >= self.config.minimum_confirmation_training_seeds
        )
        return {
            "candidate_pass": candidate_pass,
            "structural_conditions_pass": structural_conditions_pass,
            "structural_gate_eligible": structural_gate_eligible,
            "operand_mode": self.config.operand_mode,
            "foundation_gate_required": self.config.foundation_gate_required,
            "foundation_gate_passed": foundation_pass,
            "conditions": conditions,
            "qualifying_extrapolation_splits": qualifying_extrapolation_splits,
            "baseline_policy": self.config.gate.baseline_policy,
            "thresholds": self.config.gate.__dict__,
            "formal_scale_configured": formal_scale,
            "confirmation_training_seed": self.config.seed,
            "required_confirmation_training_seeds": list(
                self.config.confirmation_training_seeds
            ),
            "completed_confirmation_training_seeds": [self.config.seed],
            "stage2_unblocked": False,
            "stage2_block_reason": (
                "a single run is candidate-only; Stage 2 requires an aggregate pass "
                f"across at least {self.config.minimum_confirmation_training_seeds} "
                "independent training seeds"
            ),
        }

    def learning_gate(self) -> dict[str, object]:
        if not self.final_evaluation:
            raise RuntimeError("final evaluation must run before the learning gate")
        checks = []
        for split in self.final_evaluation["splits"].values():
            for seed_result in split["seeds"].values():
                majority = float(seed_result["majority_baseline"])
                for model_name in self.MODEL_NAMES:
                    model = seed_result["models"][model_name]
                    checks.append(
                        int(model["distinct_predicted_classes"]) > 1
                        or float(model["accuracy"]) > majority
                    )
        return {
            "passed": any(checks),
            "criterion": (
                "at least one model/split/seed has non-constant legal predictions "
                "or accuracy above the exactly balanced majority baseline"
            ),
            "candidate_only": True,
        }

    def record_run_completion(self, complete: bool, reason: str) -> None:
        self.gate_result["run_complete"] = complete
        self.gate_result["run_completion_reason"] = reason
        if complete:
            return
        self.gate_result["candidate_pass_before_run_completion_check"] = (
            self.gate_result.get("candidate_pass") is True
        )
        self.gate_result["candidate_pass"] = False
        self.gate_result["stage2_unblocked"] = False
        self.gate_result["stage2_block_reason"] = (
            "run did not reach exactly optimizer_steps with target_steps_reached"
        )

    def metrics(self) -> dict[str, object]:
        models: dict[str, object] = {}
        for name, state in self.cumulative.items():
            examples = int(state["examples"])
            updates = int(state["optimizer_updates"])
            item: dict[str, object] = {
                **state,
                "training_accuracy": int(state["correct"]) / examples if examples else None,
                "average_loss": float(state["loss_sum"]) / updates if updates else None,
                "parameter_count": self.parameter_counts[name],
                "updates_per_second": updates / self.elapsed_seconds(),
                "examples_per_second": examples / self.elapsed_seconds(),
            }
            if name != "A":
                item.update(
                    {
                        "average_node_count": (
                            int(state["node_count_sum"]) / examples if examples else None
                        ),
                        "average_maximum_tree_depth": (
                            int(state["maximum_tree_depth_sum"]) / examples
                            if examples
                            else None
                        ),
                        "average_combined_nodes": (
                            int(state["combined_nodes"]) / examples if examples else None
                        ),
                    }
                )
            models[name] = item
        curriculum = copy.deepcopy(self.curriculum_metrics)
        for stage in curriculum.values():
            attempts = int(stage["generation_attempts"])
            accepted = int(stage["accepted_examples"])
            stage["acceptance_rate"] = accepted / attempts if attempts else None
        return {
            "models": models,
            "parameter_difference": {
                "D_true_minus_A": self.parameter_counts["D_true"]
                - self.parameter_counts["A"],
                "D_sham_minus_D_true": self.parameter_counts["D_sham"]
                - self.parameter_counts["D_true"],
            },
            "D_sham_mapping": {
                "version": SHAM_MAPPING_VERSION,
                "deterministic": True,
                "content_keyed": True,
                "same_parameter_and_compose_budget_as_D_true": True,
                "C0_discriminative": False,
                "C1_limitation": (
                    "one operator source cannot be permuted and two leaves have only "
                    "one derangement; C1 is a limited control"
                ),
            },
            "latest_evaluation": self.latest_evaluation,
            "final_evaluation": self.final_evaluation,
            "gate": self.gate_result,
            "stage_boundary_evaluations": self.stage_boundary_evaluations,
            "foundation_gate": self.foundation_gate_result,
            "operand_mode": self.config.operand_mode,
            "effective_batch_size": self.config.effective_batch_size,
            "curriculum_position": self.curriculum_position(),
            "curriculum_metrics": curriculum,
            "training_content_hash_count": len(self.training_content_hashes),
            "pre_final_evaluation_content_hash_count": len(
                self.pre_final_evaluation_content_hashes
            ),
            "pre_final_evaluation_content_hash_digest": _hash_set_digest(
                self.pre_final_evaluation_content_hashes
            ),
            "historical_final_evaluation_content_hash_count": len(
                self.historical_final_evaluation_content_hashes
            ),
            "historical_final_evaluation_content_hash_digest": _hash_set_digest(
                self.historical_final_evaluation_content_hashes
            ),
            "training_duplicate_contents": self.training_duplicate_contents,
            "observed_training_shape_ids": sorted(self.observed_training_shape_ids),
        }

    def recovery_state(self) -> dict[str, object]:
        checkpoint_step = self.last_checkpoint_step
        uncheckpointed_steps = (
            self.global_step - checkpoint_step
            if checkpoint_step is not None
            else self.global_step
        )
        return {
            "semantics": "at-least-once",
            "checkpoint_step": checkpoint_step,
            "current_step": self.global_step,
            "uncheckpointed_paired_steps": uncheckpointed_steps,
            "max_replayed_paired_steps_on_crash": self.config.checkpoint_steps,
            "curriculum_position": self.curriculum_position(),
            "note": (
                "a crash resumes from the last atomic checkpoint and may replay completed "
                "steps after it; the saved curriculum position resumes exactly at that "
                "checkpoint but execution is not exact-once"
            ),
        }

    def save_checkpoint(self, kind: str = "scheduled") -> Path:
        checkpoints = self.run_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        suffix = "-bootstrap" if kind == "bootstrap" else ""
        path = checkpoints / f"checkpoint-{self.global_step:08d}{suffix}.pt"
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        payload = {
            "schema_version": 3,
            "kind": kind,
            "global_step": self.global_step,
            "curriculum_position": self.curriculum_position(),
            "elapsed_seconds": self.elapsed_seconds(),
            "models": {
                name: _cpu_tree(model.state_dict()) for name, model in self.models.items()
            },
            "optimizers": {
                name: _cpu_tree(optimizer.state_dict())
                for name, optimizer in self.optimizers.items()
            },
            "torch_rng_state": torch.get_rng_state(),
            "generator_state": self.generator.get_state(),
            "config": self.config.to_dict(),
            "cumulative": self.cumulative,
            "curriculum_metrics": self.curriculum_metrics,
            "training_content_hashes": sorted(self.training_content_hashes),
            "pre_final_evaluation_content_hashes": sorted(
                self.pre_final_evaluation_content_hashes
            ),
            "historical_final_evaluation_content_hashes": sorted(
                self.historical_final_evaluation_content_hashes
            ),
            "training_duplicate_contents": self.training_duplicate_contents,
            "observed_training_shape_ids": sorted(self.observed_training_shape_ids),
            "latest_loss": self.latest_loss,
            "latest_evaluation": self.latest_evaluation,
            "final_evaluation": self.final_evaluation,
            "gate_result": self.gate_result,
            "stage_boundary_evaluations": self.stage_boundary_evaluations,
            "foundation_gate_result": self.foundation_gate_result,
            "source_manifest": self.source_manifest,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "recovery_semantics": {
                "semantics": "at-least-once",
                "checkpoint_step": self.global_step,
                "max_replayed_paired_steps_on_crash": self.config.checkpoint_steps,
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.last_checkpoint_step = self.global_step
        atomic_write_json(
            checkpoints / "latest.json",
            {
                "checkpoint": str(path),
                "global_step": self.global_step,
                "kind": kind,
                "curriculum_position": self.curriculum_position(),
            },
        )
        self.last_checkpoint = str(path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_schema = payload.get("schema_version")
        if checkpoint_schema not in {2, 3}:
            raise RuntimeError("revised Stage 1 requires checkpoint schema version 2 or 3")
        if (
            checkpoint_schema == 2
            and payload.get("final_evaluation")
            and "historical_final_evaluation_content_hashes" not in payload
        ):
            raise RuntimeError(
                "legacy checkpoint contains final evaluation without recoverable "
                "historical final hashes; fail closed"
            )
        if payload["config"] != self.config.to_dict():
            raise RuntimeError("checkpoint config does not match the original frozen run config")
        if payload["source_manifest"]["manifest_hash"] != self.source_manifest["manifest_hash"]:
            raise RuntimeError("checkpoint source manifest does not match the frozen source snapshot")
        if payload["snapshot_manifest_hash"] != self.snapshot_manifest_hash:
            raise RuntimeError("checkpoint snapshot manifest does not match this run")
        expected_position = self.curriculum_position(int(payload["global_step"]))
        if payload["curriculum_position"] != expected_position:
            raise RuntimeError("checkpoint curriculum position does not match its global step")
        for name, model in self.models.items():
            model.load_state_dict(payload["models"][name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.backend.device)
        self.global_step = int(payload["global_step"])
        self.elapsed_before_resume = float(payload["elapsed_seconds"])
        self.process_started = time.monotonic()
        self.cumulative = payload["cumulative"]
        self.curriculum_metrics = payload["curriculum_metrics"]
        self.training_content_hashes = set(payload["training_content_hashes"])
        self.pre_final_evaluation_content_hashes = set(
            payload.get("pre_final_evaluation_content_hashes", ())
        )
        self.historical_final_evaluation_content_hashes = set(
            payload.get("historical_final_evaluation_content_hashes", ())
        )
        self.training_duplicate_contents = int(payload["training_duplicate_contents"])
        self.observed_training_shape_ids = set(payload["observed_training_shape_ids"])
        self.latest_loss = payload["latest_loss"]
        self.latest_evaluation = payload["latest_evaluation"]
        self.final_evaluation = payload["final_evaluation"]
        self.gate_result = payload["gate_result"]
        self.stage_boundary_evaluations = payload.get("stage_boundary_evaluations", {})
        self.foundation_gate_result = payload.get("foundation_gate_result", {})
        torch.set_rng_state(payload["torch_rng_state"])
        self.generator.set_state(payload["generator_state"])
        self.last_checkpoint = str(path)
        self.last_checkpoint_step = self.global_step
        if self.curriculum_position() != payload["curriculum_position"]:
            raise RuntimeError("restored curriculum position drifted from the saved checkpoint")


def latest_checkpoint(run_dir: Path) -> Path:
    pointer = run_dir / "checkpoints" / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"no latest checkpoint pointer in {run_dir}")
    data = json.loads(pointer.read_text(encoding="utf-8"))
    path = Path(data["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"latest checkpoint is missing: {path}")
    return path
