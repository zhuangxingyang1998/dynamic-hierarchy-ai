"""Paired Stage 2 R2 training, interventions, evaluation, and recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .backend import Backend, resolve_backend
from .config import ModelConfig
from .data import LeafSourceReference, MergeSourceReference, StructureSample
from .model import SmallTransformerBaseline, TrueStructureDiagnosticD
from .optim import DirectMLCompatibleAdamWCore
from .stage1_data import sham_structure
from .stage2_config import Stage2Config, Stage2ModelSpec
from .stage2_data import Stage2GeneratedBatch, Stage2OrdinaryBatch, Stage2PrecedenceFamilyGenerator
from .stage2_model import (
    Stage2MergeClassifier,
    Stage2MergeOutput,
    Stage2RecurrentFlatBaseline,
    Stage2RecurrentOutput,
    Stage2Trace,
)


TRAINABLE_CONTROLS = (
    "A-Q-param",
    "A-Q-flop",
    "A-recur",
    "B-query",
    "B-noQ-router",
    "B-sham",
    "D-true",
    "D-sham",
)
FIXED_POLICIES = {
    "F-stop": "stop",
    "F-left": "left",
    "F-right": "right",
    "F-add": "add",
    "F-sub": "sub",
}


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
    return hashlib.sha256("\n".join(sorted(values)).encode("ascii")).hexdigest()


def _model_config(spec: Stage2ModelSpec) -> ModelConfig:
    return ModelConfig(
        embedding_dim=spec.hidden_dim,
        heads=spec.heads,
        layers=spec.layers,
        feedforward_dim=spec.feedforward_dim,
        dropout=spec.dropout,
    )


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def estimate_transformer_forward_operations(
    spec: Stage2ModelSpec,
    sequence_length: int,
) -> int:
    """Count dense multiply-add sites for a padded Transformer forward."""

    hidden = spec.hidden_dim
    layer = (
        4 * sequence_length * hidden * hidden
        + 2 * sequence_length * sequence_length * hidden
        + 2 * sequence_length * hidden * spec.feedforward_dim
    )
    embedding_and_position = sequence_length * (3 * hidden + hidden * hidden)
    return embedding_and_position + spec.layers * layer + hidden * 7


def estimate_merge_forward_operations(spec: Stage2ModelSpec, leaf_count: int) -> int:
    """Count every candidate composer/router site on a full hard reduction."""

    hidden = spec.hidden_dim
    feedforward = spec.feedforward_dim
    sequence_length = 2 * leaf_count + 1
    operations = sequence_length * (3 * hidden + hidden * hidden)
    for active_nodes in range(leaf_count, 1, -1):
        candidates = active_nodes - 1
        composer = 4 * hidden * feedforward
        router = (4 * hidden + 1) * feedforward + feedforward
        stop = (2 * hidden + 1) * feedforward + feedforward
        operations += candidates * (composer + router) + stop
    operations += leaf_count * 2 * hidden
    operations += 2 * hidden * feedforward + feedforward * 7
    return operations


@dataclass(frozen=True)
class _ForwardReceipt:
    logits: torch.Tensor
    merge: Stage2MergeOutput | None = None
    recurrent: Stage2RecurrentOutput | None = None


def _oracle_edges(structure: StructureSample) -> set[tuple[int, int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    edges: set[tuple[int, int, int]] = set()
    for node in structure.nodes:
        if isinstance(node, LeafSourceReference):
            spans[node.node_id] = (node.source_index, node.source_index)
        elif isinstance(node, MergeSourceReference):
            left = spans[node.left]
            right = spans[node.right]
            span = (min(left[0], right[0]), max(left[1], right[1]))
            spans[node.node_id] = span
            edges.add((span[0], span[1], node.operator_source_index))
        else:
            raise TypeError(f"unsupported oracle node: {type(node).__name__}")
    return edges


def _trace_edges(trace: Stage2Trace) -> set[tuple[int, int, int]]:
    return {
        (int(step.source_start), int(step.source_end), int(step.operator_source_index))
        for step in trace.steps
        if step.action == "MERGE"
        and step.source_start is not None
        and step.source_end is not None
        and step.operator_source_index is not None
    }


class Stage2Trainer:
    """Train the complete R2 matrix on paired precedence-query families."""

    def __init__(self, config: Stage2Config, run_dir: Path) -> None:
        config.validate()
        self.config = config
        self.run_dir = run_dir
        self.backend: Backend = resolve_backend(
            config.device,
            config.cpu_threads,
            config.deterministic,
        )
        torch.manual_seed(config.seed)

        b_query = Stage2MergeClassifier(config.model)
        d_true = TrueStructureDiagnosticD(
            config.model.vocab_size,
            _model_config(config.model),
            output_classes=7,
        )
        self.models: dict[str, nn.Module] = {
            "A-Q-param": SmallTransformerBaseline(
                config.a_param_model.vocab_size,
                _model_config(config.a_param_model),
                output_classes=7,
            ),
            "A-Q-flop": SmallTransformerBaseline(
                config.a_flop_model.vocab_size,
                _model_config(config.a_flop_model),
                output_classes=7,
            ),
            "A-recur": Stage2RecurrentFlatBaseline(config.model),
            "B-query": b_query,
            "B-noQ-router": copy.deepcopy(b_query),
            "B-sham": copy.deepcopy(b_query),
            "D-true": d_true,
            "D-sham": copy.deepcopy(d_true),
        }
        self.models = {
            name: model.to(self.backend.device) for name, model in self.models.items()
        }
        self.optimizers = {
            name: DirectMLCompatibleAdamWCore(model.parameters(), lr=config.learning_rate)
            for name, model in self.models.items()
        }
        self.loss_fn = nn.CrossEntropyLoss()
        self.generator = Stage2PrecedenceFamilyGenerator(config.seed + 1)
        self.global_step = 0
        self.training_family_hashes: set[str] = set()
        self.training_duplicate_families = 0
        self.evaluation_family_hashes: set[str] = set()
        self.latest_evaluation: dict[str, object] = {}
        self.elapsed_before_resume = 0.0
        self.process_started = time.monotonic()
        self.last_checkpoint_step: int | None = None
        self.last_checkpoint: str | None = None
        self.parameter_counts = {
            name: _parameter_count(model) for name, model in self.models.items()
        }
        if len({self.parameter_counts[name] for name in ("B-query", "B-noQ-router", "B-sham")}) != 1:
            raise RuntimeError("B-query, B-noQ-router, and B-sham must be parameter matched")
        if self.parameter_counts["D-true"] != self.parameter_counts["D-sham"]:
            raise RuntimeError("D-true and D-sham must be parameter matched")
        self.cumulative = {
            name: {
                "optimizer_updates": 0,
                "examples": 0,
                "correct": 0,
                "loss_sum": 0.0,
                "forward_backward_seconds": 0.0,
                "router_gradient_norm_sum": 0.0,
                "candidate_scores": 0,
                "candidate_compositions": 0,
                "selected_compositions": 0,
                "recurrent_steps": 0,
            }
            for name in TRAINABLE_CONTROLS
        }
        self._initial_router = {
            name: torch.cat(
                [parameter.detach().cpu().flatten() for parameter in self.models[name].router.parameters()]
            )
            for name in ("B-query", "B-noQ-router", "B-sham")
        }

    def elapsed_seconds(self) -> float:
        return self.elapsed_before_resume + time.monotonic() - self.process_started

    def time_budget_exhausted(self) -> bool:
        return self.elapsed_seconds() >= self.config.time_budget_minutes * 60.0

    def _forward(
        self,
        name: str,
        batch: Stage2OrdinaryBatch,
        generated: Stage2GeneratedBatch,
        *,
        fixed_policy: str | None = None,
    ) -> _ForwardReceipt:
        if fixed_policy is not None:
            output = self.models["B-query"](batch, policy=fixed_policy)
            return _ForwardReceipt(output.logits, output)
        model = self.models[name]
        if name.startswith("A-Q-"):
            return _ForwardReceipt(
                model(batch.token_ids, batch.position_features, batch.attention_mask)
            )
        if name == "A-recur":
            output = model(batch)
            return _ForwardReceipt(output.logits, recurrent=output)
        if name.startswith("B-"):
            mode = {
                "B-query": "query",
                "B-noQ-router": "blind",
                "B-sham": "sham",
            }[name]
            output = model(batch, policy="learned", router_query_mode=mode)
            return _ForwardReceipt(output.logits, output)
        structure = generated.diagnostic_structure
        if name == "D-sham":
            structure = sham_structure(structure, generated.ordinary.token_ids)
        output = model(
            batch.token_ids,
            batch.position_features,
            batch.attention_mask,
            structure,
        )
        return _ForwardReceipt(output.logits)

    def train_step(self) -> dict[str, float]:
        if self.global_step >= self.config.optimizer_steps:
            raise RuntimeError("Stage 2 target optimizer steps are already complete")
        profile = self.config.train_profiles[self.global_step % len(self.config.train_profiles)]
        generated = self.generator.balanced_block(
            profile,
            blocks=self.config.families_per_stratum // 42,
            max_attempts_per_family=self.config.max_generation_attempts_per_family,
        )
        for family_hash in generated.generation.family_hashes:
            if family_hash in self.training_family_hashes:
                self.training_duplicate_families += 1
            self.training_family_hashes.add(family_hash)
        batch = generated.ordinary.to(self.backend.device)
        losses: dict[str, float] = {}
        for name in TRAINABLE_CONTROLS:
            model = self.models[name]
            optimizer = self.optimizers[name]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            first_parameter = next(model.parameters())
            self.backend.synchronize(first_parameter)
            started = time.perf_counter()
            receipt = self._forward(name, batch, generated)
            loss = self.loss_fn(receipt.logits, batch.labels)
            loss.backward()
            router_gradient_norm = 0.0
            if name.startswith("B-"):
                router_gradient_norm = sum(
                    self.backend.scalar(parameter.grad.detach().square().sum())
                    for parameter in model.router.parameters()
                    if parameter.grad is not None
                ) ** 0.5
            optimizer.step()
            self.backend.synchronize(first_parameter)
            duration = time.perf_counter() - started
            loss_value = self.backend.scalar(loss)
            correct = int(receipt.logits.detach().argmax(dim=-1).eq(batch.labels).sum().cpu().item())
            state = self.cumulative[name]
            state["optimizer_updates"] += 1
            state["examples"] += int(batch.labels.shape[0])
            state["correct"] += correct
            state["loss_sum"] += loss_value
            state["forward_backward_seconds"] += duration
            state["router_gradient_norm_sum"] += router_gradient_norm
            if receipt.merge is not None:
                state["candidate_scores"] += receipt.merge.compute.candidate_scores
                state["candidate_compositions"] += receipt.merge.compute.candidate_compositions
                state["selected_compositions"] += receipt.merge.compute.selected_compositions
                state["recurrent_steps"] += receipt.merge.compute.recurrent_steps
            elif receipt.recurrent is not None:
                state["recurrent_steps"] += receipt.recurrent.recurrent_steps
            losses[name] = loss_value
        self.global_step += 1
        if self.config.yield_ms:
            time.sleep(self.config.yield_ms / 1000.0)
        return losses

    @staticmethod
    def _paired_outcomes(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
        correct = predictions.eq(labels)
        result = {"both": 0, "add_only": 0, "sub_only": 0, "neither": 0}
        for row in range(0, labels.shape[0], 2):
            left = bool(correct[row])
            right = bool(correct[row + 1])
            key = "both" if left and right else "add_only" if left else "sub_only" if right else "neither"
            result[key] += 1
        return result

    @staticmethod
    def _trace_metrics(
        traces: tuple[Stage2Trace, ...],
        generated: Stage2GeneratedBatch,
    ) -> dict[str, object]:
        true_positives = 0
        predicted_edges = 0
        oracle_edges = 0
        exact = 0
        immediate_stop = 0
        early_stop = 0
        full_reduction = 0
        always_left = 0
        always_right = 0
        for trace, structure in zip(
            traces, generated.diagnostic_structure.samples, strict=True
        ):
            predicted = _trace_edges(trace)
            oracle = _oracle_edges(structure)
            true_positives += len(predicted & oracle)
            predicted_edges += len(predicted)
            oracle_edges += len(oracle)
            exact += predicted == oracle
            immediate_stop += bool(trace.stopped_early and not predicted)
            early_stop += bool(trace.stopped_early)
            full_reduction += bool(trace.reached_root)
            merge_steps = [step for step in trace.steps if step.action == "merge"]
            always_left += bool(merge_steps) and all(step.merge_index == 0 for step in merge_steps)
            always_right += bool(merge_steps) and all(
                step.merge_index == step.legal_merge_count - 1 for step in merge_steps
            )
        precision = true_positives / predicted_edges if predicted_edges else 0.0
        recall = true_positives / oracle_edges if oracle_edges else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        distances: list[float] = []
        identical = 0
        for row in range(0, len(traces), 2):
            left = _trace_edges(traces[row])
            right = _trace_edges(traces[row + 1])
            union = left | right
            distance = 1.0 - len(left & right) / len(union) if union else 0.0
            distances.append(distance)
            identical += left == right
        count = len(traces)
        return {
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "exact_tree_rate": exact / count,
            "mean_same_family_trace_jaccard_distance": sum(distances) / len(distances),
            "query_identical_trace_rate": identical / len(distances),
            "immediate_stop_rate": immediate_stop / count,
            "early_stop_rate": early_stop / count,
            "full_reduction_rate": full_reduction / count,
            "always_left_rate": always_left / count,
            "always_right_rate": always_right / count,
        }

    def _evaluate_profile(self, generated: Stage2GeneratedBatch) -> dict[str, object]:
        batch = generated.ordinary.to(self.backend.device)
        batch_tensor_bytes = sum(
            _tensor_bytes(value)
            for value in (
                batch.token_ids,
                batch.position_features,
                batch.attention_mask,
                batch.labels,
                batch.query_ids,
                batch.literal_source_indices,
                batch.operator_source_indices,
            )
        )
        padding_tokens = int((~batch.attention_mask.bool()).sum().detach().cpu().item())
        rows = int(batch.labels.shape[0])
        sequence_length = int(batch.token_ids.shape[1])
        leaf_count = int(batch.literal_source_indices.shape[1])
        controls: dict[str, object] = {}
        for model in self.models.values():
            model.eval()
        with torch.no_grad():
            for name in self.config.controls:
                fixed_policy = FIXED_POLICIES.get(name)
                model_name = "B-query" if fixed_policy is not None else name
                first_parameter = next(self.models[model_name].parameters())
                self.backend.synchronize(first_parameter)
                started = time.perf_counter()
                receipt = self._forward(
                    model_name,
                    batch,
                    generated,
                    fixed_policy=fixed_policy,
                )
                self.backend.synchronize(receipt.logits)
                duration = time.perf_counter() - started
                predictions = receipt.logits.argmax(dim=-1)
                correct = int(predictions.eq(batch.labels).sum().cpu().item())
                item: dict[str, object] = {
                    "correct": correct,
                    "query_rows": int(batch.labels.shape[0]),
                    "base_families": generated.generation.accepted_families,
                    "accuracy": correct / batch.labels.shape[0],
                    "cross_entropy": self.backend.scalar(
                        nn.functional.cross_entropy(receipt.logits, batch.labels)
                    ),
                    "forward_seconds": duration,
                    "examples_per_second": batch.labels.shape[0] / duration,
                    "prediction_counts": torch.bincount(
                        predictions.detach().cpu(), minlength=7
                    ).tolist(),
                    "paired_outcomes": self._paired_outcomes(
                        predictions.detach().cpu(), batch.labels.detach().cpu()
                    ),
                    "weight_source": "B-query intervention weights" if fixed_policy else name,
                }
                optimizer_source = "B-query" if fixed_policy is not None else name
                optimizer_bytes = _tensor_bytes(self.optimizers[optimizer_source].state_dict())
                parameter_bytes = sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in self.models[model_name].parameters()
                )
                compute: dict[str, object] = {
                    "input_padding_tokens": padding_tokens,
                    "recurrent_steps": 0,
                    "candidate_scores": 0,
                    "candidate_compositions": 0,
                    "selected_compositions": 0,
                    "unselected_candidate_compositions": 0,
                    "stop_scores": 0,
                    "router_action_scores": 0,
                    "parameter_count": self.parameter_counts[model_name],
                    "optimizer_state_bytes": optimizer_bytes,
                    "peak_allocated_tensor_lower_bound_bytes": (
                        parameter_bytes
                        + optimizer_bytes
                        + batch_tensor_bytes
                        + receipt.logits.numel() * receipt.logits.element_size()
                    ),
                    "peak_estimate_scope": (
                        "lower bound from parameters, optimizer tensors, input tensors, and logits; "
                        "DirectML exposes no peak allocator counter"
                    ),
                }
                if receipt.merge is not None:
                    merge_compute = receipt.merge.compute
                    compute.update(
                        {
                            "recurrent_steps": merge_compute.recurrent_steps,
                            "candidate_scores": merge_compute.candidate_scores,
                            "candidate_compositions": merge_compute.candidate_compositions,
                            "selected_compositions": merge_compute.selected_compositions,
                            "unselected_candidate_compositions": (
                                merge_compute.candidate_compositions
                                - merge_compute.selected_compositions
                            ),
                            "stop_scores": merge_compute.stop_scores,
                            "router_action_scores": merge_compute.candidate_scores,
                            "estimated_forward_operations": rows
                            * estimate_merge_forward_operations(self.config.model, leaf_count),
                        }
                    )
                    item["structure"] = self._trace_metrics(receipt.merge.traces, generated)
                elif receipt.recurrent is not None:
                    one_layer = Stage2ModelSpec(
                        **{
                            **self.config.model.__dict__,
                            "layers": 1,
                        }
                    )
                    compute.update(
                        {
                            "recurrent_steps": receipt.recurrent.recurrent_steps,
                            "stop_scores": receipt.recurrent.stop_scores,
                            "router_action_scores": 2 * receipt.recurrent.stop_scores,
                            "early_stops": receipt.recurrent.early_stops,
                            "estimated_forward_operations": receipt.recurrent.recurrent_steps
                            * estimate_transformer_forward_operations(
                                one_layer, sequence_length
                            ),
                        }
                    )
                elif name.startswith("A-Q-"):
                    spec = (
                        self.config.a_param_model
                        if name == "A-Q-param"
                        else self.config.a_flop_model
                    )
                    compute.update(
                        {
                            "recurrent_steps": rows * spec.layers,
                            "estimated_forward_operations": rows
                            * estimate_transformer_forward_operations(spec, sequence_length),
                        }
                    )
                else:
                    compose_count = rows * (leaf_count - 1)
                    compute.update(
                        {
                            "recurrent_steps": rows * self.config.model.layers,
                            "selected_compositions": compose_count,
                            "estimated_forward_operations": rows
                            * estimate_transformer_forward_operations(
                                self.config.model, sequence_length
                            )
                            + compose_count
                            * 4
                            * self.config.model.hidden_dim
                            * self.config.model.feedforward_dim,
                        }
                    )
                item["compute"] = compute
                controls[name] = item
        labels = batch.labels.detach().cpu()
        query_ids = batch.query_ids.detach().cpu()
        query_label_counts = {
            str(query): torch.bincount(labels[query_ids == query], minlength=7).tolist()
            for query in (0, 1)
        }
        query_only_correct = sum(max(counts) for counts in query_label_counts.values())
        return {
            "profile": generated.ordinary.profile_name,
            "controls": controls,
            "generation": {
                "attempts": generated.generation.attempts,
                "accepted_families": generated.generation.accepted_families,
                "label_pair_counts": generated.generation.label_pair_counts,
                "structural_rejections": generated.generation.structural_rejections,
                "equal_label_rejections": generated.generation.equal_label_rejections,
                "quota_rejections": generated.generation.quota_rejections,
                "excluded_family_rejections": generated.generation.excluded_family_rejections,
                "duplicate_family_rejections": generated.generation.duplicate_family_rejections,
                "fixed_policy_counterexample_rows": dict(
                    generated.generation.fixed_policy_counterexample_rows
                ),
            },
            "canaries": {
                "query_label_counts": query_label_counts,
                "query_only_lookup_accuracy": query_only_correct / labels.shape[0],
                "input_only_lookup_accuracy": 0.5,
                "input_only_basis": "each unique base input occurs with two unequal labels",
            },
        }

    def evaluate(self) -> dict[str, object]:
        profiles: dict[str, object] = {}
        accepted_evaluation_hashes: set[str] = set()
        excluded = set(self.training_family_hashes)
        for profile_index, profile in enumerate(self.config.evaluation_profiles):
            generator = Stage2PrecedenceFamilyGenerator(
                self.config.seed + 10000 + 1009 * profile_index
            )
            generated = generator.balanced_block(
                profile,
                blocks=self.config.evaluation_blocks,
                max_attempts_per_family=self.config.max_generation_attempts_per_family,
                excluded_family_hashes=excluded | accepted_evaluation_hashes,
            )
            hashes = set(generated.generation.family_hashes)
            if hashes & excluded or hashes & accepted_evaluation_hashes:
                raise RuntimeError("Stage 2 base-family split overlap detected")
            accepted_evaluation_hashes.update(hashes)
            profiles[profile.name] = self._evaluate_profile(generated)
        self.evaluation_family_hashes = accepted_evaluation_hashes
        self.latest_evaluation = {
            "kind": self.config.run_kind,
            "profiles": profiles,
            "train_evaluation_overlap": 0,
            "training_family_hash_count": len(self.training_family_hashes),
            "evaluation_family_hash_count": len(self.evaluation_family_hashes),
            "training_family_hash_digest": _hash_set_digest(self.training_family_hashes),
            "evaluation_family_hash_digest": _hash_set_digest(self.evaluation_family_hashes),
        }
        return self.latest_evaluation

    def budget_report(self) -> dict[str, object]:
        maximum_leaf_count = max(profile.leaf_count for profile in self.config.train_profiles)
        sequence_length = 2 * maximum_leaf_count + 1
        merge_operations = estimate_merge_forward_operations(
            self.config.model, maximum_leaf_count
        )
        param_operations = estimate_transformer_forward_operations(
            self.config.a_param_model, sequence_length
        )
        flop_operations = estimate_transformer_forward_operations(
            self.config.a_flop_model, sequence_length
        )
        b_parameters = self.parameter_counts["B-query"]
        param_difference = abs(self.parameter_counts["A-Q-param"] - b_parameters)
        optimizer_bytes = {
            name: _tensor_bytes(optimizer.state_dict())
            for name, optimizer in self.optimizers.items()
        }
        return {
            "parameter_counts": self.parameter_counts,
            "optimizer_state_bytes": optimizer_bytes,
            "A-Q-param": {
                "absolute_parameter_difference": param_difference,
                "relative_parameter_difference": param_difference / b_parameters,
                "status": "matched_within_one_percent" if param_difference / b_parameters <= 0.01 else "unmatched",
            },
            "A-Q-flop": {
                "B_full_candidate_forward_operation_estimate": merge_operations,
                "A_forward_operation_estimate": flop_operations,
                "relative_estimate_difference": abs(flop_operations - merge_operations) / merge_operations,
                "status": "operation_estimate_only; DirectML exposes no exact FLOP counter",
            },
            "A-Q-param_operation_estimate": param_operations,
            "accounting_scope": (
                "B estimate includes token/position projection, every selected and unselected "
                "candidate composition, merge and STOP routing, terminal attention, and classifier; "
                "training receipts additionally report synchronized forward/backward/update wall time"
            ),
        }

    def result(self, disposition: str) -> dict[str, object]:
        training: dict[str, object] = {}
        for name, state in self.cumulative.items():
            updates = int(state["optimizer_updates"])
            examples = int(state["examples"])
            item = dict(state)
            item["mean_loss"] = state["loss_sum"] / updates if updates else None
            item["accuracy"] = state["correct"] / examples if examples else None
            item["examples_per_second"] = (
                examples / state["forward_backward_seconds"]
                if state["forward_backward_seconds"]
                else None
            )
            item["parameter_count"] = self.parameter_counts[name]
            item["optimizer_state_bytes"] = _tensor_bytes(
                self.optimizers[name].state_dict()
            )
            if name.startswith("B-"):
                current = torch.cat(
                    [parameter.detach().cpu().flatten() for parameter in self.models[name].router.parameters()]
                )
                item["router_parameter_delta_l2"] = float(
                    (current - self._initial_router[name]).square().sum().sqrt().item()
                )
            training[name] = item
        return {
            "schema_version": 1,
            "packet": "DH-S2-R2",
            "run_kind": self.config.run_kind,
            "disposition": disposition,
            "global_step": self.global_step,
            "target_optimizer_steps": self.config.optimizer_steps,
            "elapsed_seconds": self.elapsed_seconds(),
            "config": self.config.to_dict(),
            "backend": self.backend.metadata(),
            "training": training,
            "evaluation": self.latest_evaluation,
            "budget": self.budget_report(),
            "data_isolation": {
                "training_unique_base_families": len(self.training_family_hashes),
                "training_duplicate_families": self.training_duplicate_families,
                "evaluation_unique_base_families": len(self.evaluation_family_hashes),
                "overlap": len(self.training_family_hashes & self.evaluation_family_hashes),
            },
            "recovery": {
                "semantics": "at-least-once",
                "checkpoint_step": self.last_checkpoint_step,
                "max_replayed_steps_on_crash": self.config.checkpoint_steps,
            },
            "claim_boundary": (
                "This is calibration-only engineering evidence. It cannot establish a Stage 2 "
                "candidate effect, a new training paradigm, novelty, or a formal result."
            ),
        }

    def status(self, state: str, detail: str | None = None) -> dict[str, object]:
        return {
            "schema_version": 1,
            "packet": "DH-S2-R2",
            "state": state,
            "detail": detail,
            "global_step": self.global_step,
            "target_optimizer_steps": self.config.optimizer_steps,
            "elapsed_seconds": self.elapsed_seconds(),
            "time_budget_minutes": self.config.time_budget_minutes,
            "checkpoint": self.last_checkpoint,
            "recovery_semantics": "at-least-once",
        }

    def save_checkpoint(self, kind: str = "scheduled") -> Path:
        directory = self.run_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"checkpoint-{self.global_step:08d}-{kind}.pt"
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        payload = {
            "schema_version": 1,
            "kind": kind,
            "global_step": self.global_step,
            "elapsed_seconds": self.elapsed_seconds(),
            "config": self.config.to_dict(),
            "models": {name: _cpu_tree(model.state_dict()) for name, model in self.models.items()},
            "optimizers": {
                name: _cpu_tree(optimizer.state_dict())
                for name, optimizer in self.optimizers.items()
            },
            "torch_rng_state": torch.get_rng_state(),
            "generator_state": self.generator.get_state(),
            "training_family_hashes": sorted(self.training_family_hashes),
            "training_duplicate_families": self.training_duplicate_families,
            "evaluation_family_hashes": sorted(self.evaluation_family_hashes),
            "latest_evaluation": self.latest_evaluation,
            "cumulative": self.cumulative,
            "initial_router": self._initial_router,
            "recovery": {
                "semantics": "at-least-once",
                "checkpoint_step": self.global_step,
                "max_replayed_steps_on_crash": self.config.checkpoint_steps,
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.last_checkpoint_step = self.global_step
        self.last_checkpoint = str(path)
        atomic_write_json(
            directory / "latest.json",
            {"checkpoint": str(path), "global_step": self.global_step, "kind": kind},
        )
        return path

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise RuntimeError("Stage 2 requires checkpoint schema version 1")
        if payload.get("config") != self.config.to_dict():
            raise RuntimeError("Stage 2 checkpoint config mismatch")
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
        self.training_family_hashes = set(payload["training_family_hashes"])
        self.training_duplicate_families = int(payload["training_duplicate_families"])
        self.evaluation_family_hashes = set(payload["evaluation_family_hashes"])
        self.latest_evaluation = payload["latest_evaluation"]
        self.cumulative = payload["cumulative"]
        self._initial_router = payload["initial_router"]
        torch.set_rng_state(payload["torch_rng_state"])
        self.generator.set_state(payload["generator_state"])
        self.last_checkpoint_step = self.global_step
        self.last_checkpoint = str(path)


def latest_stage2_checkpoint(run_dir: Path) -> Path:
    pointer = run_dir / "checkpoints" / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"no Stage 2 checkpoint pointer in {run_dir}")
    raw = json.loads(pointer.read_text(encoding="utf-8"))
    path = Path(raw["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"Stage 2 checkpoint is missing: {path}")
    return path
