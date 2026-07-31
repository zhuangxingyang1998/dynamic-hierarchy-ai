"""Deterministic synthetic reasoning tasks with hidden dependency annotations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import torch

from .config import DataConfig

PAD, BOS, SEP, QUERY, LPAREN, RPAREN, ADD, SUB = range(8)


@dataclass(frozen=True)
class LeafSourceReference:
    node_id: int
    source_index: int


@dataclass(frozen=True)
class MergeSourceReference:
    node_id: int
    left: int
    right: int
    operator_source_index: int


StructureNodeReference = LeafSourceReference | MergeSourceReference


@dataclass(frozen=True)
class StructureSample:
    root_id: int
    nodes: tuple[StructureNodeReference, ...]


@dataclass(frozen=True)
class StructureOnlyBatch:
    """Tree topology and source indices only; no task truth or values."""

    samples: tuple[StructureSample, ...]


@dataclass(frozen=True)
class GenerationStats:
    attempts: int
    accepted: int
    acceptance_rate: float
    label_counts: tuple[int, ...]
    structural_rejections: int
    shape_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    depth: int
    topology: str
    shape_partition: str


@dataclass(frozen=True)
class SyntheticBatch:
    token_ids: torch.Tensor
    position_features: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    task_name: str
    truth: tuple[dict[str, object], ...]
    structure: StructureOnlyBatch | None = None
    generation: GenerationStats | None = None

    def to(self, device: torch.device | str) -> "SyntheticBatch":
        return SyntheticBatch(
            token_ids=self.token_ids.to(device),
            position_features=self.position_features.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            task_name=self.task_name,
            truth=self.truth,
            structure=self.structure,
            generation=self.generation,
        )

    def split(self, size: int) -> tuple["SyntheticBatch", ...]:
        if size <= 0 or self.token_ids.shape[0] % size:
            raise ValueError("split size must be positive and divide the batch")
        batches = []
        for start in range(0, self.token_ids.shape[0], size):
            stop = start + size
            structure = (
                StructureOnlyBatch(self.structure.samples[start:stop])
                if self.structure is not None
                else None
            )
            batches.append(
                SyntheticBatch(
                    self.token_ids[start:stop],
                    self.position_features[start:stop],
                    self.attention_mask[start:stop],
                    self.labels[start:stop],
                    self.task_name,
                    self.truth[start:stop],
                    structure,
                )
            )
        return tuple(batches)


TreeShape = tuple["TreeShape", "TreeShape"] | None


def _shape_height(shape: TreeShape) -> int:
    if shape is None:
        return 0
    return 1 + max(_shape_height(shape[0]), _shape_height(shape[1]))


@lru_cache(maxsize=None)
def _all_binary_shapes(merges: int) -> tuple[TreeShape, ...]:
    if merges == 0:
        return (None,)
    shapes: list[TreeShape] = []
    for left_merges in range(merges):
        right_merges = merges - 1 - left_merges
        for left in _all_binary_shapes(left_merges):
            for right in _all_binary_shapes(right_merges):
                shapes.append((left, right))
    return tuple(shapes)


class SyntheticTaskGenerator:
    """Generates data without a hand-authored token, paragraph, or chapter ontology."""

    def __init__(self, config: DataConfig, seed: int) -> None:
        config.validate(max_length_scale=1)
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def batch(self, task_name: str, batch_size: int, length_scale: int = 1) -> SyntheticBatch:
        if length_scale < 1:
            raise ValueError("length_scale must be at least one")
        if task_name == "repeat_symbol":
            return self._repeat_symbol(batch_size, length_scale)
        if task_name == "variable_binding":
            return self._variable_binding(batch_size, length_scale)
        if task_name == "nested_expression":
            return self._nested_expression(batch_size, length_scale)
        raise ValueError(f"unknown task: {task_name}")

    def stage1_batch(
        self,
        batch_size: int,
        depth: int,
        topology: str,
        *,
        balanced_labels: bool,
        max_attempts_per_example: int,
        shape_partition: str = "train",
    ) -> SyntheticBatch:
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        if topology not in {"leaf", "skew", "balanced", "branched"}:
            raise ValueError(f"unknown topology: {topology}")
        if (depth == 0) != (topology == "leaf"):
            raise ValueError("depth zero requires leaf topology and leaf topology requires depth zero")
        if topology == "branched" and depth < 3:
            raise ValueError("branched topology requires depth at least three")
        if shape_partition not in {"train", "heldout"}:
            raise ValueError("shape_partition must be 'train' or 'heldout'")
        if balanced_labels and batch_size % self.config.expression_values:
            raise ValueError("balanced batch size must be divisible by expression_values")
        if not balanced_labels:
            if topology != "skew" or depth % self.config.expression_depth:
                raise ValueError("legacy unbalanced generation supports only configured skew scales")
            return self._legacy_nested_expression(
                batch_size,
                depth // self.config.expression_depth,
            )
        from .stage1_data import RevisedStage1Generator

        revised = RevisedStage1Generator(self.config, seed=0)
        revised.set_state(self.get_state())
        batch = revised.batch(
            batch_size,
            depth,
            topology,
            max_structural_attempts_per_example=max_attempts_per_example,
            shape_partition=shape_partition,
        )
        self.set_state(revised.get_state())
        return batch

    def get_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state)

    def _symbols(self, count: int) -> torch.Tensor:
        return torch.randint(4, self.config.vocab_size, (count,), generator=self.generator)

    def _unique_symbols(self, count: int) -> torch.Tensor:
        content_capacity = self.config.vocab_size - 4
        if count > content_capacity:
            raise ValueError(
                f"cannot draw {count} unique variables from {content_capacity} generated symbol IDs; increase vocab_size"
            )
        return torch.randperm(content_capacity, generator=self.generator)[:count].add(4)

    @staticmethod
    def _position_features(attention_mask: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = attention_mask.shape
        position = torch.arange(sequence_length, dtype=torch.float32).expand_as(attention_mask)
        lengths = attention_mask.long().sum(dim=1, keepdim=True).sub(1).clamp_min(1)
        absolute = position / lengths
        absolute = absolute * attention_mask
        segment = segment_ids.to(torch.float32)
        # A continuous local phase is useful input, not evidence that phase control works.
        phase = torch.sin(absolute * torch.pi)
        return torch.stack((absolute, segment, phase), dim=-1)

    def _repeat_symbol(self, batch_size: int, length_scale: int) -> SyntheticBatch:
        body_length = self.config.repeat_length * length_scale
        prefix_lengths = torch.randint(1, body_length, (batch_size,), generator=self.generator)
        if batch_size > 1 and torch.unique(prefix_lengths).numel() == 1:
            prefix_lengths[1] = prefix_lengths[0] % (body_length - 1) + 1
        max_length = 2 * body_length + 2
        token_ids = torch.full((batch_size, max_length), PAD, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        segments = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels, truth = [], []
        for row_index, prefix_length in enumerate(prefix_lengths.tolist()):
            body = self._symbols(body_length)
            row = torch.cat((torch.tensor([BOS]), body, torch.tensor([SEP]), body[:prefix_length], torch.tensor([QUERY])))
            valid_length = row.numel()
            token_ids[row_index, :valid_length] = row
            attention_mask[row_index, :valid_length] = True
            segments[row_index, body_length + 1 : valid_length] = 1
            labels.append(body[prefix_length])
            truth.append({"source_index": prefix_length + 1, "query_index": prefix_length, "prefix_length": prefix_length, "body": body.tolist()})
        return SyntheticBatch(token_ids, self._position_features(attention_mask, segments), attention_mask, torch.stack(labels), "repeat_symbol", tuple(truth))

    def _variable_binding(self, batch_size: int, length_scale: int) -> SyntheticBatch:
        pair_count = self.config.binding_pairs * length_scale
        rows, labels, truth = [], [], []
        for _ in range(batch_size):
            variables = self._unique_symbols(pair_count)
            values = self._symbols(pair_count)
            chosen = int(torch.randint(pair_count, (1,), generator=self.generator).item())
            bindings = torch.stack((variables, values), dim=1).flatten()
            row = torch.cat((torch.tensor([BOS]), bindings, torch.tensor([QUERY]), variables[chosen : chosen + 1]))
            rows.append(row)
            labels.append(values[chosen])
            truth.append({"query_variable": int(variables[chosen]), "bound_value": int(values[chosen]), "binding_index": chosen, "variables": variables.tolist(), "values": values.tolist()})
        token_ids = torch.stack(rows)
        segments = torch.zeros_like(token_ids)
        segments[:, -2:] = 1
        attention_mask = token_ids.ne(PAD)
        return SyntheticBatch(token_ids, self._position_features(attention_mask, segments), attention_mask, torch.stack(labels), "variable_binding", tuple(truth))

    def _legacy_nested_expression(self, batch_size: int, length_scale: int) -> SyntheticBatch:
        depth = self.config.expression_depth * length_scale
        value_start = 8
        variable_start = value_start + self.config.expression_values
        variable_tokens = torch.arange(
            variable_start,
            variable_start + self.config.expression_variables,
            dtype=torch.long,
        )
        rows: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        truths: list[dict[str, object]] = []
        structures: list[StructureSample] = []
        segment_rows: list[torch.Tensor] = []
        for _ in range(batch_size):
            binding_values = torch.randint(
                value_start,
                value_start + self.config.expression_values,
                (self.config.expression_variables,),
                generator=self.generator,
            )
            binding_order = torch.randperm(self.config.expression_variables, generator=self.generator)
            ordered_variables = variable_tokens[binding_order]
            ordered_values = binding_values[binding_order]
            bindings = torch.stack((ordered_variables, ordered_values), dim=1).flatten()

            leaf_count = depth + 1
            leaf_variables = variable_tokens[
                torch.randint(self.config.expression_variables, (leaf_count,), generator=self.generator)
            ]
            repeat_positions = torch.randperm(leaf_count, generator=self.generator)[:2]
            leaf_variables[repeat_positions[1]] = leaf_variables[repeat_positions[0]]
            nodes: list[dict[str, object]] = []
            leaf_cursor = 0

            def build(remaining_depth: int) -> int:
                nonlocal leaf_cursor
                if remaining_depth == 0:
                    node_id = len(nodes)
                    nodes.append(
                        {
                            "node_id": node_id,
                            "kind": "leaf",
                            "variable_token": int(leaf_variables[leaf_cursor]),
                        }
                    )
                    leaf_cursor += 1
                    return node_id
                deep_child = build(remaining_depth - 1)
                leaf_child = build(0)
                if int(torch.randint(2, (1,), generator=self.generator)):
                    left, right = deep_child, leaf_child
                else:
                    left, right = leaf_child, deep_child
                operator = ADD if int(torch.randint(2, (1,), generator=self.generator)) == 0 else SUB
                node_id = len(nodes)
                nodes.append(
                    {
                        "node_id": node_id,
                        "kind": "merge",
                        "left": left,
                        "right": right,
                        "operator_token": operator,
                    }
                )
                return node_id

            root_id = build(depth)
            expression_tokens: list[int] = []
            prefix_length = 1 + bindings.numel() + 1

            def serialize(node_id: int) -> None:
                node = nodes[node_id]
                if node["kind"] == "leaf":
                    node["source_index"] = prefix_length + len(expression_tokens)
                    expression_tokens.append(int(node["variable_token"]))
                    return
                expression_tokens.append(LPAREN)
                serialize(int(node["left"]))
                node["operator_source_index"] = prefix_length + len(expression_tokens)
                expression_tokens.append(int(node["operator_token"]))
                serialize(int(node["right"]))
                expression_tokens.append(RPAREN)

            serialize(root_id)
            value_by_variable = {
                int(variable_tokens[index]): int(binding_values[index]) - value_start
                for index in range(self.config.expression_variables)
            }

            def evaluate(node_id: int) -> int:
                node = nodes[node_id]
                if node["kind"] == "leaf":
                    return value_by_variable[int(node["variable_token"])]
                left_value = evaluate(int(node["left"]))
                right_value = evaluate(int(node["right"]))
                if int(node["operator_token"]) == ADD:
                    return (left_value + right_value) % self.config.expression_values
                return (left_value - right_value) % self.config.expression_values

            label = torch.tensor(value_start + evaluate(root_id), dtype=torch.long)
            row = torch.cat(
                (
                    torch.tensor([BOS]),
                    bindings,
                    torch.tensor([SEP]),
                    torch.tensor(expression_tokens),
                    torch.tensor([QUERY]),
                )
            )
            segments = torch.zeros_like(row)
            segments[prefix_length:] = 1
            rows.append(row)
            segment_rows.append(segments)
            labels.append(label)
            truths.append(
                {
                    "root_id": root_id,
                    "nodes": nodes,
                    "binding_variables": ordered_variables.tolist(),
                    "binding_values": ordered_values.tolist(),
                    "expression_depth": depth,
                    "recursive_steps": depth,
                    "combined_nodes": depth,
                    "repeated_variable": int(leaf_variables[repeat_positions[0]]),
                }
            )
            structures.append(
                StructureSample(
                    root_id=root_id,
                    nodes=tuple(
                        LeafSourceReference(
                            node_id=int(node["node_id"]),
                            source_index=int(node["source_index"]),
                        )
                        if node["kind"] == "leaf"
                        else MergeSourceReference(
                            node_id=int(node["node_id"]),
                            left=int(node["left"]),
                            right=int(node["right"]),
                            operator_source_index=int(node["operator_source_index"]),
                        )
                        for node in nodes
                    ),
                )
            )
        token_ids = torch.stack(rows)
        segment_ids = torch.stack(segment_rows)
        attention_mask = token_ids.ne(PAD)
        return SyntheticBatch(
            token_ids,
            self._position_features(attention_mask, segment_ids),
            attention_mask,
            torch.stack(labels),
            "nested_expression",
            tuple(truths),
            StructureOnlyBatch(tuple(structures)),
        )

    def _nested_expression(self, batch_size: int, length_scale: int) -> SyntheticBatch:
        return self._legacy_nested_expression(batch_size, length_scale)
