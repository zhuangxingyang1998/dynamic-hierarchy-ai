"""Leak-resistant revised Stage 1 data generation and sham structure controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import torch

from .config import DataConfig
from .data import (
    ADD,
    BOS,
    LPAREN,
    PAD,
    QUERY,
    RPAREN,
    SEP,
    SUB,
    GenerationStats,
    LeafSourceReference,
    MergeSourceReference,
    StructureOnlyBatch,
    StructureSample,
    SyntheticBatch,
    SyntheticTaskGenerator,
)

TreeShape = tuple["TreeShape", "TreeShape"] | None
SHAM_MAPPING_VERSION = "content-keyed-derangement-v1"


def canonical_shape(shape: TreeShape) -> str:
    if shape is None:
        return "L"
    return f"({canonical_shape(shape[0])},{canonical_shape(shape[1])})"


def shape_id(shape: TreeShape) -> str:
    return hashlib.sha256(canonical_shape(shape).encode("ascii")).hexdigest()[:16]


def shape_height(shape: TreeShape) -> int:
    if shape is None:
        return 0
    return 1 + max(shape_height(shape[0]), shape_height(shape[1]))


@lru_cache(maxsize=None)
def all_binary_shapes(merges: int) -> tuple[TreeShape, ...]:
    if merges == 0:
        return (None,)
    shapes: list[TreeShape] = []
    for left_merges in range(merges):
        right_merges = merges - 1 - left_merges
        for left in all_binary_shapes(left_merges):
            for right in all_binary_shapes(right_merges):
                shapes.append((left, right))
    return tuple(shapes)


@lru_cache(maxsize=None)
def shape_catalog(depth: int, topology: str) -> tuple[TreeShape, ...]:
    if depth == 0 and topology == "leaf":
        return (None,)
    if depth <= 0:
        raise ValueError("non-leaf topology requires positive depth")
    if topology == "skew":
        return tuple(shape for shape in all_binary_shapes(depth) if shape_height(shape) == depth)
    if topology == "balanced":
        child = shape_catalog(depth - 1, "balanced")[0] if depth > 1 else None
        return ((child, child),)
    if topology == "branched":
        if depth < 3:
            raise ValueError("branched topology requires depth at least three")
        return tuple(
            shape
            for shape in all_binary_shapes(depth + 1)
            if shape_height(shape) == depth
            and shape is not None
            and shape[0] is not None
            and shape[1] is not None
        )
    raise ValueError(f"unsupported topology: {topology}")


@dataclass
class StructuralTemplate:
    shape: TreeShape
    nodes: list[dict[str, int | str]]
    root_id: int
    coefficients: tuple[int, ...]


class RevisedStage1Generator:
    """Generate exactly balanced mod-7 examples without target-dependent structure."""

    def __init__(
        self,
        config: DataConfig,
        seed: int,
        operand_mode: str = "bound_variable",
    ) -> None:
        config.validate(max_length_scale=1)
        if config.expression_values != 7:
            raise ValueError("revised Stage 1 generator requires prime modulus seven")
        if operand_mode not in {"bound_variable", "literal"}:
            raise ValueError("operand_mode must be 'bound_variable' or 'literal'")
        self.config = config
        self.operand_mode = operand_mode
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def get_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state)

    def _structural_template(self, depth: int, topology: str) -> StructuralTemplate | None:
        catalog = shape_catalog(depth, topology)
        shape = catalog[int(torch.randint(len(catalog), (1,), generator=self.generator))]
        nodes: list[dict[str, int | str]] = []
        leaf_count = canonical_shape(shape).count("L")
        leaf_variables = None
        if self.operand_mode == "bound_variable":
            leaf_variables = torch.randint(
                self.config.expression_variables,
                (leaf_count,),
                generator=self.generator,
            )
            if leaf_count >= 3:
                repeat = torch.randperm(leaf_count, generator=self.generator)[:2]
                leaf_variables[repeat[1]] = leaf_variables[repeat[0]]
        leaf_cursor = 0

        def build(item: TreeShape) -> int:
            nonlocal leaf_cursor
            if item is None:
                node_id = len(nodes)
                leaf_operand = (
                    {"variable_index": int(leaf_variables[leaf_cursor])}
                    if leaf_variables is not None
                    else {"leaf_index": leaf_cursor}
                )
                nodes.append({"node_id": node_id, "kind": "leaf", **leaf_operand})
                leaf_cursor += 1
                return node_id
            left = build(item[0])
            right = build(item[1])
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

        root_id = build(shape)
        coefficients_by_node: dict[int, list[int]] = {}
        for node in nodes:
            node_id = int(node["node_id"])
            if node["kind"] == "leaf":
                coefficient_count = (
                    self.config.expression_variables
                    if self.operand_mode == "bound_variable"
                    else leaf_count
                )
                coefficients = [0] * coefficient_count
                coefficient_index = (
                    int(node["variable_index"])
                    if self.operand_mode == "bound_variable"
                    else int(node["leaf_index"])
                )
                coefficients[coefficient_index] = 1
            else:
                left = coefficients_by_node[int(node["left"])]
                right = coefficients_by_node[int(node["right"])]
                sign = 1 if int(node["operator_token"]) == ADD else -1
                coefficients = [
                    (left[index] + sign * right[index]) % 7
                    for index in range(len(left))
                ]
            coefficients_by_node[node_id] = coefficients
        root_coefficients = tuple(coefficients_by_node[root_id])
        if not any(root_coefficients):
            return None
        return StructuralTemplate(shape, nodes, root_id, root_coefficients)

    def _materialize_bound_variable(
        self,
        template: StructuralTemplate,
        target_class: int,
        topology: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], StructureSample, str]:
        value_start = 8
        variable_start = value_start + self.config.expression_values
        variable_tokens = torch.arange(
            variable_start,
            variable_start + self.config.expression_variables,
            dtype=torch.long,
        )
        values = torch.randint(
            self.config.expression_values,
            (self.config.expression_variables,),
            generator=self.generator,
        )
        nonzero = [
            index
            for index, coefficient in enumerate(template.coefficients)
            if coefficient % 7
        ]
        pivot = nonzero[int(torch.randint(len(nonzero), (1,), generator=self.generator))]
        contribution = sum(
            coefficient * int(values[index])
            for index, coefficient in enumerate(template.coefficients)
            if index != pivot
        ) % 7
        inverse = pow(template.coefficients[pivot], -1, 7)
        values[pivot] = ((target_class - contribution) * inverse) % 7

        binding_order = torch.randperm(self.config.expression_variables, generator=self.generator)
        ordered_variables = variable_tokens[binding_order]
        ordered_values = values[binding_order].add(value_start)
        bindings = torch.stack((ordered_variables, ordered_values), dim=1).flatten()
        expression_tokens: list[int] = []
        prefix_length = 1 + bindings.numel() + 1

        def serialize(node_id: int) -> None:
            node = template.nodes[node_id]
            if node["kind"] == "leaf":
                node["source_index"] = prefix_length + len(expression_tokens)
                expression_tokens.append(int(variable_tokens[int(node["variable_index"])]))
                return
            expression_tokens.append(LPAREN)
            serialize(int(node["left"]))
            node["operator_source_index"] = prefix_length + len(expression_tokens)
            expression_tokens.append(int(node["operator_token"]))
            serialize(int(node["right"]))
            expression_tokens.append(RPAREN)

        serialize(template.root_id)
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
        node_count = len(template.nodes)
        merge_count = sum(node["kind"] == "merge" for node in template.nodes)
        canonical_id = shape_id(template.shape)
        truth = {
            "operand_mode": "bound_variable",
            "root_id": template.root_id,
            "nodes": template.nodes,
            "binding_variables": ordered_variables.tolist(),
            "binding_values": ordered_values.tolist(),
            "expression_depth": shape_height(template.shape),
            "node_count": node_count,
            "merge_count": merge_count,
            "topology": topology,
            "shape_id": canonical_id,
        }
        structure = StructureSample(
            root_id=template.root_id,
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
                for node in template.nodes
            ),
        )
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "tokens": row.tolist(),
                    "label": target_class,
                    "shape_id": canonical_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        return row, segments, truth, structure, content_hash

    def _materialize_literal(
        self,
        template: StructuralTemplate,
        target_class: int,
        topology: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], StructureSample, str]:
        value_start = 8
        values = torch.randint(
            self.config.expression_values,
            (len(template.coefficients),),
            generator=self.generator,
        )
        nonzero = [
            index
            for index, coefficient in enumerate(template.coefficients)
            if coefficient % 7
        ]
        pivot = nonzero[int(torch.randint(len(nonzero), (1,), generator=self.generator))]
        contribution = sum(
            coefficient * int(values[index])
            for index, coefficient in enumerate(template.coefficients)
            if index != pivot
        ) % 7
        inverse = pow(template.coefficients[pivot], -1, 7)
        values[pivot] = ((target_class - contribution) * inverse) % 7

        expression_tokens: list[int] = []
        prefix_length = 2

        def serialize(node_id: int) -> None:
            node = template.nodes[node_id]
            if node["kind"] == "leaf":
                node["source_index"] = prefix_length + len(expression_tokens)
                leaf_value = int(values[int(node["leaf_index"])])
                expression_tokens.append(value_start + leaf_value)
                return
            expression_tokens.append(LPAREN)
            serialize(int(node["left"]))
            node["operator_source_index"] = prefix_length + len(expression_tokens)
            expression_tokens.append(int(node["operator_token"]))
            serialize(int(node["right"]))
            expression_tokens.append(RPAREN)

        serialize(template.root_id)
        row = torch.tensor([BOS, SEP, *expression_tokens, QUERY], dtype=torch.long)
        segments = torch.zeros_like(row)
        segments[prefix_length:] = 1
        canonical_id = shape_id(template.shape)
        truth = {
            "operand_mode": "literal",
            "root_id": template.root_id,
            "nodes": template.nodes,
            "literal_values": values.tolist(),
            "expression_depth": shape_height(template.shape),
            "node_count": len(template.nodes),
            "merge_count": sum(node["kind"] == "merge" for node in template.nodes),
            "topology": topology,
            "shape_id": canonical_id,
        }
        structure = StructureSample(
            root_id=template.root_id,
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
                for node in template.nodes
            ),
        )
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "tokens": row.tolist(),
                    "label": target_class,
                    "shape_id": canonical_id,
                    "operand_mode": "literal",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        return row, segments, truth, structure, content_hash

    def _materialize(
        self,
        template: StructuralTemplate,
        target_class: int,
        topology: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], StructureSample, str]:
        if self.operand_mode == "literal":
            return self._materialize_literal(template, target_class, topology)
        return self._materialize_bound_variable(template, target_class, topology)

    def batch(
        self,
        batch_size: int,
        depth: int,
        topology: str,
        *,
        max_structural_attempts_per_example: int,
        shape_partition: str,
    ) -> SyntheticBatch:
        if batch_size <= 0 or batch_size % 7:
            raise ValueError("revised batch size must be positive and divisible by seven")
        if shape_partition not in {"train", "heldout"}:
            raise ValueError("shape_partition must be 'train' or 'heldout'")
        if max_structural_attempts_per_example <= 0:
            raise ValueError("max structural attempts must be positive")

        templates: list[StructuralTemplate] = []
        attempts = 0
        structural_rejections = 0
        for _ in range(batch_size):
            for _attempt in range(max_structural_attempts_per_example):
                attempts += 1
                template = self._structural_template(depth, topology)
                if template is not None:
                    templates.append(template)
                    break
                structural_rejections += 1
            else:
                raise RuntimeError(
                    "structural generation exhausted the configured attempt limit; fail closed"
                )

        repetitions = batch_size // 7
        targets = torch.arange(7).repeat_interleave(repetitions)
        targets = targets[torch.randperm(batch_size, generator=self.generator)].tolist()
        materialized = [
            self._materialize(template, int(target), topology)
            for template, target in zip(templates, targets)
        ]
        rows, segments, truths, structures, content_hashes = zip(*materialized)
        token_ids = torch.stack(rows)
        segment_ids = torch.stack(segments)
        attention_mask = token_ids.ne(PAD)
        labels = torch.tensor(targets, dtype=torch.long)
        label_counts = torch.bincount(labels, minlength=7)
        generation = GenerationStats(
            attempts=attempts,
            accepted=batch_size,
            acceptance_rate=batch_size / attempts,
            label_counts=tuple(int(value) for value in label_counts.tolist()),
            structural_rejections=structural_rejections,
            shape_ids=tuple(str(truth["shape_id"]) for truth in truths),
            content_hashes=tuple(content_hashes),
            depth=depth,
            topology=topology,
            shape_partition=shape_partition,
        )
        return SyntheticBatch(
            token_ids,
            SyntheticTaskGenerator._position_features(attention_mask, segment_ids),
            attention_mask,
            labels,
            "nested_expression",
            tuple(truths),
            StructureOnlyBatch(tuple(structures)),
            generation,
        )

    def batch_excluding_content(
        self,
        batch_size: int,
        depth: int,
        topology: str,
        *,
        training_content_hashes: set[str],
        prior_evaluation_content_hashes: set[str],
        accepted_evaluation_hashes: set[str],
        max_structural_attempts_per_example: int,
        max_content_attempts_per_example: int,
        shape_partition: str,
        historical_final_evaluation_content_hashes: set[str] | None = None,
    ) -> tuple[SyntheticBatch, dict[str, int | float]]:
        """Build one balanced batch while actively excluding known content."""

        if batch_size <= 0 or batch_size % 7:
            raise ValueError("excluded evaluation batch size must be divisible by seven")
        if max_content_attempts_per_example <= 0:
            raise ValueError("max content attempts must be positive")
        target_per_class = batch_size // 7
        accepted_per_class = [0] * 7
        accepted_rows: list[torch.Tensor] = []
        accepted_positions: list[torch.Tensor] = []
        accepted_masks: list[torch.Tensor] = []
        accepted_labels: list[torch.Tensor] = []
        accepted_truth: list[dict[str, object]] = []
        accepted_structures: list[StructureSample] = []
        accepted_hashes: list[str] = []
        accepted_shapes: list[str] = []
        candidate_examples = 0
        structural_attempts = 0
        structural_rejections = 0
        training_exclusions = 0
        prior_evaluation_exclusions = 0
        historical_final_evaluation_exclusions = 0
        evaluation_exclusions = 0
        label_quota_rejections = 0
        hard_limit = batch_size * max_content_attempts_per_example

        while len(accepted_rows) < batch_size:
            if candidate_examples + 7 > hard_limit:
                raise RuntimeError(
                    "evaluation content exclusion exhausted the configured attempt limit; "
                    "fail closed"
                )
            candidate = self.batch(
                7,
                depth,
                topology,
                max_structural_attempts_per_example=max_structural_attempts_per_example,
                shape_partition=shape_partition,
            )
            candidate_examples += 7
            structural_attempts += candidate.generation.attempts
            structural_rejections += candidate.generation.structural_rejections
            for index, content_hash in enumerate(candidate.generation.content_hashes):
                label = int(candidate.labels[index])
                if content_hash in training_content_hashes:
                    training_exclusions += 1
                    continue
                if content_hash in prior_evaluation_content_hashes:
                    prior_evaluation_exclusions += 1
                    continue
                if content_hash in (
                    historical_final_evaluation_content_hashes or set()
                ):
                    historical_final_evaluation_exclusions += 1
                    continue
                if content_hash in accepted_evaluation_hashes:
                    evaluation_exclusions += 1
                    continue
                if accepted_per_class[label] >= target_per_class:
                    label_quota_rejections += 1
                    continue
                accepted_per_class[label] += 1
                accepted_evaluation_hashes.add(content_hash)
                accepted_rows.append(candidate.token_ids[index])
                accepted_positions.append(candidate.position_features[index])
                accepted_masks.append(candidate.attention_mask[index])
                accepted_labels.append(candidate.labels[index])
                accepted_truth.append(candidate.truth[index])
                accepted_structures.append(candidate.structure.samples[index])
                accepted_hashes.append(content_hash)
                accepted_shapes.append(candidate.generation.shape_ids[index])

        labels = torch.stack(accepted_labels)
        generation = GenerationStats(
            attempts=structural_attempts,
            accepted=batch_size,
            acceptance_rate=batch_size / candidate_examples,
            label_counts=tuple(int(value) for value in torch.bincount(labels, minlength=7)),
            structural_rejections=structural_rejections,
            shape_ids=tuple(accepted_shapes),
            content_hashes=tuple(accepted_hashes),
            depth=depth,
            topology=topology,
            shape_partition=shape_partition,
        )
        batch = SyntheticBatch(
            token_ids=torch.stack(accepted_rows),
            position_features=torch.stack(accepted_positions),
            attention_mask=torch.stack(accepted_masks),
            labels=labels,
            task_name="nested_expression",
            truth=tuple(accepted_truth),
            structure=StructureOnlyBatch(tuple(accepted_structures)),
            generation=generation,
        )
        accounting: dict[str, int | float] = {
            "candidate_examples": candidate_examples,
            "accepted": batch_size,
            "acceptance_rate": batch_size / candidate_examples,
            "training_content_exclusions": training_exclusions,
            "prior_evaluation_content_exclusions": prior_evaluation_exclusions,
            "historical_final_evaluation_content_exclusions": (
                historical_final_evaluation_exclusions
            ),
            "evaluation_content_exclusions": evaluation_exclusions,
            "label_quota_rejections": label_quota_rejections,
            "structural_attempts": structural_attempts,
            "structural_rejections": structural_rejections,
            "hard_candidate_limit": hard_limit,
        }
        return batch, accounting


def _deterministic_derangement(length: int, payload: bytes, domain: bytes) -> list[int]:
    if length <= 1:
        return list(range(length))
    for attempt in range(64):
        permutation = list(range(length))
        for index in range(length - 1, 0, -1):
            digest = hashlib.sha256(
                SHAM_MAPPING_VERSION.encode("ascii")
                + domain
                + payload
                + attempt.to_bytes(2, "little")
                + index.to_bytes(2, "little")
            ).digest()
            swap_index = int.from_bytes(digest[:8], "little") % (index + 1)
            permutation[index], permutation[swap_index] = (
                permutation[swap_index],
                permutation[index],
            )
        if all(source != destination for destination, source in enumerate(permutation)):
            return permutation
    offset = 1 + int.from_bytes(
        hashlib.sha256(domain + payload).digest()[:8],
        "little",
    ) % (length - 1)
    return [(index + offset) % length for index in range(length)]


def sham_structure(
    structure: StructureOnlyBatch,
    token_ids: torch.Tensor,
) -> StructureOnlyBatch:
    """Apply a preregistered content-keyed wrong alignment at equal compose cost."""

    rows = token_ids.detach().cpu().tolist()
    if len(rows) != len(structure.samples):
        raise ValueError("sham mapping requires one token row per structure sample")
    sham_samples: list[StructureSample] = []
    for sample, row in zip(structure.samples, rows):
        payload = bytes(int(token) for token in row)
        leaves = [node for node in sample.nodes if isinstance(node, LeafSourceReference)]
        merges = [node for node in sample.nodes if isinstance(node, MergeSourceReference)]
        leaf_sources = [node.source_index for node in leaves]
        operator_sources = [node.operator_source_index for node in merges]
        leaf_permutation = _deterministic_derangement(
            len(leaf_sources),
            payload,
            b":leaf:",
        )
        operator_permutation = _deterministic_derangement(
            len(operator_sources),
            payload,
            b":operator:",
        )
        leaf_sources = [leaf_sources[index] for index in leaf_permutation]
        operator_sources = [operator_sources[index] for index in operator_permutation]
        leaf_cursor = 0
        merge_cursor = 0
        nodes = []
        for node in sample.nodes:
            if isinstance(node, LeafSourceReference):
                nodes.append(LeafSourceReference(node.node_id, leaf_sources[leaf_cursor]))
                leaf_cursor += 1
            else:
                nodes.append(
                    MergeSourceReference(
                        node.node_id,
                        node.left,
                        node.right,
                        operator_sources[merge_cursor],
                    )
                )
                merge_cursor += 1
        sham_samples.append(StructureSample(sample.root_id, tuple(nodes)))
    return StructureOnlyBatch(tuple(sham_samples))
