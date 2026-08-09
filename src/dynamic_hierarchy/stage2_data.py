"""Leak-resistant query-family data for Stage 2 R2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from .data import (
    ADD,
    BOS,
    SUB,
    LeafSourceReference,
    MergeSourceReference,
    StructureOnlyBatch,
    StructureSample,
    SyntheticTaskGenerator,
)
from .stage2_config import Stage2Profile


QUERY_ADD_FIRST_TOKEN = 15
QUERY_SUB_FIRST_TOKEN = 16
QUERY_ADD_FIRST = 0
QUERY_SUB_FIRST = 1
LEGAL_LABEL_PAIRS = tuple((left, right) for left in range(7) for right in range(7) if left != right)


@dataclass(frozen=True)
class Stage2OrdinaryBatch:
    token_ids: torch.Tensor
    position_features: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    query_ids: torch.Tensor
    literal_source_indices: torch.Tensor
    operator_source_indices: torch.Tensor
    base_family_hashes: tuple[str, ...]
    query_row_hashes: tuple[str, ...]
    profile_name: str

    def to(self, device: torch.device | str) -> "Stage2OrdinaryBatch":
        return Stage2OrdinaryBatch(
            token_ids=self.token_ids.to(device),
            position_features=self.position_features.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            query_ids=self.query_ids.to(device),
            literal_source_indices=self.literal_source_indices.to(device),
            operator_source_indices=self.operator_source_indices.to(device),
            base_family_hashes=self.base_family_hashes,
            query_row_hashes=self.query_row_hashes,
            profile_name=self.profile_name,
        )


@dataclass(frozen=True)
class Stage2GenerationStats:
    attempts: int
    accepted_families: int
    structural_rejections: int
    equal_label_rejections: int
    quota_rejections: int
    excluded_family_rejections: int
    duplicate_family_rejections: int
    label_pair_counts: tuple[tuple[int, int, int], ...]
    family_hashes: tuple[str, ...]
    oracle_shape_ids: tuple[str, ...]
    fixed_policy_counterexample_rows: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Stage2FamilyTruth:
    base_family_hash: str
    values: tuple[int, ...]
    operator_pattern: str
    labels: tuple[int, int]
    structures: tuple[StructureSample, StructureSample]
    shape_ids: tuple[str, str]


@dataclass(frozen=True)
class Stage2GeneratedBatch:
    ordinary: Stage2OrdinaryBatch
    diagnostic_structure: StructureOnlyBatch
    family_truth: tuple[Stage2FamilyTruth, ...]
    generation: Stage2GenerationStats


@dataclass(frozen=True)
class _ActiveNode:
    node_id: int
    value: int


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _shape_id(structure: StructureSample) -> str:
    nodes: dict[int, object] = {}
    for node in structure.nodes:
        if isinstance(node, LeafSourceReference):
            nodes[node.node_id] = "leaf"
        else:
            nodes[node.node_id] = (nodes[node.left], nodes[node.right])
    return _canonical_digest(nodes[structure.root_id])


def evaluate_precedence_expression(
    values: tuple[int, ...],
    operator_pattern: str,
    query_id: int,
) -> tuple[int, StructureSample]:
    """Evaluate one completed base input and return value plus source-only tree."""

    if len(values) < 2 or len(operator_pattern) != len(values) - 1:
        raise ValueError("operator count must equal value count - 1")
    if set(operator_pattern) - {"+", "-"}:
        raise ValueError("unsupported operator symbol")
    if query_id not in {QUERY_ADD_FIRST, QUERY_SUB_FIRST}:
        raise ValueError("unsupported precedence query")
    leaf_positions = tuple(1 + 2 * index for index in range(len(values)))
    operator_positions = tuple(2 + 2 * index for index in range(len(operator_pattern)))
    active = [_ActiveNode(index, value % 7) for index, value in enumerate(values)]
    active_operators = [
        (symbol, source_index)
        for symbol, source_index in zip(operator_pattern, operator_positions, strict=True)
    ]
    references: list[LeafSourceReference | MergeSourceReference] = [
        LeafSourceReference(index, source_index)
        for index, source_index in enumerate(leaf_positions)
    ]
    preferred = "+" if query_id == QUERY_ADD_FIRST else "-"
    next_node_id = len(active)
    while active_operators:
        merge_index = next(
            (index for index, (symbol, _) in enumerate(active_operators) if symbol == preferred),
            0,
        )
        symbol, operator_source_index = active_operators.pop(merge_index)
        left = active[merge_index]
        right = active[merge_index + 1]
        value = (
            left.value + right.value if symbol == "+" else left.value - right.value
        ) % 7
        references.append(
            MergeSourceReference(
                node_id=next_node_id,
                left=left.node_id,
                right=right.node_id,
                operator_source_index=operator_source_index,
            )
        )
        active[merge_index : merge_index + 2] = [_ActiveNode(next_node_id, value)]
        next_node_id += 1
    structure = StructureSample(root_id=active[0].node_id, nodes=tuple(references))
    return active[0].value, structure


def evaluate_fixed_policy_expression(
    values: tuple[int, ...],
    operator_pattern: str,
    policy: str,
) -> tuple[int, StructureSample]:
    """Evaluate one complete expression under a deterministic fixed merge policy."""

    if policy in {"add", "sub"}:
        query = QUERY_ADD_FIRST if policy == "add" else QUERY_SUB_FIRST
        return evaluate_precedence_expression(values, operator_pattern, query)
    if policy not in {"left", "right"}:
        raise ValueError("fixed expression policy must be left, right, add, or sub")
    leaf_positions = tuple(1 + 2 * index for index in range(len(values)))
    operator_positions = tuple(2 + 2 * index for index in range(len(operator_pattern)))
    active = [_ActiveNode(index, value % 7) for index, value in enumerate(values)]
    active_operators = [
        (symbol, source_index)
        for symbol, source_index in zip(operator_pattern, operator_positions, strict=True)
    ]
    references: list[LeafSourceReference | MergeSourceReference] = [
        LeafSourceReference(index, source_index)
        for index, source_index in enumerate(leaf_positions)
    ]
    next_node_id = len(active)
    while active_operators:
        merge_index = 0 if policy == "left" else len(active_operators) - 1
        symbol, operator_source_index = active_operators.pop(merge_index)
        left = active[merge_index]
        right = active[merge_index + 1]
        value = (
            left.value + right.value if symbol == "+" else left.value - right.value
        ) % 7
        references.append(
            MergeSourceReference(
                node_id=next_node_id,
                left=left.node_id,
                right=right.node_id,
                operator_source_index=operator_source_index,
            )
        )
        active[merge_index : merge_index + 2] = [_ActiveNode(next_node_id, value)]
        next_node_id += 1
    return active[0].value, StructureSample(active[0].node_id, tuple(references))


class Stage2PrecedenceFamilyGenerator:
    """Generate base inputs first, then derive both query rows and balance families."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def get_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state)

    def balanced_block(
        self,
        profile: Stage2Profile,
        *,
        blocks: int = 1,
        max_attempts_per_family: int = 512,
        excluded_family_hashes: set[str] | None = None,
    ) -> Stage2GeneratedBatch:
        profile.validate()
        if blocks <= 0 or max_attempts_per_family <= 0:
            raise ValueError("blocks and max_attempts_per_family must be positive")
        excluded = excluded_family_hashes or set()
        target_per_pair = blocks
        pair_counts = {pair: 0 for pair in LEGAL_LABEL_PAIRS}
        accepted: list[Stage2FamilyTruth] = []
        accepted_hashes: set[str] = set()
        attempts = 0
        structural_rejections = 0
        equal_label_rejections = 0
        quota_rejections = 0
        excluded_rejections = 0
        duplicate_rejections = 0
        hard_limit = len(LEGAL_LABEL_PAIRS) * blocks * max_attempts_per_family
        while len(accepted) < len(LEGAL_LABEL_PAIRS) * blocks:
            if attempts >= hard_limit:
                raise RuntimeError("Stage 2 family generation exhausted its hard candidate limit")
            attempts += 1
            values = tuple(
                int(value)
                for value in torch.randint(
                    0,
                    7,
                    (profile.leaf_count,),
                    generator=self.generator,
                ).tolist()
            )
            add_value, add_structure = evaluate_precedence_expression(
                values, profile.operator_pattern, QUERY_ADD_FIRST
            )
            sub_value, sub_structure = evaluate_precedence_expression(
                values, profile.operator_pattern, QUERY_SUB_FIRST
            )
            if _shape_id(add_structure) == _shape_id(sub_structure):
                structural_rejections += 1
                continue
            if add_value == sub_value:
                equal_label_rejections += 1
                continue
            pair = (add_value, sub_value)
            if pair_counts[pair] >= target_per_pair:
                quota_rejections += 1
                continue
            family_hash = _canonical_digest(
                {
                    "leaf_count": profile.leaf_count,
                    "operator_pattern": profile.operator_pattern,
                    "values": values,
                }
            )
            if family_hash in excluded:
                excluded_rejections += 1
                continue
            if family_hash in accepted_hashes:
                duplicate_rejections += 1
                continue
            shape_ids = (_shape_id(add_structure), _shape_id(sub_structure))
            accepted.append(
                Stage2FamilyTruth(
                    base_family_hash=family_hash,
                    values=values,
                    operator_pattern=profile.operator_pattern,
                    labels=pair,
                    structures=(add_structure, sub_structure),
                    shape_ids=shape_ids,
                )
            )
            accepted_hashes.add(family_hash)
            pair_counts[pair] += 1

        order = torch.randperm(len(accepted), generator=self.generator).tolist()
        accepted = [accepted[index] for index in order]
        rows: list[list[int]] = []
        labels: list[int] = []
        query_ids: list[int] = []
        family_hashes: list[str] = []
        row_hashes: list[str] = []
        structures: list[StructureSample] = []
        literal_positions = [1 + 2 * index for index in range(profile.leaf_count)]
        operator_positions = [2 + 2 * index for index in range(profile.leaf_count - 1)]
        literal_sources: list[list[int]] = []
        operator_sources: list[list[int]] = []
        for family in accepted:
            prefix = [BOS]
            for index, value in enumerate(family.values):
                prefix.append(8 + value)
                if index < len(family.operator_pattern):
                    prefix.append(ADD if family.operator_pattern[index] == "+" else SUB)
            for query_id, query_token in (
                (QUERY_ADD_FIRST, QUERY_ADD_FIRST_TOKEN),
                (QUERY_SUB_FIRST, QUERY_SUB_FIRST_TOKEN),
            ):
                rows.append([*prefix, query_token])
                labels.append(family.labels[query_id])
                query_ids.append(query_id)
                family_hashes.append(family.base_family_hash)
                row_hashes.append(
                    _canonical_digest(
                        {"base_family_hash": family.base_family_hash, "query_id": query_id}
                    )
                )
                structures.append(family.structures[query_id])
                literal_sources.append(literal_positions)
                operator_sources.append(operator_positions)

        token_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        segment_ids = torch.zeros_like(token_ids, dtype=torch.long)
        ordinary = Stage2OrdinaryBatch(
            token_ids=token_ids,
            position_features=SyntheticTaskGenerator._position_features(
                attention_mask, segment_ids
            ),
            attention_mask=attention_mask,
            labels=torch.tensor(labels, dtype=torch.long),
            query_ids=torch.tensor(query_ids, dtype=torch.long),
            literal_source_indices=torch.tensor(literal_sources, dtype=torch.long),
            operator_source_indices=torch.tensor(operator_sources, dtype=torch.long),
            base_family_hashes=tuple(family_hashes),
            query_row_hashes=tuple(row_hashes),
            profile_name=profile.name,
        )
        stats = Stage2GenerationStats(
            attempts=attempts,
            accepted_families=len(accepted),
            structural_rejections=structural_rejections,
            equal_label_rejections=equal_label_rejections,
            quota_rejections=quota_rejections,
            excluded_family_rejections=excluded_rejections,
            duplicate_family_rejections=duplicate_rejections,
            label_pair_counts=tuple(
                (left, right, pair_counts[(left, right)]) for left, right in LEGAL_LABEL_PAIRS
            ),
            family_hashes=tuple(family.base_family_hash for family in accepted),
            oracle_shape_ids=tuple(
                shape_id for family in accepted for shape_id in family.shape_ids
            ),
            fixed_policy_counterexample_rows=tuple(
                (
                    policy,
                    2 * len(accepted)
                    if policy == "stop"
                    else sum(
                        _shape_id(
                            evaluate_fixed_policy_expression(
                                family.values,
                                family.operator_pattern,
                                policy,
                            )[1]
                        )
                        != oracle_shape
                        for family in accepted
                        for oracle_shape in family.shape_ids
                    ),
                )
                for policy in ("stop", "left", "right", "add", "sub")
            ),
        )
        return Stage2GeneratedBatch(
            ordinary=ordinary,
            diagnostic_structure=StructureOnlyBatch(tuple(structures)),
            family_truth=tuple(accepted),
            generation=stats,
        )
