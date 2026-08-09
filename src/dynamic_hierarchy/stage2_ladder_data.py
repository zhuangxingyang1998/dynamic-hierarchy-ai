"""Exhaustive, leak-auditable arithmetic domains for Stage 2 R5.1."""

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
from .stage2_data import (
    QUERY_ADD_FIRST_TOKEN,
    QUERY_SUB_FIRST_TOKEN,
    Stage2OrdinaryBatch,
)


ADD_OP = 0
SUB_OP = 1
ADD_FIRST = 0
SUB_FIRST = 1
N3_SALT = "DH-S2-R5.1|821501|shared-n3"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class LadderModelInput:
    values: torch.Tensor
    operators: torch.Tensor
    query_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "LadderModelInput":
        return LadderModelInput(
            values=self.values.to(device),
            operators=self.operators.to(device),
            query_ids=self.query_ids.to(device),
        )


@dataclass(frozen=True)
class LadderTargets:
    final_labels: torch.Tensor
    intermediate_labels: torch.Tensor

    def to(self, device: torch.device | str) -> "LadderTargets":
        return LadderTargets(
            final_labels=self.final_labels.to(device),
            intermediate_labels=self.intermediate_labels.to(device),
        )


@dataclass(frozen=True)
class LadderGeneratedSplit:
    model_input: LadderModelInput
    targets: LadderTargets
    family_hashes: tuple[str, ...]
    row_hashes: tuple[str, ...]
    rung: str
    split: str

    def to(self, device: torch.device | str) -> "LadderGeneratedSplit":
        return LadderGeneratedSplit(
            model_input=self.model_input.to(device),
            targets=self.targets.to(device),
            family_hashes=self.family_hashes,
            row_hashes=self.row_hashes,
            rung=self.rung,
            split=self.split,
        )


@dataclass(frozen=True)
class _Family:
    values: tuple[int, ...]
    operators: tuple[int, ...]
    family_hash: str
    labels: tuple[int, ...]
    intermediate_labels: tuple[tuple[int, ...], ...]


def evaluate_arithmetic(
    values: tuple[int, ...], operators: tuple[int, ...], query_id: int
) -> tuple[int, tuple[int, ...]]:
    if len(values) not in {2, 3} or len(operators) != len(values) - 1:
        raise ValueError("R5.1 supports only two- or three-literal expressions")
    if set(values) - set(range(7)) or set(operators) - {ADD_OP, SUB_OP}:
        raise ValueError("R5.1 values or operators are outside the legal domain")
    if query_id not in {ADD_FIRST, SUB_FIRST}:
        raise ValueError("R5.1 query_id is invalid")
    if len(values) == 2:
        result = values[0] + values[1] if operators[0] == ADD_OP else values[0] - values[1]
        return result % 7, ()
    if operators != (SUB_OP, ADD_OP):
        raise ValueError("R5.1 three-literal expressions require the '-+' pattern")
    if query_id == ADD_FIRST:
        intermediate = (values[1] + values[2]) % 7
        return (values[0] - intermediate) % 7, (intermediate,)
    intermediate = (values[0] - values[1]) % 7
    return (intermediate + values[2]) % 7, (intermediate,)


def _family(values: tuple[int, ...], operators: tuple[int, ...], paired: bool) -> _Family:
    queries = (ADD_FIRST, SUB_FIRST) if paired else (ADD_FIRST,)
    outcomes = tuple(evaluate_arithmetic(values, operators, query) for query in queries)
    return _Family(
        values=values,
        operators=operators,
        family_hash=_digest({"values": values, "operators": operators}),
        labels=tuple(item[0] for item in outcomes),
        intermediate_labels=tuple(item[1] for item in outcomes),
    )


def _rank(family_hash: str) -> str:
    return _digest({"family_hash": family_hash, "salt": N3_SALT})


def _sham_derangement(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 1 or labels.numel() < 2:
        raise ValueError("R5.1 sham labels require a nontrivial vector")
    counts = torch.bincount(labels, minlength=7)
    maximum = int(counts.max().item())
    if maximum * 2 > labels.numel():
        raise RuntimeError("R5.1 sham target histogram cannot be deranged")
    order = torch.argsort(labels, stable=True)
    sorted_labels = labels[order]
    rotated = torch.roll(sorted_labels, shifts=-maximum)
    candidate = torch.empty_like(labels)
    candidate[order] = rotated
    if not bool(torch.all(candidate != labels).detach().cpu().item()):
        raise RuntimeError("R5.1 sham derangement construction failed")
    return candidate


def sham_intermediate_labels(
    labels: torch.Tensor,
    query_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a stable derangement, preserving each query's label histogram."""

    labels_cpu = labels.detach().cpu()
    if query_ids is None:
        return _sham_derangement(labels_cpu).to(labels.device)
    queries_cpu = query_ids.detach().cpu()
    if queries_cpu.shape != labels_cpu.shape:
        raise ValueError("R5.1 sham query IDs must match the label vector")
    candidate = torch.empty_like(labels_cpu)
    for query in torch.unique(queries_cpu, sorted=True).tolist():
        mask = queries_cpu == int(query)
        candidate[mask] = _sham_derangement(labels_cpu[mask])
    if not bool(torch.all(candidate != labels_cpu).item()):
        raise RuntimeError("R5.1 grouped sham derangement construction failed")
    return candidate.to(labels.device)


class ArithmeticLadderData:
    """Own one salted N3 partition and materialize reserve batches lazily."""

    def __init__(self, seed: int) -> None:
        if seed != 821501:
            raise ValueError("R5.1 partition salt is frozen to seed 821501")
        self.seed = seed
        self._memberships = self._build_memberships()
        self._cache: dict[tuple[str, str], LadderGeneratedSplit] = {}
        self.partition_digest = _digest(
            {
                rung: {
                    split: [family.family_hash for family in families]
                    for split, families in splits.items()
                }
                for rung, splits in self._memberships.items()
            }
        )
        self._validate_memberships()

    def _build_memberships(self) -> dict[str, dict[str, tuple[_Family, ...]]]:
        binary = tuple(
            _family((left, right), (operator,), paired=False)
            for operator in (ADD_OP, SUB_OP)
            for left in range(7)
            for right in range(7)
        )
        n3_by_labels = {
            (left, right): []
            for left in range(7)
            for right in range(7)
            if left != right
        }
        for left in range(7):
            for middle in range(7):
                for right in range(7):
                    family = _family(
                        (left, middle, right), (SUB_OP, ADD_OP), paired=True
                    )
                    pair = (family.labels[0], family.labels[1])
                    if pair[0] != pair[1]:
                        n3_by_labels[pair].append(family)
        n3_splits = {split: [] for split in ("train", "validation", "reserve")}
        for label_pair in sorted(n3_by_labels):
            ordered = sorted(n3_by_labels[label_pair], key=lambda family: _rank(family.family_hash))
            n3_splits["train"].extend(ordered[:5])
            n3_splits["validation"].extend(ordered[5:6])
            n3_splits["reserve"].extend(ordered[6:7])
        shared = {split: tuple(families) for split, families in n3_splits.items()}
        return {
            "binary": {"train": binary, "fit": binary},
            "shared-n3": shared,
        }

    def _validate_memberships(self) -> None:
        binary = self._memberships["binary"]["train"]
        if len(binary) != 98 or len({item.family_hash for item in binary}) != 98:
            raise RuntimeError("R5.1 binary domain is not exactly 98 unique facts")
        expected = {"train": 210, "validation": 42, "reserve": 42}
        split_sets: dict[str, set[str]] = {}
        for split, count in expected.items():
            families = self._memberships["shared-n3"][split]
            hashes = {item.family_hash for item in families}
            if len(families) != count or len(hashes) != count:
                raise RuntimeError(f"R5.1 shared-n3/{split} count or uniqueness failed")
            pair_counts: dict[tuple[int, int], int] = {}
            for family in families:
                pair = (family.labels[0], family.labels[1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            target = 5 if split == "train" else 1
            if len(pair_counts) != 42 or set(pair_counts.values()) != {target}:
                raise RuntimeError(f"R5.1 shared-n3/{split} label pairs are unbalanced")
            split_sets[split] = hashes
        if any(
            split_sets[left] & split_sets[right]
            for left, right in (("train", "validation"), ("train", "reserve"), ("validation", "reserve"))
        ):
            raise RuntimeError("R5.1 shared N3 partition overlaps")

    @staticmethod
    def _membership_rung(rung: str) -> str:
        if rung == "binary":
            return "binary"
        if rung in {"fixed-add", "fixed-sub", "paired"}:
            return "shared-n3"
        raise ValueError(f"unsupported R5.1 rung: {rung}")

    def family_hashes(self, rung: str, split: str) -> tuple[str, ...]:
        membership = self._membership_rung(rung)
        return tuple(item.family_hash for item in self._memberships[membership][split])

    def is_materialized(self, rung: str, split: str) -> bool:
        return (rung, split) in self._cache

    def batch(self, rung: str, split: str) -> LadderGeneratedSplit:
        key = (rung, split)
        if key in self._cache:
            return self._cache[key]
        membership = self._membership_rung(rung)
        if split not in self._memberships[membership]:
            raise ValueError(f"unsupported R5.1 split: {rung}/{split}")
        families = self._memberships[membership][split]
        rows: list[_Family] = []
        queries: list[int] = []
        if rung == "paired":
            query_groups = (ADD_FIRST, SUB_FIRST)
        elif rung == "fixed-sub":
            query_groups = (SUB_FIRST,)
        else:
            query_groups = (ADD_FIRST,)
        for query in query_groups:
            for family in families:
                rows.append(family)
                queries.append(query)
        values = torch.tensor([family.values for family in rows], dtype=torch.long)
        operators = torch.tensor([family.operators for family in rows], dtype=torch.long)
        query_ids = torch.tensor(queries, dtype=torch.long)
        final_labels = torch.tensor(
            [
                family.labels[query] if membership == "shared-n3" else family.labels[0]
                for family, query in zip(rows, queries, strict=True)
            ],
            dtype=torch.long,
        )
        if values.shape[1] == 3:
            intermediate_labels = torch.tensor(
                [family.intermediate_labels[query] for family, query in zip(rows, queries, strict=True)],
                dtype=torch.long,
            )
        else:
            intermediate_labels = torch.empty((len(rows), 0), dtype=torch.long)
        family_hashes = tuple(family.family_hash for family in rows)
        row_hashes = tuple(
            _digest({"family_hash": family.family_hash, "query_id": query})
            for family, query in zip(rows, queries, strict=True)
        )
        generated = LadderGeneratedSplit(
            model_input=LadderModelInput(values, operators, query_ids),
            targets=LadderTargets(final_labels, intermediate_labels),
            family_hashes=family_hashes,
            row_hashes=row_hashes,
            rung=rung,
            split=split,
        )
        self._validate_batch(generated)
        self._cache[key] = generated
        return generated

    @staticmethod
    def _validate_batch(batch: LadderGeneratedSplit) -> None:
        rows = batch.model_input.values.shape[0]
        if batch.targets.final_labels.shape != (rows,) or len(batch.row_hashes) != rows:
            raise RuntimeError("R5.1 batch row dimensions are inconsistent")
        solved = []
        intermediates = []
        for values, operators, query in zip(
            batch.model_input.values.tolist(),
            batch.model_input.operators.tolist(),
            batch.model_input.query_ids.tolist(),
            strict=True,
        ):
            answer, intermediate = evaluate_arithmetic(tuple(values), tuple(operators), int(query))
            solved.append(answer)
            intermediates.append(intermediate)
        if solved != batch.targets.final_labels.tolist():
            raise RuntimeError("R5.1 exact solver disagrees with final labels")
        if batch.targets.intermediate_labels.numel() and [list(item) for item in intermediates] != batch.targets.intermediate_labels.tolist():
            raise RuntimeError("R5.1 exact solver disagrees with intermediate labels")

    def partition_evidence(self) -> dict[str, object]:
        result: dict[str, object] = {"partition_digest": self.partition_digest, "rungs": {}}
        for rung, splits in self._memberships.items():
            result["rungs"][rung] = {
                split: {
                    "families": len(families),
                    "family_hash_digest": _digest(sorted(item.family_hash for item in families)),
                }
                for split, families in splits.items()
            }
        result["within_shared_n3_overlap"] = 0
        result["split_salt"] = N3_SALT
        result["canonical_order"] = "ordered label pair, then salted SHA256; all ADD rows then all SUB rows"
        return result


def _majority_label(labels: list[int]) -> int:
    counts = torch.bincount(torch.tensor(labels), minlength=7).tolist()
    return int(max(range(7), key=lambda index: (counts[index], -index)))


def two_literal_lookup_accuracies(
    train: LadderGeneratedSplit, evaluation: LadderGeneratedSplit
) -> dict[str, float]:
    if train.model_input.values.shape[1] != 3 or evaluation.model_input.values.shape[1] != 3:
        raise ValueError("R5.1 two-literal canaries require N3 batches")
    result: dict[str, float] = {}
    for positions in ((0, 1), (0, 2), (1, 2)):
        grouped: dict[tuple[int, int, int], list[int]] = {}
        by_query: dict[int, list[int]] = {}
        for values, query, label in zip(
            train.model_input.values.tolist(),
            train.model_input.query_ids.tolist(),
            train.targets.final_labels.tolist(),
            strict=True,
        ):
            key = (values[positions[0]], values[positions[1]], int(query))
            grouped.setdefault(key, []).append(int(label))
            by_query.setdefault(int(query), []).append(int(label))
        lookup = {key: _majority_label(labels) for key, labels in grouped.items()}
        fallback = {query: _majority_label(labels) for query, labels in by_query.items()}
        correct = 0
        for values, query, label in zip(
            evaluation.model_input.values.tolist(),
            evaluation.model_input.query_ids.tolist(),
            evaluation.targets.final_labels.tolist(),
            strict=True,
        ):
            key = (values[positions[0]], values[positions[1]], int(query))
            prediction = lookup.get(key, fallback[int(query)])
            correct += int(prediction == int(label))
        result[f"{positions[0]}-{positions[1]}"] = correct / len(evaluation.targets.final_labels)
    return result


def batch_evidence(batch: LadderGeneratedSplit) -> dict[str, object]:
    labels = batch.targets.final_labels
    label_counts = torch.bincount(labels.cpu(), minlength=7).tolist()
    query_counts = torch.bincount(batch.model_input.query_ids.cpu(), minlength=2).tolist()
    by_query: dict[int, list[int]] = {}
    by_family: dict[str, list[int]] = {}
    for family_hash, query, label in zip(
        batch.family_hashes,
        batch.model_input.query_ids.tolist(),
        labels.tolist(),
        strict=True,
    ):
        by_query.setdefault(int(query), []).append(int(label))
        by_family.setdefault(family_hash, []).append(int(label))
    query_correct = sum(max(torch.bincount(torch.tensor(items), minlength=7).tolist()) for items in by_query.values())
    input_correct = sum(max(torch.bincount(torch.tensor(items), minlength=7).tolist()) for items in by_family.values())
    intermediate_counts = (
        torch.bincount(batch.targets.intermediate_labels.reshape(-1).cpu(), minlength=7).tolist()
        if batch.targets.intermediate_labels.numel()
        else [0] * 7
    )
    return {
        "rung": batch.rung,
        "split": batch.split,
        "rows": int(labels.shape[0]),
        "unique_families": len(set(batch.family_hashes)),
        "label_counts": label_counts,
        "query_counts": query_counts,
        "intermediate_label_counts": intermediate_counts,
        "family_hash_digest": _digest(sorted(set(batch.family_hashes))),
        "row_hash_digest": _digest(sorted(batch.row_hashes)),
        "exact_solver_accuracy": 1.0,
        "query_only_lookup_accuracy": query_correct / len(labels),
        "input_only_lookup_accuracy": input_correct / len(labels),
    }


def to_stage2_oracle_batch(
    batch: LadderGeneratedSplit,
    *,
    merge_query_ids: torch.Tensor | None = None,
) -> tuple[Stage2OrdinaryBatch, StructureOnlyBatch]:
    """Serialize R5 input into the existing B-oracle execution interface."""

    values = batch.model_input.values.cpu()
    operators = batch.model_input.operators.cpu()
    answer_queries = batch.model_input.query_ids.cpu()
    structure_queries = answer_queries if merge_query_ids is None else merge_query_ids.cpu()
    if structure_queries.shape != answer_queries.shape:
        raise ValueError("R5.1 bridge structure-query shape mismatch")
    rows: list[list[int]] = []
    literal_sources: list[list[int]] = []
    operator_sources: list[list[int]] = []
    structures: list[StructureSample] = []
    for row_values, row_operators, answer_query, structure_query in zip(
        values.tolist(), operators.tolist(), answer_queries.tolist(), structure_queries.tolist(), strict=True
    ):
        tokens = [BOS]
        for index, value in enumerate(row_values):
            tokens.append(8 + int(value))
            if index < len(row_operators):
                tokens.append(ADD if row_operators[index] == ADD_OP else SUB)
        tokens.append(QUERY_ADD_FIRST_TOKEN if answer_query == ADD_FIRST else QUERY_SUB_FIRST_TOKEN)
        rows.append(tokens)
        literals = [1 + 2 * index for index in range(len(row_values))]
        op_sources = [2 + 2 * index for index in range(len(row_operators))]
        literal_sources.append(literals)
        operator_sources.append(op_sources)
        references: list[LeafSourceReference | MergeSourceReference] = [
            LeafSourceReference(index, source) for index, source in enumerate(literals)
        ]
        if len(row_values) == 2:
            references.append(MergeSourceReference(2, 0, 1, op_sources[0]))
            root_id = 2
        elif structure_query == ADD_FIRST:
            references.append(MergeSourceReference(3, 1, 2, op_sources[1]))
            references.append(MergeSourceReference(4, 0, 3, op_sources[0]))
            root_id = 4
        else:
            references.append(MergeSourceReference(3, 0, 1, op_sources[0]))
            references.append(MergeSourceReference(4, 3, 2, op_sources[1]))
            root_id = 4
        structures.append(StructureSample(root_id, tuple(references)))
    token_ids = torch.tensor(rows, dtype=torch.long)
    attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
    segment_ids = torch.zeros_like(token_ids)
    ordinary = Stage2OrdinaryBatch(
        token_ids=token_ids,
        position_features=SyntheticTaskGenerator._position_features(attention_mask, segment_ids),
        attention_mask=attention_mask,
        labels=batch.targets.final_labels.cpu(),
        query_ids=answer_queries,
        literal_source_indices=torch.tensor(literal_sources, dtype=torch.long),
        operator_source_indices=torch.tensor(operator_sources, dtype=torch.long),
        base_family_hashes=batch.family_hashes,
        query_row_hashes=batch.row_hashes,
        profile_name=f"r5-bridge-{batch.rung}-{batch.split}",
    )
    return ordinary, StructureOnlyBatch(tuple(structures))
