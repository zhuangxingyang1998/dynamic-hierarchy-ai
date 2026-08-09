"""Exhaustive Stage 2 R6 families, split receipts, and partner maps."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import torch

from .stage2_congruence_config import R6_PACKET, R6_PARTITION_DIGEST, R6_SEED
from .stage2_ladder_data import (
    ADD_FIRST,
    ADD_OP,
    SUB_FIRST,
    SUB_OP,
    ArithmeticLadderData,
    LadderGeneratedSplit,
    LadderModelInput,
    LadderTargets,
)


R6_SPLIT_RULE = "(f+2*i_add+3*i_sub)%7"
R6_SPLIT_DIGESTS = {
    "train": "f019acf6bcad4e9d007cc1301b7ee3082d6d176ebbfde88863a269d2522addc1",
    "validation": "c07618d26bd3011701f02c8e9bcc23cdb8e0b7995870a3097c374922ef53d20e",
    "reserve": "7d75bf9c3f8601157669f7411d6d0048110a75a3ecff1f9e3c19760f8a9addfb",
}


def digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class CongruenceFamily:
    values: tuple[int, int, int]
    operators: tuple[int, int]
    family_hash: str
    final_label: int
    add_intermediate: int
    sub_intermediate: int
    split_code: int


def evaluate_congruence(
    values: tuple[int, int, int], query_id: int
) -> tuple[int, int]:
    if len(values) != 3 or set(values) - set(range(7)):
        raise ValueError("R6 requires three modulo-seven literals")
    a, b, c = values
    if query_id == ADD_FIRST:
        intermediate = (a + b) % 7
        return (intermediate - c) % 7, intermediate
    if query_id == SUB_FIRST:
        intermediate = (b - c) % 7
        return (a + intermediate) % 7, intermediate
    raise ValueError("R6 query_id is invalid")


def _family(values: tuple[int, int, int]) -> CongruenceFamily:
    add_final, add_intermediate = evaluate_congruence(values, ADD_FIRST)
    sub_final, sub_intermediate = evaluate_congruence(values, SUB_FIRST)
    if add_final != sub_final:
        raise RuntimeError("R6 fixed queries disagree on final arithmetic")
    split_code = (add_final + 2 * add_intermediate + 3 * sub_intermediate) % 7
    return CongruenceFamily(
        values=values,
        operators=(ADD_OP, SUB_OP),
        family_hash=digest({"values": values, "operators": (ADD_OP, SUB_OP)}),
        final_label=add_final,
        add_intermediate=add_intermediate,
        sub_intermediate=sub_intermediate,
        split_code=split_code,
    )


def _split_name(code: int) -> str:
    if code == 0:
        return "validation"
    if code == 1:
        return "reserve"
    return "train"


class StateCongruenceData:
    """Own the complete R6 domain and lazily materialize query batches."""

    def __init__(self, seed: int = R6_SEED) -> None:
        if seed != R6_SEED:
            raise ValueError("R6 data seed is frozen to 821601")
        self.seed = seed
        memberships: dict[str, list[CongruenceFamily]] = {
            "train": [],
            "validation": [],
            "reserve": [],
        }
        for a in range(7):
            for b in range(7):
                for c in range(7):
                    family = _family((a, b, c))
                    memberships[_split_name(family.split_code)].append(family)
        self._memberships = {
            split: tuple(sorted(rows, key=lambda item: item.family_hash))
            for split, rows in memberships.items()
        }
        self._cache: dict[tuple[int, str], LadderGeneratedSplit] = {}
        partition_payload = {
            "packet": R6_PACKET,
            "split_rule": R6_SPLIT_RULE,
            "splits": {
                split: sorted(item.family_hash for item in rows)
                for split, rows in self._memberships.items()
            },
        }
        self.partition_digest = digest(partition_payload)
        self._validate()

    def _validate(self) -> None:
        if self.partition_digest != R6_PARTITION_DIGEST:
            raise RuntimeError("R6 partition digest changed")
        expected = {"train": 245, "validation": 49, "reserve": 49}
        joint: set[tuple[int, int, int]] = set()
        split_sets: dict[str, set[str]] = {}
        for split, count in expected.items():
            rows = self._memberships[split]
            hashes = {item.family_hash for item in rows}
            if len(rows) != count or len(hashes) != count:
                raise RuntimeError(f"R6 {split} count or uniqueness changed")
            expected_per_label = 35 if split == "train" else 7
            for field in ("final_label", "add_intermediate", "sub_intermediate"):
                counts = Counter(getattr(item, field) for item in rows)
                if counts != Counter({label: expected_per_label for label in range(7)}):
                    raise RuntimeError(f"R6 {split}/{field} is unbalanced")
            observed_digest = digest(sorted(hashes))
            if observed_digest != R6_SPLIT_DIGESTS[split]:
                raise RuntimeError(f"R6 {split} ID digest changed")
            split_sets[split] = hashes
            joint.update(
                (item.final_label, item.add_intermediate, item.sub_intermediate)
                for item in rows
            )
        if len(joint) != 343:
            raise RuntimeError("R6 arithmetic coordinate map is not bijective")
        if any(
            split_sets[left] & split_sets[right]
            for left, right in (
                ("train", "validation"),
                ("train", "reserve"),
                ("validation", "reserve"),
            )
        ):
            raise RuntimeError("R6 split identities overlap")
        r5 = ArithmeticLadderData(821501)
        r5_hashes = set()
        for split in ("train", "validation", "reserve"):
            r5_hashes.update(r5.family_hashes("fixed-add", split))
        if set().union(*split_sets.values()) & r5_hashes:
            raise RuntimeError("R6 base identities overlap R5")

    def is_materialized(self, query_id: int, split: str) -> bool:
        return (query_id, split) in self._cache

    def family_hashes(self, split: str) -> tuple[str, ...]:
        return tuple(item.family_hash for item in self._memberships[split])

    def batch(
        self, query_id: int, split: str, *, materialize: bool = True
    ) -> LadderGeneratedSplit:
        if query_id not in {ADD_FIRST, SUB_FIRST}:
            raise ValueError("R6 query_id is invalid")
        key = (query_id, split)
        if materialize and key in self._cache:
            return self._cache[key]
        if split not in self._memberships:
            raise ValueError(f"R6 split is invalid: {split}")
        families = self._memberships[split]
        values = torch.tensor([item.values for item in families], dtype=torch.long)
        operators = torch.tensor(
            [item.operators for item in families], dtype=torch.long
        )
        queries = torch.full((len(families),), query_id, dtype=torch.long)
        labels = torch.tensor(
            [item.final_label for item in families], dtype=torch.long
        )
        intermediates = torch.tensor(
            [
                (item.add_intermediate if query_id == ADD_FIRST else item.sub_intermediate,)
                for item in families
            ],
            dtype=torch.long,
        )
        hashes = tuple(item.family_hash for item in families)
        row_hashes = tuple(
            digest({"base_family_id": family_hash, "query_id": query_id})
            for family_hash in hashes
        )
        batch = LadderGeneratedSplit(
            model_input=LadderModelInput(values, operators, queries),
            targets=LadderTargets(labels, intermediates),
            family_hashes=hashes,
            row_hashes=row_hashes,
            rung="fixed-add" if query_id == ADD_FIRST else "fixed-sub",
            split=split,
        )
        self._validate_batch(batch)
        if materialize:
            self._cache[key] = batch
        return batch

    @staticmethod
    def _validate_batch(batch: LadderGeneratedSplit) -> None:
        solved = []
        intermediate = []
        for values, query in zip(
            batch.model_input.values.tolist(),
            batch.model_input.query_ids.tolist(),
            strict=True,
        ):
            answer, first = evaluate_congruence(tuple(values), int(query))
            solved.append(answer)
            intermediate.append([first])
        if solved != batch.targets.final_labels.tolist():
            raise RuntimeError("R6 exact solver disagrees with final labels")
        if intermediate != batch.targets.intermediate_labels.tolist():
            raise RuntimeError("R6 exact solver disagrees with intermediate labels")

    def partition_evidence(self) -> dict[str, object]:
        return {
            "partition_digest": self.partition_digest,
            "split_rule": R6_SPLIT_RULE,
            "splits": {
                split: {
                    "families": len(rows),
                    "family_hash_digest": digest(
                        sorted(item.family_hash for item in rows)
                    ),
                    "final_counts": _counts(item.final_label for item in rows),
                    "add_intermediate_counts": _counts(
                        item.add_intermediate for item in rows
                    ),
                    "sub_intermediate_counts": _counts(
                        item.sub_intermediate for item in rows
                    ),
                }
                for split, rows in self._memberships.items()
            },
            "joint_coordinate_count": 343,
            "r5_overlap": 0,
        }


def _counts(values) -> list[int]:
    result = Counter(int(value) for value in values)
    return [result[label] for label in range(7)]


def _majority(labels: list[int]) -> int:
    counts = Counter(labels)
    return min(range(7), key=lambda label: (-counts[label], label))


def shortcut_canaries(
    train: LadderGeneratedSplit, evaluation: LadderGeneratedSplit
) -> dict[str, object]:
    result: dict[str, object] = {"train_fitted": {}}
    for width in (0, 1, 2):
        views = ((),) if width == 0 else tuple(combinations(range(3), width))
        for positions in views:
            grouped: dict[tuple[int, ...], list[int]] = {}
            for values, query, label in zip(
                train.model_input.values.tolist(),
                train.model_input.query_ids.tolist(),
                train.targets.final_labels.tolist(),
                strict=True,
            ):
                key = tuple(values[index] for index in positions) + (int(query),)
                grouped.setdefault(key, []).append(int(label))
            lookup = {key: _majority(labels) for key, labels in grouped.items()}
            correct = 0
            for values, query, label in zip(
                evaluation.model_input.values.tolist(),
                evaluation.model_input.query_ids.tolist(),
                evaluation.targets.final_labels.tolist(),
                strict=True,
            ):
                key = tuple(values[index] for index in positions) + (int(query),)
                correct += int(lookup[key] == int(label))
            name = "query-only" if not positions else "-".join(map(str, positions))
            result["train_fitted"][name] = {
                "correct": correct,
                "rows": len(evaluation.targets.final_labels),
                "accuracy": correct / len(evaluation.targets.final_labels),
            }
    evaluation_only: dict[str, object] = {}
    for positions in combinations(range(3), 2):
        grouped: dict[tuple[int, ...], list[int]] = {}
        for values, query, label in zip(
            evaluation.model_input.values.tolist(),
            evaluation.model_input.query_ids.tolist(),
            evaluation.targets.final_labels.tolist(),
            strict=True,
        ):
            key = tuple(values[index] for index in positions) + (int(query),)
            grouped.setdefault(key, []).append(int(label))
        correct = sum(max(Counter(labels).values()) for labels in grouped.values())
        evaluation_only["-".join(map(str, positions))] = {
            "correct": correct,
            "rows": len(evaluation.targets.final_labels),
            "accuracy": correct / len(evaluation.targets.final_labels),
        }
    result["evaluation_only_leakage"] = evaluation_only
    result["exact_solver"] = {"correct": len(evaluation.targets.final_labels), "accuracy": 1.0}
    return result


def partner_source_indices(
    batch: LadderGeneratedSplit, mode: str, schedule_index: int
) -> torch.Tensor:
    if batch.split != "train" or len(batch.targets.final_labels) != 245:
        raise ValueError("R6 partner maps require the 245-row train batch")
    if not 1 <= schedule_index <= 34:
        raise ValueError("R6 schedule index must be in [1, 34]")
    labels = batch.targets.intermediate_labels[:, 0].cpu()
    groups: dict[int, list[int]] = {}
    for label in range(7):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten().tolist()
        indices.sort(key=lambda index: batch.family_hashes[index])
        if len(indices) != 35:
            raise RuntimeError("R6 train intermediate class is not size 35")
        groups[label] = indices
    source = torch.empty(245, dtype=torch.long)
    if mode == "self-duplicate":
        return torch.arange(245, dtype=torch.long)
    if mode == "congruence-true":
        for label, indices in groups.items():
            for rank, target in enumerate(indices):
                source[target] = indices[(rank + schedule_index) % 35]
    elif mode == "mixed-counterfactual":
        value_offset = 1 + ((schedule_index - 1) % 6)
        rank_offset = (schedule_index - 1) // 6
        for label, indices in groups.items():
            source_group = groups[(label + value_offset) % 7]
            for rank, target in enumerate(indices):
                source[target] = source_group[(rank + rank_offset) % 35]
    else:
        raise ValueError(f"R6 partner mode is invalid: {mode}")
    expected = torch.arange(245)
    if not torch.equal(torch.sort(source).values, expected):
        raise RuntimeError("R6 partner map is not bijective")
    if bool(torch.any(source == expected).item()):
        raise RuntimeError("R6 nonidentity partner map contains a self-map")
    source_labels = labels[source]
    if mode == "congruence-true" and not torch.equal(source_labels, labels):
        raise RuntimeError("R6 true partner changed an intermediate value")
    if mode == "mixed-counterfactual" and not bool(
        torch.all(source_labels != labels).item()
    ):
        raise RuntimeError("R6 mixed partner preserved an intermediate value")
    return source


def partner_map_receipt(
    batch: LadderGeneratedSplit, mode: str, schedule_index: int
) -> dict[str, object]:
    source = partner_source_indices(batch, mode, schedule_index)
    labels = batch.targets.intermediate_labels[:, 0].cpu()
    transitions = [[0] * 7 for _ in range(7)]
    for target_index, source_index in enumerate(source.tolist()):
        transitions[int(labels[target_index])][int(labels[source_index])] += 1

    visited = [False] * len(source)
    cycle_lengths: list[int] = []
    source_list = source.tolist()
    for start in range(len(source_list)):
        if visited[start]:
            continue
        current = start
        length = 0
        while not visited[current]:
            visited[current] = True
            length += 1
            current = source_list[current]
        cycle_lengths.append(length)
    cycle_lengths.sort()
    return {
        "query_id": int(batch.model_input.query_ids[0]),
        "mode": mode,
        "schedule_index": schedule_index,
        "map_digest": digest(
            {
                "query_id": int(batch.model_input.query_ids[0]),
                "mode": mode,
                "schedule_index": schedule_index,
                "source_indices": source_list,
            }
        ),
        "source_use_counts": [1] * len(source_list),
        "value_transitions": transitions,
        "cycle_count": len(cycle_lengths),
        "cycle_lengths": cycle_lengths,
    }


def counterfactual_labels(
    batch: LadderGeneratedSplit,
    target_indices: torch.Tensor,
    source_intermediate_labels: torch.Tensor,
) -> torch.Tensor:
    target_indices_cpu = target_indices.detach().cpu()
    source_values = source_intermediate_labels.detach().cpu()
    values = batch.model_input.values[target_indices_cpu]
    queries = batch.model_input.query_ids[target_indices_cpu]
    if source_values.shape != queries.shape:
        raise ValueError("R6 counterfactual source values have the wrong shape")
    add = (source_values - values[:, 2]) % 7
    sub = (values[:, 0] + source_values) % 7
    return torch.where(queries == ADD_FIRST, add, sub).long()


def index_model_input(
    model_input: LadderModelInput, indices: torch.Tensor
) -> LadderModelInput:
    return LadderModelInput(
        values=model_input.values[indices],
        operators=model_input.operators[indices],
        query_ids=model_input.query_ids[indices],
    )
