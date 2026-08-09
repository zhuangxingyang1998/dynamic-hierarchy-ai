"""Read-only hidden-state interventions for the completed Stage 2 R5.1 run."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .stage2_ladder_config import Stage2LadderConfig, load_stage2_ladder_config
from .stage2_ladder_data import (
    ADD_FIRST,
    ADD_OP,
    SUB_FIRST,
    SUB_OP,
    ArithmeticLadderData,
    LadderGeneratedSplit,
    sham_intermediate_labels,
)
from .stage2_ladder_model import ArithmeticComposerModel


PACKET_ID = "DH-S2-R5D-R1"
CANONICAL_CHECKPOINT_SHA256 = (
    "18327E373F937D353297811DB60C7180B9B3823FE49B4E7CDB09EE27D6EFD489"
)
CANONICAL_PARTITION_DIGEST = (
    "1701144f08fe7b7ee72b30b210c4922a14a3a4da69694ebb092db0c2cbace2d1"
)
CANONICAL_ARTIFACT_HASHES = {
    "frozen_config": "4B64023623B3DE1AC23D06E718ADA1C9BB639085CF95688A4EAC1FED03D5DCA7",
    "result": "3A56C05A3A8566A3E1E0AFEE1628329B7DCC83DD0C37EC336D67053F103C3B1B",
    "ledger": "8847E7BD0F1098A930B9C8DD725D2AA0DCBD197B91E1FED90286132AE7961ACD",
    "checkpoint": CANONICAL_CHECKPOINT_SHA256,
}
FIXED_BRANCHES = tuple(
    f"fixed-{query}-{mode}"
    for query in ("add", "sub")
    for mode in ("root", "teacher", "aux-true", "aux-sham")
)
EXPECTED_MODEL_NAMES = ("binary-root", *FIXED_BRANCHES)


@dataclass(frozen=True)
class FixedReplay:
    first_state: torch.Tensor
    intermediate_logits: torch.Tensor
    root_logits: torch.Tensor


@dataclass(frozen=True)
class CanonicalInputs:
    config: Stage2LadderConfig
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    result: dict[str, Any]
    ledger: dict[str, Any]
    artifact_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"diagnostic JSON root must be an object: {path}")
    return raw


def _latest_checkpoint(run_dir: Path) -> Path:
    receipt = _json_file(run_dir / "checkpoints" / "latest.json")
    checkpoint = Path(str(receipt["checkpoint"]))
    if not checkpoint.is_file():
        candidate = run_dir / "checkpoints" / checkpoint.name
        if not candidate.is_file():
            raise FileNotFoundError(f"canonical R5.1 checkpoint is missing: {checkpoint}")
        checkpoint = candidate
    return checkpoint


def canonical_artifact_hashes(run_dir: Path) -> tuple[dict[str, str], Path]:
    checkpoint = _latest_checkpoint(run_dir)
    paths = {
        "frozen_config": run_dir / "frozen-config.json",
        "result": run_dir / "result.json",
        "ledger": run_dir / "r5-evaluation-ledger.json",
        "checkpoint": checkpoint,
    }
    return {name: sha256_file(path) for name, path in paths.items()}, checkpoint


def load_canonical_inputs(run_dir: Path) -> CanonicalInputs:
    hashes, checkpoint_path = canonical_artifact_hashes(run_dir)
    if hashes != CANONICAL_ARTIFACT_HASHES:
        raise RuntimeError(
            f"canonical R5.1 artifact hashes changed: observed={hashes}"
        )
    config = load_stage2_ladder_config(run_dir / "frozen-config.json")
    if config.run_kind != "calibration_only" or config.seed != 821501:
        raise RuntimeError("state diagnostic requires the frozen R5.1 calibration")
    result = _json_file(run_dir / "result.json")
    ledger = _json_file(run_dir / "r5-evaluation-ledger.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("packet") != "DH-S2-R5.1":
        raise RuntimeError("state diagnostic checkpoint schema or packet mismatch")
    if checkpoint.get("config") != config.to_dict():
        raise RuntimeError("state diagnostic checkpoint config mismatch")
    if checkpoint.get("partition_digest") != CANONICAL_PARTITION_DIGEST:
        raise RuntimeError("state diagnostic partition digest mismatch")
    if checkpoint.get("global_round") != 600:
        raise RuntimeError("state diagnostic requires the round-600 checkpoint")
    if checkpoint.get("final_disposition") != "fixed_query_failed":
        raise RuntimeError("state diagnostic requires fixed_query_failed checkpoint")
    if result.get("disposition") != "fixed_query_failed" or result.get("global_round") != 600:
        raise RuntimeError("canonical R5.1 result receipt mismatch")
    if result.get("evaluation_ledger") != ledger:
        raise RuntimeError("canonical result and evaluation ledger disagree")
    if checkpoint.get("ledger_snapshot") != ledger:
        raise RuntimeError("canonical checkpoint and evaluation ledger disagree")
    names = tuple(str(name) for name in checkpoint.get("model_names", ()))
    if set(names) != set(EXPECTED_MODEL_NAMES) or any(
        name.startswith("paired-") for name in names
    ):
        raise RuntimeError("state diagnostic checkpoint model set is invalid")
    return CanonicalInputs(
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        result=result,
        ledger=ledger,
        artifact_hashes=hashes,
    )


@torch.inference_mode()
def replay_fixed(
    model: ArithmeticComposerModel,
    batch: LadderGeneratedSplit,
    *,
    intermediate_state: torch.Tensor | None = None,
) -> FixedReplay:
    if batch.split != "validation" or batch.rung not in {"fixed-add", "fixed-sub"}:
        raise ValueError("state replay accepts fixed-query validation only")
    values = batch.model_input.values
    operators = batch.model_input.operators
    queries = batch.model_input.query_ids
    if values.shape != (42, 3) or operators.shape != (42, 2) or queries.shape != (42,):
        raise ValueError("state replay requires exactly 42 three-literal rows")
    query = int(queries[0].item())
    if bool(torch.any(queries != query).item()) or query not in {ADD_FIRST, SUB_FIRST}:
        raise ValueError("state replay requires one constant legal query")
    if bool(torch.any(operators[:, 0] != SUB_OP).item()) or bool(
        torch.any(operators[:, 1] != ADD_OP).item()
    ):
        raise ValueError("state replay requires the '-+' operator pattern")

    literal_states = model.literal_embedding(values)
    operator_states = model.operator_embedding(operators)
    query_states = model.query_embedding(queries)
    if query == ADD_FIRST:
        first = model._compose(
            literal_states[:, 1], literal_states[:, 2], operator_states[:, 1]
        )
        consumed = first if intermediate_state is None else intermediate_state
        root = model._compose(
            literal_states[:, 0], consumed, operator_states[:, 0]
        )
    else:
        first = model._compose(
            literal_states[:, 0], literal_states[:, 1], operator_states[:, 0]
        )
        consumed = first if intermediate_state is None else intermediate_state
        root = model._compose(
            consumed, literal_states[:, 2], operator_states[:, 1]
        )
    if consumed.shape != first.shape:
        raise ValueError("intermediate state override shape mismatch")
    return FixedReplay(
        first_state=first,
        intermediate_logits=model._logits(first, query_states),
        root_logits=model._logits(root, query_states),
    )


def same_label_permutation(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.detach().cpu()
    if labels.ndim != 1:
        raise ValueError("same-label transplant labels must be one-dimensional")
    permutation = torch.empty_like(labels)
    for label in torch.unique(labels, sorted=True).tolist():
        indices = torch.nonzero(labels == int(label), as_tuple=False).flatten()
        if indices.numel() < 2:
            raise RuntimeError("same-label transplant requires at least two rows per label")
        permutation[indices] = torch.roll(indices, shifts=-1)
    expected = torch.arange(labels.numel())
    if not torch.equal(torch.sort(permutation).values, expected):
        raise RuntimeError("same-label transplant is not bijective")
    if bool(torch.any(permutation == expected).item()) or not torch.equal(
        labels[permutation], labels
    ):
        raise RuntimeError("same-label transplant invariant failed")
    return permutation


def sham_state_permutation(
    labels: torch.Tensor, query_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.detach().cpu()
    queries = query_ids.detach().cpu()
    sham_labels = sham_intermediate_labels(labels, queries).cpu()
    permutation = torch.empty_like(labels)
    for label in torch.unique(labels, sorted=True).tolist():
        targets = torch.nonzero(sham_labels == int(label), as_tuple=False).flatten()
        sources = torch.nonzero(labels == int(label), as_tuple=False).flatten()
        if targets.numel() != sources.numel():
            raise RuntimeError("sham transplant label histogram changed")
        permutation[targets] = sources
    expected = torch.arange(labels.numel())
    if not torch.equal(torch.sort(permutation).values, expected):
        raise RuntimeError("sham state transplant is not bijective")
    if bool(torch.any(sham_labels == labels).item()) or not torch.equal(
        labels[permutation], sham_labels
    ):
        raise RuntimeError("sham state transplant semantics are inconsistent")
    return sham_labels, permutation


def counterfactual_labels(
    batch: LadderGeneratedSplit, intermediate_labels: torch.Tensor
) -> torch.Tensor:
    values = batch.model_input.values.detach().cpu()
    queries = batch.model_input.query_ids.detach().cpu()
    intermediate = intermediate_labels.detach().cpu()
    if intermediate.shape != queries.shape:
        raise ValueError("counterfactual intermediate labels have the wrong shape")
    add = (values[:, 0] - intermediate) % 7
    sub = (intermediate + values[:, 2]) % 7
    return torch.where(queries == ADD_FIRST, add, sub).long()


def _answer_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    counterfactual: torch.Tensor | None = None,
) -> dict[str, Any]:
    predictions = logits.argmax(dim=-1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    result: dict[str, Any] = {
        "accuracy": float((predictions == labels_cpu).float().mean().item()),
        "cross_entropy": float(F.cross_entropy(logits, labels).detach().cpu().item()),
        "predictions": predictions.tolist(),
        "prediction_counts": torch.bincount(predictions, minlength=7).tolist(),
    }
    if counterfactual is not None:
        result["counterfactual_accuracy"] = float(
            (predictions == counterfactual.detach().cpu()).float().mean().item()
        )
        result["counterfactual_labels"] = counterfactual.detach().cpu().tolist()
    return result


def _transition_metrics(
    reference_predictions: list[int],
    candidate_predictions: list[int],
    labels: torch.Tensor,
) -> dict[str, int]:
    labels_list = labels.detach().cpu().tolist()
    changed = repaired = damaged = stable_correct = stable_wrong = 0
    for reference, candidate, label in zip(
        reference_predictions, candidate_predictions, labels_list, strict=True
    ):
        reference_correct = reference == label
        candidate_correct = candidate == label
        changed += int(reference != candidate)
        repaired += int(not reference_correct and candidate_correct)
        damaged += int(reference_correct and not candidate_correct)
        stable_correct += int(reference_correct and candidate_correct)
        stable_wrong += int(not reference_correct and not candidate_correct)
    return {
        "prediction_changed_rows": changed,
        "wrong_to_correct_rows": repaired,
        "correct_to_wrong_rows": damaged,
        "stable_correct_rows": stable_correct,
        "stable_wrong_rows": stable_wrong,
    }


def _state_geometry(
    model: ArithmeticComposerModel,
    first_state: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    embeddings = model.literal_embedding.weight
    normalized_states = F.normalize(first_state, dim=-1)
    normalized_embeddings = F.normalize(embeddings, dim=-1)
    cosine = normalized_states @ normalized_embeddings.transpose(0, 1)
    euclidean = torch.cdist(first_state, embeddings)
    cosine_labels = cosine.argmax(dim=-1)
    euclidean_labels = euclidean.argmin(dim=-1)
    row_indices = torch.arange(labels.shape[0])
    true_cosine = cosine[row_indices, labels]
    wrong_cosine = cosine.clone()
    wrong_cosine[row_indices, labels] = -torch.inf
    margin = true_cosine - wrong_cosine.max(dim=-1).values
    true_distance = euclidean[row_indices, labels]
    geometry = {
        "intermediate_label_counts": torch.bincount(labels.cpu(), minlength=7).tolist(),
        "cosine_nearest_accuracy": float(
            (cosine_labels == labels).float().mean().detach().cpu().item()
        ),
        "euclidean_nearest_accuracy": float(
            (euclidean_labels == labels).float().mean().detach().cpu().item()
        ),
        "mean_true_cosine_similarity": float(true_cosine.mean().detach().cpu().item()),
        "mean_true_euclidean_distance": float(true_distance.mean().detach().cpu().item()),
        "mean_true_vs_best_wrong_cosine_margin": float(
            margin.mean().detach().cpu().item()
        ),
    }
    return geometry, cosine_labels, euclidean_labels


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


@torch.inference_mode()
def analyze_branch(
    branch: str,
    model: ArithmeticComposerModel,
    batch: LadderGeneratedSplit,
    canonical_validation: dict[str, Any],
) -> dict[str, Any]:
    if branch not in FIXED_BRANCHES:
        raise ValueError(f"unsupported state diagnostic branch: {branch}")
    model.eval()
    labels = batch.targets.intermediate_labels[:, 0]
    answers = batch.targets.final_labels
    ordinary = model(batch.model_input)
    teacher = model(
        batch.model_input,
        teacher_intermediate_labels=labels,
    )
    learned = replay_fixed(model, batch)
    canonical_state = model.literal_embedding(labels)
    true_canonical = replay_fixed(
        model, batch, intermediate_state=canonical_state
    )
    geometry, cosine_labels, euclidean_labels = _state_geometry(
        model, learned.first_state, labels
    )
    readout_labels = learned.intermediate_logits.argmax(dim=-1)
    readout_canonical = replay_fixed(
        model,
        batch,
        intermediate_state=model.literal_embedding(readout_labels),
    )
    nearest_canonical = replay_fixed(
        model,
        batch,
        intermediate_state=model.literal_embedding(cosine_labels),
    )
    same_indices = same_label_permutation(labels)
    same_label = replay_fixed(
        model,
        batch,
        intermediate_state=learned.first_state[same_indices],
    )
    sham_labels, sham_indices = sham_state_permutation(
        labels, batch.model_input.query_ids
    )
    sham_counterfactual = counterfactual_labels(batch, sham_labels)
    sham_state = replay_fixed(
        model,
        batch,
        intermediate_state=learned.first_state[sham_indices],
    )
    sham_canonical = replay_fixed(
        model,
        batch,
        intermediate_state=model.literal_embedding(sham_labels),
    )
    paths = {
        "learned": _answer_metrics(learned.root_logits, answers),
        "true_canonical": _answer_metrics(true_canonical.root_logits, answers),
        "readout_canonical": _answer_metrics(readout_canonical.root_logits, answers),
        "nearest_canonical": _answer_metrics(nearest_canonical.root_logits, answers),
        "same_label_transplant": _answer_metrics(same_label.root_logits, answers),
        "sham_state_transplant": _answer_metrics(
            sham_state.root_logits,
            answers,
            counterfactual=sham_counterfactual,
        ),
        "sham_canonical": _answer_metrics(
            sham_canonical.root_logits,
            answers,
            counterfactual=sham_counterfactual,
        ),
    }
    canonical_path = "true_canonical" if branch.endswith("teacher") else "learned"
    expected_accuracy = float(canonical_validation["accuracy"])
    expected_cross_entropy = float(canonical_validation["cross_entropy"])
    canonical_metrics = paths[canonical_path]
    expected_correct = round(expected_accuracy * len(answers))
    observed_correct = sum(
        prediction == answer
        for prediction, answer in zip(
            canonical_metrics["predictions"], answers.tolist(), strict=True
        )
    )
    ordinary_difference = float(
        (ordinary.root_logits - learned.root_logits).abs().max().item()
    )
    intermediate_difference = float(
        (ordinary.intermediate_logits[0] - learned.intermediate_logits).abs().max().item()
    )
    teacher_difference = float(
        (teacher.root_logits - true_canonical.root_logits).abs().max().item()
    )
    intermediate_accuracy = float(
        (readout_labels == labels).float().mean().item()
    )
    path_predictions = {
        name: metrics["predictions"] for name, metrics in paths.items()
    }
    transitions_from_learned = {
        name: _transition_metrics(
            path_predictions["learned"], predictions, answers
        )
        for name, predictions in path_predictions.items()
        if name != "learned"
    }
    canonical_predictions = path_predictions[canonical_path]
    learned_predictions = path_predictions["learned"]
    rows = []
    for index, (learned_prediction, canonical_prediction, answer) in enumerate(
        zip(learned_predictions, canonical_predictions, answers.tolist(), strict=True)
    ):
        if learned_prediction == answer and canonical_prediction == answer:
            continue
        rows.append(
            {
                "index": index,
                "family_hash": batch.family_hashes[index],
                "row_hash": batch.row_hashes[index],
                "values": batch.model_input.values[index].tolist(),
                "query_id": int(batch.model_input.query_ids[index].item()),
                "true_intermediate": int(labels[index].item()),
                "true_answer": int(answer),
                "readout_intermediate": int(readout_labels[index].item()),
                "cosine_nearest_intermediate": int(cosine_labels[index].item()),
                "euclidean_nearest_intermediate": int(euclidean_labels[index].item()),
                "sham_intermediate": int(sham_labels[index].item()),
                "predictions": {
                    name: int(predictions[index])
                    for name, predictions in path_predictions.items()
                },
            }
        )
    reproduction = {
        "canonical_runtime_path": canonical_path,
        "ordinary_replay_max_abs_difference": ordinary_difference,
        "intermediate_replay_max_abs_difference": intermediate_difference,
        "teacher_replay_max_abs_difference": teacher_difference,
        "ledger_accuracy": expected_accuracy,
        "observed_accuracy": canonical_metrics["accuracy"],
        "ledger_correct_rows": expected_correct,
        "observed_correct_rows": observed_correct,
        "ledger_cross_entropy": expected_cross_entropy,
        "observed_cross_entropy": canonical_metrics["cross_entropy"],
        "cross_entropy_abs_difference": abs(
            canonical_metrics["cross_entropy"] - expected_cross_entropy
        ),
    }
    invariants = {
        "rows_exact": len(answers) == 42,
        "ordinary_replay_matches": ordinary_difference <= 1e-6,
        "intermediate_replay_matches": intermediate_difference <= 1e-6,
        "teacher_replay_matches": teacher_difference <= 1e-6,
        "ledger_correct_rows_match": observed_correct == expected_correct,
        "ledger_cross_entropy_matches": reproduction[
            "cross_entropy_abs_difference"
        ]
        <= 1e-4,
        "same_label_bijection": len(set(same_indices.tolist())) == len(labels),
        "same_label_no_self_map": bool(
            torch.all(same_indices != torch.arange(len(labels))).item()
        ),
        "sham_bijection": len(set(sham_indices.tolist())) == len(labels),
        "sham_changes_every_label": bool(torch.all(sham_labels != labels).item()),
        "finite_metrics": _all_finite(
            {"geometry": geometry, "paths": paths, "reproduction": reproduction}
        ),
        "no_parameter_gradients": all(
            parameter.grad is None for parameter in model.parameters()
        ),
    }
    return {
        "branch": branch,
        "rung": batch.rung,
        "split": batch.split,
        "rows": len(answers),
        "intermediate_readout_accuracy": intermediate_accuracy,
        "geometry": geometry,
        "paths": paths,
        "transitions_from_learned": transitions_from_learned,
        "reproduction": reproduction,
        "transplants": {
            "same_label_source_indices": same_indices.tolist(),
            "sham_labels": sham_labels.tolist(),
            "sham_source_indices": sham_indices.tolist(),
        },
        "error_rows": rows,
        "invariants": invariants,
    }


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "src/dynamic_hierarchy/stage2_state_diagnostic.py",
        "src/dynamic_hierarchy/stage2_ladder_model.py",
        "src/dynamic_hierarchy/stage2_ladder_data.py",
        "scripts/run_stage2_state_diagnostic.py",
        "docs/stage2-r5-state-diagnostic-packet-r1.md",
    )
    return {path: sha256_file(root / path) for path in paths}


def run_canonical_state_diagnostic(run_dir: Path) -> dict[str, Any]:
    inputs = load_canonical_inputs(run_dir)
    data = ArithmeticLadderData(inputs.config.seed)
    if data.partition_digest != CANONICAL_PARTITION_DIGEST:
        raise RuntimeError("state diagnostic data reconstruction changed")
    reserve_before = {
        rung: data.is_materialized(rung, "reserve")
        for rung in ("fixed-add", "fixed-sub")
    }
    branches: dict[str, Any] = {}
    with torch.inference_mode():
        for branch in FIXED_BRANCHES:
            query = "fixed-add" if branch.startswith("fixed-add-") else "fixed-sub"
            batch = data.batch(query, "validation")
            model = ArithmeticComposerModel(inputs.config.model)
            model.load_state_dict(inputs.checkpoint["models"][branch])
            validation = inputs.ledger["rungs"]["fixed"]["branches"][branch][
                "validation"
            ]
            branches[branch] = analyze_branch(
                branch, model, batch, validation
            )
    reserve_after = {
        rung: data.is_materialized(rung, "reserve")
        for rung in ("fixed-add", "fixed-sub")
    }
    hashes_after, checkpoint_after = canonical_artifact_hashes(run_dir)
    branch_invariants = all(
        all(bool(value) for value in branch["invariants"].values())
        for branch in branches.values()
    )
    global_invariants = {
        "all_fixed_branches_present": set(branches) == set(FIXED_BRANCHES),
        "paired_models_absent": not any(
            name.startswith("paired-")
            for name in inputs.checkpoint["model_names"]
        ),
        "branch_invariants_passed": branch_invariants,
        "no_optimizer_constructed": True,
        "no_optimizer_updates": True,
        "reserve_unmaterialized_before": not any(reserve_before.values()),
        "reserve_unmaterialized_after": not any(reserve_after.values()),
        "artifact_hashes_unchanged": hashes_after == inputs.artifact_hashes,
        "checkpoint_path_unchanged": checkpoint_after.resolve()
        == inputs.checkpoint_path.resolve(),
    }
    status = (
        "diagnostic_complete"
        if all(bool(value) for value in global_invariants.values())
        else "implementation_invalid"
    )
    return {
        "schema_version": 1,
        "packet": PACKET_ID,
        "status": status,
        "posthoc_diagnostic": True,
        "input": {
            "canonical_run": str(run_dir),
            "checkpoint": str(inputs.checkpoint_path),
            "checkpoint_global_round": inputs.checkpoint["global_round"],
            "checkpoint_disposition": inputs.checkpoint["final_disposition"],
            "partition_digest": data.partition_digest,
            "artifact_hashes_before": inputs.artifact_hashes,
            "artifact_hashes_after": hashes_after,
        },
        "data_access": {
            "splits": ["validation"],
            "fixed_add_rows_per_branch": 42,
            "fixed_sub_rows_per_branch": 42,
            "reserve_materialized_before": reserve_before,
            "reserve_materialized_after": reserve_after,
        },
        "branches": branches,
        "global_invariants": global_invariants,
        "execution": {"optimizer_updates": 0, "backward_calls": 0},
        "source_manifest": _source_manifest(),
        "claim_boundary": {
            "amends_r5_result": False,
            "new_training": False,
            "reserve_evaluated": False,
            "learned_routing_tested": False,
            "causal_interventions_are_posthoc": True,
            "r6_training_authorized": False,
        },
    }
