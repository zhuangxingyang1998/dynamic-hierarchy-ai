from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import psutil
import torch

from dynamic_hierarchy import stage1_campaign as stage1_campaign_module
from dynamic_hierarchy.data import (
    ADD,
    SUB,
    LeafSourceReference,
    MergeSourceReference,
    StructureOnlyBatch,
)
from dynamic_hierarchy.model import SmallTransformerBaseline, TrueStructureDiagnosticD
from dynamic_hierarchy.process_registry import register_worker_pid
from dynamic_hierarchy.provenance import source_manifest
from dynamic_hierarchy.resource_guard import ResourceGuard, ResourceSample
from dynamic_hierarchy.run_lock import PerRunMutex
from dynamic_hierarchy.snapshot import create_snapshot, snapshot_sources
from dynamic_hierarchy.stage1_campaign import (
    campaign_seed_freshness,
    create_campaign_package,
    materialize_campaign_run,
    verify_campaign_package,
)
from dynamic_hierarchy.stage1_config import (
    CurriculumStage,
    Stage1Config,
    TrainingProfile,
    load_stage1_config,
    stage1_config_digest,
    validated_experiment_compatibility_spec_digest,
    validated_experiment_spec_digest,
)
from dynamic_hierarchy.stage1_confirmation import (
    aggregate_confirmation,
    run_completion_checks,
    run_completion_evidence,
)
from dynamic_hierarchy.stage1_integrity import (
    formal_seed_freshness,
    verify_result_manifests,
)
from dynamic_hierarchy.stage1_data import (
    SHAM_MAPPING_VERSION,
    RevisedStage1Generator,
    shape_catalog,
    shape_height,
    shape_id,
    sham_structure,
)
from dynamic_hierarchy.stage1_runtime import Stage1Trainer
from scripts import run_stage1_confirmation_sequence as confirmation_sequence
from scripts import (
    run_stage1_confirmation_sequence_v2 as confirmation_sequence_v2,
)
from scripts import (
    run_stage1_confirmation_sequence_v3 as confirmation_sequence_v3,
)
from scripts import (
    run_stage1_confirmation_sequence_v4 as confirmation_sequence_v4,
)
from scripts.run_stage1_confirmation_sequence import (
    SequenceAbort,
    verify_formal_result,
)
from scripts.stage1_worker import (
    classify_run_outcome,
    finalize_training_attempt,
    validate_candidate_prerequisite,
)


ROOT = Path(__file__).resolve().parents[1]


def create_test_campaign(
    temporary_root: Path,
) -> tuple[Path, Stage1Config, dict[str, object], Path]:
    project_root = temporary_root / "project"
    for source in snapshot_sources(ROOT):
        relative = source.relative_to(ROOT)
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    config_path = (
        project_root
        / "configs"
        / "stage1-revised-literal-formal-confirmation-v2-directml.json"
    )
    config = load_stage1_config(config_path)
    candidate = ROOT / config.candidate_prerequisite_result_path
    verification = validate_candidate_prerequisite(config, candidate)
    campaign_root = temporary_root / "campaign"
    manifest = create_campaign_package(
        project_root,
        campaign_root,
        config_path,
        config,
        candidate,
        verification,
    )
    return campaign_root, config, manifest, project_root


def tiny_stage1_config(device: str = "cpu") -> Stage1Config:
    config = load_stage1_config(
        ROOT
        / "configs"
        / (
            "stage1-revised-smoke-directml.json"
            if device == "directml"
            else "stage1-revised-smoke-cpu.json"
        )
    )
    return config


def tiny_literal_config(device: str = "cpu") -> Stage1Config:
    base = load_stage1_config(
        ROOT / "configs" / "stage1-revised-literal-candidate-directml.json"
    )
    curriculum = (
        CurriculumStage("literal_c0", 1, (TrainingProfile("c0", 0, "leaf"),)),
        CurriculumStage(
            "literal_c1",
            1,
            (
                TrainingProfile("c0", 0, "leaf"),
                TrainingProfile("c1", 1, "skew"),
            ),
        ),
        CurriculumStage(
            "literal_depth_2",
            1,
            (
                TrainingProfile("c0", 0, "leaf"),
                TrainingProfile("c1", 1, "skew"),
                TrainingProfile("d2s", 2, "skew"),
                TrainingProfile("d2b", 2, "balanced"),
            ),
        ),
        CurriculumStage(
            "literal_depth_3",
            1,
            (
                TrainingProfile("c0", 0, "leaf"),
                TrainingProfile("c1", 1, "skew"),
                TrainingProfile("d3s", 3, "skew"),
                TrainingProfile("d3b", 3, "balanced"),
            ),
        ),
        CurriculumStage(
            "literal_rehearsal",
            1,
            (
                TrainingProfile("c0", 0, "leaf"),
                TrainingProfile("c1", 1, "skew"),
                TrainingProfile("d2s", 2, "skew"),
                TrainingProfile("d2b", 2, "balanced"),
                TrainingProfile("d3s", 3, "skew"),
                TrainingProfile("d3b", 3, "balanced"),
            ),
        ),
    )
    config = replace(
        base,
        device=device,
        deterministic=device == "cpu",
        optimizer_steps=5,
        microbatch_size=14,
        gradient_accumulation=4,
        yield_ms=0,
        curriculum=curriculum,
        foundation_gate_required=False,
        foundation_eval_examples=14,
        foundation_eval_batch_size=7,
        final_eval_examples_per_seed=14,
        final_eval_batch_size=7,
        eval_seeds=(11003,),
    )
    config.validate()
    return config


def recompute_label(truth: dict[str, object]) -> int:
    values = {
        int(variable): int(value) - 8
        for variable, value in zip(
            truth["binding_variables"],
            truth["binding_values"],
        )
    }
    node_values: dict[int, int] = {}
    for node in truth["nodes"]:
        node_id = int(node["node_id"])
        if node["kind"] == "leaf":
            variable_token = 15 + int(node["variable_index"])
            node_values[node_id] = values[variable_token]
        else:
            left = node_values[int(node["left"])]
            right = node_values[int(node["right"])]
            node_values[node_id] = (
                left + right
                if int(node["operator_token"]) == ADD
                else left - right
            ) % 7
    return node_values[int(truth["root_id"])]


def recompute_literal_label(truth: dict[str, object]) -> int:
    literal_values = [int(value) for value in truth["literal_values"]]
    node_values: dict[int, int] = {}
    for node in truth["nodes"]:
        node_id = int(node["node_id"])
        if node["kind"] == "leaf":
            node_values[node_id] = literal_values[int(node["leaf_index"])]
        else:
            left = node_values[int(node["left"])]
            right = node_values[int(node["right"])]
            node_values[node_id] = (
                left + right
                if int(node["operator_token"]) == ADD
                else left - right
            ) % 7
    return node_values[int(truth["root_id"])]


def complete_result_skeleton(config: Stage1Config) -> dict[str, object]:
    target = config.optimizer_steps
    examples = target * config.effective_batch_size
    return {
        "schema_version": 3,
        "state": "completed",
        "reason": "target_steps_reached",
        "global_step": target,
        "target_steps": target,
        "run_eligible_for_aggregation": True,
        "config": config.to_dict(),
        "config_digest": stage1_config_digest(config.to_dict()),
        "validated_experiment_spec_digest": validated_experiment_spec_digest(
            config
        ),
        "validated_experiment_compatibility_spec_digest": (
            validated_experiment_compatibility_spec_digest(config)
        ),
        "formal_final_attempt": {
            "required": config.formal_evaluation is True,
            "state": "completed" if config.formal_evaluation else "not_required",
        },
        "metrics": {
            "curriculum_position": {"complete": True},
            "models": {
                name: {
                    "optimizer_updates": target,
                    "examples": examples,
                }
                for name in ("A", "D_true", "D_sham")
            },
        },
    }


def attach_manifest_evidence(
    result: dict[str, object],
    run_dir: Path,
) -> Path:
    snapshot_root = run_dir / "snapshot"
    if (snapshot_root / "snapshot-manifest.json").is_file():
        snapshot_manifest = json.loads(
            (snapshot_root / "snapshot-manifest.json").read_text(
                encoding="utf-8"
            )
        )
    else:
        snapshot_manifest = create_snapshot(ROOT, snapshot_root)
    result["manifest"] = source_manifest(snapshot_root)
    result["snapshot_manifest"] = snapshot_manifest
    result["snapshot_manifest_hash"] = snapshot_manifest["manifest_hash"]
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def complete_formal_result(config: Stage1Config) -> dict[str, object]:
    result = complete_result_skeleton(config)
    result.update(
        {
            "candidate_prerequisite": {
                "required": True,
                "passed": True,
                "expected": {
                    "config_digest": config.candidate_prerequisite_config_digest,
                    "manifest_hash": config.candidate_prerequisite_manifest_hash,
                    "snapshot_manifest_hash": (
                        config.candidate_prerequisite_snapshot_manifest_hash
                    ),
                    "result_digest": config.candidate_prerequisite_result_digest,
                    "experiment_spec_digest": (
                        config.candidate_prerequisite_experiment_spec_digest
                    ),
                    "compatibility_spec_digest": (
                        config.candidate_prerequisite_compatibility_spec_digest
                    ),
                },
            },
            "foundation_gate": {"passed": True},
            "learning_gate": {"passed": True},
            "candidate_gate": {
                "candidate_pass": True,
                "stage2_unblocked": False,
            },
            "final_evaluation": {
                "kind": "formal_confirmation",
                "examples_per_split_seed": config.final_eval_examples_per_seed,
                "evaluation_seeds": list(config.eval_seeds),
                "overlap_audit": {
                    "all_content_disjoint": True,
                    "all_shape_rules_valid": True,
                },
            },
        }
    )
    return result


class RevisedStageOneDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tiny_stage1_config()

    def test_exact_mod7_balance_and_independent_label_recomputation(self) -> None:
        generator = RevisedStage1Generator(self.config.data, seed=101)
        cases = (
            (0, "leaf"),
            (1, "skew"),
            (2, "balanced"),
            (3, "branched"),
        )
        for depth, topology in cases:
            batch = generator.batch(
                14,
                depth,
                topology,
                max_structural_attempts_per_example=256,
                shape_partition="heldout" if topology == "branched" else "train",
            )
            self.assertEqual(batch.labels.tolist().count(0), 2)
            self.assertEqual(
                torch.bincount(batch.labels, minlength=7).tolist(),
                [2] * 7,
            )
            self.assertEqual(batch.generation.label_counts, (2,) * 7)
            self.assertEqual(batch.generation.accepted, 14)
            self.assertEqual(
                [recompute_label(truth) for truth in batch.truth],
                batch.labels.tolist(),
            )
            self.assertTrue(all(0 <= label < 7 for label in batch.labels.tolist()))

    def test_generation_is_deterministic_including_hashes_and_rejections(self) -> None:
        left = RevisedStage1Generator(self.config.data, seed=103).batch(
            14,
            3,
            "branched",
            max_structural_attempts_per_example=256,
            shape_partition="heldout",
        )
        right = RevisedStage1Generator(self.config.data, seed=103).batch(
            14,
            3,
            "branched",
            max_structural_attempts_per_example=256,
            shape_partition="heldout",
        )
        self.assertTrue(torch.equal(left.token_ids, right.token_ids))
        self.assertTrue(torch.equal(left.labels, right.labels))
        self.assertEqual(left.truth, right.truth)
        self.assertEqual(left.generation, right.generation)

    def test_literal_exact_balance_recomputation_and_value_tokens(self) -> None:
        generator = RevisedStage1Generator(
            self.config.data,
            seed=104,
            operand_mode="literal",
        )
        for depth, topology in ((0, "leaf"), (1, "skew"), (3, "balanced")):
            batch = generator.batch(
                14,
                depth,
                topology,
                max_structural_attempts_per_example=256,
                shape_partition="train",
            )
            self.assertEqual(torch.bincount(batch.labels, minlength=7).tolist(), [2] * 7)
            self.assertEqual(
                [recompute_literal_label(truth) for truth in batch.truth],
                batch.labels.tolist(),
            )
            for truth, row, structure in zip(
                batch.truth,
                batch.token_ids.tolist(),
                batch.structure.samples,
            ):
                self.assertEqual(truth["operand_mode"], "literal")
                self.assertNotIn("binding_values", truth)
                for node in structure.nodes:
                    if isinstance(node, LeafSourceReference):
                        self.assertIn(row[node.source_index], range(8, 15))

    def test_literal_structure_only_batch_contains_source_references_only(self) -> None:
        batch = RevisedStage1Generator(
            self.config.data,
            seed=106,
            operand_mode="literal",
        ).batch(
            14,
            3,
            "balanced",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        forbidden = {"literal_values", "leaf_index", "value", "target", "label"}
        for sample in batch.structure.samples:
            for node in sample.nodes:
                self.assertTrue(forbidden.isdisjoint(vars(node)))

    def test_evaluation_content_exclusion_is_deterministic_and_disjoint(self) -> None:
        blocked_training_batch = RevisedStage1Generator(
            self.config.data,
            seed=111,
            operand_mode="literal",
        ).batch(
            7,
            3,
            "balanced",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        training_hashes = set(blocked_training_batch.generation.content_hashes)

        def generate() -> tuple[object, dict[str, int | float], set[str]]:
            accepted: set[str] = set()
            batch, accounting = RevisedStage1Generator(
                self.config.data,
                seed=111,
                operand_mode="literal",
            ).batch_excluding_content(
                14,
                3,
                "balanced",
                training_content_hashes=training_hashes,
                prior_evaluation_content_hashes=set(),
                accepted_evaluation_hashes=accepted,
                max_structural_attempts_per_example=256,
                max_content_attempts_per_example=16,
                shape_partition="train",
            )
            return batch, accounting, accepted

        left, left_accounting, left_hashes = generate()
        right, right_accounting, right_hashes = generate()
        self.assertTrue(training_hashes.isdisjoint(left_hashes))
        self.assertEqual(left_accounting["training_content_exclusions"], 7)
        self.assertEqual(left_hashes, right_hashes)
        self.assertEqual(left_accounting, right_accounting)
        self.assertTrue(torch.equal(left.token_ids, right.token_ids))
        self.assertEqual(torch.bincount(left.labels, minlength=7).tolist(), [2] * 7)

        first_evaluation_batch = RevisedStage1Generator(
            self.config.data,
            seed=117,
            operand_mode="literal",
        ).batch(
            7,
            3,
            "balanced",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        prior_evaluation_hashes = set(first_evaluation_batch.generation.content_hashes)
        second_batch, second_accounting = RevisedStage1Generator(
            self.config.data,
            seed=117,
            operand_mode="literal",
        ).batch_excluding_content(
            7,
            3,
            "balanced",
            training_content_hashes=set(),
            prior_evaluation_content_hashes=prior_evaluation_hashes,
            accepted_evaluation_hashes=set(),
            max_structural_attempts_per_example=256,
            max_content_attempts_per_example=16,
            shape_partition="train",
        )
        self.assertEqual(
            second_accounting["prior_evaluation_content_exclusions"],
            7,
        )
        self.assertTrue(
            set(first_evaluation_batch.generation.content_hashes).isdisjoint(
                second_batch.generation.content_hashes
            )
        )

    def test_evaluation_content_exclusion_fails_closed_on_exhaustion(self) -> None:
        fixed = RevisedStage1Generator(
            self.config.data,
            seed=119,
            operand_mode="literal",
        ).batch(
            7,
            3,
            "balanced",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        generator = RevisedStage1Generator(
            self.config.data,
            seed=121,
            operand_mode="literal",
        )
        with patch.object(generator, "batch", return_value=fixed):
            with self.assertRaisesRegex(RuntimeError, "content exclusion exhausted"):
                generator.batch_excluding_content(
                    14,
                    3,
                    "balanced",
                    training_content_hashes=set(fixed.generation.content_hashes),
                    prior_evaluation_content_hashes=set(),
                    accepted_evaluation_hashes=set(),
                    max_structural_attempts_per_example=256,
                    max_content_attempts_per_example=1,
                    shape_partition="train",
                )

    def test_depth1_keeps_add_and_subtract_after_zero_coefficient_filter(self) -> None:
        batch = RevisedStage1Generator(self.config.data, seed=105).batch(
            98,
            1,
            "skew",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        operators = {
            int(node["operator_token"])
            for truth in batch.truth
            for node in truth["nodes"]
            if node["kind"] == "merge"
        }
        self.assertEqual(operators, {ADD, SUB})
        self.assertLess(batch.generation.structural_rejections, batch.generation.accepted)

    def test_structural_rejection_limit_fails_closed(self) -> None:
        generator = RevisedStage1Generator(self.config.data, seed=107)
        with patch.object(generator, "_structural_template", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "attempt limit"):
                generator.batch(
                    7,
                    1,
                    "skew",
                    max_structural_attempts_per_example=2,
                    shape_partition="train",
                )

    def test_topology_catalogs_are_valid_and_ood_shapes_are_disjoint(self) -> None:
        skew = shape_catalog(3, "skew")
        balanced = shape_catalog(3, "balanced")
        branched = shape_catalog(3, "branched")
        self.assertTrue(all(shape_height(shape) == 3 for shape in skew))
        self.assertEqual(len(balanced), 1)
        self.assertTrue(
            all(
                shape is not None and shape[0] is not None and shape[1] is not None
                for shape in branched
            )
        )
        train_ids = {
            shape_id(shape)
            for depth, topology in ((1, "skew"), (2, "skew"), (2, "balanced"), (3, "skew"), (3, "balanced"))
            for shape in shape_catalog(depth, topology)
        }
        heldout_shape_ids = {shape_id(shape) for shape in branched}
        depth_ood_ids = {shape_id(shape) for shape in shape_catalog(5, "skew")}
        self.assertTrue(train_ids.isdisjoint(heldout_shape_ids))
        self.assertTrue(train_ids.isdisjoint(depth_ood_ids))

    def test_structure_only_batch_has_no_values_targets_or_rejection_metadata(self) -> None:
        batch = RevisedStage1Generator(self.config.data, seed=109).batch(
            14,
            3,
            "balanced",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        forbidden = {
            "binding_values",
            "binding_variables",
            "label",
            "labels",
            "target",
            "targets",
            "value",
            "values",
            "attempts",
            "rejections",
            "operator_token",
            "variable_token",
        }
        self.assertIsInstance(batch.structure, StructureOnlyBatch)
        self.assertEqual({field.name for field in fields(StructureOnlyBatch)}, {"samples"})
        for sample in batch.structure.samples:
            self.assertEqual(set(vars(sample)), {"root_id", "nodes"})
            for node in sample.nodes:
                self.assertTrue(forbidden.isdisjoint(vars(node)))
                self.assertIsInstance(node, (LeafSourceReference, MergeSourceReference))

    def test_sham_preserves_topology_and_counts_but_permutes_sources(self) -> None:
        batch = RevisedStage1Generator(self.config.data, seed=113).batch(
            7,
            3,
            "balanced",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        sham = sham_structure(batch.structure, batch.token_ids)
        for true_sample, sham_sample in zip(batch.structure.samples, sham.samples):
            self.assertEqual(true_sample.root_id, sham_sample.root_id)
            self.assertEqual(len(true_sample.nodes), len(sham_sample.nodes))
            self.assertNotEqual(true_sample.nodes, sham_sample.nodes)
            for true_node, sham_node in zip(true_sample.nodes, sham_sample.nodes):
                self.assertEqual(true_node.node_id, sham_node.node_id)
                if isinstance(true_node, MergeSourceReference):
                    self.assertEqual((true_node.left, true_node.right), (sham_node.left, sham_node.right))

    def test_content_keyed_sham_varies_and_declares_low_depth_limits(self) -> None:
        batch = RevisedStage1Generator(
            self.config.data,
            seed=125,
            operand_mode="literal",
        ).batch(
            98,
            3,
            "balanced",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        sham = sham_structure(batch.structure, batch.token_ids)
        leaf_mappings = {
            tuple(
                node.source_index
                for node in sample.nodes
                if isinstance(node, LeafSourceReference)
            )
            for sample in sham.samples
        }
        self.assertGreater(len(leaf_mappings), 1)
        for true_sample, sham_sample in zip(batch.structure.samples, sham.samples):
            true_leaves = [
                node.source_index
                for node in true_sample.nodes
                if isinstance(node, LeafSourceReference)
            ]
            sham_leaves = [
                node.source_index
                for node in sham_sample.nodes
                if isinstance(node, LeafSourceReference)
            ]
            self.assertTrue(
                all(left != right for left, right in zip(true_leaves, sham_leaves))
            )

        c0 = RevisedStage1Generator(
            self.config.data,
            seed=126,
            operand_mode="literal",
        ).batch(
            7,
            0,
            "leaf",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        self.assertEqual(sham_structure(c0.structure, c0.token_ids), c0.structure)
        self.assertEqual(SHAM_MAPPING_VERSION, "content-keyed-derangement-v1")


class RevisedStageOneRuntimeTests(unittest.TestCase):
    def test_output_heads_are_exactly_seven_legal_classes(self) -> None:
        config = tiny_stage1_config()
        batch = RevisedStage1Generator(config.data, seed=127).batch(
            7,
            3,
            "skew",
            max_structural_attempts_per_example=256,
            shape_partition="train",
        )
        model_a = SmallTransformerBaseline(64, config.model_a, output_classes=7)
        model_d = TrueStructureDiagnosticD(64, config.model_d, output_classes=7)
        logits_a = model_a(batch.token_ids, batch.position_features, batch.attention_mask)
        diagnostics = model_d(
            batch.token_ids,
            batch.position_features,
            batch.attention_mask,
            batch.structure,
        )
        self.assertEqual(logits_a.shape, (7, 7))
        self.assertEqual(diagnostics.logits.shape, (7, 7))
        self.assertEqual(diagnostics.node_counts, (7,) * 7)
        self.assertEqual(diagnostics.maximum_tree_depths, (3,) * 7)
        self.assertEqual(diagnostics.combined_nodes, (3,) * 7)
        self.assertEqual(diagnostics.compose_module_calls, 3)

    def test_d_true_and_d_sham_begin_identical_and_train_with_equal_updates(self) -> None:
        config = tiny_stage1_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "snapshot-test",
            )
            self.assertEqual(
                trainer.parameter_counts["D_true"],
                trainer.parameter_counts["D_sham"],
            )
            for true_parameter, sham_parameter in zip(
                trainer.models["D_true"].parameters(),
                trainer.models["D_sham"].parameters(),
            ):
                self.assertTrue(torch.equal(true_parameter, sham_parameter))
            trainer.train_pair()
            self.assertEqual(
                {trainer.cumulative[name]["optimizer_updates"] for name in trainer.MODEL_NAMES},
                {1},
            )
            self.assertEqual(
                {trainer.cumulative[name]["examples"] for name in trainer.MODEL_NAMES},
                {7},
            )
            self.assertEqual(
                trainer.cumulative["D_true"]["compose_module_calls"],
                trainer.cumulative["D_sham"]["compose_module_calls"],
            )
            stage = trainer.curriculum_metrics["binding_lookup"]
            self.assertEqual(stage["label_counts"], [1] * 7)

    def test_literal_d_sham_compute_matches_d_true(self) -> None:
        config = tiny_literal_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "literal-sham-test",
            )
            trainer.train_pair()
            self.assertEqual(
                trainer.cumulative["D_true"]["compose_module_calls"],
                trainer.cumulative["D_sham"]["compose_module_calls"],
            )
            self.assertEqual(
                trainer.cumulative["D_true"]["combined_nodes"],
                trainer.cumulative["D_sham"]["combined_nodes"],
            )
            self.assertEqual(
                trainer.cumulative["D_true"]["optimizer_updates"],
                trainer.cumulative["D_sham"]["optimizer_updates"],
            )

    def test_curriculum_transition_checkpoint_and_exact_saved_position_resume(self) -> None:
        config = tiny_stage1_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            trainer = Stage1Trainer(config, run_dir, ROOT, "snapshot-test")
            self.assertEqual(trainer.curriculum_position()["stage_name"], "binding_lookup")
            trainer.train_pair()
            self.assertEqual(trainer.curriculum_position()["stage_name"], "depth_1")
            trainer.train_pair()
            self.assertEqual(trainer.curriculum_position()["stage_name"], "depth_2")
            checkpoint = trainer.save_checkpoint()
            restored = Stage1Trainer(config, run_dir, ROOT, "snapshot-test")
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.global_step, 2)
            self.assertEqual(restored.curriculum_position(), trainer.curriculum_position())
            self.assertEqual(restored.curriculum_metrics, trainer.curriculum_metrics)
            self.assertTrue(
                torch.equal(restored.generator.get_state(), trainer.generator.get_state())
            )
            restored.train_pair()
            self.assertEqual(restored.curriculum_position()["stage_name"], "depth_3")

    def test_literal_stage_boundary_evaluation_checkpoint_and_resume(self) -> None:
        config = tiny_literal_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            trainer = Stage1Trainer(config, run_dir, ROOT, "literal-boundary-test")
            trainer.train_pair()
            c0_boundary = trainer.evaluate_pending_stage_boundary()
            self.assertEqual(set(c0_boundary["tasks"]), {"C0"})
            trainer.train_pair()
            c1_boundary = trainer.evaluate_pending_stage_boundary()
            self.assertEqual(set(c1_boundary["tasks"]), {"C0", "C1"})
            self.assertEqual(c1_boundary["tasks"]["C0"]["label_counts"], [2] * 7)
            self.assertEqual(c1_boundary["tasks"]["C1"]["label_counts"], [2] * 7)
            checkpoint = trainer.save_checkpoint()

            restored = Stage1Trainer(config, run_dir, ROOT, "literal-boundary-test")
            restored.load_checkpoint(checkpoint)
            self.assertEqual(
                restored.stage_boundary_evaluations,
                trainer.stage_boundary_evaluations,
            )
            self.assertEqual(
                restored.pre_final_evaluation_content_hashes,
                trainer.pre_final_evaluation_content_hashes,
            )
            self.assertIsNone(restored.evaluate_pending_stage_boundary())

    def test_large_evaluation_accounting_and_per_class_counts(self) -> None:
        base = tiny_stage1_config()
        config = replace(
            base,
            final_eval_examples_per_seed=1029,
            final_eval_batch_size=147,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "snapshot-test",
            )
            split = config.evaluation_splits[0]
            result, hashes, shapes = trainer._evaluate_split_seed(
                split,
                config.eval_seeds[0],
                1029,
                147,
            )
            self.assertEqual(result["label_counts"], [147] * 7)
            self.assertEqual(len(hashes), 1029)
            self.assertGreater(len(shapes), 0)
            self.assertEqual(sum(result["paired_outcomes"].values()), 1029)
            for model in result["models"].values():
                self.assertEqual(sum(model["prediction_counts"]), 1029)
                self.assertGreater(model["cross_entropy"], 0)

    def test_final_evaluation_actively_excludes_training_and_prior_eval_content(self) -> None:
        config = tiny_literal_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "active-exclusion-test",
            )
            first_split = config.evaluation_splits[0]
            first_seed = config.eval_seeds[0]
            preview = RevisedStage1Generator(
                config.data,
                trainer._evaluation_generator_seed(first_split, first_seed),
                operand_mode="literal",
            ).batch(
                7,
                first_split.depth,
                first_split.topology,
                max_structural_attempts_per_example=256,
                shape_partition=first_split.shape_partition,
            )
            trainer.training_content_hashes.update(preview.generation.content_hashes)
            result = trainer.evaluate_final_gate()
            first_result = result["splits"][first_split.name]["seeds"][str(first_seed)]
            exclusions = first_result["generation"]["content_exclusion"]
            self.assertEqual(exclusions["training_content_exclusions"], 7)
            self.assertTrue(first_result["generation"]["active_content_exclusion"])
            self.assertTrue(result["overlap_audit"]["all_content_disjoint"])
            self.assertTrue(
                all(
                    count == 0
                    for count in result["overlap_audit"][
                        "pairwise_evaluation_content_overlap"
                    ].values()
                )
            )

    def test_heartbeat_hashes_are_excluded_from_final_evaluation(self) -> None:
        config = tiny_literal_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "heartbeat-exclusion-test",
            )
            trainer.evaluate_heartbeat()
            self.assertEqual(
                len(trainer.pre_final_evaluation_content_hashes),
                config.heartbeat_examples,
            )
            result = trainer.evaluate_final_gate()
            first_split = config.evaluation_splits[0]
            first_seed = str(config.eval_seeds[0])
            exclusions = result["splits"][first_split.name]["seeds"][first_seed][
                "generation"
            ]["content_exclusion"]
            self.assertGreaterEqual(
                result["overlap_audit"][
                    "pre_final_evaluation_content_hash_count"
                ],
                config.heartbeat_examples,
            )
            self.assertTrue(result["overlap_audit"]["all_content_disjoint"])
            self.assertTrue(
                all(
                    count == 0
                    for count in result["overlap_audit"][
                        "evaluation_overlap_with_pre_final"
                    ].values()
                )
            )

    def test_stop_final_checkpoint_resume_preserves_historical_final_hashes(self) -> None:
        config = tiny_literal_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            trainer = Stage1Trainer(
                config,
                run_dir,
                ROOT,
                "stop-final-resume-test",
            )
            trainer.evaluate_final_gate()
            first_final_hashes = set(
                trainer.historical_final_evaluation_content_hashes
            )
            self.assertTrue(first_final_hashes)
            trainer.record_run_completion(False, "user_stop")
            checkpoint = trainer.save_checkpoint("final")

            restored = Stage1Trainer(
                config,
                run_dir,
                ROOT,
                "stop-final-resume-test",
            )
            restored.load_checkpoint(checkpoint)
            self.assertEqual(
                restored.historical_final_evaluation_content_hashes,
                first_final_hashes,
            )
            second_final = restored.evaluate_final_gate()
            overlap_audit = second_final["overlap_audit"]
            self.assertEqual(
                overlap_audit[
                    "historical_final_evaluation_content_hash_count"
                ],
                len(first_final_hashes),
            )
            self.assertTrue(
                all(
                    count == 0
                    for count in overlap_audit[
                        "evaluation_overlap_with_historical_final"
                    ].values()
                )
            )
            self.assertTrue(overlap_audit["all_content_disjoint"])
            self.assertGreater(
                len(restored.historical_final_evaluation_content_hashes),
                len(first_final_hashes),
            )

    def test_formal_evaluation_emits_complete_paired_sample_masks(self) -> None:
        base = tiny_literal_config()
        config = replace(
            base,
            formal_evaluation=True,
            requires_candidate_pass=True,
            final_eval_examples_per_seed=10010,
            final_eval_batch_size=70,
            eval_seeds=(51047, 61051, 71059),
        )
        config.validate()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "formal-paired-data-test",
            )
            split = config.evaluation_splits[0]
            result, hashes, _ = trainer._evaluate_split_seed(
                split,
                config.eval_seeds[0],
                14,
                7,
                enforce_content_exclusion=True,
                training_content_hashes=set(),
                prior_evaluation_content_hashes=set(),
                accepted_evaluation_hashes=set(),
            )
            paired = result["paired_sample_data"]
            self.assertEqual(paired["sample_count"], 14)
            self.assertEqual(len(paired["correctness_masks"]), 14)
            self.assertEqual(paired["content_hash_digest"], result["content_hash_digest"])
            self.assertEqual(len(hashes), 14)
            self.assertTrue(all(0 <= mask <= 7 for mask in paired["correctness_masks"]))

    def _synthetic_evaluation(self, config: Stage1Config, passing: bool) -> dict[str, object]:
        splits = {}
        for split in config.evaluation_splits:
            a_accuracy = 0.45
            d_true_accuracy = 0.55 if passing else 0.44
            d_sham_accuracy = 0.43
            seeds = {
                str(seed): {
                    "models": {
                        "A": {
                            "accuracy": a_accuracy,
                            "distinct_predicted_classes": 7,
                        },
                        "D_true": {
                            "accuracy": d_true_accuracy,
                            "distinct_predicted_classes": 7,
                        },
                        "D_sham": {
                            "accuracy": d_sham_accuracy,
                            "distinct_predicted_classes": 7,
                        },
                    },
                    "majority_baseline": 1 / 7,
                }
                for seed in config.eval_seeds
            }
            splits[split.name] = {"seeds": seeds}
        return {
            "splits": splits,
            "overlap_audit": {
                "all_content_disjoint": True,
                "all_shape_rules_valid": True,
            },
        }

    def test_candidate_gate_pass_fail_and_stage2_stays_blocked(self) -> None:
        config = tiny_literal_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "snapshot-test",
            )
            passed = trainer.compute_gate(self._synthetic_evaluation(config, True))
            failed = trainer.compute_gate(self._synthetic_evaluation(config, False))
            self.assertTrue(passed["candidate_pass"])
            self.assertFalse(passed["stage2_unblocked"])
            self.assertFalse(failed["candidate_pass"])
            self.assertIn("D_true_over_D_sham", " ".join(passed["conditions"]))

    def test_posthoc_baseline_policy_separates_a_sanity_from_structural_effect(self) -> None:
        base = tiny_literal_config()
        config = replace(
            base,
            gate=replace(
                base.gate,
                baseline_policy="privileged_structure_posthoc_v1",
            ),
        )
        config.validate()
        evaluation = self._synthetic_evaluation(config, True)
        for split in config.evaluation_splits:
            for seed_result in evaluation["splits"][split.name]["seeds"].values():
                seed_result["models"]["A"]["accuracy"] = 1 / 7
        id_split = next(
            split
            for split in config.evaluation_splits
            if split.category == "in_distribution"
        )
        for seed_result in evaluation["splits"][id_split.name]["seeds"].values():
            seed_result["models"]["A"]["accuracy"] = 0.45

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "posthoc-gate-test",
            )
            passed = trainer.compute_gate(evaluation)
            self.assertTrue(passed["candidate_pass"])
            self.assertEqual(
                passed["baseline_policy"],
                "privileged_structure_posthoc_v1",
            )
            self.assertTrue(
                passed["conditions"][
                    "A_above_majority_on_at_least_one_in_distribution_split"
                ]
            )

            for seed_result in evaluation["splits"][id_split.name]["seeds"].values():
                seed_result["models"]["A"]["accuracy"] = 1 / 7
            self.assertFalse(trainer.compute_gate(evaluation)["candidate_pass"])

            for seed_result in evaluation["splits"][id_split.name]["seeds"].values():
                seed_result["models"]["A"]["accuracy"] = 0.45
            ood_split = next(
                split
                for split in config.evaluation_splits
                if split.category != "in_distribution"
            )
            for seed_result in evaluation["splits"][ood_split.name]["seeds"].values():
                seed_result["models"]["A"]["distinct_predicted_classes"] = 1
            self.assertFalse(trainer.compute_gate(evaluation)["candidate_pass"])

    def test_bound_variable_axis_is_not_structural_gate_eligible(self) -> None:
        config = tiny_stage1_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(config, Path(temporary_directory), ROOT, "bound-axis")
            result = trainer.compute_gate(self._synthetic_evaluation(config, True))
            self.assertTrue(result["structural_conditions_pass"])
            self.assertFalse(result["structural_gate_eligible"])
            self.assertFalse(result["candidate_pass"])

    def test_incomplete_run_is_fail_closed_for_gate_and_aggregation(self) -> None:
        config = tiny_literal_config()
        result = complete_result_skeleton(config)
        self.assertTrue(all(run_completion_checks(result).values()))
        result["state"] = "incomplete"
        result["reason"] = "time_budget_reached"
        result["global_step"] = config.optimizer_steps - 1
        result["run_eligible_for_aggregation"] = False
        checks = run_completion_checks(result)
        self.assertFalse(checks["state_completed"])
        self.assertFalse(checks["target_reason"])
        self.assertFalse(checks["exact_observed_step"])
        self.assertFalse(checks["run_marked_eligible"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "incomplete-gate-test",
            )
            trainer.gate_result = {
                "candidate_pass": True,
                "stage2_unblocked": False,
            }
            trainer.record_run_completion(False, "user_stop")
            self.assertFalse(trainer.gate_result["candidate_pass"])
            self.assertFalse(trainer.gate_result["run_complete"])
        self.assertEqual(
            classify_run_outcome("target_steps_reached", 5, 5),
            ("completed", True),
        )
        self.assertEqual(
            classify_run_outcome("time_budget_reached", 4, 5),
            ("incomplete", False),
        )
        self.assertEqual(
            classify_run_outcome("user_stop", 4, 5),
            ("incomplete", False),
        )
        self.assertEqual(
            classify_run_outcome("target_steps_reached", 4, 5),
            ("incomplete", False),
        )

    def test_formal_stop_and_timeout_do_not_touch_final_holdout(self) -> None:
        config = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )

        class FakeTrainer:
            def __init__(self, step: int) -> None:
                self.global_step = step
                self.final_evaluation: dict[str, object] = {}
                self.gate_result: dict[str, object] = {}
                self.final_calls = 0
                self.learning_calls = 0
                self.saved: list[str] = []
                self.completions: list[tuple[bool, str]] = []

            def evaluate_final_gate(self) -> None:
                self.final_calls += 1
                self.final_evaluation = {"kind": "formal_confirmation"}
                self.gate_result = {
                    "candidate_pass": True,
                    "stage2_unblocked": False,
                }

            def learning_gate(self) -> dict[str, object]:
                self.learning_calls += 1
                raise AssertionError(
                    "incomplete formal training must not evaluate learning gate"
                )

            def record_run_completion(self, complete: bool, reason: str) -> None:
                self.completions.append((complete, reason))
                if not complete:
                    self.gate_result = {
                        "candidate_pass": False,
                        "stage2_unblocked": False,
                    }

            def save_checkpoint(self, kind: str) -> Path:
                self.saved.append(kind)
                return Path(f"checkpoint-{kind}.pt")

        for reason in ("user_stop", "time_budget_reached"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                trainer = FakeTrainer(config.optimizer_steps - 1)
                state, complete, learning_gate, final_attempt = (
                    finalize_training_attempt(
                    trainer,
                    config,
                    reason,
                    Path(temporary),
                    )
                )
                self.assertEqual(state, "incomplete")
                self.assertFalse(complete)
                self.assertEqual(trainer.final_calls, 0)
                self.assertEqual(trainer.learning_calls, 0)
                self.assertEqual(trainer.final_evaluation, {})
                self.assertEqual(trainer.saved, ["incomplete"])
                self.assertEqual(learning_gate["state"], "not_evaluated")
                self.assertIs(learning_gate["passed"], False)
                self.assertEqual(final_attempt["state"], "not_started")
                self.assertFalse(
                    (Path(temporary) / "formal-final-attempt.json").exists()
                )

    def test_formal_final_is_once_only_and_recovers_without_reuse(self) -> None:
        config = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )

        class FakeTrainer:
            def __init__(self) -> None:
                self.global_step = config.optimizer_steps
                self.final_evaluation: dict[str, object] = {}
                self.gate_result: dict[str, object] = {}
                self.final_calls = 0
                self.saved: list[str] = []

            def evaluate_final_gate(self) -> None:
                self.final_calls += 1
                self.final_evaluation = {"kind": "formal_confirmation"}
                self.gate_result = {
                    "candidate_pass": True,
                    "stage2_unblocked": False,
                }

            def learning_gate(self) -> dict[str, object]:
                return {"passed": True}

            def record_run_completion(self, complete: bool, reason: str) -> None:
                self.gate_result["run_complete"] = complete

            def save_checkpoint(self, kind: str) -> Path:
                self.saved.append(kind)
                return Path(f"checkpoint-{kind}.pt")

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = FakeTrainer()
            _, complete, _, marker = finalize_training_attempt(
                trainer,
                config,
                "target_steps_reached",
                run_dir,
            )
            self.assertTrue(complete)
            self.assertEqual(trainer.final_calls, 1)
            self.assertEqual(marker["state"], "completed")

            _, recovered, _, recovered_marker = finalize_training_attempt(
                trainer,
                config,
                "target_steps_reached",
                run_dir,
            )
            self.assertTrue(recovered)
            self.assertEqual(trainer.final_calls, 1)
            self.assertTrue(
                recovered_marker["recovered_without_holdout_reuse"]
            )

    def test_resume_time_budget_is_per_worker_session(self) -> None:
        config = tiny_literal_config()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage1Trainer(config, Path(temporary), ROOT, "session-budget")
            trainer.elapsed_before_resume = 10_000.0
            trainer.process_started = time.monotonic()
            elapsed_tolerance_seconds = 1e-9
            self.assertGreaterEqual(
                trainer.elapsed_seconds() + elapsed_tolerance_seconds,
                10_000.0,
            )
            self.assertLess(trainer.session_elapsed_seconds(), 1.0)

    def test_legacy_completion_fields_are_located_but_not_current_schema(self) -> None:
        config = tiny_literal_config()
        result = complete_result_skeleton(config)
        result["schema_version"] = 2
        del result["global_step"]
        del result["target_steps"]
        del result["run_eligible_for_aggregation"]
        result["checkpoint_recovery"] = {
            "current_step": config.optimizer_steps,
        }

        evidence = run_completion_evidence(result)
        self.assertEqual(
            evidence["actual_step_field"],
            "checkpoint_recovery.current_step",
        )
        self.assertEqual(evidence["target_step_field"], "config.optimizer_steps")
        self.assertFalse(evidence["aggregation_eligibility_explicit"])
        self.assertTrue(evidence["target_completion_observed"])
        checks = run_completion_checks(result)
        self.assertTrue(
            all(
                value
                for name, value in checks.items()
                if name != "run_marked_eligible"
            )
        )
        self.assertFalse(checks["run_marked_eligible"])

    def test_candidate_prerequisite_is_pinned_to_result_config_and_source(self) -> None:
        candidate_config = tiny_literal_config()
        candidate_result = complete_result_skeleton(candidate_config)
        candidate_result.update(
            {
                "foundation_gate": {"passed": True},
                "candidate_gate": {
                    "candidate_pass": True,
                    "stage2_unblocked": False,
                },
                "learning_gate": {"passed": True},
                "final_evaluation": {
                    "examples_per_split_seed": (
                        candidate_config.final_eval_examples_per_seed
                    ),
                    "evaluation_seeds": list(candidate_config.eval_seeds),
                },
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate_run = Path(temporary_directory)
            snapshot_manifest = create_snapshot(
                ROOT,
                candidate_run / "snapshot",
            )
            candidate_result["manifest"] = source_manifest(
                candidate_run / "snapshot"
            )
            candidate_result["snapshot_manifest_hash"] = snapshot_manifest[
                "manifest_hash"
            ]
            result_path = candidate_run / "candidate-result.json"
            result_bytes = json.dumps(
                candidate_result,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            result_path.write_bytes(result_bytes)
            unconfigured_formal = load_stage1_config(
                ROOT / "configs" / "stage1-revised-literal-formal-directml.json"
            )
            formal = replace(
                candidate_config,
                formal_evaluation=True,
                requires_candidate_pass=True,
                final_eval_examples_per_seed=10010,
                final_eval_batch_size=70,
                eval_seeds=(51047, 61051, 71059),
            )
            formal.validate()
            experiment_spec_digest = validated_experiment_spec_digest(
                candidate_config
            )
            pinned = replace(
                formal,
                candidate_prerequisite_config_digest=stage1_config_digest(
                    candidate_result["config"]
                ),
                candidate_prerequisite_manifest_hash=candidate_result[
                    "manifest"
                ]["manifest_hash"],
                candidate_prerequisite_snapshot_manifest_hash=(
                    snapshot_manifest["manifest_hash"]
                ),
                candidate_prerequisite_result_digest=hashlib.sha256(
                    result_bytes
                ).hexdigest(),
                candidate_prerequisite_experiment_spec_digest=(
                    experiment_spec_digest
                ),
                candidate_prerequisite_result_path="candidate-result.json",
                candidate_prerequisite_compatibility_spec_digest=(
                    validated_experiment_compatibility_spec_digest(
                        candidate_config
                    )
                ),
            )
            pinned.validate()
            verification = validate_candidate_prerequisite(pinned, result_path)
            self.assertTrue(verification["passed"])
            self.assertTrue(all(verification["checks"].values()))

            incompatible_formal = replace(pinned, learning_rate=0.002)
            incompatible_formal.validate()
            with self.assertRaisesRegex(RuntimeError, "specification"):
                validate_candidate_prerequisite(
                    incompatible_formal,
                    result_path,
                )

            strict_boolean_result = json.loads(result_bytes.decode("utf-8"))
            strict_boolean_result["candidate_gate"]["candidate_pass"] = "false"
            strict_boolean_bytes = json.dumps(
                strict_boolean_result,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            result_path.write_bytes(strict_boolean_bytes)
            strict_boolean_pinned = replace(
                pinned,
                candidate_prerequisite_result_digest=hashlib.sha256(
                    strict_boolean_bytes
                ).hexdigest(),
            )
            with self.assertRaisesRegex(RuntimeError, "pinned completion"):
                validate_candidate_prerequisite(
                    strict_boolean_pinned,
                    result_path,
                )

            legacy_result = json.loads(result_bytes.decode("utf-8"))
            legacy_result["schema_version"] = 2
            legacy_result["checkpoint_recovery"] = {
                "current_step": candidate_config.optimizer_steps,
            }
            del legacy_result["global_step"]
            del legacy_result["target_steps"]
            del legacy_result["run_eligible_for_aggregation"]
            legacy_bytes = json.dumps(
                legacy_result,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            result_path.write_bytes(legacy_bytes)
            legacy_pinned = replace(
                pinned,
                candidate_prerequisite_result_digest=hashlib.sha256(
                    legacy_bytes
                ).hexdigest(),
            )
            with self.assertRaisesRegex(RuntimeError, "pinned completion"):
                validate_candidate_prerequisite(legacy_pinned, result_path)

            result_path.write_bytes(result_bytes)
            candidate_result["config"]["optimizer_steps"] += 1
            result_path.write_text(json.dumps(candidate_result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pinned completion"):
                validate_candidate_prerequisite(pinned, result_path)

            with self.assertRaisesRegex(RuntimeError, "fully pinned"):
                validate_candidate_prerequisite(unconfigured_formal, result_path)

    def test_validated_experiment_spec_digest_binds_scientific_fields(self) -> None:
        candidate = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-posthoc-revalidation-directml.json"
        )
        candidate_digest = validated_experiment_spec_digest(candidate)
        formal_variant = replace(
            candidate,
            seed=82427,
            formal_evaluation=True,
            requires_candidate_pass=True,
            final_eval_examples_per_seed=10010,
            final_eval_batch_size=70,
            eval_seeds=(51047, 61051, 71059),
            time_budget_minutes=300,
            checkpoint_minutes=10,
            cpu_pause_percent=90,
            cpu_resume_percent=80,
        )
        formal_variant.validate()
        self.assertEqual(
            validated_experiment_spec_digest(formal_variant),
            candidate_digest,
        )
        self.assertNotEqual(
            validated_experiment_spec_digest(
                replace(candidate, learning_rate=0.002)
            ),
            candidate_digest,
        )
        self.assertNotEqual(
            validated_experiment_spec_digest(
                replace(
                    candidate,
                    gate=replace(
                        candidate.gate,
                        minimum_d_advantage_extrapolation=0.03,
                    ),
                )
            ),
            candidate_digest,
        )
        self.assertNotEqual(
            validated_experiment_spec_digest(
                replace(
                    candidate,
                    model_a=replace(
                        candidate.model_a,
                        feedforward_dim=96,
                    ),
                )
            ),
            candidate_digest,
        )

    def test_formal_confirmation_config_matches_passing_candidate(self) -> None:
        result_path = (
            ROOT / "runs" / "stage1-20260730T152137Z" / "result.json"
        )
        candidate_result = json.loads(result_path.read_text(encoding="utf-8"))
        formal = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )
        self.assertEqual(formal.optimizer_steps, 8000)
        self.assertTrue(formal.formal_evaluation)
        self.assertEqual(formal.final_eval_examples_per_seed, 10010)
        self.assertEqual(
            candidate_result["validated_experiment_spec_digest"],
            formal.candidate_prerequisite_experiment_spec_digest,
        )
        self.assertEqual(
            validated_experiment_compatibility_spec_digest(
                candidate_result["config"]
            ),
            validated_experiment_compatibility_spec_digest(formal),
        )
        self.assertEqual(
            validated_experiment_compatibility_spec_digest(formal),
            formal.candidate_prerequisite_compatibility_spec_digest,
        )
        verification = validate_candidate_prerequisite(formal, result_path)
        self.assertTrue(verification["passed"])
        self.assertTrue(all(verification["checks"].values()))

    def test_formal_confirmation_seeds_are_fresh_from_nonformal_work(self) -> None:
        formal_path = (
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )
        formal = load_stage1_config(formal_path)
        registered = {
            *formal.confirmation_training_seeds,
            *formal.eval_seeds,
            formal.foundation_eval_seed,
        }
        self.assertEqual(len(registered), 12)
        self.assertNotIn(82421, formal.confirmation_training_seeds)

        observed: set[int] = set()

        def collect_seed_values(value: object, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child in value.items():
                    collect_seed_values(child, str(child_key))
            elif isinstance(value, list):
                if key.endswith("seeds"):
                    observed.update(
                        item
                        for item in value
                        if type(item) is int
                    )
                else:
                    for child in value:
                        collect_seed_values(child, key)
            elif type(value) is int and (
                key == "seed" or key.endswith("_seed")
            ):
                observed.add(value)

        for path in (ROOT / "configs").glob("*.json"):
            if path.resolve() == formal_path.resolve():
                continue
            collect_seed_values(json.loads(path.read_text(encoding="utf-8")))
        for path in (ROOT / "runs").glob("*/result.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("config", {}).get("formal_evaluation") is True:
                continue
            collect_seed_values(result)
        self.assertTrue(registered.isdisjoint(observed))
        runtime_gate = formal_seed_freshness(ROOT, formal_path, formal)
        self.assertTrue(runtime_gate["passed"])
        self.assertEqual(runtime_gate["overlap"], [])

    def test_formal_seed_freshness_rejects_history_and_internal_overlap(self) -> None:
        formal = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )
        with self.assertRaisesRegex(ValueError, "pairwise disjoint"):
            replace(
                formal,
                foundation_eval_seed=formal.confirmation_training_seeds[0],
            ).validate()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs"
            configs.mkdir()
            formal_path = configs / "formal.json"
            formal_path.write_text(
                json.dumps(formal.to_dict()),
                encoding="utf-8",
            )
            historical = replace(
                tiny_literal_config(),
                seed=formal.eval_seeds[0],
            )
            (configs / "historical.json").write_text(
                json.dumps(historical.to_dict()),
                encoding="utf-8",
            )
            freshness = formal_seed_freshness(root, formal_path, formal)
            self.assertFalse(freshness["passed"])
            self.assertEqual(freshness["overlap"], [formal.eval_seeds[0]])

    def test_formal_manifest_evidence_is_recomputed_and_fail_closed(self) -> None:
        formal = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )
        result = complete_formal_result(formal)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "stage1-formal-evidence"
            result_path = attach_manifest_evidence(result, run_dir)
            checks = verify_result_manifests(result, run_dir)
            self.assertTrue(all(checks.values()))
            verify_formal_result(result_path, formal, formal.seed)

            source_file = next(
                path
                for path in (run_dir / "snapshot" / "src").rglob("*.py")
            )
            source_file.write_text(
                source_file.read_text(encoding="utf-8") + "\n# corruption\n",
                encoding="utf-8",
            )
            corrupted = verify_result_manifests(result, run_dir)
            self.assertFalse(
                corrupted["snapshot_manifest_file_hashes_match"]
            )
            self.assertFalse(
                corrupted["source_manifest_recomputed_from_snapshot"]
            )
            with self.assertRaisesRegex(SequenceAbort, "manifest"):
                verify_formal_result(result_path, formal, formal.seed)

    def test_sequential_confirmation_verifier_fails_closed(self) -> None:
        formal = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )
        seed = formal.confirmation_training_seeds[0]
        run_config = replace(formal, seed=seed)
        result = complete_result_skeleton(run_config)
        result.update(
            {
                "candidate_prerequisite": {
                    "required": True,
                    "passed": True,
                    "expected": {
                        "config_digest": formal.candidate_prerequisite_config_digest,
                        "manifest_hash": formal.candidate_prerequisite_manifest_hash,
                        "snapshot_manifest_hash": (
                            formal.candidate_prerequisite_snapshot_manifest_hash
                        ),
                        "result_digest": formal.candidate_prerequisite_result_digest,
                        "experiment_spec_digest": (
                            formal.candidate_prerequisite_experiment_spec_digest
                        ),
                        "compatibility_spec_digest": (
                            formal.candidate_prerequisite_compatibility_spec_digest
                        ),
                    },
                },
                "foundation_gate": {"passed": True},
                "learning_gate": {"passed": True},
                "candidate_gate": {
                    "candidate_pass": True,
                    "stage2_unblocked": False,
                },
                "final_evaluation": {
                    "kind": "formal_confirmation",
                    "examples_per_split_seed": 10010,
                    "evaluation_seeds": list(formal.eval_seeds),
                    "overlap_audit": {
                        "all_content_disjoint": True,
                        "all_shape_rules_valid": True,
                    },
                },
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "stage1-formal-verifier"
            result_path = attach_manifest_evidence(result, run_dir)
            verify_formal_result(result_path, formal, seed)

            result["run_eligible_for_aggregation"] = "true"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(SequenceAbort, "eligible"):
                verify_formal_result(result_path, formal, seed)

            result["run_eligible_for_aggregation"] = True
            result["candidate_gate"]["candidate_pass"] = False
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(SequenceAbort, "candidate_gate"):
                verify_formal_result(result_path, formal, seed)

    def test_sequential_runner_direct_cli_entrypoint_loads(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_stage1_confirmation_sequence.py"),
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--aggregate-output", completed.stdout)

    def test_launcher_wait_does_not_depend_on_inherited_pipe_eof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "launch-receipt.json"
            child_code = "import time; time.sleep(3)"
            parent_code = (
                "import subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', "
                f"{child_code!r}]); "
                "print('launcher-exited')"
            )
            started = time.monotonic()
            confirmation_sequence._run_launcher_command(
                [sys.executable, "-c", parent_code],
                receipt_path,
                "inherited-handle-test",
            )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 2.0)
            stdout_path = receipt_path.with_name(
                f"{receipt_path.stem}.launcher.stdout.log"
            )
            self.assertIn(
                "launcher-exited",
                stdout_path.read_text(encoding="utf-8"),
            )
            # The simulated worker intentionally inherits the log handles.
            # Let it exit before TemporaryDirectory removes those files on Windows.
            time.sleep(3.2)

    def test_sequential_runner_fails_closed_when_coordinator_is_live(self) -> None:
        with patch.object(
            confirmation_sequence.PerRunMutex,
            "acquire",
            side_effect=RuntimeError("held"),
        ):
            with self.assertRaisesRegex(
                SequenceAbort,
                "coordinator is already active",
            ):
                confirmation_sequence.run_sequence(
                    ROOT
                    / "configs"
                    / "stage1-revised-literal-formal-confirmation-directml.json",
                    ROOT / "runs" / "unused-test-sequence.json",
                    ROOT / "runs" / "unused-test-aggregate.json",
                    0.01,
                )
        self.assertFalse((ROOT / "runs" / "unused-test-sequence.json").exists())

    def test_sequence_waits_for_worker_exit_after_terminal_result(self) -> None:
        with (
            patch.object(
                confirmation_sequence,
                "_run_is_live",
                side_effect=[True, True, False],
            ) as live,
            patch.object(confirmation_sequence.time, "sleep") as sleep,
        ):
            confirmation_sequence._wait_for_worker_exit(
                ROOT / "runs" / "synthetic-formal-run",
                0.01,
                timeout_seconds=1.0,
            )
        self.assertEqual(live.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_partial_launch_shell_rebuild_is_strict_and_preserves_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake_root = Path(temporary)
            runs = fake_root / "runs"
            clean_run = runs / "stage1-formal-01-1-test"
            attach_root = clean_run / "snapshot"
            create_snapshot(ROOT, attach_root)
            (clean_run / ".launch.lock").write_bytes(b"")
            (clean_run / "stdout.log").write_bytes(b"")
            (clean_run / "stderr.log").write_bytes(b"")
            with patch.object(confirmation_sequence, "ROOT", fake_root):
                safe, evidence = (
                    confirmation_sequence._partial_shell_is_safe_to_rebuild(
                        clean_run
                    )
                )
                self.assertTrue(safe, evidence)
                confirmation_sequence._rebuild_partial_shell(clean_run)
                self.assertFalse(clean_run.exists())

                evidenced_run = runs / "stage1-formal-01-2-test"
                create_snapshot(ROOT, evidenced_run / "snapshot")
                (evidenced_run / "status.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(SequenceAbort, "contains evidence"):
                    confirmation_sequence._rebuild_partial_shell(evidenced_run)
                self.assertTrue((evidenced_run / "status.json").is_file())

    def test_recoverable_incomplete_requires_checkpoint_and_no_final_holdout(self) -> None:
        formal = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir()
            checkpoint = checkpoints / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            (checkpoints / "latest.json").write_text(
                json.dumps({"checkpoint": str(checkpoint)}),
                encoding="utf-8",
            )
            result = complete_result_skeleton(formal)
            result.update(
                {
                    "state": "incomplete",
                    "reason": "time_budget_reached",
                    "global_step": formal.optimizer_steps - 1,
                    "run_eligible_for_aggregation": False,
                    "final_evaluation": {},
                    "formal_final_attempt": {
                        "required": True,
                        "state": "not_started",
                    },
                }
            )
            checks = confirmation_sequence._recoverable_incomplete_checks(
                result,
                run_dir,
                formal,
            )
            self.assertTrue(all(checks.values()))
            result["final_evaluation"] = {"kind": "formal_confirmation"}
            self.assertFalse(
                confirmation_sequence._recoverable_incomplete_checks(
                    result,
                    run_dir,
                    formal,
                )["formal_holdout_untouched"]
            )

    def test_legacy_failed_finalization_is_narrow_and_archived_byte_exact(self) -> None:
        formal = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )
        seed = formal.confirmation_training_seeds[1]
        run_config = replace(formal, seed=seed)
        step = formal.optimizer_steps - 1

        def build_artifact(run_dir: Path) -> tuple[dict[str, object], Path]:
            result = complete_formal_result(run_config)
            result.update(
                {
                    "state": "failed",
                    "reason": (
                        confirmation_sequence
                        .LEGACY_INCOMPLETE_FINALIZATION_ERROR
                    ),
                    "global_step": step,
                    "target_steps": formal.optimizer_steps,
                    "run_eligible_for_aggregation": False,
                    "candidate_gate": {
                        "run_complete": False,
                        "run_completion_reason": (
                            confirmation_sequence
                            .LEGACY_INCOMPLETE_FINALIZATION_ERROR
                        ),
                        "candidate_pass_before_run_completion_check": False,
                        "candidate_pass": False,
                        "stage2_unblocked": False,
                        "stage2_block_reason": (
                            "run did not reach exactly optimizer_steps with "
                            "target_steps_reached"
                        ),
                    },
                    "formal_final_attempt": {
                        "required": True,
                        "state": "not_started",
                    },
                }
            )
            result.pop("final_evaluation")
            result.pop("learning_gate")
            result.pop("foundation_gate")
            result["metrics"]["curriculum_position"] = {"complete": False}
            for model in result["metrics"]["models"].values():
                model["optimizer_updates"] = step
                model["examples"] = step * formal.effective_batch_size

            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir(parents=True)
            checkpoint = checkpoints / f"checkpoint-{step:08d}.pt"
            checkpoint.write_bytes(b"checkpoint evidence")
            (checkpoints / "latest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "global_step": step,
                        "kind": "emergency",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "state": "failed",
                        "step": step,
                        "error": (
                            confirmation_sequence
                            .LEGACY_INCOMPLETE_FINALIZATION_ERROR
                        ),
                    }
                ),
                encoding="utf-8",
            )
            result_path = attach_manifest_evidence(result, run_dir)
            serialized_result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            return serialized_result, result_path

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "stage1-formal-legacy-stop"
            result, result_path = build_artifact(run_dir)
            checks = (
                confirmation_sequence._legacy_failed_finalization_checks(
                    result,
                    run_dir,
                    formal,
                    seed,
                )
            )
            self.assertTrue(all(checks.values()), checks)

            control = run_dir / "control"
            control.mkdir()
            stop = control / "STOP"
            stop.write_text("soft stop", encoding="utf-8")
            self.assertTrue(
                confirmation_sequence._requires_explicit_stop_resume(
                    result,
                    run_dir,
                    legacy_failed=True,
                )
            )
            self.assertTrue(result_path.is_file())
            stop.unlink()
            self.assertFalse(
                confirmation_sequence._requires_explicit_stop_resume(
                    result,
                    run_dir,
                    legacy_failed=True,
                )
            )

            original_bytes = result_path.read_bytes()
            original_digest = hashlib.sha256(original_bytes).hexdigest()
            archived = confirmation_sequence._archive_incomplete_result(
                result_path,
                result,
            )
            self.assertFalse(result_path.exists())
            self.assertEqual(archived.read_bytes(), original_bytes)
            self.assertEqual(
                hashlib.sha256(archived.read_bytes()).hexdigest(),
                original_digest,
            )
            self.assertNotIn(":", archived.name)

        mutations = {
            "holdout": lambda value: value.__setitem__(
                "final_evaluation",
                {"kind": "formal_confirmation"},
            ),
            "config": lambda value: value["config"].__setitem__(
                "learning_rate",
                0.5,
            ),
            "manifest": lambda value: value["manifest"].__setitem__(
                "manifest_hash",
                "0" * 64,
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / f"stage1-formal-tampered-{name}"
                result, result_path = build_artifact(run_dir)
                mutate(result)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                checks = (
                    confirmation_sequence._legacy_failed_finalization_checks(
                        result,
                        run_dir,
                        formal,
                        seed,
                    )
                )
                self.assertFalse(all(checks.values()), checks)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "stage1-formal-tampered-marker"
            result, _ = build_artifact(run_dir)
            (run_dir / "formal-final-attempt.json").write_text(
                json.dumps({"state": "started"}),
                encoding="utf-8",
            )
            checks = confirmation_sequence._legacy_failed_finalization_checks(
                result,
                run_dir,
                formal,
                seed,
            )
            self.assertFalse(checks["legacy_marker_not_started"])

    def test_sequential_runner_never_overlaps_and_stops_on_first_failure(self) -> None:
        formal = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-directml.json"
        )

        def result_for(seed: int, passing: bool) -> dict[str, object]:
            run_config = replace(formal, seed=seed)
            result = complete_formal_result(run_config)
            result["candidate_gate"]["candidate_pass"] = passing
            return result

        with tempfile.TemporaryDirectory() as temporary_directory:
            fake_root = Path(temporary_directory)
            candidate_destination = (
                fake_root / formal.candidate_prerequisite_result_path
            )
            candidate_destination.parent.mkdir(parents=True)
            shutil.copy2(
                ROOT / formal.candidate_prerequisite_result_path,
                candidate_destination,
            )
            shutil.copytree(
                (ROOT / formal.candidate_prerequisite_result_path).parent
                / "snapshot",
                candidate_destination.parent / "snapshot",
            )
            state_path = fake_root / "runs" / "sequence.json"
            aggregate_path = fake_root / "runs" / "aggregate.json"
            active = False
            launched: list[int] = []
            resumed: list[int] = []
            waits_by_seed: dict[int, int] = {}

            def fake_launch(
                config_path: Path,
                candidate_path: Path,
                seed: int,
                run_dir: Path,
                receipt_path: Path,
            ) -> None:
                nonlocal active
                self.assertFalse(active, "a second run started before terminal wait")
                active = True
                launched.append(seed)
                run_dir.mkdir(parents=True)
                (run_dir / "pid.json").write_text(
                    json.dumps({"pid": 999_999_999}),
                    encoding="utf-8",
                )

            def fake_resume(run_dir: Path, receipt_path: Path) -> None:
                nonlocal active
                self.assertFalse(active)
                active = True
                resumed.append(int(run_dir.name.split("-")[3]))

            def fake_wait(run_dir: Path, poll_seconds: float) -> Path:
                nonlocal active
                self.assertTrue(active)
                active = False
                seed = launched[-1]
                waits_by_seed[seed] = waits_by_seed.get(seed, 0) + 1
                if len(launched) == 1 and waits_by_seed[seed] == 1:
                    checkpoints = run_dir / "checkpoints"
                    checkpoints.mkdir()
                    checkpoint = checkpoints / "checkpoint.pt"
                    checkpoint.write_bytes(b"checkpoint")
                    (checkpoints / "latest.json").write_text(
                        json.dumps({"checkpoint": str(checkpoint)}),
                        encoding="utf-8",
                    )
                    incomplete = complete_formal_result(
                        replace(formal, seed=seed)
                    )
                    incomplete.update(
                        {
                            "state": "incomplete",
                            "reason": "time_budget_reached",
                            "global_step": formal.optimizer_steps - 1,
                            "run_eligible_for_aggregation": False,
                            "final_evaluation": {},
                            "candidate_gate": {
                                "candidate_pass": False,
                                "stage2_unblocked": False,
                            },
                            "formal_final_attempt": {
                                "required": True,
                                "state": "not_started",
                            },
                        }
                    )
                    return attach_manifest_evidence(incomplete, run_dir)
                result_path = attach_manifest_evidence(
                    result_for(seed, len(launched) == 1),
                    run_dir,
                )
                return result_path

            with (
                patch.object(confirmation_sequence, "ROOT", fake_root),
                patch.object(confirmation_sequence, "_launch", fake_launch),
                patch.object(confirmation_sequence, "_resume", fake_resume),
                patch.object(
                    confirmation_sequence,
                    "_wait_for_terminal",
                    fake_wait,
                ),
            ):
                with self.assertRaisesRegex(SequenceAbort, "candidate_gate"):
                    confirmation_sequence.run_sequence(
                        ROOT
                        / "configs"
                        / "stage1-revised-literal-formal-confirmation-directml.json",
                        state_path,
                        aggregate_path,
                        0.01,
                    )
            self.assertEqual(
                launched,
                list(formal.confirmation_training_seeds[:2]),
            )
            self.assertEqual(resumed, [formal.confirmation_training_seeds[0]])
            self.assertFalse(active)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["state"], "failed_closed")
            first_entry = state["runs"][str(formal.confirmation_training_seeds[0])]
            self.assertEqual(
                first_entry["incomplete_attempts"][0]["reason"],
                "time_budget_reached",
            )

    def test_schema_booleans_reject_string_values(self) -> None:
        config = tiny_literal_config()
        result = complete_result_skeleton(config)
        result["run_eligible_for_aggregation"] = "true"
        self.assertFalse(
            run_completion_checks(result)["run_marked_eligible"]
        )
        with self.assertRaisesRegex(ValueError, "formal_evaluation"):
            replace(config, formal_evaluation="false").validate()
        split = replace(
            config.evaluation_splits[0],
            required_above_majority="false",
        )
        with self.assertRaisesRegex(ValueError, "required_above_majority"):
            split.validate()

    def test_literal_foundation_gate_pass_and_fail(self) -> None:
        base = tiny_literal_config()
        config = replace(
            base,
            foundation_gate_required=True,
            foundation_eval_examples=700,
            foundation_eval_batch_size=70,
        )
        config.validate()
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "literal-foundation-gate",
            )

            def boundary(a_c1: float) -> dict[str, object]:
                return {
                    "stage_name": "literal_rehearsal",
                    "global_step": 5,
                    "examples_per_task": 700,
                    "tasks": {
                        "C0": {
                            "models": {
                                "A": {"accuracy": 0.99},
                                "D_true": {"accuracy": 1.0},
                                "D_sham": {"accuracy": 0.0},
                            }
                        },
                        "C1": {
                            "models": {
                                "A": {"accuracy": a_c1},
                                "D_true": {"accuracy": 0.99},
                                "D_sham": {"accuracy": 0.0},
                            }
                        },
                    },
                }

            self.assertTrue(trainer.compute_foundation_gate(boundary(0.98))["passed"])
            failed = trainer.compute_foundation_gate(boundary(0.979))
            self.assertFalse(failed["passed"])
            self.assertFalse(failed["conditions"]["C1:A"])

    def test_formal_config_preserves_confirmation_scale(self) -> None:
        config = load_stage1_config(
            ROOT / "configs" / "stage1-revised-formal-directml.json"
        )
        self.assertTrue(config.formal_evaluation)
        self.assertGreaterEqual(config.final_eval_examples_per_seed, 10_000)
        self.assertGreaterEqual(len(config.confirmation_training_seeds), 8)
        self.assertEqual(config.data.expression_values, 7)
        self.assertTrue(
            set(config.eval_seeds).isdisjoint({11003, 22003, 33013, 44017})
        )

    def test_literal_candidate_and_formal_configs_are_separate(self) -> None:
        candidate = load_stage1_config(
            ROOT / "configs" / "stage1-revised-literal-candidate-directml.json"
        )
        formal = load_stage1_config(
            ROOT / "configs" / "stage1-revised-literal-formal-directml.json"
        )
        self.assertEqual(candidate.operand_mode, "literal")
        self.assertFalse(candidate.formal_evaluation)
        self.assertTrue(candidate.foundation_gate_required)
        self.assertTrue(formal.formal_evaluation)
        self.assertGreaterEqual(formal.final_eval_examples_per_seed, 10_000)
        self.assertGreaterEqual(len(formal.confirmation_training_seeds), 8)
        self.assertEqual(formal.eval_seeds, (51047, 61051, 71059))
        self.assertFalse(formal.candidate_prerequisite_config_digest)
        self.assertFalse(
            formal.candidate_prerequisite_experiment_spec_digest
        )

        structural_candidate = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-structural-candidate-directml.json"
        )
        stage_steps = {
            stage.name: stage.steps for stage in structural_candidate.curriculum
        }
        self.assertEqual(structural_candidate.optimizer_steps, 8000)
        self.assertEqual(stage_steps["literal_c0"], 200)
        self.assertEqual(stage_steps["literal_c1"], 1600)
        self.assertGreater(stage_steps["literal_depth_2"], 200)
        self.assertGreater(stage_steps["literal_depth_3"], 200)
        self.assertGreater(stage_steps["literal_rehearsal"], 800)
        c1_profiles = structural_candidate.curriculum[1].profiles
        self.assertEqual(
            [profile.depth for profile in c1_profiles].count(0),
            1,
        )
        self.assertEqual(
            [profile.depth for profile in c1_profiles].count(1),
            3,
        )
        self.assertEqual(structural_candidate.model_a.embedding_dim, 32)
        self.assertEqual(structural_candidate.model_d.embedding_dim, 32)
        self.assertEqual(structural_candidate.learning_rate, 0.001)
        self.assertNotEqual(
            structural_candidate.curriculum,
            formal.curriculum,
            "formal curriculum must not mirror an unrun candidate",
        )

        posthoc_candidate = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-posthoc-revalidation-directml.json"
        )
        self.assertFalse(posthoc_candidate.formal_evaluation)
        self.assertEqual(posthoc_candidate.optimizer_steps, 8000)
        self.assertEqual(posthoc_candidate.seed, 82421)
        self.assertEqual(posthoc_candidate.eval_seeds, (92041, 92051))
        self.assertEqual(posthoc_candidate.foundation_eval_seed, 92063)
        self.assertEqual(
            posthoc_candidate.gate.baseline_policy,
            "privileged_structure_posthoc_v1",
        )
        self.assertEqual(
            formal.gate.baseline_policy,
            "joint_all_required_v1",
            "formal policy stays frozen until post-hoc candidate validation",
        )
        self.assertNotEqual(
            validated_experiment_spec_digest(formal),
            validated_experiment_spec_digest(posthoc_candidate),
            "the current formal config must remain incompatible and blocked",
        )

    def test_eight_training_seed_aggregate_is_the_only_stage2_gate(self) -> None:
        base = load_stage1_config(
            ROOT / "configs" / "stage1-revised-literal-formal-directml.json"
        )
        config = replace(
            base,
            candidate_prerequisite_config_digest="candidate-config",
            candidate_prerequisite_manifest_hash="candidate-source",
            candidate_prerequisite_snapshot_manifest_hash="candidate-snapshot",
            candidate_prerequisite_result_digest="candidate-result",
            candidate_prerequisite_experiment_spec_digest=(
                validated_experiment_spec_digest(base)
            ),
            candidate_prerequisite_result_path="candidate-result.json",
            candidate_prerequisite_compatibility_spec_digest=(
                validated_experiment_compatibility_spec_digest(base)
            ),
        )
        config.validate()
        evidence_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(evidence_temporary.cleanup)
        evidence_run_dir = Path(evidence_temporary.name) / "formal-evidence"
        evidence_snapshot = create_snapshot(
            ROOT,
            evidence_run_dir / "snapshot",
        )
        evidence_source = source_manifest(evidence_run_dir / "snapshot")
        evidence_result_path = evidence_run_dir / "result.json"

        def aggregate_checked(
            values: list[dict[str, object]],
        ) -> dict[str, object]:
            return aggregate_confirmation(
                values,
                result_paths=[evidence_result_path] * len(values),
            )

        masks = [7] * 4004 + [3] * 1001 + [2] * 2002 + [0] * 3003
        self.assertEqual(len(masks), 10010)
        results = []
        for seed in config.confirmation_training_seeds:
            run_config_object = replace(config, seed=seed)
            run_config = run_config_object.to_dict()
            split_results = {}
            for split in config.evaluation_splits:
                seed_results = {}
                for evaluation_seed in config.eval_seeds:
                    digest = f"digest-{seed}-{split.name}-{evaluation_seed}"
                    seed_results[str(evaluation_seed)] = {
                        "models": {
                            "A": {"correct": 5005},
                            "D_true": {"correct": 7007},
                            "D_sham": {"correct": 4004},
                        },
                        "content_hash_count": 10010,
                        "content_hash_digest": digest,
                        "paired_sample_data": {
                            "schema_version": 1,
                            "sample_count": 10010,
                            "content_hash_digest": digest,
                            "correctness_masks": masks,
                        },
                    }
                split_results[split.name] = {"seeds": seed_results}
            result = complete_result_skeleton(run_config_object)
            result.update(
                {
                    "config": run_config,
                    "config_digest": stage1_config_digest(run_config),
                    "candidate_prerequisite": {
                        "required": True,
                        "passed": True,
                        "expected": {
                            "config_digest": "candidate-config",
                            "manifest_hash": "candidate-source",
                            "snapshot_manifest_hash": "candidate-snapshot",
                            "result_digest": "candidate-result",
                            "experiment_spec_digest": (
                                validated_experiment_spec_digest(config)
                            ),
                            "compatibility_spec_digest": (
                                validated_experiment_compatibility_spec_digest(
                                    config
                                )
                            ),
                        },
                    },
                    "candidate_gate": {
                        "candidate_pass": True,
                        "stage2_unblocked": False,
                    },
                    "manifest": evidence_source,
                    "snapshot_manifest": evidence_snapshot,
                    "snapshot_manifest_hash": evidence_snapshot["manifest_hash"],
                    "final_evaluation": {
                        "kind": "formal_confirmation",
                        "examples_per_split_seed": 10010,
                        "evaluation_seeds": list(config.eval_seeds),
                        "splits": split_results,
                        "overlap_audit": {
                            "all_content_disjoint": True,
                            "all_shape_rules_valid": True,
                        },
                    },
                }
            )
            results.append(result)
        aggregate = aggregate_checked(results)
        self.assertTrue(aggregate["stage2_unblocked"])
        self.assertEqual(
            aggregate["statistical_plan"]["multiplicity_correction"],
            "bonferroni",
        )
        self.assertEqual(aggregate["statistical_plan"]["comparison_count"], 8)
        self.assertTrue(
            all(aggregate["statistical_conditions"].values())
        )
        self.assertFalse(aggregate_checked(results[:-1])["stage2_unblocked"])
        results[0]["validated_experiment_spec_digest"] = "wrong"
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        results[0]["validated_experiment_spec_digest"] = (
            validated_experiment_spec_digest(config)
        )
        results[0]["candidate_prerequisite"]["required"] = "true"
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        results[0]["candidate_prerequisite"]["required"] = True
        results[0]["candidate_gate"]["candidate_pass"] = False
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        results[0]["candidate_gate"]["candidate_pass"] = "false"
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        results[0]["candidate_gate"]["candidate_pass"] = True
        results[0]["run_eligible_for_aggregation"] = "true"
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        results[0]["run_eligible_for_aggregation"] = True
        results[0]["final_evaluation"]["overlap_audit"][
            "all_content_disjoint"
        ] = "false"
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        results[0]["final_evaluation"]["overlap_audit"][
            "all_content_disjoint"
        ] = True
        results[0]["global_step"] -= 1
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        results[0]["global_step"] += 1
        results[0]["schema_version"] = 2
        results[0]["checkpoint_recovery"] = {
            "current_step": config.optimizer_steps,
        }
        del results[0]["global_step"]
        del results[0]["target_steps"]
        del results[0]["run_eligible_for_aggregation"]
        legacy_aggregate = aggregate_checked(results)
        self.assertFalse(legacy_aggregate["stage2_unblocked"])
        self.assertFalse(
            legacy_aggregate["run_checks"][str(config.seed)][
                "current_result_schema"
            ]
        )
        results[0]["schema_version"] = 3
        results[0]["global_step"] = config.optimizer_steps
        results[0]["target_steps"] = config.optimizer_steps
        results[0]["run_eligible_for_aggregation"] = True
        first_split_name = config.evaluation_splits[0].name
        weak_masks = [1] * 5005 + [0] * 5005
        first_split_seeds = results[0]["final_evaluation"]["splits"][
            first_split_name
        ]["seeds"]
        for seed_result in first_split_seeds.values():
            seed_result["paired_sample_data"]["correctness_masks"] = weak_masks
            seed_result["models"]["A"]["correct"] = 5005
            seed_result["models"]["D_true"]["correct"] = 0
            seed_result["models"]["D_sham"]["correct"] = 0
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])
        for seed_result in first_split_seeds.values():
            seed_result["paired_sample_data"]["correctness_masks"] = masks
            seed_result["models"]["A"]["correct"] = 5005
            seed_result["models"]["D_true"]["correct"] = 7007
            seed_result["models"]["D_sham"]["correct"] = 4004
        del results[0]["final_evaluation"]["splits"][
            first_split_name
        ]["seeds"][str(config.eval_seeds[0])]["paired_sample_data"]
        self.assertFalse(aggregate_checked(results)["stage2_unblocked"])

    def test_resource_guard_uses_consecutive_samples_and_hysteresis(self) -> None:
        guard = ResourceGuard(85, 75, 6, 8, pressure_samples=3, recovery_samples=2)
        high = ResourceSample(cpu_percent=90, available_ram_gb=10)
        self.assertFalse(guard.observe(high))
        self.assertFalse(guard.observe(high))
        self.assertTrue(guard.observe(high))
        recovered = ResourceSample(cpu_percent=70, available_ram_gb=9)
        self.assertTrue(guard.observe(recovered))
        self.assertFalse(guard.observe(recovered))

    def test_snapshot_includes_revised_docs_and_excludes_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "project"
            destination = Path(temporary_directory) / "run" / "snapshot"
            for relative in (
                "pyproject.toml",
                "docs/research-protocol.md",
                "docs/stage1-revised-protocol-addendum.md",
                "configs/stage1-revised.json",
                "src/dynamic_hierarchy/stage1_data.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("intentional\n", encoding="utf-8")
            ignored = root / "runs" / "old.json"
            ignored.parent.mkdir(parents=True, exist_ok=True)
            ignored.write_text("generated", encoding="utf-8")
            manifest = create_snapshot(root, destination)
            self.assertIn("docs/research-protocol.md", manifest["files"])
            self.assertIn("docs/stage1-revised-protocol-addendum.md", manifest["files"])
            self.assertNotIn("runs/old.json", manifest["files"])

    def test_campaign_v2_snapshot_is_canonical_across_runs_and_worktree_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            campaign_root, config, manifest, project_root = (
                create_test_campaign(temporary_root)
            )
            checks, verified_manifest = verify_campaign_package(
                campaign_root,
                config,
            )
            self.assertTrue(all(checks.values()), checks)
            self.assertEqual(
                verified_manifest["manifest_hash"],
                manifest["manifest_hash"],
            )
            canonical_snapshot = campaign_root / "canonical-snapshot"
            canonical_worker = (
                canonical_snapshot / "scripts" / "stage1_worker.py"
            ).read_bytes()
            project_worker = project_root / "scripts" / "stage1_worker.py"
            project_worker.write_bytes(
                project_worker.read_bytes() + b"\n# later worktree change\n"
            )

            first_run = temporary_root / "stage1-formal-v2-01-test"
            second_run = temporary_root / "stage1-formal-v2-02-test"
            materialize_campaign_run(campaign_root, first_run, config)
            materialize_campaign_run(campaign_root, second_run, config)
            first_manifest = (
                first_run / "snapshot" / "snapshot-manifest.json"
            ).read_bytes()
            second_manifest = (
                second_run / "snapshot" / "snapshot-manifest.json"
            ).read_bytes()
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                json.loads(first_manifest)["manifest_hash"],
                manifest["snapshot_manifest_hash"],
            )
            self.assertEqual(
                (
                    first_run / "snapshot" / "scripts" / "stage1_worker.py"
                ).read_bytes(),
                canonical_worker,
            )
            self.assertNotEqual(project_worker.read_bytes(), canonical_worker)
            manifest_files = json.loads(first_manifest)["files"]
            self.assertIn(
                "campaign/environment-receipt.json",
                manifest_files,
            )
            self.assertIn(
                "campaign/candidate-identity.json",
                manifest_files,
            )
            for required in (
                "configs/stage1-revised-literal-formal-confirmation-v2-directml.json",
                "docs/stage1-formal-v2-campaign-plan.md",
                "scripts/run_stage1_confirmation_sequence_v2.py",
                "src/dynamic_hierarchy/stage1_campaign.py",
            ):
                self.assertIn(required, manifest_files)

    def test_campaign_v2_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            campaign_root, config, _, _ = create_test_campaign(temporary_root)
            source = (
                campaign_root
                / "canonical-snapshot"
                / "scripts"
                / "stage1_worker.py"
            )
            source.write_bytes(source.read_bytes() + b"\n# corruption\n")
            checks, _ = verify_campaign_package(campaign_root, config)
            self.assertFalse(
                checks["snapshot_manifest_file_hashes_match"]
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "canonical campaign integrity failed",
            ):
                materialize_campaign_run(
                    campaign_root,
                    temporary_root / "stage1-formal-v2-corrupt",
                    config,
                )

    def test_campaign_v2_result_must_match_canonical_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            campaign_root, config, manifest, _ = create_test_campaign(
                temporary_root
            )
            seed = config.confirmation_training_seeds[0]
            run_config = replace(config, seed=seed)
            run_dir = temporary_root / "stage1-formal-v2-result"
            materialize_campaign_run(campaign_root, run_dir, config)
            result = complete_formal_result(run_config)
            result_path = attach_manifest_evidence(result, run_dir)

            verified = confirmation_sequence_v2.verify_campaign_result(
                result_path,
                config,
                seed,
                campaign_root,
            )
            self.assertEqual(
                verified["manifest"]["manifest_hash"],
                manifest["source_manifest_hash"],
            )
            self.assertEqual(
                verified["snapshot_manifest_hash"],
                manifest["snapshot_manifest_hash"],
            )

            result["manifest"]["manifest_hash"] = "0" * 64
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(SequenceAbort, "manifest"):
                confirmation_sequence_v2.verify_campaign_result(
                    result_path,
                    config,
                    seed,
                    campaign_root,
                )

    def test_campaign_v2_seeds_are_fresh_and_legacy_queue_is_excluded(
        self,
    ) -> None:
        config_path = (
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-v2-directml.json"
        )
        config = load_stage1_config(config_path)
        self.assertEqual(
            validated_experiment_compatibility_spec_digest(config),
            config.candidate_prerequisite_compatibility_spec_digest,
        )
        legacy_state_path = (
            ROOT / "runs" / "stage1-literal-formal-sequence.json"
        )
        legacy_state = json.loads(
            legacy_state_path.read_text(encoding="utf-8")
        )
        self.assertTrue(
            set(config.confirmation_training_seeds).isdisjoint(
                legacy_state["training_seeds"]
            )
        )
        with self.assertRaisesRegex(
            SequenceAbort,
            "not an isolated campaign v2 state",
        ):
            confirmation_sequence_v2._load_state(legacy_state_path)
        self.assertNotEqual(
            confirmation_sequence_v2.DEFAULT_STATE.resolve(),
            legacy_state_path.resolve(),
        )
        self.assertTrue(
            confirmation_sequence_v2.RUN_PREFIX.startswith(
                "stage1-formal-v2-"
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs"
            configs.mkdir()
            isolated_config = configs / "formal-v2.json"
            isolated_config.write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            runs = root / "runs"
            runs.mkdir()
            (runs / "stage1-literal-formal-sequence.json").write_text(
                json.dumps(
                    {"training_seeds": legacy_state["training_seeds"]}
                ),
                encoding="utf-8",
            )
            freshness = campaign_seed_freshness(
                root,
                isolated_config,
                config,
            )
            self.assertTrue(freshness["passed"], freshness)
            self.assertEqual(freshness["registered_seed_count"], 12)
            self.assertEqual(freshness["overlap"], [])

            (configs / "historical.json").write_text(
                json.dumps(
                    {"seed": config.confirmation_training_seeds[0]}
                ),
                encoding="utf-8",
            )
            collision = campaign_seed_freshness(
                root,
                isolated_config,
                config,
            )
            self.assertFalse(collision["passed"])
            self.assertEqual(
                collision["overlap"],
                [config.confirmation_training_seeds[0]],
            )

    def test_campaign_v2_pause_classification_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            control = run_dir / "control"
            control.mkdir()
            stop = control / "STOP"
            stop.write_text("stop", encoding="utf-8")
            result = {"state": "incomplete", "reason": "user_stop"}
            self.assertTrue(
                confirmation_sequence_v2._user_stop_requires_resume(
                    result,
                    run_dir,
                )
            )
            self.assertFalse(
                confirmation_sequence_v2._user_stop_requires_resume(
                    {"state": "failed", "reason": "user_stop"},
                    run_dir,
                )
            )
            self.assertFalse(
                confirmation_sequence_v2._user_stop_requires_resume(
                    {
                        "state": "incomplete",
                        "reason": "time_budget_reached",
                    },
                    run_dir,
                )
            )
            stop.unlink()
            self.assertFalse(
                confirmation_sequence_v2._user_stop_requires_resume(
                    result,
                    run_dir,
                )
            )

    def test_campaign_v3_control_files_do_not_poison_seed_freshness(
        self,
    ) -> None:
        config_path = (
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-v3-directml.json"
        )
        config = load_stage1_config(config_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs"
            configs.mkdir()
            isolated_config = configs / "formal-v3.json"
            isolated_config.write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            fresh = campaign_seed_freshness(
                root,
                isolated_config,
                config,
            )
            self.assertTrue(fresh["passed"], fresh)
            self.assertEqual(fresh["overlap"], [])

            state_path = root / "runs" / "formal-v3-sequence.json"
            aggregate_path = root / "runs" / "formal-v3-aggregate.json"
            receipt_path = confirmation_sequence_v3._launch_receipt_path(
                state_path,
                config.confirmation_training_seeds[0],
            )
            receipt_path.parent.mkdir()
            receipt_path.write_text(
                json.dumps(
                    {"training_seed": config.confirmation_training_seeds[0]}
                ),
                encoding="utf-8",
            )
            poisoned = campaign_seed_freshness(
                root,
                isolated_config,
                config,
            )
            self.assertFalse(poisoned["passed"])
            self.assertEqual(
                poisoned["overlap"],
                [config.confirmation_training_seeds[0]],
            )
            recovered = campaign_seed_freshness(
                root,
                isolated_config,
                config,
                excluded_files=(
                    confirmation_sequence_v3._freshness_excluded_files(
                        state_path,
                        aggregate_path,
                        config,
                    )
                ),
            )
            self.assertTrue(recovered["passed"], recovered)

    def test_retired_campaign_seeds_are_historical_after_v4_completion(
        self,
    ) -> None:
        if not confirmation_sequence_v4.DEFAULT_STATE.is_file():
            self.skipTest("completed campaign-v4 evidence is absent")
        v4_state = json.loads(
            confirmation_sequence_v4.DEFAULT_STATE.read_text(encoding="utf-8")
        )
        if (
            v4_state.get("state") != "completed"
            or len(v4_state.get("runs", {})) != 8
        ):
            self.skipTest("campaign v4 is not an exact completed eight-run set")

        v2_path = (
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-v2-directml.json"
        )
        v2 = load_stage1_config(v2_path)
        v2_state: dict[str, object] = {}
        if confirmation_sequence_v2.DEFAULT_STATE.is_file():
            v2_state = json.loads(
                confirmation_sequence_v2.DEFAULT_STATE.read_text(
                    encoding="utf-8"
                )
            )
        v2_freshness = campaign_seed_freshness(
            ROOT,
            v2_path,
            v2,
            excluded_roots=(
                confirmation_sequence_v2.DEFAULT_CAMPAIGN_ROOT,
                *confirmation_sequence_v2._state_run_roots(v2_state),
            ),
            excluded_files=(
                confirmation_sequence_v2.DEFAULT_STATE,
                confirmation_sequence_v2.DEFAULT_AGGREGATE,
            ),
        )
        self.assertFalse(v2_freshness["passed"], v2_freshness)
        self.assertEqual(
            v2_freshness["overlap"],
            v2_freshness["registered_seeds"],
        )

        v3_path = (
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-v3-directml.json"
        )
        v3 = load_stage1_config(v3_path)
        v3_state: dict[str, object] = {}
        if confirmation_sequence_v3.DEFAULT_STATE.is_file():
            v3_state = json.loads(
                confirmation_sequence_v3.DEFAULT_STATE.read_text(
                    encoding="utf-8"
                )
            )
        v3_freshness = campaign_seed_freshness(
            ROOT,
            v3_path,
            v3,
            excluded_roots=(
                confirmation_sequence_v3.DEFAULT_CAMPAIGN_ROOT,
                *confirmation_sequence_v3._state_run_roots(v3_state),
            ),
            excluded_files=confirmation_sequence_v3._freshness_excluded_files(
                confirmation_sequence_v3.DEFAULT_STATE,
                confirmation_sequence_v3.DEFAULT_AGGREGATE,
                v3,
            ),
        )
        self.assertFalse(v3_freshness["passed"], v3_freshness)
        self.assertEqual(
            v3_freshness["overlap"],
            v3_freshness["registered_seeds"],
        )

    def test_campaign_v3_launch_and_resume_use_canonical_snapshot(
        self,
    ) -> None:
        seed = 971401
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "stage1-formal-v3-test"
            launcher = run_dir / "snapshot" / "scripts" / "start_stage1.ps1"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("# canonical launcher", encoding="utf-8")
            candidate = Path(temporary) / "candidate.json"
            candidate.write_text("{}", encoding="utf-8")
            receipt_path = Path(temporary) / "launch-receipt.json"
            commands: list[list[str]] = []

            def fake_launcher(
                command: list[str],
                receipt: Path,
                label: str,
            ) -> None:
                commands.append(command)
                is_resume = "-ResumeRun" in command
                receipt.write_text(
                    json.dumps(
                        {
                            "run_dir": str(run_dir.resolve()),
                            "training_seed": seed,
                            "prepared_campaign_run": True,
                            "resume": is_resume,
                            "runtime_python": str(
                                Path(sys.executable).resolve()
                            ),
                        }
                    ),
                    encoding="utf-8",
                )

            with patch.object(
                confirmation_sequence_v3,
                "_run_launcher_command",
                side_effect=fake_launcher,
            ):
                confirmation_sequence_v3._launch_prepared(
                    "configs/formal-v3.json",
                    candidate,
                    seed,
                    run_dir,
                    receipt_path,
                )
                confirmation_sequence_v3._resume_prepared(
                    run_dir,
                    receipt_path,
                    seed,
                )

            self.assertEqual(len(commands), 2)
            for command in commands:
                file_index = command.index("-File") + 1
                runtime_index = command.index("-RuntimePython") + 1
                self.assertEqual(
                    Path(command[file_index]).resolve(),
                    launcher.resolve(),
                )
                self.assertEqual(
                    Path(command[runtime_index]).resolve(),
                    Path(sys.executable).resolve(),
                )
            self.assertNotIn(str(ROOT / "scripts" / "start_stage1.ps1"), commands[0])
            with self.assertRaisesRegex(
                SequenceAbort,
                "canonical coordinator",
            ):
                confirmation_sequence_v3.run_campaign(
                    ROOT
                    / "configs"
                    / "stage1-revised-literal-formal-confirmation-v3-directml.json",
                    Path(temporary) / "campaign",
                    Path(temporary) / "state.json",
                    Path(temporary) / "aggregate.json",
                    0.01,
                )

    def test_campaign_v3_live_environment_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign_root, config, _, _ = create_test_campaign(
                Path(temporary)
            )
            checks, _ = verify_campaign_package(
                campaign_root,
                config,
                verify_live_environment=True,
            )
            self.assertTrue(all(checks.values()), checks)
            with patch.object(
                stage1_campaign_module,
                "_environment_identity",
                return_value={"project_root": "drifted"},
            ):
                drifted, _ = verify_campaign_package(
                    campaign_root,
                    config,
                    verify_live_environment=True,
                )
            self.assertFalse(drifted["live_environment_identity"])

    def test_campaign_v4_fresh_seeds_exclude_own_control_plane(self) -> None:
        config_path = (
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-v4-directml.json"
        )
        config = load_stage1_config(config_path)
        state: dict[str, object] = {}
        if confirmation_sequence_v4.DEFAULT_STATE.is_file():
            state = json.loads(
                confirmation_sequence_v4.DEFAULT_STATE.read_text(
                    encoding="utf-8"
                )
            )
        freshness = campaign_seed_freshness(
            ROOT,
            config_path,
            config,
            excluded_roots=(
                confirmation_sequence_v4.DEFAULT_CAMPAIGN_ROOT,
                *confirmation_sequence_v4._state_run_roots(state),
            ),
            excluded_files=confirmation_sequence_v4._freshness_excluded_files(
                confirmation_sequence_v4.DEFAULT_STATE,
                confirmation_sequence_v4.DEFAULT_AGGREGATE,
                config,
            ),
        )
        self.assertTrue(freshness["passed"], freshness)
        self.assertEqual(freshness["overlap"], [])
        v3 = load_stage1_config(
            ROOT
            / "configs"
            / "stage1-revised-literal-formal-confirmation-v3-directml.json"
        )
        v3_seeds = {
            *v3.confirmation_training_seeds,
            *v3.eval_seeds,
            v3.foundation_eval_seed,
        }
        self.assertTrue(
            set(freshness["registered_seeds"]).isdisjoint(v3_seeds)
        )

    def test_campaign_v4_canonical_import_roots_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "canonical-snapshot"
            create_snapshot(ROOT, snapshot)
            coordinator = (
                snapshot
                / "scripts"
                / "run_stage1_confirmation_sequence_v4.py"
            )
            environment = os.environ.copy()
            environment["DYNAMIC_HIERARCHY_PROJECT_ROOT"] = str(ROOT)
            audit_code = (
                "import runpy\n"
                f"runpy.run_path({str(coordinator)!r}, run_name='audit')\n"
                "import dynamic_hierarchy.stage1_campaign as campaign\n"
                "import scripts.run_stage1_confirmation_sequence as sequence\n"
                "print(campaign.__file__)\n"
                "print(sequence.__file__)\n"
            )
            audited = subprocess.run(
                [sys.executable, "-c", audit_code],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                audited.returncode,
                0,
                audited.stderr or audited.stdout,
            )
            module_paths = [
                Path(line).resolve()
                for line in audited.stdout.splitlines()
                if line.strip()
            ]
            self.assertEqual(len(module_paths), 2)
            module_paths[0].relative_to((snapshot / "src").resolve())
            module_paths[1].relative_to((snapshot / "scripts").resolve())

            leak_code = (
                "import dynamic_hierarchy.stage1_campaign\n"
                "import runpy\n"
                f"runpy.run_path({str(coordinator)!r}, run_name='audit')\n"
            )
            leaked = subprocess.run(
                [sys.executable, "-c", leak_code],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(leaked.returncode, 0)
            self.assertIn("imported mutable module", leaked.stderr)

    def test_snapshot_launcher_preserves_campaign_identity_on_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "stage1-formal-v3-launcher"
            snapshot = run_dir / "snapshot"
            scripts = snapshot / "scripts"
            configs = snapshot / "configs"
            scripts.mkdir(parents=True)
            configs.mkdir()
            launcher = scripts / "start_stage1.ps1"
            shutil.copy2(ROOT / "scripts" / "start_stage1.ps1", launcher)
            worker = scripts / "stage1_worker.py"
            worker.write_text(
                "import argparse\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--run-dir')\n"
                "parser.add_argument('--config')\n"
                "parser.parse_known_args()\n",
                encoding="utf-8",
            )
            (configs / "formal-v3.json").write_text("{}", encoding="utf-8")
            (run_dir / "campaign-receipt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            candidate = Path(temporary) / "candidate.json"
            candidate.write_text("{}", encoding="utf-8")
            first_receipt = Path(temporary) / "first-receipt.json"
            common = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher),
            ]
            first = subprocess.run(
                [
                    *common,
                    "-Config",
                    "configs/formal-v3.json",
                    "-TrainingSeed",
                    "971401",
                    "-CandidateResult",
                    str(candidate),
                    "-PreparedRunDir",
                    str(run_dir),
                    "-LaunchReceipt",
                    str(first_receipt),
                    "-RuntimePython",
                    str(Path(sys.executable).resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr or first.stdout)
            deadline = time.monotonic() + 10
            first_record = json.loads(first_receipt.read_text(encoding="utf-8"))
            while (
                psutil.pid_exists(first_record["launcher_pid"])
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)

            resume_receipt = Path(temporary) / "resume-receipt.json"
            resumed = subprocess.run(
                [
                    *common,
                    "-ResumeRun",
                    str(run_dir),
                    "-LaunchReceipt",
                    str(resume_receipt),
                    "-RuntimePython",
                    str(Path(sys.executable).resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                resumed.returncode,
                0,
                resumed.stderr or resumed.stdout,
            )
            receipt = json.loads(resume_receipt.read_text(encoding="utf-8"))
            self.assertTrue(receipt["prepared_campaign_run"])
            self.assertTrue(receipt["resume"])
            self.assertEqual(
                Path(receipt["runtime_python"]).resolve(),
                Path(sys.executable).resolve(),
            )
            deadline = time.monotonic() + 10
            while (
                psutil.pid_exists(receipt["launcher_pid"])
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)

    def test_worker_pid_registration_preserves_launcher_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            pid_path = run_dir / "pid.json"
            pid_path.write_text(
                '{"pid": 111, "launcher_pid": 111, "worker_pid": null, '
                '"launch_id": "launch-test", "config": "snapshot/config.json"}',
                encoding="utf-8",
            )
            register_worker_pid(
                run_dir,
                "launch-test",
                worker_pid=222,
                attempts=1,
                delay_seconds=0,
            )
            record = __import__("json").loads(pid_path.read_text(encoding="utf-8"))
            self.assertEqual(record["launcher_pid"], 111)
            self.assertEqual(record["worker_pid"], 222)

    @unittest.skipUnless(__import__("os").name == "nt", "Windows named mutex test")
    def test_per_run_mutex_rejects_concurrent_worker_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            primary = PerRunMutex(run_dir)
            primary.acquire()
            outcomes: list[str] = []

            def contend() -> None:
                contender = PerRunMutex(run_dir)
                try:
                    contender.acquire()
                except RuntimeError as error:
                    outcomes.append(str(error))
                else:
                    outcomes.append("unexpectedly acquired")
                    contender.release()

            thread = threading.Thread(target=contend)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertIn("another Stage 1 worker holds run mutex", outcomes[0])
            primary.release()
            with PerRunMutex(run_dir) as recovered:
                self.assertTrue(recovered.metadata()["owned"])


if __name__ == "__main__":
    unittest.main()
