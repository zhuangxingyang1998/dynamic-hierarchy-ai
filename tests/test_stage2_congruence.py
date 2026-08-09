from __future__ import annotations

import json
import importlib.util
import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
import dynamic_hierarchy.stage2_congruence_model as congruence_model_module

from dynamic_hierarchy.stage2_congruence_config import (
    R6_PARTITION_DIGEST,
    Stage2CongruenceConfig,
    load_stage2_congruence_config,
)
from dynamic_hierarchy.stage2_congruence_data import (
    R6_SPLIT_DIGESTS,
    StateCongruenceData,
    counterfactual_labels,
    digest,
    partner_map_receipt,
    partner_source_indices,
    shortcut_canaries,
)
from dynamic_hierarchy.stage2_congruence_model import StateCongruenceModel
from dynamic_hierarchy.stage2_congruence_runtime import (
    BRANCH_ORDER,
    Stage2CongruenceTrainer,
    atomic_write_json,
    existing_initialization_identity,
    latest_stage2_congruence_checkpoint,
    load_or_create_source_snapshot,
    sha256_file,
)
from dynamic_hierarchy.stage2_ladder_config import LadderModelSpec
from dynamic_hierarchy.stage2_ladder_data import (
    ADD_FIRST,
    SUB_FIRST,
    LadderModelInput,
)
from scripts.run_stage2_congruence import (
    _can_recover_missing_initial_checkpoint,
    _exit_code,
    _initialization_recovery_permissions,
    main as run_stage2_congruence_main,
    _result_output_path,
    _validate_run_dir,
    _validate_run_lifecycle,
)


ROOT = Path(__file__).resolve().parents[1]
INHERITED_CHECKPOINT = ROOT / (
    "runs/stage2-r5-ladder-directml-821501/checkpoints/"
    "r5-00000600-final.pt"
)


class Stage2CongruenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = StateCongruenceData()

    @staticmethod
    def _set_branch_gate(
        metrics: dict[str, object],
        *,
        semantic: bool,
        confidence: bool = True,
    ) -> None:
        pair = metrics["pair_receipt"]
        all_labels = [int(value) for value in pair["counterfactual_answers"]]
        same = [
            int(source) == int(target)
            for source, target in zip(
                pair["source_values"], pair["target_values"], strict=True
            )
        ]
        nonself = [
            keep and int(source) != int(target)
            for keep, source, target in zip(
                same,
                pair["source_indices"],
                pair["target_indices"],
                strict=True,
            )
        ]
        labels_by_name = {
            "ordinary": [
                int(pair["target_answers"][index * 49]) for index in range(49)
            ],
            "same_value": [
                label for label, keep in zip(all_labels, same, strict=True) if keep
            ],
            "nonself_same_value": [
                label
                for label, keep in zip(all_labels, nonself, strict=True)
                if keep
            ],
            "all_state_counterfactual": all_labels,
            "wrong_state_counterfactual": [
                label for label, keep in zip(all_labels, same, strict=True) if not keep
            ],
        }
        for name, labels in labels_by_name.items():
            item = metrics[name]
            predictions = (
                list(labels) if semantic else [(label + 1) % 7 for label in labels]
            )
            item["predictions"] = predictions
            item["correct"] = sum(
                prediction == label
                for prediction, label in zip(predictions, labels, strict=True)
            )
            item["accuracy"] = item["correct"] / len(labels)
            item["prediction_class_counts"] = [
                predictions.count(label) for label in range(7)
            ]
            item["target_class_counts"] = [labels.count(label) for label in range(7)]
            item["predicted_classes"] = sum(
                value > 0 for value in item["prediction_class_counts"]
            )
            item["nll"] = [0.0 if confidence else 1.0] * len(labels)
            item["cross_entropy"] = sum(item["nll"]) / len(labels)
        pair["predictions"] = list(
            metrics["all_state_counterfactual"]["predictions"]
        )
        metrics["all_state_predictions"] = list(pair["predictions"])
        metrics["matrix_digest"] = digest(
            {
                "target_indices": pair["target_indices"],
                "source_indices": pair["source_indices"],
                "pair_ids": pair["pair_ids"],
                "target_family_ids": pair["target_family_ids"],
                "source_family_ids": pair["source_family_ids"],
                "source_values": pair["source_values"],
                "target_values": pair["target_values"],
                "target_answers": pair["target_answers"],
                "predictions": pair["predictions"],
                "targets": pair["counterfactual_answers"],
            }
        )
        metrics["error_pairs"] = [
            [
                int(pair["target_indices"][index]),
                int(pair["source_indices"][index]),
                int(prediction),
                int(pair["counterfactual_answers"][index]),
                pair["pair_ids"][index],
            ]
            for index, prediction in enumerate(pair["predictions"])
            if prediction != int(pair["counterfactual_answers"][index])
        ]
        metrics["semantic_accuracy_passed"] = semantic
        metrics["confidence_passed"] = confidence
        metrics["full_gate_passed"] = semantic and confidence

    def test_calibration_config_is_fully_frozen(self) -> None:
        config = load_stage2_congruence_config(
            ROOT / "configs/stage2-r6-congruence-directml.json"
        )
        self.assertEqual(config.steps, 306)
        self.assertEqual(config.checkpoint_steps, 17)
        self.assertEqual(config.partition_digest, R6_PARTITION_DIGEST)
        with self.assertRaisesRegex(ValueError, "fully frozen"):
            replace(config, steps=305).validate()
        with self.assertRaisesRegex(ValueError, "deterministic"):
            replace(Stage2CongruenceConfig(), device="directml").validate()

    def test_exhaustive_partition_and_r5_isolation(self) -> None:
        evidence = self.data.partition_evidence()
        self.assertEqual(evidence["partition_digest"], R6_PARTITION_DIGEST)
        self.assertEqual(evidence["joint_coordinate_count"], 343)
        self.assertEqual(evidence["r5_overlap"], 0)
        for split, rows in (("train", 245), ("validation", 49), ("reserve", 49)):
            receipt = evidence["splits"][split]
            self.assertEqual(receipt["families"], rows)
            self.assertEqual(receipt["family_hash_digest"], R6_SPLIT_DIGESTS[split])
            expected = [35] * 7 if split == "train" else [7] * 7
            self.assertEqual(receipt["final_counts"], expected)
            self.assertEqual(receipt["add_intermediate_counts"], expected)
            self.assertEqual(receipt["sub_intermediate_counts"], expected)
        self.assertFalse(self.data.is_materialized(ADD_FIRST, "reserve"))
        self.assertFalse(self.data.is_materialized(SUB_FIRST, "reserve"))

    def test_shortcut_canaries_match_preregistered_counts(self) -> None:
        for query in (ADD_FIRST, SUB_FIRST):
            train = self.data.batch(query, "train")
            for split in ("validation", "reserve"):
                evaluation = self.data.batch(query, split)
                receipt = shortcut_canaries(train, evaluation)
                self.assertEqual(receipt["train_fitted"]["query-only"]["correct"], 7)
                for view in ("0", "1", "2"):
                    self.assertEqual(receipt["train_fitted"][view]["correct"], 7)
                for view in ("0-1", "0-2", "1-2"):
                    self.assertEqual(receipt["train_fitted"][view]["correct"], 0)
                    self.assertEqual(receipt["evaluation_only_leakage"][view]["correct"], 49)
                self.assertEqual(receipt["exact_solver"]["correct"], 49)

    def test_all_partner_maps_are_bijective_and_cover_nine_cycles(self) -> None:
        for query in (ADD_FIRST, SUB_FIRST):
            batch = self.data.batch(query, "train")
            labels = batch.targets.intermediate_labels[:, 0]
            for schedule in range(1, 35):
                true = partner_source_indices(batch, "congruence-true", schedule)
                mixed = partner_source_indices(batch, "mixed-counterfactual", schedule)
                self.assertEqual(torch.sort(true).values.tolist(), list(range(245)))
                self.assertEqual(torch.sort(mixed).values.tolist(), list(range(245)))
                self.assertTrue(torch.all(true != torch.arange(245)))
                self.assertTrue(torch.equal(labels[true], labels))
                self.assertTrue(torch.all(labels[mixed] != labels))
                true_receipt = partner_map_receipt(
                    batch, "congruence-true", schedule
                )
                mixed_receipt = partner_map_receipt(
                    batch, "mixed-counterfactual", schedule
                )
                self.assertEqual(true_receipt["source_use_counts"], [1] * 245)
                self.assertEqual(mixed_receipt["source_use_counts"], [1] * 245)
                self.assertEqual(sum(true_receipt["cycle_lengths"]), 245)
                self.assertEqual(sum(mixed_receipt["cycle_lengths"]), 245)
            counts = [0] * 34
            for step in range(306):
                counts[step % 34] += 1
            self.assertEqual(counts, [9] * 34)

    def test_counterfactual_targets_follow_source_value(self) -> None:
        for query in (ADD_FIRST, SUB_FIRST):
            batch = self.data.batch(query, "train")
            source = partner_source_indices(batch, "mixed-counterfactual", 1)
            source_values = batch.targets.intermediate_labels[source, 0]
            actual = counterfactual_labels(batch, torch.arange(245), source_values)
            values = batch.model_input.values
            expected = (
                (source_values - values[:, 2]) % 7
                if query == ADD_FIRST
                else (values[:, 0] + source_values) % 7
            )
            self.assertEqual(actual.tolist(), expected.tolist())
            self.assertTrue(torch.all(actual != batch.targets.final_labels))

    def test_model_accepts_only_fixed_r6_interfaces(self) -> None:
        model = StateCongruenceModel(
            LadderModelSpec(hidden_dim=16, feedforward_dim=32, dropout=0.0)
        )
        batch = self.data.batch(ADD_FIRST, "validation")
        output = model(batch.model_input)
        self.assertEqual(output.first_state.shape, (49, 16))
        self.assertEqual(output.ordinary_logits.shape, (49, 7))
        wrong = LadderModelInput(
            values=batch.model_input.values,
            operators=torch.flip(batch.model_input.operators, dims=(1,)),
            query_ids=batch.model_input.query_ids,
        )
        with self.assertRaisesRegex(ValueError, r"\+-"):
            model(wrong)
        mixed_queries = batch.model_input.query_ids.clone()
        mixed_queries[-1] = SUB_FIRST
        with self.assertRaisesRegex(ValueError, "fixed query"):
            model(
                LadderModelInput(
                    batch.model_input.values,
                    batch.model_input.operators,
                    mixed_queries,
                )
            )

    def test_matched_path_keeps_source_producer_gradient(self) -> None:
        torch.manual_seed(31)
        model = StateCongruenceModel(
            LadderModelSpec(hidden_dim=16, feedforward_dim=32, dropout=0.0)
        )
        batch = self.data.batch(ADD_FIRST, "train")
        source = partner_source_indices(batch, "congruence-true", 1)
        output = model(batch.model_input, source_indices=source)
        output.first_state.retain_grad()
        ordinary = F.cross_entropy(output.ordinary_logits, batch.targets.final_labels)
        intervention = F.cross_entropy(
            output.intervention_logits, batch.targets.final_labels
        )
        ((ordinary + intervention) / 2.0).backward()
        self.assertIsNotNone(output.first_state.grad)
        self.assertGreater(float(output.first_state.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.literal_embedding.weight.grad)
        self.assertIsNotNone(model.composer[0].weight.grad)
        self.assertIsNotNone(model.readout[0].weight.grad)
        self_output = model(
            batch.model_input,
            source_indices=partner_source_indices(batch, "self-duplicate", 1),
        )
        self.assertLessEqual(
            float(
                (self_output.ordinary_logits - self_output.intervention_logits)
                .abs()
                .max()
                .detach()
            ),
            1e-6,
        )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_validation_decision_prioritizes_semantic_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2CongruenceTrainer(
                Stage2CongruenceConfig(),
                Path(temporary),
                allow_create_run_instance=True,
            )
            metrics = {
                branch: trainer._evaluate_branch(
                    branch,
                    trainer.data.batch(
                        0 if branch.startswith("fixed-add-") else 1,
                        "validation",
                    ),
                )
                for branch in BRANCH_ORDER
            }
            for item in metrics.values():
                self._set_branch_gate(item, semantic=False)
            self._set_branch_gate(metrics["fixed-add-root"], semantic=True)
            self.assertEqual(
                trainer.validation_decision(metrics, True),
                "task_ceiling",
            )
            self._set_branch_gate(metrics["fixed-add-root"], semantic=False)
            self._set_branch_gate(
                metrics["fixed-add-congruence-true"], semantic=True
            )
            self._set_branch_gate(
                metrics["fixed-sub-congruence-true"], semantic=True
            )
            self._set_branch_gate(
                metrics["fixed-sub-self-duplicate"], semantic=True
            )
            self.assertEqual(
                trainer.validation_decision(metrics, True),
                "control_sufficient",
            )
            self._set_branch_gate(
                metrics["fixed-sub-self-duplicate"], semantic=False
            )
            self._set_branch_gate(
                metrics["fixed-add-mixed-counterfactual"], semantic=True
            )
            self.assertEqual(
                trainer.validation_decision(metrics, True),
                "valid_augmentation_non_specific",
            )
            self.assertEqual(
                trainer.validation_decision(metrics, False),
                "implementation_invalid",
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_runtime_checkpoint_gate_and_denominators(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            self.assertTrue(trainer.initialization_receipt["distinct_parameter_objects"])
            trainer.train_step()
            checks = trainer._training_receipt_checks(1)
            self.assertTrue(all(checks.values()), checks)
            self.assertEqual(
                trainer.operation_counts["fixed-add-root"],
                {
                    "first_compositions": 1,
                    "outer_compositions": 1,
                    "readouts": 1,
                    "ce_terms": 1,
                },
            )
            self.assertEqual(
                trainer.operation_counts["fixed-add-congruence-true"],
                {
                    "first_compositions": 1,
                    "outer_compositions": 2,
                    "readouts": 2,
                    "ce_terms": 2,
                },
            )
            self.assertEqual(
                trainer.partner_counts["fixed-add-congruence-true"][0], 1
            )
            self.assertEqual(
                trainer.partner_counts["fixed-sub-congruence-true"][0], 1
            )
            self.assertEqual(
                trainer.partner_counts["fixed-add-mixed-counterfactual"][0], 1
            )
            self.assertEqual(
                trainer.partner_counts["fixed-sub-mixed-counterfactual"][0], 1
            )
            checkpoint = trainer.save_checkpoint()
            restored = Stage2CongruenceTrainer(config, run_dir)
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.stage_step, 1)
            ledger = restored.run_gate()
            self.assertEqual(ledger["true_reserve_state"], "not_opened")
            self.assertFalse(restored.data.is_materialized(ADD_FIRST, "reserve"))
            self.assertFalse(restored.data.is_materialized(SUB_FIRST, "reserve"))
            branch = ledger["validation"]["fixed-add-root"]
            self.assertEqual(branch["ordinary"]["rows"], 49)
            self.assertEqual(branch["same_value"]["rows"], 343)
            self.assertEqual(branch["nonself_same_value"]["rows"], 294)
            self.assertEqual(branch["all_state_counterfactual"]["rows"], 2401)
            self.assertEqual(branch["wrong_state_counterfactual"]["rows"], 2058)
            self.assertTrue(branch["evidence_passed"])
            self.assertEqual(len(branch["pair_receipt"]["pair_ids"]), 2401)
            self.assertEqual(len(set(branch["pair_receipt"]["pair_ids"])), 2401)
            self.assertEqual(
                ledger["validation_binding"], restored._state_binding()
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_stale_validation_ledger_is_rejected(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            trainer.train_step()
            checkpoint = trainer.save_checkpoint("pre-gate")
            trainer.run_gate()
            ledger = json.loads(trainer.ledger_path.read_text(encoding="utf-8"))
            ledger["validation_binding"]["stage_step"] = 0
            atomic_write_json(trainer.ledger_path, ledger)
            restored = Stage2CongruenceTrainer(config, run_dir)
            with self.assertRaisesRegex(RuntimeError, "another model state"):
                restored.load_checkpoint(checkpoint)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_completed_resume_is_read_only_and_validation_replays_exactly(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            trainer.train_step()
            trainer.run_gate()
            checkpoint = trainer.save_checkpoint("post-gate")
            ledger_before = digest(trainer.ledger)
            restored = Stage2CongruenceTrainer(config, run_dir)
            with mock.patch.object(
                restored,
                "_evaluate_branch",
                side_effect=AssertionError("completed resume reevaluated validation"),
            ):
                restored.load_checkpoint(checkpoint)
            self.assertTrue(restored.is_complete)
            self.assertEqual(digest(restored.ledger), ledger_before)
            replay = restored.verify_completed_validation_replay()
            self.assertTrue(replay["matched"], replay)
            self.assertTrue(replay["ledger_unchanged"], replay)

    def test_legacy_or_partial_ledger_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            atomic_write_json(
                run_dir / "r6-evaluation-ledger.json",
                {
                    "schema_version": 1,
                    "packet": "DH-S2-R6-R2",
                },
            )
            with self.assertRaisesRegex(
                RuntimeError, "existing evidence|malformed"
            ):
                Stage2CongruenceTrainer(
                    Stage2CongruenceConfig(),
                    run_dir,
                    allow_create_run_instance=True,
                )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_conclusion_tamper_is_rederived_and_rejected(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            trainer.train_step()
            trainer.run_gate()
            ledger = copy.deepcopy(trainer.ledger)
            ledger["research_disposition"] = "state_congruence_signal"
            atomic_write_json(trainer.ledger_path, ledger)
            with self.assertRaisesRegex(RuntimeError, "state combination"):
                Stage2CongruenceTrainer(config, run_dir)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_metric_tamper_cannot_extend_a_pre_gate_checkpoint(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            trainer.train_step()
            checkpoint = trainer.save_checkpoint("pre-gate")
            trainer.run_gate()
            ledger = copy.deepcopy(trainer.ledger)
            self._set_branch_gate(
                ledger["validation"]["fixed-add-root"], semantic=True
            )
            ledger["validation_digest"] = digest(ledger["validation"])
            ledger["validation_disposition"] = "task_ceiling"
            ledger["research_disposition"] = "task_ceiling"
            atomic_write_json(trainer.ledger_path, ledger)
            restored = Stage2CongruenceTrainer(config, run_dir)
            with self.assertRaisesRegex(RuntimeError, "exact replay"):
                restored.load_checkpoint(checkpoint)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_foreign_and_unverified_checkpoint_paths_are_rejected(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as left_temp, tempfile.TemporaryDirectory() as right_temp:
            left_dir = Path(left_temp)
            right_dir = Path(right_temp)
            left = Stage2CongruenceTrainer(
                config, left_dir, allow_create_run_instance=True
            )
            right = Stage2CongruenceTrainer(
                config, right_dir, allow_create_run_instance=True
            )
            self.assertNotEqual(left.run_instance_digest, right.run_instance_digest)
            left.train_step()
            right.train_step()
            left_checkpoint = left.save_checkpoint("left")
            right.save_checkpoint("right")
            with self.assertRaisesRegex(RuntimeError, "verified latest"):
                right.load_checkpoint(left_checkpoint)
            receipt_path = right_dir / "checkpoints/latest.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["checkpoint"] = str(left_checkpoint.resolve())
            receipt["checkpoint_sha256"] = sha256_file(left_checkpoint)
            atomic_write_json(receipt_path, receipt)
            with self.assertRaisesRegex(RuntimeError, "run-relative"):
                latest_stage2_congruence_checkpoint(right_dir)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_illegal_stranded_state_cannot_train(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            ledger = copy.deepcopy(trainer.ledger)
            ledger["true_reserve_state"] = "reserve_stranded"
            atomic_write_json(trainer.ledger_path, ledger)
            with self.assertRaisesRegex(RuntimeError, "state combination"):
                Stage2CongruenceTrainer(config, run_dir)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_implementation_invalid_is_nonzero_and_never_opens_reserve(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2CongruenceTrainer(
                config,
                Path(temporary),
                allow_create_run_instance=True,
            )
            trainer.train_step()
            trainer.operation_counts["fixed-add-root"]["readouts"] += 1
            ledger = trainer.run_gate()
            self.assertEqual(ledger["execution_disposition"], "implementation_invalid")
            self.assertIsNone(ledger["research_disposition"])
            self.assertEqual(ledger["true_reserve_state"], "not_opened")
            self.assertEqual(_exit_code("implementation_invalid"), 2)
            self.assertFalse(trainer.data.is_materialized(ADD_FIRST, "reserve"))
            self.assertFalse(trainer.data.is_materialized(SUB_FIRST, "reserve"))

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_smoke_cannot_materialize_an_eligible_reserve(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2CongruenceTrainer(
                config,
                Path(temporary),
                allow_create_run_instance=True,
            )
            trainer.train_step()
            with mock.patch.object(
                Stage2CongruenceTrainer,
                "validation_decision",
                return_value="reserve_eligible",
            ):
                ledger = trainer.run_gate()
            self.assertEqual(ledger["true_reserve_state"], "not_opened")
            self.assertEqual(
                ledger["smoke_reserve_disposition"],
                "eligible_but_forbidden_in_smoke",
            )
            self.assertIsNone(ledger["research_disposition"])
            self.assertFalse(trainer.data.is_materialized(ADD_FIRST, "reserve"))
            self.assertFalse(trainer.data.is_materialized(SUB_FIRST, "reserve"))

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_reserve_open_crash_becomes_stranded(self) -> None:
        config = replace(
            Stage2CongruenceConfig(),
            run_kind="test-calibration",
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with mock.patch.object(
                Stage2CongruenceConfig, "validate", return_value=None
            ):
                trainer = Stage2CongruenceTrainer(
                    config, run_dir, allow_create_run_instance=True
                )
                trainer.train_step()
                checkpoint = trainer.save_checkpoint("pre-gate")
                validation = {
                    branch: trainer._evaluate_branch(
                        branch,
                        trainer.data.batch(
                            0 if branch.startswith("fixed-add-") else 1,
                            "validation",
                        ),
                    )
                    for branch in BRANCH_ORDER
                }
                for metrics in validation.values():
                    self._set_branch_gate(metrics, semantic=False)
                for branch in (
                    "fixed-add-congruence-true",
                    "fixed-sub-congruence-true",
                ):
                    self._set_branch_gate(validation[branch], semantic=True)
                ledger = copy.deepcopy(trainer.ledger)
                ledger.update(
                    {
                        "validation_state": "complete",
                        "validation": validation,
                        "validation_digest": digest(validation),
                        "validation_binding": trainer._state_binding(),
                        "validation_disposition": "reserve_eligible",
                        "invariants": {
                            **trainer._invariants(),
                            "all_validation_metrics_finite": True,
                        },
                        "true_reserve_state": "reserve_opened",
                        "execution_disposition": None,
                        "research_disposition": None,
                    }
                )
                for execution in (
                    None,
                    "completed",
                    "implementation_invalid",
                    "reserve_stranded",
                ):
                    for research in (
                        None,
                        "state_congruence_signal",
                        "state_congruence_failed",
                    ):
                        candidate = copy.deepcopy(ledger)
                        candidate["execution_disposition"] = execution
                        candidate["research_disposition"] = research
                        trainer.ledger = candidate
                        if execution is None and research is None:
                            trainer._validate_ledger_semantics()
                        else:
                            with self.assertRaises(RuntimeError):
                                trainer._validate_ledger_semantics()
                invalid = copy.deepcopy(ledger)
                invalid["execution_disposition"] = "completed"
                invalid["research_disposition"] = "state_congruence_signal"
                atomic_write_json(trainer.ledger_path, invalid)
                with self.assertRaises(RuntimeError):
                    Stage2CongruenceTrainer(config, run_dir)
                preserved = json.loads(
                    trainer.ledger_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    preserved["true_reserve_state"], "reserve_opened"
                )
                self.assertEqual(
                    preserved["execution_disposition"], "completed"
                )
                trainer.ledger = ledger
                atomic_write_json(trainer.ledger_path, ledger)
                pending = Stage2CongruenceTrainer(config, run_dir)
                self.assertFalse(pending.is_complete)
                self.assertEqual(
                    pending.ledger["true_reserve_state"], "reserve_opened"
                )
                replay_rows = {
                    branch: copy.deepcopy(metrics)
                    for branch, metrics in validation.items()
                }
                with mock.patch.object(
                    pending,
                    "_evaluate_branch",
                    side_effect=lambda branch, batch: replay_rows[branch],
                ):
                    pending.load_checkpoint(checkpoint)
                stranded = pending
                self.assertTrue(stranded.reserve_stranded)
                self.assertTrue(stranded.is_complete)
                self.assertEqual(
                    stranded.ledger["true_reserve_state"], "reserve_stranded"
                )
                self.assertIsNone(stranded.research_disposition)
                with self.assertRaisesRegex(RuntimeError, "outside the frozen stage"):
                    stranded.train_step()

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_completed_reserve_extension_requires_exact_model_replay(self) -> None:
        config = replace(Stage2CongruenceConfig(), run_kind="test-calibration")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            Stage2CongruenceConfig, "validate", return_value=None
        ):
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            trainer.train_step()
            checkpoint = trainer.save_checkpoint("pre-gate")
            validation = {
                branch: trainer._evaluate_branch(
                    branch,
                    trainer.data.batch(
                        ADD_FIRST if branch.startswith("fixed-add-") else SUB_FIRST,
                        "validation",
                    ),
                )
                for branch in BRANCH_ORDER
            }
            for metrics in validation.values():
                self._set_branch_gate(metrics, semantic=False)
            for branch in (
                "fixed-add-congruence-true",
                "fixed-sub-congruence-true",
            ):
                self._set_branch_gate(validation[branch], semantic=True)
            invariants = {
                **trainer._invariants(),
                "all_validation_metrics_finite": True,
            }
            actual_reserve = {
                branch: trainer._evaluate_branch(
                    branch,
                    trainer.data.batch(branch_query, "reserve"),
                )
                for branch, branch_query in (
                    ("fixed-add-congruence-true", ADD_FIRST),
                    ("fixed-sub-congruence-true", SUB_FIRST),
                )
            }
            tampered_reserve = copy.deepcopy(actual_reserve)
            for metrics in tampered_reserve.values():
                self._set_branch_gate(metrics, semantic=True)
            ledger = copy.deepcopy(trainer.ledger)
            ledger.update(
                {
                    "validation_state": "complete",
                    "validation": validation,
                    "validation_digest": digest(validation),
                    "validation_binding": trainer._state_binding(),
                    "validation_disposition": "reserve_eligible",
                    "invariants": invariants,
                    "true_reserve_state": "complete",
                    "true_reserve": tampered_reserve,
                    "true_reserve_digest": digest(tampered_reserve),
                    "true_reserve_binding": trainer._state_binding(),
                    "execution_disposition": "completed",
                    "research_disposition": "state_congruence_signal",
                }
            )
            atomic_write_json(trainer.ledger_path, ledger)
            restored = Stage2CongruenceTrainer(config, run_dir)

            def replay(branch, batch):
                if batch.split == "validation":
                    return copy.deepcopy(validation[branch])
                return copy.deepcopy(actual_reserve[branch])

            with mock.patch.object(
                restored, "_evaluate_branch", side_effect=replay
            ):
                with self.assertRaisesRegex(RuntimeError, "reserve failed exact replay"):
                    restored.load_checkpoint(checkpoint)

            legitimate = copy.deepcopy(ledger)
            legitimate["true_reserve"] = actual_reserve
            legitimate["true_reserve_digest"] = digest(actual_reserve)
            legitimate["research_disposition"] = trainer._reserve_research_decision(
                actual_reserve
            )
            atomic_write_json(trainer.ledger_path, legitimate)
            verified = Stage2CongruenceTrainer(config, run_dir)
            real_evaluate = verified._evaluate_branch

            def legitimate_replay(branch, batch):
                if batch.split == "validation":
                    return copy.deepcopy(validation[branch])
                return real_evaluate(branch, batch)

            before = digest(verified.ledger)
            with mock.patch.object(
                verified, "_evaluate_branch", side_effect=legitimate_replay
            ):
                verified.load_checkpoint(checkpoint)
            self.assertTrue(verified.is_complete)
            self.assertEqual(digest(verified.ledger), before)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_real_call_counter_detects_extra_readout(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2CongruenceTrainer(
                config,
                Path(temporary),
                allow_create_run_instance=True,
            )
            model = trainer.models["fixed-add-root"]
            original = model._logits

            def extra_readout(state, query):
                output = original(state, query)
                original(state, query)
                return output

            with mock.patch.object(model, "_logits", side_effect=extra_readout):
                trainer.train_step()
            self.assertEqual(
                trainer.operation_counts["fixed-add-root"]["readouts"], 2
            )
            ledger = trainer.run_gate()
            self.assertEqual(
                ledger["execution_disposition"], "implementation_invalid"
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_real_call_counter_detects_balanced_compose_drift(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2CongruenceTrainer(
                config,
                Path(temporary),
                allow_create_run_instance=True,
            )
            model = trainer.models["fixed-add-root"]
            original_outer = model.outer_logits

            def no_first(model_input):
                return model.literal_embedding(model_input.values[:, 0])

            def extra_outer(model_input, intermediate_state):
                logits = original_outer(model_input, intermediate_state)
                literals = model.literal_embedding(model_input.values)
                operators = model.operator_embedding(model_input.operators)
                prior_stage = model._active_compose_stage
                model._active_compose_stage = "outer"
                try:
                    model._compose(
                        literals[:, 0], literals[:, 1], operators[:, 0]
                    )
                finally:
                    model._active_compose_stage = prior_stage
                return logits

            with mock.patch.object(model, "first_states", side_effect=no_first), mock.patch.object(
                model, "outer_logits", side_effect=extra_outer
            ):
                trainer.train_step()
            self.assertEqual(
                trainer.operation_counts["fixed-add-root"]["first_compositions"],
                0,
            )
            self.assertEqual(
                trainer.operation_counts["fixed-add-root"]["outer_compositions"],
                2,
            )
            ledger = trainer.run_gate()
            self.assertEqual(
                ledger["execution_disposition"], "implementation_invalid"
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_orphan_checkpoint_does_not_block_replay(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            initial = trainer.save_checkpoint("initial")
            trainer.train_step()
            real_atomic_write = atomic_write_json

            def fail_latest(path, value):
                if path.name == "latest.json":
                    raise RuntimeError("injected latest publication failure")
                return real_atomic_write(path, value)

            with mock.patch(
                "dynamic_hierarchy.stage2_congruence_runtime.atomic_write_json",
                side_effect=fail_latest,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    trainer.save_checkpoint("orphan")
            self.assertEqual(
                latest_stage2_congruence_checkpoint(run_dir).resolve(),
                initial.resolve(),
            )
            restored = Stage2CongruenceTrainer(config, run_dir)
            restored.load_checkpoint(initial)
            restored.train_step()
            replay = restored.save_checkpoint("replayed")
            self.assertNotEqual(replay.name, initial.name)
            self.assertGreaterEqual(
                len(list((run_dir / "checkpoints").glob("*.pt"))), 3
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_missing_initial_checkpoint_is_recoverable_only_without_evidence(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "frozen-config.json").write_text("{}", encoding="utf-8")
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            self.assertTrue(_can_recover_missing_initial_checkpoint(run_dir))
            trainer.save_checkpoint("initial-recovery")
            self.assertFalse(_can_recover_missing_initial_checkpoint(run_dir))

    def test_fresh_manifest_creation_rejects_every_existing_evidence_kind(self) -> None:
        for name, is_directory in (
            ("status.json", False),
            ("failure.json", False),
            ("result.json", False),
            ("attempt-results", True),
            ("r6-evaluation-ledger.json", False),
            ("checkpoints", True),
            ("STOP", False),
            ("unknown-evidence.bin", False),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                path = run_dir / name
                if is_directory:
                    path.mkdir()
                else:
                    path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "existing evidence"):
                    Stage2CongruenceTrainer(
                        Stage2CongruenceConfig(),
                        run_dir,
                        allow_create_run_instance=True,
                    )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_source_snapshot_is_bound_everywhere_and_tamper_fails(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            source_digest = trainer.source_snapshot_digest
            self.assertEqual(
                trainer.run_instance["manifest"]["source_snapshot_digest"],
                source_digest,
            )
            self.assertEqual(
                trainer.ledger["source_snapshot_digest"], source_digest
            )
            checkpoint = trainer.save_checkpoint("source-bound")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            latest = json.loads(
                (run_dir / "checkpoints" / "latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["source_snapshot_digest"], source_digest)
            self.assertEqual(latest["source_snapshot_digest"], source_digest)
            self.assertEqual(
                trainer.status("running")["source_snapshot_digest"],
                source_digest,
            )
            self.assertEqual(
                trainer.result("calibration_incomplete")[
                    "source_snapshot_digest"
                ],
                source_digest,
            )
            frozen_runner = (
                run_dir / "snapshot" / "scripts" / "run_stage2_congruence.py"
            )
            frozen_runner.write_bytes(frozen_runner.read_bytes() + b"\n# tampered\n")
            with self.assertRaisesRegex(RuntimeError, "source file hash"):
                Stage2CongruenceTrainer(config, run_dir)
            self.assertEqual(existing_initialization_identity(run_dir), {})

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_source_snapshot_rejects_foreign_import_path(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            with mock.patch.object(
                congruence_model_module,
                "__file__",
                str(run_dir / "foreign" / "stage2_congruence_model.py"),
            ):
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    Stage2CongruenceTrainer(config, run_dir)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_pre_round_zero_initialization_boundaries_are_recoverable(self) -> None:
        config = Stage2CongruenceConfig()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            atomic_write_json(run_dir / "frozen-config.json", config.to_dict())
            self.assertEqual(
                _initialization_recovery_permissions(run_dir), (True, True)
            )
            trainer = Stage2CongruenceTrainer(
                config,
                run_dir,
                allow_create_source_snapshot=True,
                allow_create_run_instance=True,
            )
            trainer.save_checkpoint("initial-recovery")
            self.assertFalse(_can_recover_missing_initial_checkpoint(run_dir))

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            atomic_write_json(run_dir / "frozen-config.json", config.to_dict())
            source = load_or_create_source_snapshot(run_dir, allow_create=True)
            self.assertEqual(
                _initialization_recovery_permissions(run_dir), (False, True)
            )
            trainer = Stage2CongruenceTrainer(
                config,
                run_dir,
                allow_create_source_snapshot=False,
                allow_create_run_instance=True,
            )
            identity = existing_initialization_identity(run_dir)
            self.assertEqual(
                identity["source_snapshot_digest"], source["source_digest"]
            )
            self.assertEqual(
                identity["run_instance_digest"], trainer.run_instance_digest
            )
            self.assertTrue(_can_recover_missing_initial_checkpoint(run_dir))

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            atomic_write_json(run_dir / "frozen-config.json", config.to_dict())
            partial = run_dir / f".snapshot.{'a' * 32}.tmp"
            partial.mkdir()
            (partial / "partial.py").write_text("partial", encoding="ascii")
            self.assertEqual(
                _initialization_recovery_permissions(run_dir), (True, True)
            )
            self.assertFalse(partial.exists())

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            atomic_write_json(run_dir / "frozen-config.json", config.to_dict())
            (run_dir / "checkpoints").mkdir()
            with self.assertRaisesRegex(RuntimeError, "without a run instance"):
                _initialization_recovery_permissions(run_dir)

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_constructor_failure_status_retains_available_source_identity(self) -> None:
        config_path = ROOT / "configs" / "stage2-r6-smoke-cpu.json"
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def fail_after_snapshot(config, target, **kwargs):
                load_or_create_source_snapshot(
                    target,
                    allow_create=bool(kwargs["allow_create_source_snapshot"]),
                )
                raise RuntimeError("injected constructor failure after snapshot")

            argv = [
                "run_stage2_congruence.py",
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "scripts.run_stage2_congruence.Stage2CongruenceTrainer",
                side_effect=fail_after_snapshot,
            ):
                with self.assertRaisesRegex(RuntimeError, "after snapshot"):
                    run_stage2_congruence_main()
            failure = json.loads(
                (run_dir / "failure.json").read_text(encoding="utf-8")
            )
            status = json.loads(
                (run_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                failure["source_snapshot_digest"],
                status["source_snapshot_digest"],
            )
            self.assertNotIn("run_instance_digest", failure)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"

            def fail_after_instance(config, target, **kwargs):
                Stage2CongruenceTrainer(config, target, **kwargs)
                raise RuntimeError("injected constructor failure after instance")

            argv = [
                "run_stage2_congruence.py",
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "scripts.run_stage2_congruence.Stage2CongruenceTrainer",
                side_effect=fail_after_instance,
            ):
                with self.assertRaisesRegex(RuntimeError, "after instance"):
                    run_stage2_congruence_main()
            failure = json.loads(
                (run_dir / "failure.json").read_text(encoding="utf-8")
            )
            status = json.loads(
                (run_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                failure["source_snapshot_digest"],
                status["source_snapshot_digest"],
            )
            self.assertEqual(
                failure["run_instance_digest"],
                status["run_instance_digest"],
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_round_zero_orphan_is_recoverable_without_collision(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            real_atomic_write = atomic_write_json

            def fail_latest(path, value):
                if path.name == "latest.json":
                    raise RuntimeError("injected initial latest failure")
                return real_atomic_write(path, value)

            with mock.patch(
                "dynamic_hierarchy.stage2_congruence_runtime.atomic_write_json",
                side_effect=fail_latest,
            ):
                with self.assertRaisesRegex(RuntimeError, "initial latest"):
                    trainer.save_checkpoint("initial")
            self.assertTrue(_can_recover_missing_initial_checkpoint(run_dir))
            self.assertEqual(
                len(list((run_dir / "checkpoints").glob("*.pt"))), 1
            )
            restored = Stage2CongruenceTrainer(config, run_dir)
            recovery = restored.save_checkpoint("initial-recovery")
            self.assertTrue(recovery.is_file())
            self.assertEqual(
                latest_stage2_congruence_checkpoint(run_dir), recovery.resolve()
            )
            self.assertEqual(
                len(list((run_dir / "checkpoints").glob("*.pt"))), 2
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_gate_rebuilds_pair_identity_labels_nll_and_accuracy(self) -> None:
        config = Stage2CongruenceConfig()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2CongruenceTrainer(
                config, Path(temporary), allow_create_run_instance=True
            )
            batch = trainer.data.batch(ADD_FIRST, "validation")
            original = trainer._evaluate_branch("fixed-add-root", batch)
            trainer._derive_branch_gate(
                original, batch, config.max_cross_entropy
            )

            def refresh_matrix(metrics):
                pair = metrics["pair_receipt"]
                metrics["matrix_digest"] = digest(
                    {
                        "target_indices": pair["target_indices"],
                        "source_indices": pair["source_indices"],
                        "pair_ids": pair["pair_ids"],
                        "target_family_ids": pair["target_family_ids"],
                        "source_family_ids": pair["source_family_ids"],
                        "source_values": pair["source_values"],
                        "target_values": pair["target_values"],
                        "target_answers": pair["target_answers"],
                        "predictions": pair["predictions"],
                        "targets": pair["counterfactual_answers"],
                    }
                )

            for name, mutate in (
                (
                    "duplicate pair ID",
                    lambda item: item["pair_receipt"]["pair_ids"].__setitem__(
                        1, item["pair_receipt"]["pair_ids"][0]
                    ),
                ),
                (
                    "wrong family ID",
                    lambda item: item["pair_receipt"][
                        "source_family_ids"
                    ].__setitem__(0, "wrong-family"),
                ),
                (
                    "wrong source value",
                    lambda item: item["pair_receipt"]["source_values"].__setitem__(
                        0, (item["pair_receipt"]["source_values"][0] + 1) % 7
                    ),
                ),
                (
                    "wrong counterfactual target",
                    lambda item: item["pair_receipt"][
                        "counterfactual_answers"
                    ].__setitem__(
                        0,
                        (item["pair_receipt"]["counterfactual_answers"][0] + 1)
                        % 7,
                    ),
                ),
            ):
                with self.subTest(name=name):
                    attacked = copy.deepcopy(original)
                    mutate(attacked)
                    refresh_matrix(attacked)
                    with self.assertRaisesRegex(RuntimeError, "frozen data"):
                        trainer._derive_branch_gate(
                            attacked, batch, config.max_cross_entropy
                        )

            negative = copy.deepcopy(original)
            negative["ordinary"]["nll"][0] = -1.0
            negative["ordinary"]["cross_entropy"] = sum(
                negative["ordinary"]["nll"]
            ) / 49
            with self.assertRaisesRegex(RuntimeError, "NLL"):
                trainer._derive_branch_gate(
                    negative, batch, config.max_cross_entropy
                )

            inaccurate = copy.deepcopy(original)
            inaccurate["ordinary"]["accuracy"] = 1.0
            with self.assertRaisesRegex(RuntimeError, "accuracy"):
                trainer._derive_branch_gate(
                    inaccurate, batch, config.max_cross_entropy
                )

    @unittest.skipIf(
        importlib.util.find_spec("torch_directml") is not None,
        "full frozen-budget reconstruction runs once in the CPU suite",
    )
    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_full_306_step_budget_reconstructs_from_checkpoint(self) -> None:
        config = replace(
            Stage2CongruenceConfig(),
            steps=306,
            checkpoint_steps=17,
            time_budget_minutes=30.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            for _ in range(306):
                trainer.train_step()
            checks = trainer._training_receipt_checks(306)
            self.assertTrue(all(checks.values()), checks)
            for counts in trainer.partner_counts.values():
                self.assertEqual(counts, [9] * 34)
            checkpoint = trainer.save_checkpoint("full-budget")
            restored = Stage2CongruenceTrainer(config, run_dir)
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored._state_binding(), trainer._state_binding())
            self.assertTrue(
                all(restored._training_receipt_checks(306).values())
            )

    def test_run_directory_lifecycle_and_result_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _validate_run_lifecycle(run_dir, False)
            (run_dir / "frozen-config.json").write_text("{}", encoding="utf-8")
            (run_dir / "run-instance.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "requires --resume"):
                _validate_run_lifecycle(run_dir, False)
            _validate_run_lifecycle(run_dir, True)
            (run_dir / "result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                _validate_run_lifecycle(run_dir, False)
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                _validate_run_lifecycle(run_dir, True)
            self.assertEqual(
                _result_output_path(run_dir, "completed", 1),
                run_dir / "result.json",
            )
            self.assertEqual(
                _result_output_path(run_dir, "calibration_incomplete", 1).parent,
                run_dir / "attempt-results",
            )

    @unittest.skipUnless(INHERITED_CHECKPOINT.is_file(), "retained R5 checkpoint is local")
    def test_partial_checkpoint_can_resume_to_gate(self) -> None:
        config = replace(Stage2CongruenceConfig(), steps=2)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer = Stage2CongruenceTrainer(
                config, run_dir, allow_create_run_instance=True
            )
            trainer.train_step()
            checkpoint = trainer.save_checkpoint("incomplete")
            restored = Stage2CongruenceTrainer(config, run_dir)
            restored.load_checkpoint(checkpoint)
            self.assertFalse(restored.is_complete)
            restored.train_step()
            self.assertTrue(restored.needs_gate)
            restored.run_gate()
            self.assertTrue(restored.is_complete)

    def test_calibration_run_dir_is_canonical(self) -> None:
        config = load_stage2_congruence_config(
            ROOT / "configs/stage2-r6-congruence-directml.json"
        )
        _validate_run_dir(
            Path(config.canonical_run_dir), config.run_kind, config.canonical_run_dir
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            _validate_run_dir(
                Path("runs/not-r6"), config.run_kind, config.canonical_run_dir
            )


if __name__ == "__main__":
    unittest.main()
