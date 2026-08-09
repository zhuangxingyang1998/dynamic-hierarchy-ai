from __future__ import annotations

import json
import unittest
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from dynamic_hierarchy.data import MergeSourceReference
from dynamic_hierarchy.stage2_config import (
    Stage2ModelSpec,
    Stage2Profile,
    stage2_config_from_dict,
)
from dynamic_hierarchy.stage2_data import (
    LEGAL_LABEL_PAIRS,
    QUERY_ADD_FIRST,
    QUERY_SUB_FIRST,
    Stage2OrdinaryBatch,
    Stage2PrecedenceFamilyGenerator,
    evaluate_precedence_expression,
)
from dynamic_hierarchy.stage2_model import (
    Stage2MergeClassifier,
    Stage2RecurrentFlatBaseline,
    straight_through_select,
)
from dynamic_hierarchy.stage2_runtime import Stage2Trainer
from scripts.run_stage2 import _freeze_config


class Stage2DataTests(unittest.TestCase):
    def test_default_config_round_trips_typed_profiles(self) -> None:
        config = stage2_config_from_dict({})
        self.assertTrue(config.train_profiles)
        self.assertTrue(all(isinstance(profile, Stage2Profile) for profile in config.train_profiles))

    def test_r2_to_dict_omits_r3_only_fields_when_loaded_from_legacy_dict(self) -> None:
        legacy_r2 = {
            "revision": "stage2-r2",
            "run_kind": "smoke",
            "seed": 821101,
            "device": "cpu",
            "deterministic": True,
            "cpu_threads": 1,
            "optimizer_steps": 1,
            "learning_rate": 0.001,
            "families_per_stratum": 42,
            "max_generation_attempts_per_family": 512,
            "checkpoint_steps": 1,
            "evaluation_blocks": 1,
            "time_budget_minutes": 5.0,
            "yield_ms": 0,
            "cpu_pause_percent": 90.0,
            "cpu_resume_percent": 75.0,
            "ram_pause_gb": 4.0,
            "ram_resume_gb": 6.0,
            "pressure_samples": 3,
            "recovery_samples": 2,
            "controls": [
                "A-Q-param",
                "A-Q-flop",
                "A-recur",
                "B-query",
                "B-noQ-router",
                "B-sham",
                "F-stop",
                "F-left",
                "F-right",
                "F-add",
                "F-sub",
                "D-true",
                "D-sham",
            ],
            "train_profiles": [
                {
                    "name": "tiny_train",
                    "leaf_count": 3,
                    "operator_pattern": "-+",
                    "category": "train",
                }
            ],
            "evaluation_profiles": [
                {
                    "name": "tiny_eval",
                    "leaf_count": 3,
                    "operator_pattern": "-+",
                    "category": "in_distribution",
                }
            ],
            "model": {
                "vocab_size": 64,
                "hidden_dim": 8,
                "heads": 2,
                "layers": 1,
                "feedforward_dim": 16,
                "dropout": 0.0,
                "temperature": 1.0,
            },
            "a_param_model": {
                "vocab_size": 64,
                "hidden_dim": 8,
                "heads": 2,
                "layers": 3,
                "feedforward_dim": 16,
                "dropout": 0.0,
                "temperature": 1.0,
            },
            "a_flop_model": {
                "vocab_size": 64,
                "hidden_dim": 8,
                "heads": 2,
                "layers": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
                "temperature": 1.0,
            },
        }
        config = stage2_config_from_dict(legacy_r2)
        serialized = config.to_dict()
        self.assertEqual(config.phase, "routing")
        self.assertNotIn("phase", serialized)
        for key in (
            "feasibility_min_accuracy",
            "feasibility_max_cross_entropy",
            "routing_required_iid_accuracy",
            "routing_min_advantage_over_blind_and_sham",
            "routing_min_advantage_over_best_fixed",
            "routing_min_exact_tree_rate",
            "routing_max_query_identical_trace_rate",
            "routing_min_ood_advantage",
        ):
            self.assertNotIn(key, serialized)
        self.assertTrue(
            all("blocks" not in profile for profile in serialized["train_profiles"])
        )
        self.assertTrue(
            all("blocks" not in profile for profile in serialized["evaluation_profiles"])
        )

    def test_r3_omits_and_r4_serializes_profile_block_counts(self) -> None:
        r3 = stage2_config_from_dict({"revision": "stage2-r3"}).to_dict()
        self.assertTrue(all("blocks" not in item for item in r3["train_profiles"]))
        self.assertTrue(all("blocks" not in item for item in r3["evaluation_profiles"]))

        r4 = stage2_config_from_dict({"revision": "stage2-r4"}).to_dict()
        self.assertEqual([item["blocks"] for item in r4["train_profiles"]], [5, 35])
        self.assertEqual(
            [item["blocks"] for item in r4["evaluation_profiles"]],
            [1, 7, 1, 7],
        )

    def test_profile_rejects_precedence_insensitive_pattern(self) -> None:
        with self.assertRaisesRegex(ValueError, "precedence-insensitive"):
            Stage2Profile("bad", 4, "++-", "train").validate()
        Stage2Profile("good", 4, "-++", "train").validate()

    def test_queries_produce_different_tree_and_value(self) -> None:
        values = (1, 2, 3)
        add_value, add_tree = evaluate_precedence_expression(values, "-+", QUERY_ADD_FIRST)
        sub_value, sub_tree = evaluate_precedence_expression(values, "-+", QUERY_SUB_FIRST)
        self.assertEqual(add_value, 3)
        self.assertEqual(sub_value, 2)
        self.assertNotEqual(add_tree, sub_tree)

    def test_complete_block_is_family_paired_and_exactly_balanced(self) -> None:
        profile = Stage2Profile("block", 4, "-+-", "train")
        generated = Stage2PrecedenceFamilyGenerator(20260809).balanced_block(profile)
        ordinary = generated.ordinary
        self.assertEqual(generated.generation.accepted_families, 42)
        self.assertEqual(ordinary.token_ids.shape[0], 84)
        self.assertEqual(
            {(left, right): count for left, right, count in generated.generation.label_pair_counts},
            {pair: 1 for pair in LEGAL_LABEL_PAIRS},
        )
        for row in range(0, ordinary.token_ids.shape[0], 2):
            self.assertEqual(
                ordinary.token_ids[row, :-1].tolist(),
                ordinary.token_ids[row + 1, :-1].tolist(),
            )
            self.assertEqual(ordinary.base_family_hashes[row], ordinary.base_family_hashes[row + 1])
            self.assertNotEqual(int(ordinary.labels[row]), int(ordinary.labels[row + 1]))
            self.assertEqual(int(ordinary.query_ids[row]), QUERY_ADD_FIRST)
            self.assertEqual(int(ordinary.query_ids[row + 1]), QUERY_SUB_FIRST)
        self.assertEqual(len(set(generated.generation.family_hashes)), 42)
        self.assertEqual(len(set(ordinary.query_row_hashes)), 84)
        counterexamples = dict(generated.generation.fixed_policy_counterexample_rows)
        self.assertEqual(counterexamples["stop"], 84)
        self.assertEqual(counterexamples["add"], 42)
        self.assertEqual(counterexamples["sub"], 42)
        self.assertTrue(all(count > 0 for count in counterexamples.values()))
        for query_id in (QUERY_ADD_FIRST, QUERY_SUB_FIRST):
            query_labels = ordinary.labels[ordinary.query_ids == query_id]
            self.assertEqual(torch.bincount(query_labels, minlength=7).tolist(), [6] * 7)

    def test_base_hash_depends_on_input_not_profile_name(self) -> None:
        left = Stage2Profile("left_name", 4, "-+-", "train")
        right = Stage2Profile("right_name", 4, "-+-", "train")
        left_batch = Stage2PrecedenceFamilyGenerator(91).balanced_block(left)
        right_batch = Stage2PrecedenceFamilyGenerator(91).balanced_block(right)
        self.assertEqual(left_batch.generation.family_hashes, right_batch.generation.family_hashes)

    def test_ordinary_view_has_no_oracle_structure_or_truth(self) -> None:
        ordinary_fields = {item.name for item in fields(Stage2OrdinaryBatch)}
        self.assertNotIn("structure", ordinary_fields)
        self.assertNotIn("truth", ordinary_fields)
        self.assertNotIn("intermediate_values", ordinary_fields)

    def test_family_exclusion_happens_before_return(self) -> None:
        profile = Stage2Profile("exclude", 4, "-+-", "train")
        first = Stage2PrecedenceFamilyGenerator(71).balanced_block(profile)
        excluded = set(first.generation.family_hashes)
        second = Stage2PrecedenceFamilyGenerator(71).balanced_block(
            profile,
            excluded_family_hashes=excluded,
        )
        self.assertTrue(excluded.isdisjoint(second.generation.family_hashes))
        self.assertGreater(second.generation.excluded_family_rejections, 0)


class Stage2ModelTests(unittest.TestCase):
    @staticmethod
    def _small_batch():
        profile = Stage2Profile("model", 4, "-+-", "train")
        return Stage2PrecedenceFamilyGenerator(19).balanced_block(profile).ordinary

    def test_straight_through_selector_is_hard_forward_and_soft_backward(self) -> None:
        candidates = torch.tensor([[1.0, 2.0], [30.0, 40.0]], requires_grad=True)
        logits = torch.tensor([2.0, 1.0], requires_grad=True)
        probabilities = torch.softmax(logits, dim=0)
        selected = straight_through_select(candidates, probabilities, 0)
        self.assertTrue(torch.equal(selected.detach(), candidates[0].detach()))
        selected.sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_unselected_candidates_cannot_change_hard_forward_value(self) -> None:
        probabilities = torch.tensor([0.7, 0.2, 0.1], requires_grad=True)
        original = torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]])
        perturbed = original.clone()
        perturbed[1:] = 1000000.0
        selected = straight_through_select(original, probabilities, 0)
        changed = straight_through_select(perturbed, probabilities, 0)
        self.assertTrue(torch.equal(selected.detach(), changed.detach()))

    def test_all_fixed_policies_execute_and_report_hard_traces(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32))
        for policy in ("stop", "left", "right", "add", "sub"):
            output = model(batch, policy=policy)
            self.assertEqual(output.logits.shape, (84, 7))
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertEqual(len(output.traces), 84)
        stopped = model(batch, policy="stop")
        self.assertTrue(all(trace.stopped_early for trace in stopped.traces))
        self.assertEqual(stopped.compute.selected_compositions, 0)
        self.assertEqual(stopped.compute.recurrent_steps, 84)
        self.assertEqual(stopped.compute.candidate_compositions, 84 * 3)
        self.assertEqual(stopped.compute.candidate_scores, 84 * 4)
        self.assertEqual(stopped.compute.stop_scores, 84)

    def test_left_policy_recomputes_contiguous_adjacency(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32))
        trace = model(batch, policy="left").traces[0]
        self.assertEqual(
            [
                (step.source_start, step.source_end, step.operator_source_index)
                for step in trace.steps
            ],
            [(1, 3, 2), (1, 5, 4), (1, 7, 6)],
        )

    def test_learned_ties_choose_lowest_stable_merge_action(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32))
        with torch.no_grad():
            for module in (model.router, model.stop_router):
                for parameter in module.parameters():
                    parameter.zero_()
        output = model(batch, policy="learned")
        self.assertTrue(
            all(
                all(step.merge_index == 0 for step in trace.steps)
                for trace in output.traces
            )
        )

    def test_short_merge_budget_fails_closed(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32))
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            model(batch, policy="left", merge_budget=1)

    def test_query_blind_router_has_identical_pair_traces(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32))
        output = model(batch, policy="learned", router_query_mode="blind")
        for row in range(0, len(output.traces), 2):
            left = tuple((step.action, step.operator_source_index) for step in output.traces[row].steps)
            right = tuple(
                (step.action, step.operator_source_index) for step in output.traces[row + 1].steps
            )
            self.assertEqual(left, right)

    def test_router_receives_gradients_on_nonstopping_path(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32))
        with torch.no_grad():
            model.stop_router[-1].weight.zero_()
            model.stop_router[-1].bias.fill_(-20.0)
        output = model(batch, policy="learned")
        loss = torch.nn.functional.cross_entropy(output.logits, batch.labels)
        loss.backward()
        router_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.router.parameters()
            if parameter.grad is not None
        )
        composer_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.composer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(router_grad, 0.0)
        self.assertGreater(composer_grad, 0.0)

    def test_compute_only_recurrence_has_trainable_hard_halting(self) -> None:
        batch = self._small_batch()
        model = Stage2RecurrentFlatBaseline(
            Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32)
        )
        with torch.no_grad():
            model.halting_router[-1].weight.zero_()
            model.halting_router[-1].bias.copy_(torch.tensor([0.0, 1.0]))
        output = model(batch)
        self.assertEqual(output.logits.shape, (84, 7))
        self.assertEqual(output.recurrent_steps, 84)
        self.assertEqual(output.early_stops, 84)
        torch.nn.functional.cross_entropy(output.logits, batch.labels).backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.halting_router.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_r3_learned_b_has_no_stop_router_and_always_reaches_root(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(
            Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32),
            allow_stop=False,
        )
        self.assertFalse(hasattr(model, "stop_router"))
        output = model(batch, policy="learned")
        self.assertEqual(output.compute.stop_scores, 0)
        self.assertTrue(all(not trace.stopped_early for trace in output.traces))
        self.assertTrue(all(trace.reached_root for trace in output.traces))
        loss = torch.nn.functional.cross_entropy(output.logits, batch.labels)
        loss.backward()
        router_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.router.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(router_grad, 0.0)

    def test_r3_oracle_trace_matches_source_only_tree_with_selected_only_compute(self) -> None:
        profile = Stage2Profile("oracle", 4, "-+-", "train")
        generated = Stage2PrecedenceFamilyGenerator(23).balanced_block(profile)
        batch = generated.ordinary
        model = Stage2MergeClassifier(
            Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32),
            allow_stop=False,
        )
        output = model(
            batch,
            policy="oracle",
            oracle_structure=generated.diagnostic_structure,
            forced_compute_mode="selected_only",
        )
        def oracle_edges(row: int) -> set[tuple[int, int, int]]:
            result: set[tuple[int, int, int]] = set()
            spans: dict[int, tuple[int, int]] = {}
            for node in generated.diagnostic_structure.samples[row].nodes:
                if isinstance(node, MergeSourceReference):
                    left = spans[node.left]
                    right = spans[node.right]
                    result.add((min(left[0], right[0]), max(left[1], right[1]), node.operator_source_index))
                    spans[node.node_id] = (min(left[0], right[0]), max(left[1], right[1]))
                else:
                    spans[node.node_id] = (node.source_index, node.source_index)
            return result
        for row, trace in enumerate(output.traces):
            predicted = {
                (step.source_start, step.source_end, step.operator_source_index)
                for step in trace.steps
                if step.action == "MERGE"
            }
            self.assertEqual(predicted, oracle_edges(row))
        rows = batch.token_ids.shape[0]
        merges_per_row = batch.literal_source_indices.shape[1] - 1
        self.assertEqual(output.compute.stop_scores, 0)
        self.assertEqual(output.compute.candidate_scores, 0)
        self.assertEqual(output.compute.selected_compositions, rows * merges_per_row)
        self.assertEqual(output.compute.candidate_compositions, rows * merges_per_row)
        with self.assertRaisesRegex(TypeError, "StructureOnlyBatch"):
            model(batch, policy="oracle", oracle_structure=object())

    def test_fixed_candidate_matched_compute_keeps_all_candidates_explicit(self) -> None:
        batch = self._small_batch()
        model = Stage2MergeClassifier(
            Stage2ModelSpec(hidden_dim=16, heads=4, feedforward_dim=32),
            allow_stop=False,
        )
        output = model(batch, policy="left", forced_compute_mode="candidate_matched")
        rows = batch.token_ids.shape[0]
        self.assertEqual(output.compute.selected_compositions, rows * 3)
        self.assertEqual(output.compute.candidate_compositions, rows * (3 + 2 + 1))
        self.assertEqual(output.compute.candidate_scores, rows * (3 + 2 + 1))
        self.assertEqual(output.compute.stop_scores, 0)
        self.assertGreater(output.compute.candidate_compositions, output.compute.selected_compositions)


class Stage2RuntimeTests(unittest.TestCase):
    @staticmethod
    def _config():
        model = {
            "vocab_size": 64,
            "hidden_dim": 8,
            "heads": 2,
            "layers": 1,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "temperature": 1.0,
        }
        return stage2_config_from_dict(
            {
                "optimizer_steps": 2,
                "cpu_threads": 1,
                "yield_ms": 0,
                "model": model,
                "a_param_model": {**model, "layers": 3},
                "a_flop_model": {**model, "layers": 2},
                "train_profiles": [
                    {
                        "name": "tiny_train",
                        "leaf_count": 3,
                        "operator_pattern": "-+",
                        "category": "train",
                    }
                ],
                "evaluation_profiles": [
                    {
                        "name": "tiny_eval",
                        "leaf_count": 3,
                        "operator_pattern": "-+",
                        "category": "in_distribution",
                    }
                ],
            }
        )

    @staticmethod
    def _r3_feasibility_config():
        model = {
            "vocab_size": 64,
            "hidden_dim": 8,
            "heads": 2,
            "layers": 1,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "temperature": 1.0,
        }
        return stage2_config_from_dict(
            {
                "revision": "stage2-r3",
                "phase": "feasibility",
                "optimizer_steps": 3,
                "cpu_threads": 1,
                "yield_ms": 0,
                "model": model,
                "a_param_model": {**model, "layers": 3},
            }
        )

    @staticmethod
    def _r4_feasibility_config():
        model = {
            "vocab_size": 64,
            "hidden_dim": 8,
            "heads": 2,
            "layers": 1,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "temperature": 1.0,
        }
        return stage2_config_from_dict(
            {
                "revision": "stage2-r4",
                "phase": "feasibility",
                "optimizer_steps": 3,
                "cpu_threads": 1,
                "yield_ms": 0,
                "model": model,
                "a_param_model": {**model, "layers": 3},
            }
        )

    def test_complete_control_matrix_evaluates_without_family_leakage(self) -> None:
        with TemporaryDirectory() as temporary:
            trainer = Stage2Trainer(self._config(), Path(temporary))
            losses = trainer.train_step()
            evaluation = trainer.evaluate()
            self.assertEqual(set(losses), set(trainer.models))
            self.assertEqual(
                set(evaluation["profiles"]["tiny_eval"]["controls"]),
                set(trainer.config.controls),
            )
            self.assertEqual(evaluation["train_evaluation_overlap"], 0)
            controls = evaluation["profiles"]["tiny_eval"]["controls"]
            self.assertEqual(controls["F-add"]["structure"]["exact_tree_rate"], 0.5)
            self.assertEqual(controls["F-sub"]["structure"]["exact_tree_rate"], 0.5)
            self.assertEqual(controls["F-stop"]["structure"]["exact_tree_rate"], 0.0)
            self.assertEqual(
                evaluation["profiles"]["tiny_eval"]["canaries"]["query_only_lookup_accuracy"],
                1 / 7,
            )

    def test_frozen_config_resume_uses_json_canonical_types(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _freeze_config(run_dir, self._config().to_dict())
            _freeze_config(run_dir, self._config().to_dict())

    def test_checkpoint_restores_next_dataset_position_and_optimizer_state(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            original = Stage2Trainer(self._config(), run_dir)
            original.train_step()
            checkpoint = original.save_checkpoint()
            restored = Stage2Trainer(self._config(), run_dir)
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.global_step, 1)
            self.assertEqual(restored.training_family_hashes, original.training_family_hashes)
            original_losses = original.train_step()
            restored_losses = restored.train_step()
            for name in original_losses:
                self.assertAlmostEqual(original_losses[name], restored_losses[name], places=6)

    def test_r3_fixed_pools_are_reused_and_checkpoint_restore_keeps_next_update(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            original = Stage2Trainer(self._r3_feasibility_config(), run_dir)
            self.assertEqual(set(original.fixed_train_pools), {"r3_train_n3", "r3_train_n4"})
            self.assertEqual(len(original.training_family_hashes), 84)
            first = original.fixed_train_pool_evidence
            self.assertTrue(all("profile" not in item for item in first.values()))
            original.train_step()
            self.assertEqual(original.training_family_exposures, 42)
            self.assertEqual(original.training_repeated_family_exposures, 0)
            checkpoint = original.save_checkpoint()
            restored = Stage2Trainer(self._r3_feasibility_config(), run_dir)
            self.assertEqual(restored.fixed_train_pool_evidence, first)
            restored.load_checkpoint(checkpoint)
            original_losses = original.train_step()
            restored_losses = restored.train_step()
            for name in original_losses:
                self.assertAlmostEqual(original_losses[name], restored_losses[name], places=6)
            original_losses = original.train_step()
            restored_losses = restored.train_step()
            for name in original_losses:
                self.assertAlmostEqual(original_losses[name], restored_losses[name], places=6)
            self.assertEqual(original.training_family_exposures, 126)
            self.assertEqual(original.training_repeated_family_exposures, 42)
            self.assertEqual(
                restored.training_repeated_family_exposures,
                original.training_repeated_family_exposures,
            )

    def test_r3_feasibility_gate_passes_and_fails_closed(self) -> None:
        trainer = Stage2Trainer(self._r3_feasibility_config(), Path("."))
        trainer.latest_evaluation = {
            "profiles": {
                "r3_eval_n3": {
                    "controls": {
                        "B-oracle": {
                            "accuracy": 0.5,
                            "cross_entropy": 1.5,
                            "prediction_counts": [1] * 7,
                        },
                        "D-true": {
                            "accuracy": 0.6,
                            "cross_entropy": 1.4,
                            "prediction_counts": [2] * 7,
                        },
                    }
                },
                "r3_eval_n4": {
                    "controls": {
                        "B-oracle": {
                            "accuracy": 0.7,
                            "cross_entropy": 1.0,
                            "prediction_counts": [3] * 7,
                        },
                        "D-true": {
                            "accuracy": 0.8,
                            "cross_entropy": 0.9,
                            "prediction_counts": [4] * 7,
                        },
                    }
                },
            }
        }
        gate = trainer._feasibility_gate()
        self.assertTrue(gate["passed"])
        trainer.latest_evaluation["profiles"]["r3_eval_n4"]["controls"]["D-true"]["cross_entropy"] = float("inf")
        failed = trainer._feasibility_gate()
        self.assertFalse(failed["passed"])
        self.assertIn("r3_eval_n4:D-true:gate_failed", failed["failures"])
        trainer.latest_evaluation["profiles"]["r3_eval_n4"]["controls"]["D-true"][
            "prediction_counts"
        ] = [1] * 8
        malformed = trainer._feasibility_gate()
        self.assertFalse(malformed["passed"])

    def test_r4_fixed_partition_schedule_and_one_epoch_are_exact(self) -> None:
        with TemporaryDirectory() as temporary:
            trainer = Stage2Trainer(self._r4_feasibility_config(), Path(temporary))
            self.assertEqual(len(trainer.fixed_train_pools), 40)
            self.assertEqual(len(trainer.fixed_train_schedule), 40)
            self.assertEqual(len(set(trainer.fixed_train_schedule)), 40)
            self.assertEqual(len(trainer.training_family_hashes), 1680)
            profile_blocks = {"r4_train_n3": 0, "r4_train_n4": 0}
            for generated in trainer.fixed_train_pools.values():
                profile_blocks[generated.ordinary.profile_name] += 1
                self.assertEqual(generated.generation.accepted_families, 42)
                self.assertEqual(len(generated.ordinary.query_row_hashes), 84)
                self.assertTrue(
                    all(count == 1 for _, _, count in generated.generation.label_pair_counts)
                )
            self.assertEqual(profile_blocks, {"r4_train_n3": 5, "r4_train_n4": 35})

            for step in range(40):
                trainer.global_step = step
                trainer._training_batch_for_step()
            self.assertEqual(trainer.training_family_exposures, 1680)
            self.assertEqual(trainer.training_repeated_family_exposures, 0)
            self.assertEqual(set(trainer.training_family_exposure_counts.values()), {1})

    def test_r4_checkpoint_restores_schedule_position_and_exposure_counts(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            original = Stage2Trainer(self._r4_feasibility_config(), run_dir)
            original.train_step()
            checkpoint = original.save_checkpoint()
            restored = Stage2Trainer(self._r4_feasibility_config(), run_dir)
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.fixed_train_schedule, original.fixed_train_schedule)
            self.assertEqual(
                restored.training_family_exposure_counts,
                original.training_family_exposure_counts,
            )
            original_losses = original.train_step()
            restored_losses = restored.train_step()
            for name in original_losses:
                self.assertAlmostEqual(original_losses[name], restored_losses[name], places=6)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            first_hash = next(iter(payload["training_family_exposure_counts"]))
            payload["training_family_exposure_counts"][first_hash] += 1
            tampered = run_dir / "tampered.pt"
            torch.save(payload, tampered)
            with self.assertRaisesRegex(RuntimeError, "exposure counts"):
                restored.load_checkpoint(tampered)

    @staticmethod
    def _synthetic_feasibility_profile(generated, *, passing: bool):
        accuracy = 0.75 if passing else 0.25
        cross_entropy = 1.0 if passing else 2.0
        controls = {
            name: {
                "accuracy": accuracy,
                "cross_entropy": cross_entropy,
                "prediction_counts": [1] * 7,
            }
            for name in ("A-Q-param", "B-oracle", "D-true")
        }
        controls["B-oracle"].update(
            {
                "selection_path": "forced_selected_only",
                "structure": {
                    "exact_tree_rate": 1.0,
                    "edge_f1": 1.0,
                    "full_reduction_rate": 1.0,
                    "immediate_stop_rate": 0.0,
                    "early_stop_rate": 0.0,
                },
                "compute": {
                    "stop_scores": 0,
                    "candidate_compositions": 1,
                    "selected_compositions": 1,
                    "unselected_candidate_compositions": 0,
                },
            }
        )
        return {
            "profile": generated.ordinary.profile_name,
            "controls": controls,
        }

    def test_r4_validation_failure_keeps_reserve_unopened(self) -> None:
        with TemporaryDirectory() as temporary:
            trainer = Stage2Trainer(self._r4_feasibility_config(), Path(temporary))
            evaluated: list[str] = []

            def failing(generated):
                evaluated.append(generated.ordinary.profile_name)
                return self._synthetic_feasibility_profile(generated, passing=False)

            trainer._evaluate_profile = failing
            result = trainer.evaluate()
            self.assertEqual(evaluated, ["r4_validation_n3", "r4_validation_n4"])
            self.assertFalse(result["reserve_opened"])
            self.assertEqual(result["reserve_family_hash_count"], 0)
            self.assertEqual(len(trainer.evaluation_family_hashes), 336)
            self.assertFalse(result["gate"]["passed"])
            repeated = trainer.evaluate()
            self.assertEqual(
                json.loads(json.dumps(repeated)), json.loads(json.dumps(result))
            )
            self.assertEqual(evaluated, ["r4_validation_n3", "r4_validation_n4"])

    def test_r4_passing_validation_opens_complete_disjoint_reserve(self) -> None:
        with TemporaryDirectory() as temporary:
            trainer = Stage2Trainer(self._r4_feasibility_config(), Path(temporary))
            evaluated: list[str] = []

            def passing(generated):
                evaluated.append(generated.ordinary.profile_name)
                return self._synthetic_feasibility_profile(generated, passing=True)

            trainer._evaluate_profile = passing
            result = trainer.evaluate()
            self.assertEqual(
                evaluated,
                [
                    "r4_validation_n3",
                    "r4_validation_n4",
                    "r4_reserve_n3",
                    "r4_reserve_n4",
                ],
            )
            self.assertTrue(result["reserve_opened"])
            self.assertEqual(result["validation_family_hash_count"], 336)
            self.assertEqual(result["reserve_family_hash_count"], 336)
            self.assertEqual(len(trainer.evaluation_family_hashes), 672)
            self.assertEqual(
                len(
                    trainer.training_family_hashes
                    | trainer.validation_family_hashes
                    | trainer.reserve_family_hashes
                ),
                2352,
            )
            self.assertTrue(result["gate"]["passed"])
            repeated = trainer.evaluate()
            self.assertEqual(
                json.loads(json.dumps(repeated)), json.loads(json.dumps(result))
            )
            self.assertEqual(len(evaluated), 4)

    def test_r4_interrupted_open_reserve_cannot_be_replayed(self) -> None:
        with TemporaryDirectory() as temporary:
            trainer = Stage2Trainer(self._r4_feasibility_config(), Path(temporary))
            trainer.reserve_opened = True
            trainer._write_r4_evaluation_ledger("reserve_opened")
            with self.assertRaisesRegex(RuntimeError, "cannot be replayed"):
                trainer.evaluate()

    def test_r4_gate_requires_causal_oracle_structure_receipts(self) -> None:
        with TemporaryDirectory() as temporary:
            trainer = Stage2Trainer(self._r4_feasibility_config(), Path(temporary))
            generated = next(iter(trainer.fixed_train_pools.values()))
            profile_result = self._synthetic_feasibility_profile(
                generated, passing=True
            )
            profile_result["controls"]["B-oracle"]["structure"][
                "exact_tree_rate"
            ] = 0.0
            trainer.latest_evaluation = {
                "profiles": {
                    profile.name: {
                        **profile_result,
                        "profile": profile.name,
                    }
                    for profile in trainer.config.evaluation_profiles
                    if profile.category == "validation"
                }
            }
            validation_profiles = tuple(
                profile
                for profile in trainer.config.evaluation_profiles
                if profile.category == "validation"
            )
            gate = trainer._feasibility_gate(validation_profiles)
            self.assertFalse(gate["passed"])
            self.assertFalse(
                gate["profiles"]["r4_validation_n3"]["B-oracle"][
                    "causal_structure_valid"
                ]
            )

    def test_r3_thresholds_are_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            stage2_config_from_dict(
                {
                    "revision": "stage2-r3",
                    "phase": "feasibility",
                    "feasibility_min_accuracy": 0.49,
                }
            )

    def test_r4_calibration_contract_and_thresholds_are_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 600"):
            stage2_config_from_dict(
                {
                    "revision": "stage2-r4",
                    "run_kind": "calibration_only",
                    "optimizer_steps": 599,
                }
            )
        with self.assertRaisesRegex(ValueError, "frozen"):
            stage2_config_from_dict(
                {
                    "revision": "stage2-r4",
                    "feasibility_max_cross_entropy": 1.51,
                }
            )
        with self.assertRaisesRegex(ValueError, "DirectML"):
            stage2_config_from_dict(
                {
                    "revision": "stage2-r4",
                    "run_kind": "calibration_only",
                    "seed": 821401,
                    "optimizer_steps": 600,
                    "device": "cpu",
                    "deterministic": True,
                    "checkpoint_steps": 25,
                    "time_budget_minutes": 30,
                    "yield_ms": 2,
                }
            )
        with self.assertRaisesRegex(ValueError, "learning rate"):
            stage2_config_from_dict(
                {
                    "revision": "stage2-r4",
                    "run_kind": "calibration_only",
                    "seed": 821401,
                    "optimizer_steps": 600,
                    "device": "directml",
                    "deterministic": False,
                    "learning_rate": 0.123,
                    "checkpoint_steps": 25,
                    "time_budget_minutes": 30,
                    "yield_ms": 2,
                }
            )

    def test_r3_routing_train_step_fails_closed(self) -> None:
        model = {
            "vocab_size": 64,
            "hidden_dim": 8,
            "heads": 2,
            "layers": 1,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "temperature": 1.0,
        }
        config = stage2_config_from_dict(
            {
                "revision": "stage2-r3",
                "phase": "routing",
                "optimizer_steps": 1,
                "cpu_threads": 1,
                "yield_ms": 0,
                "model": model,
                "a_param_model": {**model, "layers": 3},
                "a_flop_model": {**model, "layers": 2},
                "controls": [
                    "A-Q-param",
                    "A-Q-flop",
                    "A-recur",
                    "B-query",
                    "B-noQ-router",
                    "B-sham",
                    "B-oracle",
                    "F-stop",
                    "F-left",
                    "F-right",
                    "F-add",
                    "F-sub",
                    "D-true",
                    "D-sham",
                ],
            }
        )
        with TemporaryDirectory() as temporary:
            trainer = Stage2Trainer(config, Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "ReadyForRouting"):
                trainer.train_step()


if __name__ == "__main__":
    unittest.main()
