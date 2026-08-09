from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

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


if __name__ == "__main__":
    unittest.main()
