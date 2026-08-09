import json
import tempfile
import unittest
from pathlib import Path

import torch

from dynamic_hierarchy.stage2_ladder_config import (
    LadderModelSpec,
    Stage2LadderConfig,
    load_stage2_ladder_config,
    stage2_ladder_config_from_dict,
)
from dynamic_hierarchy.stage2_ladder_data import (
    ADD_FIRST,
    ArithmeticLadderData,
    LadderModelInput,
    batch_evidence,
    sham_intermediate_labels,
    two_literal_lookup_accuracies,
)
from dynamic_hierarchy.stage2_ladder_model import (
    ArithmeticComposerModel,
    bridge_root_logits,
    model_state_digest,
)
from dynamic_hierarchy.stage2_ladder_runtime import Stage2LadderTrainer


ROOT = Path(__file__).resolve().parents[1]


def smoke_config(**overrides):
    raw = {
        "revision": "stage2-r5.1",
        "phase": "arithmetic_ladder",
        "run_kind": "smoke",
        "seed": 821501,
        "device": "cpu",
        "deterministic": True,
        "cpu_threads": 1,
        "rung1_steps": 1,
        "rung2_steps": 1,
        "rung3_steps": 1,
        "checkpoint_steps": 1,
        "yield_ms": 0,
        "model": {"hidden_dim": 8, "feedforward_dim": 16, "dropout": 0.0},
    }
    raw.update(overrides)
    return stage2_ladder_config_from_dict(raw)


class Stage2LadderConfigTests(unittest.TestCase):
    def test_frozen_calibration_config_loads_and_rejects_changes(self) -> None:
        config = load_stage2_ladder_config(ROOT / "configs/stage2-r5-ladder-directml.json")
        self.assertEqual(config.rung1_steps, 300)
        self.assertEqual(config.min_final_accuracy, 1.0)
        changed = config.to_dict()
        changed["rung2_steps"] = 301
        with self.assertRaisesRegex(ValueError, "fully frozen"):
            stage2_ladder_config_from_dict(changed)

    def test_unknown_config_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown R5 config fields"):
            stage2_ladder_config_from_dict({"mystery": 1})


class Stage2LadderDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = ArithmeticLadderData(821501)

    def test_domains_and_common_partition_are_exact(self) -> None:
        self.assertEqual(len(self.data.family_hashes("binary", "train")), 98)
        self.assertEqual(len(self.data.family_hashes("paired", "train")), 210)
        self.assertEqual(len(self.data.family_hashes("paired", "validation")), 42)
        self.assertEqual(len(self.data.family_hashes("paired", "reserve")), 42)
        for split in ("train", "validation", "reserve"):
            self.assertEqual(
                self.data.family_hashes("fixed-add", split),
                self.data.family_hashes("fixed-sub", split),
            )
            self.assertEqual(
                self.data.family_hashes("fixed-add", split),
                self.data.family_hashes("paired", split),
            )
        train = set(self.data.family_hashes("paired", "train"))
        validation = set(self.data.family_hashes("paired", "validation"))
        reserve = set(self.data.family_hashes("paired", "reserve"))
        self.assertFalse(train & validation or train & reserve or validation & reserve)

    def test_balancing_hash_domains_and_row_order(self) -> None:
        paired = self.data.batch("paired", "validation")
        evidence = batch_evidence(paired)
        self.assertEqual(evidence["rows"], 84)
        self.assertEqual(evidence["label_counts"], [12] * 7)
        self.assertEqual(evidence["query_counts"], [42, 42])
        self.assertEqual(evidence["query_only_lookup_accuracy"], 1 / 7)
        self.assertEqual(evidence["input_only_lookup_accuracy"], 0.5)
        self.assertEqual(paired.model_input.query_ids[:42].tolist(), [0] * 42)
        self.assertEqual(paired.model_input.query_ids[42:].tolist(), [1] * 42)
        self.assertEqual(paired.family_hashes[:42], paired.family_hashes[42:])
        for index in range(42):
            self.assertNotEqual(paired.row_hashes[index], paired.row_hashes[index + 42])

    def test_exact_solver_and_two_literal_canaries_pass_every_split(self) -> None:
        for rung in ("fixed-add", "fixed-sub", "paired"):
            train = self.data.batch(rung, "train")
            for split in ("validation", "reserve"):
                evaluated = self.data.batch(rung, split)
                self.assertEqual(batch_evidence(evaluated)["exact_solver_accuracy"], 1.0)
                canaries = two_literal_lookup_accuracies(train, evaluated)
                self.assertTrue(all(value <= 0.5 for value in canaries.values()))

    def test_sham_is_histogram_matched_and_changes_every_row(self) -> None:
        for rung in ("fixed-add", "fixed-sub", "paired"):
            batch = self.data.batch(rung, "train")
            labels = batch.targets.intermediate_labels[:, 0]
            queries = batch.model_input.query_ids
            sham = sham_intermediate_labels(labels, queries)
            self.assertTrue(torch.all(sham != labels))
            for query in torch.unique(queries).tolist():
                mask = queries == int(query)
                self.assertEqual(
                    torch.bincount(sham[mask], minlength=7).tolist(),
                    torch.bincount(labels[mask], minlength=7).tolist(),
                )

    def test_sham_mapping_uses_frozen_stable_tie_order(self) -> None:
        labels = torch.tensor([2, 0, 2, 1, 0, 1], dtype=torch.long)
        self.assertEqual(
            sham_intermediate_labels(labels).tolist(),
            [0, 1, 0, 2, 1, 2],
        )

    def test_reserve_batch_is_lazy(self) -> None:
        fresh = ArithmeticLadderData(821501)
        self.assertFalse(fresh.is_materialized("paired", "reserve"))
        fresh.batch("paired", "validation")
        self.assertFalse(fresh.is_materialized("paired", "reserve"))


class Stage2LadderModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = ArithmeticLadderData(821501)
        torch.manual_seed(7)
        self.model = ArithmeticComposerModel(
            LadderModelSpec(hidden_dim=16, feedforward_dim=32, dropout=0.0)
        )

    def test_model_accepts_only_input_without_targets(self) -> None:
        generated = self.data.batch("fixed-add", "validation")
        with self.assertRaisesRegex(TypeError, "LadderModelInput"):
            self.model(generated)
        output = self.model(generated.model_input)
        self.assertEqual(output.root_logits.shape, (42, 7))

    def test_bridge_matches_existing_b_execution_path(self) -> None:
        for rung in ("binary", "fixed-add", "fixed-sub", "paired"):
            split = "fit" if rung == "binary" else "validation"
            generated = self.data.batch(rung, split)
            with torch.no_grad():
                pure = self.model(generated.model_input).root_logits
                bridged = bridge_root_logits(self.model, generated)
            self.assertLessEqual(float((pure - bridged).abs().max()), 1e-5)

    def test_n3_executes_only_selected_query_groups(self) -> None:
        calls = []
        handle = self.model.composer.register_forward_hook(
            lambda *_: calls.append(1)
        )
        try:
            self.model(self.data.batch("fixed-add", "validation").model_input)
            self.assertEqual(len(calls), 2)
            calls.clear()
            self.model(self.data.batch("paired", "validation").model_input)
            self.assertEqual(len(calls), 4)
        finally:
            handle.remove()

    def test_teacher_and_wrong_tree_are_explicit_interventions(self) -> None:
        generated = self.data.batch("paired", "validation")
        ordinary = self.model(generated.model_input).root_logits
        teacher = self.model(
            generated.model_input,
            teacher_intermediate_labels=generated.targets.intermediate_labels[:, 0],
        ).root_logits
        opposite = self.model(
            generated.model_input,
            merge_query_ids=1 - generated.model_input.query_ids,
        ).root_logits
        self.assertFalse(torch.equal(ordinary, teacher))
        self.assertFalse(torch.equal(ordinary, opposite))


class Stage2LadderRuntimeTests(unittest.TestCase):
    @staticmethod
    def _passing_metrics(branch, batch):
        paired = branch.startswith("paired-")
        return {
            "branch": branch,
            "accuracy": 1.0,
            "cross_entropy": 0.0,
            "prediction_counts": [1] * 7,
            "predicted_classes": 7,
            "per_query_accuracy": {"add": 1.0, "sub": 1.0} if paired else {"add": 1.0},
            "paired_family_both_correct": 1.0 if paired else None,
            "intermediate_accuracy_report_only": 1.0,
            "two_literal_lookup_canaries": {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0},
            "bridge_max_abs_difference": 0.0,
            "interventions": {
                "correct_tree_accuracy": 1.0,
                "opposite_tree_accuracy": 0.0,
                "fixed_left_accuracy": 0.5,
                "fixed_right_accuracy": 0.5,
                "best_fixed_tree_accuracy": 0.5,
                "correct_minus_opposite": 1.0,
            } if paired else None,
            "data": {"exact_solver_accuracy": 1.0},
            "answer_passed": True,
            "structure_passed": True,
            "passed": True,
        }

    def test_rung1_failure_creates_no_downstream_models(self) -> None:
        config = smoke_config(
            min_final_accuracy=1.0,
            atomic_max_cross_entropy=0.0,
            required_predicted_classes=7,
            bridge_max_abs_difference=1e-5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2LadderTrainer(config, Path(temporary))
            trainer.train_step()
            trainer.run_gate()
            self.assertTrue(trainer.is_complete)
            self.assertEqual(trainer.final_disposition, "representation_fit_failed")
            self.assertEqual(set(trainer.models), {"binary-root"})

    def test_passed_rung1_clones_all_fixed_models_before_optimizers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2LadderTrainer(smoke_config(), Path(temporary))
            trainer.train_step()
            trainer.run_gate()
            self.assertEqual(trainer.current_rung, "fixed")
            self.assertEqual(set(trainer.active_branches), set(trainer.FIXED_BRANCHES))
            receipt = trainer.initialization_groups["fixed"]
            self.assertTrue(receipt["identical"])
            self.assertEqual(set(receipt["branch_digests"].values()), {trainer.rung1_state_digest})

    def test_fixed_root_failure_keeps_its_reserve_and_paired_models_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Stage2LadderTrainer(smoke_config(), Path(temporary))
            trainer.train_step()
            trainer.run_gate()
            trainer.train_step()
            evaluated = []

            def evaluate(branch, batch):
                evaluated.append((branch, batch.split))
                result = self._passing_metrics(branch, batch)
                if branch == "fixed-add-root":
                    result["answer_passed"] = False
                    result["passed"] = False
                return result

            trainer._evaluate_branch = evaluate
            trainer.run_gate()
            self.assertEqual(trainer.final_disposition, "fixed_query_failed")
            self.assertFalse(any(name.startswith("paired-") for name in trainer.models))
            self.assertNotIn(("fixed-add-root", "reserve"), evaluated)
            self.assertEqual(
                trainer.ledger["rungs"]["fixed"]["branches"]["fixed-add-root"]["reserve_state"],
                "validation_failed",
            )

    def test_opened_reserve_interruption_cannot_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = smoke_config()
            trainer = Stage2LadderTrainer(config, path)
            trainer.train_step()
            trainer.run_gate()
            trainer.train_step()

            def interrupt(branch, batch):
                if branch == "fixed-add-root" and batch.split == "reserve":
                    raise RuntimeError("simulated reserve interruption")
                return self._passing_metrics(branch, batch)

            trainer._evaluate_branch = interrupt
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                trainer.run_gate()
            raw = json.loads((path / "r5-evaluation-ledger.json").read_text())
            self.assertEqual(
                raw["rungs"]["fixed"]["branches"]["fixed-add-root"]["reserve_state"],
                "reserve_opened",
            )
            with self.assertRaisesRegex(RuntimeError, "cannot replay"):
                Stage2LadderTrainer(config, path)

    def test_paired_answer_pass_opens_reserve_and_structure_failure_is_decorative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            trainer = Stage2LadderTrainer(smoke_config(), path)
            trainer.train_step()
            trainer.run_gate()
            trainer.train_step()
            trainer._evaluate_branch = self._passing_metrics
            trainer.run_gate()
            trainer.train_step()
            evaluated = []

            def evaluate(branch, batch):
                evaluated.append((branch, batch.split))
                result = self._passing_metrics(branch, batch)
                if branch == "paired-root" and batch.split == "validation":
                    result["structure_passed"] = False
                    result["passed"] = False
                return result

            trainer._evaluate_branch = evaluate
            trainer.run_gate()
            root = trainer.ledger["rungs"]["paired"]["branches"]["paired-root"]
            self.assertIn(("paired-root", "reserve"), evaluated)
            self.assertTrue(root["reserve_admission_passed"])
            self.assertFalse(root["passed"])
            self.assertEqual(trainer.final_disposition, "structure_decorative")

    def test_completed_branch_evidence_resumes_without_reopening_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = smoke_config()
            trainer = Stage2LadderTrainer(config, path)
            trainer.train_step()
            trainer.run_gate()
            trainer.train_step()
            checkpoint = trainer.save_checkpoint("pre-gate")
            entry = {
                "state": "unopened",
                "kind": "validation_then_reserve",
                "branches": {},
            }
            trainer.ledger["rungs"]["fixed"] = entry
            trainer._evaluate_branch = self._passing_metrics
            completed = trainer._gate_branch_with_reserve(
                entry, "fixed-add-root"
            )
            self.assertEqual(completed["reserve_state"], "complete")

            restored = Stage2LadderTrainer(config, path)
            restored.load_checkpoint(checkpoint)
            evaluated = []

            def evaluate(branch, batch):
                evaluated.append((branch, batch.split))
                return self._passing_metrics(branch, batch)

            restored._evaluate_branch = evaluate
            restored.run_gate()
            self.assertFalse(
                any(branch == "fixed-add-root" for branch, _ in evaluated)
            )
            self.assertEqual(
                restored.ledger["rungs"]["fixed"]["branches"]["fixed-add-root"],
                completed,
            )

    def test_completed_binary_gate_resumes_without_reevaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = smoke_config()
            trainer = Stage2LadderTrainer(config, path)
            trainer.train_step()
            checkpoint = trainer.save_checkpoint("pre-gate")
            trainer.run_gate()
            self.assertEqual(
                trainer.ledger["rungs"]["binary"]["state"], "complete"
            )

            restored = Stage2LadderTrainer(config, path)
            restored.load_checkpoint(checkpoint)

            def unexpected(*_):
                raise AssertionError("completed binary evidence was reevaluated")

            restored._evaluate_branch = unexpected
            restored.run_gate()
            self.assertEqual(restored.current_rung, "fixed")
            self.assertEqual(set(restored.active_branches), set(restored.FIXED_BRANCHES))

    def test_checkpoint_restores_next_binary_update(self) -> None:
        config = smoke_config(rung1_steps=3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            original = Stage2LadderTrainer(config, path)
            original.train_step()
            checkpoint = original.save_checkpoint()
            restored = Stage2LadderTrainer(config, path)
            restored.load_checkpoint(checkpoint)
            expected = original.train_step()
            actual = restored.train_step()
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                self.assertAlmostEqual(expected[name], actual[name], places=6)


if __name__ == "__main__":
    unittest.main()
