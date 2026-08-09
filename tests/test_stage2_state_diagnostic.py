import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from dynamic_hierarchy.stage2_ladder_config import LadderModelSpec
from dynamic_hierarchy.stage2_ladder_data import ArithmeticLadderData
from dynamic_hierarchy.stage2_ladder_model import ArithmeticComposerModel
from dynamic_hierarchy.stage2_state_diagnostic import (
    analyze_branch,
    counterfactual_labels,
    replay_fixed,
    same_label_permutation,
    sham_state_permutation,
)
from scripts.run_stage2_state_diagnostic import _validate_output_path


class Stage2StateDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.data = ArithmeticLadderData(821501)
        self.model = ArithmeticComposerModel(
            LadderModelSpec(hidden_dim=16, feedforward_dim=32, dropout=0.0)
        )

    def test_replay_matches_ordinary_and_teacher_paths(self) -> None:
        for rung in ("fixed-add", "fixed-sub"):
            batch = self.data.batch(rung, "validation")
            ordinary = self.model(batch.model_input)
            replay = replay_fixed(self.model, batch)
            self.assertLessEqual(
                float(
                    (ordinary.root_logits - replay.root_logits)
                    .abs()
                    .max()
                    .detach()
                ),
                1e-6,
            )
            labels = batch.targets.intermediate_labels[:, 0]
            teacher = self.model(
                batch.model_input, teacher_intermediate_labels=labels
            )
            canonical = replay_fixed(
                self.model,
                batch,
                intermediate_state=self.model.literal_embedding(labels),
            )
            self.assertLessEqual(
                float(
                    (teacher.root_logits - canonical.root_logits)
                    .abs()
                    .max()
                    .detach()
                ),
                1e-6,
            )

    def test_same_label_and_sham_transplants_are_bijective(self) -> None:
        for rung in ("fixed-add", "fixed-sub"):
            batch = self.data.batch(rung, "validation")
            labels = batch.targets.intermediate_labels[:, 0]
            expected = torch.arange(len(labels))
            same = same_label_permutation(labels)
            self.assertEqual(torch.sort(same).values.tolist(), expected.tolist())
            self.assertTrue(torch.all(same != expected))
            self.assertTrue(torch.equal(labels[same], labels))
            sham_labels, sham = sham_state_permutation(
                labels, batch.model_input.query_ids
            )
            self.assertEqual(torch.sort(sham).values.tolist(), expected.tolist())
            self.assertTrue(torch.all(sham_labels != labels))
            self.assertTrue(torch.equal(labels[sham], sham_labels))
            self.assertEqual(
                torch.bincount(sham_labels, minlength=7).tolist(),
                torch.bincount(labels, minlength=7).tolist(),
            )

    def test_counterfactual_labels_use_the_substituted_value(self) -> None:
        for rung in ("fixed-add", "fixed-sub"):
            batch = self.data.batch(rung, "validation")
            intermediate = (batch.targets.intermediate_labels[:, 0] + 1) % 7
            actual = counterfactual_labels(batch, intermediate)
            values = batch.model_input.values
            if rung == "fixed-add":
                expected = (values[:, 0] - intermediate) % 7
            else:
                expected = (intermediate + values[:, 2]) % 7
            self.assertEqual(actual.tolist(), expected.tolist())

    def test_branch_analysis_is_finite_and_does_not_materialize_reserve(self) -> None:
        batch = self.data.batch("fixed-add", "validation")
        ordinary = self.model(batch.model_input).root_logits
        ledger = {
            "accuracy": float(
                (
                    ordinary.argmax(dim=-1) == batch.targets.final_labels
                ).float().mean().item()
            ),
            "cross_entropy": float(
                F.cross_entropy(ordinary, batch.targets.final_labels).item()
            ),
        }
        result = analyze_branch(
            "fixed-add-root", self.model, batch, ledger
        )
        self.assertTrue(all(result["invariants"].values()))
        self.assertEqual(result["rows"], 42)
        for transition in result["transitions_from_learned"].values():
            self.assertEqual(
                transition["wrong_to_correct_rows"]
                + transition["correct_to_wrong_rows"]
                + transition["stable_correct_rows"]
                + transition["stable_wrong_rows"],
                42,
            )
        self.assertFalse(self.data.is_materialized("fixed-add", "reserve"))
        self.assertFalse(self.data.is_materialized("fixed-sub", "reserve"))

    def test_output_must_be_new_and_outside_canonical_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                _validate_output_path(run, run / "diagnostic.json")
            output = root / "diagnostic.json"
            _validate_output_path(run, output)
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                _validate_output_path(run, output)


if __name__ == "__main__":
    unittest.main()
