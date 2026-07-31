from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from dynamic_hierarchy.config import DataConfig, ExperimentConfig, ModelConfig
from dynamic_hierarchy.data import SyntheticTaskGenerator
from dynamic_hierarchy.hierarchy import CandidateHierarchyController
from dynamic_hierarchy.model import SmallTransformerBaseline
from dynamic_hierarchy.provenance import source_manifest
from dynamic_hierarchy.optim import DirectMLCompatibleAdamWCore
from dynamic_hierarchy.reporting import fallback_observability, summarize_measurements
from dynamic_hierarchy.training import train


class StageZeroTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_config = DataConfig(vocab_size=32, repeat_length=5, binding_pairs=3)

    def test_data_is_reproducible(self) -> None:
        left = SyntheticTaskGenerator(self.data_config, seed=7).batch("repeat_symbol", 4)
        right = SyntheticTaskGenerator(self.data_config, seed=7).batch("repeat_symbol", 4)
        self.assertTrue(torch.equal(left.token_ids, right.token_ids))
        self.assertTrue(torch.equal(left.labels, right.labels))
        self.assertEqual(left.truth, right.truth)

    def test_variable_bindings_are_unique_and_unambiguous(self) -> None:
        batch = SyntheticTaskGenerator(self.data_config, seed=11).batch("variable_binding", 3)
        self.assertEqual(batch.token_ids.shape, (3, 9))
        self.assertEqual(batch.position_features.shape, (3, 9, 3))
        self.assertEqual(batch.labels.shape, (3,))
        self.assertEqual(len(batch.truth), 3)
        self.assertTrue(torch.all(batch.attention_mask))
        for row, truth in enumerate(batch.truth):
            variables = truth["variables"]
            self.assertEqual(len(variables), len(set(variables)))
            self.assertEqual(variables.count(truth["query_variable"]), 1)
            self.assertEqual(batch.labels[row].item(), truth["bound_value"])
            self.assertEqual(truth["values"][truth["binding_index"]], truth["bound_value"])

    def test_repeat_queries_vary_and_padding_masks_are_correct(self) -> None:
        batch = SyntheticTaskGenerator(self.data_config, seed=23).batch("repeat_symbol", 8)
        lengths = batch.attention_mask.long().sum(dim=1)
        self.assertGreater(torch.unique(lengths).numel(), 1)
        self.assertTrue(torch.any(~batch.attention_mask))
        for row, truth in enumerate(batch.truth):
            valid_length = lengths[row].item()
            self.assertEqual(batch.token_ids[row, valid_length - 1].item(), 3)
            self.assertTrue(torch.all(batch.token_ids[row, valid_length:] == 0))
            self.assertTrue(torch.all(~batch.attention_mask[row, valid_length:]))
            self.assertGreaterEqual(truth["prefix_length"], 1)
            self.assertEqual(batch.labels[row].item(), truth["body"][truth["query_index"]])

    def test_two_and_four_times_lengths_fit_and_validate(self) -> None:
        generator = SyntheticTaskGenerator(self.data_config, seed=29)
        for scale in (2, 4):
            for task in ("repeat_symbol", "variable_binding"):
                batch = generator.batch(task, 3, length_scale=scale)
                self.assertEqual(batch.token_ids.shape[0], 3)
                self.assertTrue(torch.all(batch.labels >= 4))
        ExperimentConfig(data=self.data_config, eval_length_scales=(1, 2, 4)).validate()

    def test_configuration_rejects_invalid_tasks_shapes_and_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "tasks must not be empty"):
            ExperimentConfig(tasks=()).validate()
        with self.assertRaisesRegex(ValueError, "tasks must not contain duplicates"):
            ExperimentConfig(tasks=("repeat_symbol", "repeat_symbol")).validate()
        with self.assertRaisesRegex(ValueError, "divisible"):
            ExperimentConfig(model=ModelConfig(embedding_dim=10, heads=3)).validate()
        with self.assertRaisesRegex(ValueError, "vocab_size is too small"):
            ExperimentConfig(data=DataConfig(vocab_size=12, repeat_length=3, binding_pairs=3), eval_length_scales=(1, 2, 4)).validate()
        with self.assertRaisesRegex(ValueError, "repeat_length must be at least 3"):
            ExperimentConfig(data=DataConfig(repeat_length=2)).validate()
        ExperimentConfig(device="directml", deterministic=False).validate()
        with self.assertRaisesRegex(ValueError, "deterministic=false"):
            ExperimentConfig(device="directml").validate()
        with self.assertRaisesRegex(ValueError, "warmup_steps must be a nonnegative"):
            ExperimentConfig(warmup_steps=-1).validate()
        with self.assertRaisesRegex(ValueError, "deterministic must be a boolean"):
            ExperimentConfig(deterministic=1).validate()
        with self.assertRaisesRegex(ValueError, "device must be"):
            ExperimentConfig(device="cuda").validate()

    def test_source_manifest_ignores_caches_and_install_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative_path in (
                "pyproject.toml",
                "requirements-cpu.lock",
                "requirements-directml.lock",
                "configs/smoke.json",
                "scripts/train.py",
                "src/dynamic_hierarchy/model.py",
                "tests/test_model.py",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source\n", encoding="utf-8")
            before = source_manifest(root)
            for relative_path in (
                "src/dynamic_hierarchy/__pycache__/model.cpython-312.pyc",
                "src/dynamic_hierarchy_ai.egg-info/PKG-INFO",
                ".venv/cache.txt",
                "runs/run.json",
                "data/generated.json",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"generated")
            after = source_manifest(root)
            self.assertEqual(before, after)
            self.assertEqual(set(after["files"]), {
                "pyproject.toml",
                "requirements-cpu.lock",
                "requirements-directml.lock",
                "configs/smoke.json",
                "scripts/train.py",
                "src/dynamic_hierarchy/model.py",
                "tests/test_model.py",
            })

    def test_train_reports_every_task_and_length_scale(self) -> None:
        config = ExperimentConfig(
            seed=37,
            steps=2,
            warmup_steps=1,
            batch_size=2,
            eval_batches=1,
            eval_length_scales=(1, 2, 4),
            cpu_threads=1,
            data=self.data_config,
            model=ModelConfig(embedding_dim=24, heads=4, layers=1, feedforward_dim=48),
        )
        metrics = train(config)
        self.assertEqual(set(metrics.train_by_task), {"repeat_symbol", "variable_binding"})
        self.assertEqual(set(metrics.evaluation_by_task_and_scale), {"repeat_symbol", "variable_binding"})
        self.assertEqual(metrics.performance.backend, "cpu")
        self.assertGreater(metrics.performance.parameter_count, 0)
        self.assertGreater(metrics.performance.training_seconds, 0.0)
        self.assertEqual(metrics.performance.warmup_steps, 1)
        self.assertTrue(metrics.performance.backward_completed)
        self.assertTrue(metrics.performance.deterministic_requested)
        self.assertTrue(metrics.performance.deterministic_algorithms_enabled)
        self.assertIn("after final optimizer.step", metrics.performance.timing_barrier)
        self.assertGreater(metrics.performance.steps_per_second, 0.0)
        for task_name in config.tasks:
            self.assertEqual(metrics.train_by_task[task_name].total, config.batch_size)
            scales = metrics.evaluation_by_task_and_scale[task_name]
            self.assertEqual(set(scales), {"1", "2", "4"})
            for accuracy in scales.values():
                self.assertEqual(accuracy.total, config.batch_size * config.eval_batches)

    def test_model_forward_shape(self) -> None:
        batch = SyntheticTaskGenerator(self.data_config, seed=13).batch("repeat_symbol", 2)
        model = SmallTransformerBaseline(32, ModelConfig(embedding_dim=24, heads=4, layers=1, feedforward_dim=48))
        self.assertEqual(model(batch.token_ids, batch.position_features, batch.attention_mask).shape, (2, 32))

    def test_one_training_update_changes_a_parameter(self) -> None:
        torch.manual_seed(17)
        batch = SyntheticTaskGenerator(self.data_config, seed=17).batch("variable_binding", 4)
        model = SmallTransformerBaseline(32, ModelConfig(embedding_dim=24, heads=4, layers=1, feedforward_dim=48))
        before = model.classifier.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = nn.CrossEntropyLoss()(model(batch.token_ids, batch.position_features, batch.attention_mask), batch.labels)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight.detach()))

    def test_adamw_core_matches_torch_state_for_25_steps(self) -> None:
        reference = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        candidate = torch.nn.Parameter(reference.detach().clone())
        settings = {
            "lr": 0.007,
            "betas": (0.83, 0.97),
            "eps": 2e-7,
            "weight_decay": 0.13,
        }
        reference_optimizer = torch.optim.AdamW([reference], foreach=False, **settings)
        candidate_optimizer = DirectMLCompatibleAdamWCore([candidate], **settings)
        for step in range(25):
            gradient = torch.tensor([0.25 + step * 0.01, -0.5 + step * 0.005], dtype=torch.float64)
            reference.grad = gradient.clone()
            candidate.grad = gradient.clone()
            reference_optimizer.step()
            candidate_optimizer.step()
        self.assertTrue(torch.allclose(reference, candidate, atol=1e-12, rtol=1e-12))
        reference_state = reference_optimizer.state[reference]
        candidate_state = candidate_optimizer.state[candidate]
        self.assertEqual(int(reference_state["step"]), candidate_state["step"])
        self.assertTrue(torch.allclose(reference_state["exp_avg"], candidate_state["exp_avg"], atol=1e-12, rtol=1e-12))
        self.assertTrue(
            torch.allclose(reference_state["exp_avg_sq"], candidate_state["exp_avg_sq"], atol=1e-12, rtol=1e-12)
        )

    def test_adamw_core_closure_enables_gradients(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = DirectMLCompatibleAdamWCore([parameter], lr=0.1)
        calls = 0

        def closure() -> torch.Tensor:
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            loss = parameter.square().sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertEqual(calls, 1)
        self.assertTrue(loss.requires_grad)
        self.assertLess(parameter.item(), 2.0)

    def test_reporting_marks_directml_fallback_as_unknown(self) -> None:
        status = fallback_observability("directml", [])
        self.assertEqual(status["status"], "unknown")
        self.assertIn("no public DirectML fallback counter", status["detail"])
        self.assertIn("no Python warnings observed", status["detail"])
        summary = summarize_measurements([3.0, 1.0, 2.0])
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)
        self.assertEqual(summary["approx_p95"], 3.0)

    def test_hierarchy_interface_only_proposes_operations(self) -> None:
        controller = CandidateHierarchyController(hidden_dim=8)
        proposal = controller(torch.zeros(2, 5, 8), torch.zeros(2, 5, 3))
        self.assertEqual(proposal.operations, ("MERGE", "STOP"))
        self.assertEqual(proposal.phase.shape, (2, 5))
        self.assertEqual(proposal.operation_scores.shape, (2, 5, 2))
        self.assertTrue(torch.all((proposal.phase >= 0) & (proposal.phase <= 1)))


if __name__ == "__main__":
    unittest.main()
