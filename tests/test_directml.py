from __future__ import annotations

import importlib.util
import unittest

import torch

from dynamic_hierarchy.backend import resolve_backend
from dynamic_hierarchy.config import DataConfig, ExperimentConfig, ModelConfig
from dynamic_hierarchy.optim import DirectMLCompatibleAdamWCore
from dynamic_hierarchy.training import train
from test_stage1 import ROOT, tiny_literal_config, tiny_stage1_config
from dynamic_hierarchy.stage1_runtime import Stage1Trainer

import tempfile
from pathlib import Path


DIRECTML_AVAILABLE = importlib.util.find_spec("torch_directml") is not None


@unittest.skipUnless(DIRECTML_AVAILABLE, "torch-directml is installed only in .venv-directml")
class DirectMLBackendTests(unittest.TestCase):
    def test_directml_tensor_and_backward(self) -> None:
        backend = resolve_backend("directml", cpu_threads=1, deterministic=False)
        value = torch.tensor([2.0], device=backend.device, requires_grad=True)
        loss = (value * value).sum()
        loss.backward()
        self.assertEqual(value.grad.detach().cpu().tolist(), [4.0])
        self.assertEqual(backend.name, "directml")

    def test_directml_optimizer_state_stays_on_device(self) -> None:
        backend = resolve_backend("directml", cpu_threads=1, deterministic=False)
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=backend.device))
        optimizer = DirectMLCompatibleAdamWCore(
            [parameter],
            lr=0.007,
            betas=(0.83, 0.97),
            eps=2e-7,
            weight_decay=0.13,
        )
        for step in range(25):
            parameter.grad = torch.tensor([0.25 + step * 0.01, -0.5 + step * 0.005], device=backend.device)
            optimizer.step()
        backend.synchronize(parameter)
        state = optimizer.state[parameter]
        self.assertEqual(state["step"], 25)
        self.assertEqual(state["exp_avg"].device.type, "privateuseone")
        self.assertEqual(state["exp_avg_sq"].device.type, "privateuseone")

    def test_directml_tiny_full_train_records_final_barrier(self) -> None:
        metrics = train(
            ExperimentConfig(
                device="directml",
                deterministic=False,
                steps=2,
                warmup_steps=1,
                batch_size=2,
                eval_batches=1,
                eval_length_scales=(1,),
                cpu_threads=1,
                data=DataConfig(vocab_size=32, repeat_length=5, binding_pairs=3),
                model=ModelConfig(embedding_dim=24, heads=4, layers=1, feedforward_dim=48),
            )
        )
        self.assertEqual(metrics.performance.backend, "directml")
        self.assertTrue(metrics.performance.backward_completed)
        self.assertFalse(metrics.performance.deterministic_algorithms_enabled)
        self.assertEqual(metrics.performance.warmup_steps, 1)
        self.assertIn("classifier.weight", metrics.performance.timing_barrier)
        self.assertIn("final optimizer write", metrics.performance.synchronization_method)

    def test_directml_stage1_pair_checkpoint_and_resume(self) -> None:
        config = tiny_stage1_config(device="directml")
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            trainer = Stage1Trainer(config, run_dir, ROOT, "directml-snapshot-test")
            trainer.evaluate()
            trainer.save_checkpoint("bootstrap")
            trainer.train_pair()
            checkpoint = trainer.save_checkpoint()
            restored = Stage1Trainer(config, run_dir, ROOT, "directml-snapshot-test")
            restored.load_checkpoint(checkpoint)
            restored.train_pair()
            self.assertEqual(restored.global_step, 2)
            self.assertEqual(restored.cumulative["A"]["examples"], 14)
            self.assertEqual(restored.cumulative["D_true"]["examples"], 14)
            self.assertEqual(restored.cumulative["D_sham"]["examples"], 14)
            for optimizer in restored.optimizers.values():
                for state in optimizer.state.values():
                    self.assertEqual(state["exp_avg"].device.type, "privateuseone")
                    self.assertEqual(state["exp_avg_sq"].device.type, "privateuseone")

    def test_directml_literal_stage1_smoke_and_boundary_eval(self) -> None:
        config = tiny_literal_config(device="directml")
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = Stage1Trainer(
                config,
                Path(temporary_directory),
                ROOT,
                "directml-literal-test",
            )
            trainer.train_pair()
            boundary = trainer.evaluate_pending_stage_boundary()
            self.assertEqual(boundary["tasks"]["C0"]["label_counts"], [2] * 7)
            self.assertEqual(trainer.cumulative["A"]["examples"], 56)
            self.assertEqual(
                trainer.cumulative["D_true"]["compose_module_calls"],
                trainer.cumulative["D_sham"]["compose_module_calls"],
            )


if __name__ == "__main__":
    unittest.main()
