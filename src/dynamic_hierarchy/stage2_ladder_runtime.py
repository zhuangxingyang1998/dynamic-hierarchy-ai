"""Gated training, one-shot reserve evidence, and recovery for Stage 2 R5.1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .backend import resolve_backend
from .optim import DirectMLCompatibleAdamWCore
from .stage2_ladder_config import Stage2LadderConfig
from .stage2_ladder_data import (
    ADD_FIRST,
    SUB_FIRST,
    ArithmeticLadderData,
    LadderGeneratedSplit,
    batch_evidence,
    sham_intermediate_labels,
    two_literal_lookup_accuracies,
)
from .stage2_ladder_model import (
    ArithmeticComposerModel,
    bridge_root_logits,
    model_state_digest,
    parameter_count,
)


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


class Stage2LadderTrainer:
    FIXED_BRANCHES = tuple(
        f"fixed-{query}-{mode}"
        for query in ("add", "sub")
        for mode in ("root", "teacher", "aux-true", "aux-sham")
    )
    PAIRED_BRANCHES = ("paired-root", "paired-aux-true", "paired-aux-sham")

    def __init__(self, config: Stage2LadderConfig, run_dir: Path) -> None:
        config.validate()
        self.config = config
        self.run_dir = run_dir
        self.backend = resolve_backend(
            config.device, config.cpu_threads, deterministic=config.deterministic
        )
        torch.manual_seed(config.seed)
        self.data = ArithmeticLadderData(config.seed)
        self.config_digest = _digest(config.to_dict())
        self.ledger_path = run_dir / "r5-evaluation-ledger.json"
        self.ledger = self._load_or_initialize_ledger()
        self.models: dict[str, ArithmeticComposerModel] = {}
        self.optimizers: dict[str, DirectMLCompatibleAdamWCore] = {}
        self.branch_modes: dict[str, str] = {}
        self.initialization_groups: dict[str, dict[str, object]] = {}
        self.cumulative: dict[str, dict[str, float | int]] = {}
        self.rung1_state: dict[str, torch.Tensor] | None = None
        self.rung1_state_digest: str | None = None
        self.current_rung = "binary"
        self.active_branches = ("binary-root",)
        self.stage_step = 0
        self.global_round = 0
        self.final_disposition: str | None = None
        self.process_started = time.monotonic()
        self.elapsed_before_resume = 0.0
        self.last_checkpoint: str | None = None
        self._create_binary_model()

    @staticmethod
    def _empty_cumulative() -> dict[str, float | int]:
        return {
            "optimizer_updates": 0,
            "examples": 0,
            "correct": 0,
            "root_loss_sum": 0.0,
            "objective_loss_sum": 0.0,
            "forward_backward_seconds": 0.0,
        }

    def _register_model(
        self,
        name: str,
        mode: str,
        state: dict[str, torch.Tensor],
    ) -> None:
        if name in self.models:
            return
        model = ArithmeticComposerModel(self.config.model)
        model.load_state_dict(state)
        model = model.to(self.backend.device)
        self.models[name] = model
        self.optimizers[name] = DirectMLCompatibleAdamWCore(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
        self.branch_modes[name] = mode
        self.cumulative[name] = self._empty_cumulative()

    def _create_binary_model(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed + 101)
            model = ArithmeticComposerModel(self.config.model)
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        digest = model_state_digest(model)
        self._register_model("binary-root", "root", state)
        self.initialization_groups["binary"] = {
            "seed": self.config.seed + 101,
            "initial_state_digest": digest,
            "parameter_count": parameter_count(model),
        }

    def _clone_from_rung1(self, group: str, names: tuple[str, ...]) -> None:
        if self.rung1_state is None or self.rung1_state_digest is None:
            raise RuntimeError("R5.1 cannot create downstream models before Rung 1 passes")
        branch_digests: dict[str, str] = {}
        for name in names:
            if name.endswith("teacher"):
                mode = "teacher"
            elif name.endswith("aux-true"):
                mode = "aux-true"
            elif name.endswith("aux-sham"):
                mode = "aux-sham"
            else:
                mode = "root"
            self._register_model(name, mode, self.rung1_state)
            branch_digests[name] = model_state_digest(self.models[name])
        self.initialization_groups[group] = {
            "inherited_rung1_digest": self.rung1_state_digest,
            "branch_digests": branch_digests,
            "identical": all(
                digest == self.rung1_state_digest for digest in branch_digests.values()
            ),
            "optimizer_created_after_clone": True,
            "parameter_count": parameter_count(self.models[names[0]]),
        }
        if not self.initialization_groups[group]["identical"]:
            raise RuntimeError(f"R5.1 {group} branches did not inherit identical Rung 1 state")

    def _load_or_initialize_ledger(self) -> dict[str, object]:
        if not self.ledger_path.is_file():
            return {
                "schema_version": 1,
                "packet": "DH-S2-R5.1",
                "config_digest": self.config_digest,
                "partition_digest": self.data.partition_digest,
                "rungs": {},
            }
        raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise RuntimeError("R5.1 evaluation ledger is malformed")
        if raw.get("packet") != "DH-S2-R5.1":
            raise RuntimeError("R5.1 evaluation ledger packet mismatch")
        if raw.get("config_digest") != self.config_digest:
            raise RuntimeError("R5.1 evaluation ledger config mismatch")
        if raw.get("partition_digest") != self.data.partition_digest:
            raise RuntimeError("R5.1 evaluation ledger partition mismatch")
        rungs = raw.get("rungs")
        if not isinstance(rungs, dict):
            raise RuntimeError("R5.1 evaluation ledger lacks rung evidence")
        for rung in rungs.values():
            if not isinstance(rung, dict):
                raise RuntimeError("R5.1 evaluation ledger rung is malformed")
            for branch in rung.get("branches", {}).values():
                if isinstance(branch, dict) and branch.get("reserve_state") == "reserve_opened":
                    raise RuntimeError(
                        "R5.1 reserve opened without completion; one-shot reserve cannot replay"
                    )
        return raw

    def _write_ledger(self) -> None:
        atomic_write_json(self.ledger_path, self.ledger)

    def elapsed_seconds(self) -> float:
        return self.elapsed_before_resume + (time.monotonic() - self.process_started)

    def time_budget_exhausted(self) -> bool:
        return self.elapsed_seconds() >= self.config.time_budget_minutes * 60.0

    @property
    def is_complete(self) -> bool:
        return self.current_rung == "complete"

    def _target_steps(self) -> int:
        return {
            "binary": self.config.rung1_steps,
            "fixed": self.config.rung2_steps,
            "paired": self.config.rung3_steps,
        }[self.current_rung]

    @property
    def needs_gate(self) -> bool:
        return not self.is_complete and self.stage_step >= self._target_steps()

    def _batch_for_branch(self, branch: str, split: str) -> LadderGeneratedSplit:
        if branch == "binary-root":
            return self.data.batch("binary", "fit" if split != "train" else "train")
        if branch.startswith("fixed-add-"):
            return self.data.batch("fixed-add", split)
        if branch.startswith("fixed-sub-"):
            return self.data.batch("fixed-sub", split)
        if branch.startswith("paired-"):
            return self.data.batch("paired", split)
        raise ValueError(f"unknown R5.1 branch: {branch}")

    def _forward_for_mode(
        self,
        branch: str,
        batch: LadderGeneratedSplit,
    ):
        mode = self.branch_modes[branch]
        teacher = (
            batch.targets.intermediate_labels[:, 0]
            if mode == "teacher"
            else None
        )
        return self.models[branch](
            batch.model_input,
            teacher_intermediate_labels=teacher,
        )

    def train_step(self) -> dict[str, float]:
        if self.is_complete or self.needs_gate:
            raise RuntimeError("R5.1 train_step called outside an open stage")
        losses: dict[str, float] = {}
        for branch in self.active_branches:
            source_batch = self._batch_for_branch(branch, "train")
            sham_targets = None
            if self.branch_modes[branch] == "aux-sham":
                sham_targets = sham_intermediate_labels(
                    source_batch.targets.intermediate_labels[:, 0],
                    source_batch.model_input.query_ids,
                ).to(self.backend.device)
            batch = source_batch.to(self.backend.device)
            model = self.models[branch]
            optimizer = self.optimizers[branch]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            self.backend.synchronize(next(model.parameters()))
            started = time.perf_counter()
            output = self._forward_for_mode(branch, batch)
            root_loss = F.cross_entropy(output.root_logits, batch.targets.final_labels)
            objective = root_loss
            mode = self.branch_modes[branch]
            if mode in {"aux-true", "aux-sham"}:
                if len(output.intermediate_logits) != 1:
                    raise RuntimeError("R5.1 auxiliary branch lacks first-merge logits")
                targets = batch.targets.intermediate_labels[:, 0]
                if mode == "aux-sham":
                    if sham_targets is None:
                        raise RuntimeError("R5.1 sham targets were not prepared")
                    targets = sham_targets
                objective = objective + self.config.auxiliary_weight * F.cross_entropy(
                    output.intermediate_logits[0], targets
                )
            objective.backward()
            optimizer.step()
            self.backend.synchronize(next(model.parameters()))
            elapsed = time.perf_counter() - started
            predictions = output.root_logits.detach().argmax(dim=-1)
            correct = int(
                (predictions == batch.targets.final_labels).sum().detach().cpu().item()
            )
            root_value = self.backend.scalar(root_loss)
            objective_value = self.backend.scalar(objective)
            cumulative = self.cumulative[branch]
            cumulative["optimizer_updates"] += 1
            cumulative["examples"] += int(batch.targets.final_labels.shape[0])
            cumulative["correct"] += correct
            cumulative["root_loss_sum"] += root_value
            cumulative["objective_loss_sum"] += objective_value
            cumulative["forward_backward_seconds"] += elapsed
            losses[branch] = objective_value
        self.stage_step += 1
        self.global_round += 1
        if self.config.yield_ms:
            time.sleep(self.config.yield_ms / 1000.0)
        return losses

    @staticmethod
    def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        return float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu().item())

    def _interventions(
        self,
        branch: str,
        batch: LadderGeneratedSplit,
        correct_logits: torch.Tensor,
    ) -> dict[str, object]:
        model = self.models[branch]
        queries = batch.model_input.query_ids
        with torch.no_grad():
            opposite_logits = model(
                batch.model_input,
                merge_query_ids=1 - queries,
            ).root_logits
            left_logits = model(
                batch.model_input,
                merge_query_ids=torch.full_like(queries, SUB_FIRST),
            ).root_logits
            right_logits = model(
                batch.model_input,
                merge_query_ids=torch.full_like(queries, ADD_FIRST),
            ).root_logits
        labels = batch.targets.final_labels
        correct_accuracy = self._accuracy(correct_logits, labels)
        opposite_accuracy = self._accuracy(opposite_logits, labels)
        left_accuracy = self._accuracy(left_logits, labels)
        right_accuracy = self._accuracy(right_logits, labels)
        return {
            "correct_tree_accuracy": correct_accuracy,
            "opposite_tree_accuracy": opposite_accuracy,
            "fixed_left_accuracy": left_accuracy,
            "fixed_right_accuracy": right_accuracy,
            "best_fixed_tree_accuracy": max(left_accuracy, right_accuracy),
            "correct_minus_opposite": correct_accuracy - opposite_accuracy,
        }

    def _evaluate_branch(self, branch: str, batch: LadderGeneratedSplit) -> dict[str, object]:
        device_batch = batch.to(self.backend.device)
        model = self.models[branch]
        model.eval()
        with torch.no_grad():
            output = self._forward_for_mode(branch, device_batch)
            loss = F.cross_entropy(output.root_logits, device_batch.targets.final_labels)
            predictions = output.root_logits.argmax(dim=-1)
            prediction_counts = torch.bincount(predictions.cpu(), minlength=7).tolist()
            self.backend.synchronize(next(model.parameters()))
        labels = device_batch.targets.final_labels
        accuracy = self._accuracy(output.root_logits, labels)
        per_query: dict[str, float] = {}
        for query, label in ((ADD_FIRST, "add"), (SUB_FIRST, "sub")):
            mask = device_batch.model_input.query_ids == query
            if bool(mask.any().detach().cpu().item()):
                per_query[label] = self._accuracy(output.root_logits[mask], labels[mask])
        both_correct = None
        if branch.startswith("paired-"):
            correct_by_family: dict[str, list[bool]] = {}
            row_correct = (predictions == labels).detach().cpu().tolist()
            for family_hash, correct in zip(batch.family_hashes, row_correct, strict=True):
                correct_by_family.setdefault(family_hash, []).append(bool(correct))
            if not all(len(items) == 2 for items in correct_by_family.values()):
                raise RuntimeError("R5.1 paired evaluation lost family pairing")
            both_correct = sum(all(items) for items in correct_by_family.values()) / len(correct_by_family)
        train_batch = self._batch_for_branch(branch, "train")
        canaries = (
            two_literal_lookup_accuracies(train_batch, batch)
            if batch.model_input.values.shape[1] == 3
            else {}
        )
        bridge_difference = None
        if self.branch_modes[branch] != "teacher":
            with torch.no_grad():
                bridged = bridge_root_logits(model, batch)
                bridge_difference = float(
                    (output.root_logits.detach().cpu() - bridged.detach().cpu()).abs().max().item()
                )
        interventions = None
        if branch.startswith("paired-") and self.branch_modes[branch] != "teacher":
            interventions = self._interventions(
                branch,
                device_batch,
                output.root_logits,
            )
        intermediate_accuracy = None
        if output.intermediate_logits:
            intermediate_accuracy = self._accuracy(
                output.intermediate_logits[0],
                device_batch.targets.intermediate_labels[:, 0],
            )
        metrics: dict[str, object] = {
            "branch": branch,
            "accuracy": accuracy,
            "cross_entropy": self.backend.scalar(loss),
            "prediction_counts": prediction_counts,
            "predicted_classes": sum(count > 0 for count in prediction_counts),
            "per_query_accuracy": per_query,
            "paired_family_both_correct": both_correct,
            "intermediate_accuracy_report_only": intermediate_accuracy,
            "two_literal_lookup_canaries": canaries,
            "bridge_max_abs_difference": bridge_difference,
            "interventions": interventions,
            "data": batch_evidence(batch),
        }
        metrics["answer_passed"] = self._answer_metrics_pass(branch, metrics)
        metrics["structure_passed"] = self._structure_metrics_pass(branch, metrics)
        metrics["passed"] = bool(metrics["answer_passed"] and metrics["structure_passed"])
        return metrics

    def _answer_metrics_pass(self, branch: str, metrics: dict[str, object]) -> bool:
        accuracy = float(metrics.get("accuracy", float("nan")))
        cross_entropy = float(metrics.get("cross_entropy", float("nan")))
        predicted_classes = int(metrics.get("predicted_classes", 0))
        data = metrics.get("data")
        exact_solver = data.get("exact_solver_accuracy") if isinstance(data, dict) else None
        threshold = (
            self.config.atomic_max_cross_entropy
            if branch == "binary-root"
            else self.config.recursive_max_cross_entropy
        )
        canaries = metrics.get("two_literal_lookup_canaries")
        canaries_pass = not isinstance(canaries, dict) or all(
            float(value) <= self.config.max_partial_lookup_accuracy
            for value in canaries.values()
        )
        bridge = metrics.get("bridge_max_abs_difference")
        bridge_pass = bridge is None or float(bridge) <= self.config.bridge_max_abs_difference
        per_query = metrics.get("per_query_accuracy")
        per_query_pass = not branch.startswith("paired-") or (
            isinstance(per_query, dict)
            and set(per_query) == {"add", "sub"}
            and all(float(value) >= self.config.min_final_accuracy for value in per_query.values())
        )
        both = metrics.get("paired_family_both_correct")
        both_pass = not branch.startswith("paired-") or (
            both is not None and float(both) >= self.config.min_paired_both_correct
        )
        return bool(
            math.isfinite(accuracy)
            and math.isfinite(cross_entropy)
            and accuracy >= self.config.min_final_accuracy
            and cross_entropy <= threshold
            and predicted_classes >= self.config.required_predicted_classes
            and exact_solver == 1.0
            and canaries_pass
            and bridge_pass
            and per_query_pass
            and both_pass
        )

    def _structure_metrics_pass(self, branch: str, metrics: dict[str, object]) -> bool:
        if branch != "paired-root":
            return True
        if self.config.run_kind == "smoke":
            return True
        interventions = metrics.get("interventions")
        if not isinstance(interventions, dict):
            return False
        return bool(
            float(interventions["opposite_tree_accuracy"])
            <= self.config.max_opposite_tree_accuracy
            and float(interventions["best_fixed_tree_accuracy"])
            <= self.config.max_fixed_tree_accuracy
            and float(interventions["correct_minus_opposite"])
            >= self.config.min_structure_accuracy_drop
        )

    def _gate_branch_with_reserve(
        self,
        rung_entry: dict[str, object],
        branch: str,
    ) -> dict[str, object]:
        branches = rung_entry.setdefault("branches", {})
        if not isinstance(branches, dict):
            raise RuntimeError("R5.1 ledger branch map is malformed")
        existing = branches.get(branch)
        if isinstance(existing, dict) and existing.get("state") == "complete":
            return existing
        validation = self._evaluate_branch(branch, self._batch_for_branch(branch, "validation"))
        reserve_admission_passed = bool(
            validation["answer_passed"]
            if branch == "paired-root"
            else validation["passed"]
        )
        entry: dict[str, object] = {
            "state": "unopened",
            "validation": validation,
            "validation_passed": bool(validation["passed"]),
            "reserve_admission_passed": reserve_admission_passed,
            "reserve_state": "unopened",
            "reserve": None,
            "passed": False,
        }
        branches[branch] = entry
        if not reserve_admission_passed:
            entry["state"] = "complete"
            entry["reserve_state"] = "validation_failed"
            self._write_ledger()
            return entry
        entry["reserve_state"] = "reserve_opened"
        self._write_ledger()
        reserve = self._evaluate_branch(branch, self._batch_for_branch(branch, "reserve"))
        entry["reserve"] = reserve
        entry["reserve_state"] = "complete"
        entry["passed"] = bool(validation["passed"] and reserve["passed"])
        entry["state"] = "complete"
        self._write_ledger()
        return entry

    def run_gate(self) -> dict[str, object]:
        if not self.needs_gate:
            raise RuntimeError("R5.1 gate requested before its exposure budget completed")
        rungs = self.ledger["rungs"]
        if not isinstance(rungs, dict):
            raise RuntimeError("R5.1 ledger rung map is malformed")
        existing = rungs.get(self.current_rung)
        if isinstance(existing, dict) and existing.get("state") == "complete":
            if self.current_rung == "binary" and bool(existing.get("passed")):
                self.rung1_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.models["binary-root"].state_dict().items()
                }
                self.rung1_state_digest = model_state_digest(self.models["binary-root"])
                if self.rung1_state_digest != existing.get("passed_state_digest"):
                    raise RuntimeError("R5.1 completed Rung 1 evidence does not match checkpoint weights")
            self._transition(existing)
            return existing
        if self.current_rung == "binary":
            metrics = self._evaluate_branch("binary-root", self.data.batch("binary", "fit"))
            passed = bool(metrics["passed"])
            entry = {
                "state": "complete",
                "kind": "full_domain_fit_and_bridge",
                "branches": {
                    "binary-root": {
                        "state": "complete",
                        "fit": metrics,
                        "passed": passed,
                        "reserve_state": "not_applicable",
                    }
                },
                "passed": passed,
            }
            rungs["binary"] = entry
            if passed:
                self.rung1_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.models["binary-root"].state_dict().items()
                }
                self.rung1_state_digest = model_state_digest(self.models["binary-root"])
                entry["passed_state_digest"] = self.rung1_state_digest
            self._write_ledger()
            self._transition(entry)
            return entry
        entry = existing if isinstance(existing, dict) else {
            "state": "unopened",
            "kind": "validation_then_reserve",
            "branches": {},
        }
        rungs[self.current_rung] = entry
        for branch in self.active_branches:
            self._gate_branch_with_reserve(entry, branch)
        branches = entry["branches"]
        entry["passed_branches"] = [
            branch for branch in self.active_branches if bool(branches[branch]["passed"])
        ]
        entry["state"] = "complete"
        self._write_ledger()
        self._transition(entry)
        return entry

    def _transition(self, entry: dict[str, object]) -> None:
        if self.current_rung == "binary":
            if not bool(entry.get("passed")):
                self.current_rung = "complete"
                self.active_branches = ()
                self.final_disposition = "representation_fit_failed"
                return
            self._clone_from_rung1("fixed", self.FIXED_BRANCHES)
            self.current_rung = "fixed"
            self.active_branches = self.FIXED_BRANCHES
            self.stage_step = 0
            return
        if self.current_rung == "fixed":
            passed = set(entry.get("passed_branches", ()))
            if not {"fixed-add-root", "fixed-sub-root"} <= passed:
                self.current_rung = "complete"
                self.active_branches = ()
                self.final_disposition = "fixed_query_failed"
                return
            self._clone_from_rung1("paired", self.PAIRED_BRANCHES)
            self.current_rung = "paired"
            self.active_branches = self.PAIRED_BRANCHES
            self.stage_step = 0
            return
        if self.current_rung != "paired":
            raise RuntimeError("R5.1 transition encountered an unknown rung")
        root = entry.get("branches", {}).get("paired-root", {})
        validation = root.get("validation", {}) if isinstance(root, dict) else {}
        reserve = root.get("reserve", {}) if isinstance(root, dict) else {}
        if not isinstance(validation, dict):
            validation = {}
        if not isinstance(reserve, dict):
            reserve = {}
        answer_passed = bool(
            validation.get("answer_passed") and reserve.get("answer_passed")
        )
        structure_passed = bool(
            validation.get("structure_passed") and reserve.get("structure_passed")
        )
        if not answer_passed:
            disposition = "paired_query_failed"
        elif not structure_passed:
            disposition = "structure_decorative"
        else:
            disposition = "ladder_pass"
        self.current_rung = "complete"
        self.active_branches = ()
        self.final_disposition = disposition

    def training_report(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, cumulative in self.cumulative.items():
            updates = int(cumulative["optimizer_updates"])
            examples = int(cumulative["examples"])
            train_rows = len(self._batch_for_branch(name, "train").targets.final_labels)
            result[name] = {
                **cumulative,
                "accuracy": int(cumulative["correct"]) / examples if examples else 0.0,
                "mean_root_loss": float(cumulative["root_loss_sum"]) / updates if updates else None,
                "mean_objective_loss": float(cumulative["objective_loss_sum"]) / updates if updates else None,
                "parameter_count": parameter_count(self.models[name]),
                "mode": self.branch_modes[name],
                "train_rows_per_update": train_rows,
                "per_row_exposures": updates,
            }
        return result

    def result(self, disposition: str) -> dict[str, object]:
        if disposition != "calibration_incomplete" and not self.is_complete:
            raise RuntimeError("R5.1 completed result requested before terminal gate")
        return {
            "schema_version": 1,
            "packet": "DH-S2-R5.1",
            "revision": self.config.revision,
            "phase": self.config.phase,
            "run_kind": self.config.run_kind,
            "disposition": disposition,
            "config": self.config.to_dict(),
            "backend": self.backend.metadata(),
            "elapsed_seconds": self.elapsed_seconds(),
            "global_round": self.global_round,
            "current_rung": self.current_rung,
            "stage_step": self.stage_step,
            "training": self.training_report(),
            "evaluation_ledger": self.ledger,
            "data": self.data.partition_evidence(),
            "initialization_groups": self.initialization_groups,
            "rung1_passed_state_digest": self.rung1_state_digest,
            "recovery": {
                "semantics": "at-least-once training; one-shot reserve fails closed after opening",
                "last_checkpoint": self.last_checkpoint,
            },
            "claim_boundary": {
                "learned_routing_trained": False,
                "continuous_phase_trained": False,
                "auxiliary_and_teacher_are_diagnostic": True,
                "single_seed_candidate_claim": False,
                "ladder_pass_means_ready_for_routing_design_only": True,
            },
        }

    def status(self, state: str, detail: str | None = None) -> dict[str, object]:
        return {
            "schema_version": 1,
            "packet": "DH-S2-R5.1",
            "state": state,
            "detail": detail,
            "current_rung": self.current_rung,
            "stage_step": self.stage_step,
            "stage_target": None if self.is_complete else self._target_steps(),
            "global_round": self.global_round,
            "active_branches": list(self.active_branches),
            "elapsed_seconds": self.elapsed_seconds(),
            "final_disposition": self.final_disposition,
        }

    def save_checkpoint(self, kind: str = "scheduled") -> Path:
        directory = self.run_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"r5-{self.global_round:08d}-{kind}.pt"
        payload = {
            "schema_version": 1,
            "packet": "DH-S2-R5.1",
            "config": self.config.to_dict(),
            "config_digest": self.config_digest,
            "partition_digest": self.data.partition_digest,
            "current_rung": self.current_rung,
            "active_branches": self.active_branches,
            "stage_step": self.stage_step,
            "global_round": self.global_round,
            "final_disposition": self.final_disposition,
            "model_names": tuple(self.models),
            "models": {name: _cpu_tree(model.state_dict()) for name, model in self.models.items()},
            "optimizers": {name: _cpu_tree(optimizer.state_dict()) for name, optimizer in self.optimizers.items()},
            "branch_modes": self.branch_modes,
            "initialization_groups": self.initialization_groups,
            "cumulative": self.cumulative,
            "rung1_state": _cpu_tree(self.rung1_state),
            "rung1_state_digest": self.rung1_state_digest,
            "ledger_snapshot": self.ledger,
            "ledger_digest": _digest(self.ledger),
            "torch_rng_state": torch.get_rng_state(),
            "elapsed_seconds": self.elapsed_seconds(),
        }
        torch.save(payload, path)
        atomic_write_json(
            directory / "latest.json",
            {"checkpoint": str(path), "global_round": self.global_round, "kind": kind},
        )
        self.last_checkpoint = str(path)
        return path

    @staticmethod
    def _ledger_extends_pre_gate_checkpoint(
        checkpoint_ledger: object,
        current_ledger: dict[str, object],
        rung: str,
        active_branches: tuple[str, ...],
    ) -> bool:
        if not isinstance(checkpoint_ledger, dict):
            return False
        for key in ("schema_version", "packet", "config_digest", "partition_digest"):
            if checkpoint_ledger.get(key) != current_ledger.get(key):
                return False
        previous_rungs = checkpoint_ledger.get("rungs")
        current_rungs = current_ledger.get("rungs")
        if not isinstance(previous_rungs, dict) or not isinstance(current_rungs, dict):
            return False
        for name in set(previous_rungs) | set(current_rungs):
            if name != rung and previous_rungs.get(name) != current_rungs.get(name):
                return False

        previous = previous_rungs.get(rung)
        current = current_rungs.get(rung)
        if not isinstance(current, dict):
            return False
        if previous is not None and not isinstance(previous, dict):
            return False
        if rung == "binary":
            if previous is not None or active_branches != ("binary-root",):
                return False
            branches = current.get("branches")
            if not isinstance(branches, dict) or set(branches) != {"binary-root"}:
                return False
            branch = branches["binary-root"]
            if not isinstance(branch, dict):
                return False
            passed = bool(branch.get("passed"))
            expected_keys = {"state", "kind", "branches", "passed"}
            if passed:
                expected_keys.add("passed_state_digest")
            return bool(
                set(current) == expected_keys
                and current.get("state") == "complete"
                and current.get("kind") == "full_domain_fit_and_bridge"
                and bool(current.get("passed")) == passed
                and branch.get("state") == "complete"
                and branch.get("reserve_state") == "not_applicable"
                and isinstance(branch.get("fit"), dict)
                and (
                    not passed
                    or isinstance(current.get("passed_state_digest"), str)
                )
            )
        previous = previous or {
            "state": "unopened",
            "kind": "validation_then_reserve",
            "branches": {},
        }
        if previous.get("kind") != current.get("kind"):
            return False
        previous_branches = previous.get("branches")
        current_branches = current.get("branches")
        if not isinstance(previous_branches, dict) or not isinstance(current_branches, dict):
            return False
        if not set(previous_branches) <= set(current_branches):
            return False
        if any(current_branches[name] != value for name, value in previous_branches.items()):
            return False
        if not set(current_branches) <= set(active_branches):
            return False
        for value in current_branches.values():
            if not isinstance(value, dict) or value.get("state") != "complete":
                return False
            if value.get("reserve_state") not in {"complete", "validation_failed"}:
                return False

        state = current.get("state")
        if state == "unopened":
            return set(current) <= {"state", "kind", "branches"}
        if state != "complete" or set(current_branches) != set(active_branches):
            return False
        expected_passed = [
            name for name in active_branches if bool(current_branches[name].get("passed"))
        ]
        return current.get("passed_branches") == expected_passed

    def _ensure_checkpoint_models(self, names: tuple[str, ...]) -> None:
        fixed = tuple(name for name in names if name.startswith("fixed-"))
        paired = tuple(name for name in names if name.startswith("paired-"))
        if fixed:
            self._clone_from_rung1("fixed", fixed)
        if paired:
            self._clone_from_rung1("paired", paired)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("packet") != "DH-S2-R5.1":
            raise RuntimeError("R5.1 checkpoint schema or packet mismatch")
        if payload.get("config_digest") != self.config_digest or payload.get("config") != self.config.to_dict():
            raise RuntimeError("R5.1 checkpoint config mismatch")
        if payload.get("partition_digest") != self.data.partition_digest:
            raise RuntimeError("R5.1 checkpoint partition mismatch")
        checkpoint_ledger = payload.get("ledger_digest")
        current_ledger = _digest(self.ledger)
        if checkpoint_ledger != current_ledger:
            rung = str(payload.get("current_rung"))
            stage_step = int(payload.get("stage_step", -1))
            target = {"binary": self.config.rung1_steps, "fixed": self.config.rung2_steps, "paired": self.config.rung3_steps}.get(rung)
            active = tuple(str(name) for name in payload.get("active_branches", ()))
            snapshot = payload.get("ledger_snapshot")
            if (
                target is None
                or stage_step != target
                or _digest(snapshot) != checkpoint_ledger
                or not self._ledger_extends_pre_gate_checkpoint(
                    snapshot, self.ledger, rung, active
                )
            ):
                raise RuntimeError("R5.1 checkpoint and evaluation ledger disagree")
        self.rung1_state = payload.get("rung1_state")
        self.rung1_state_digest = payload.get("rung1_state_digest")
        names = tuple(str(name) for name in payload.get("model_names", ()))
        self._ensure_checkpoint_models(names)
        if set(names) != set(self.models):
            raise RuntimeError("R5.1 checkpoint model set is inconsistent")
        if payload.get("initialization_groups") != self.initialization_groups:
            raise RuntimeError("R5.1 inherited-state receipts do not reconstruct")
        for name in names:
            self.models[name].load_state_dict(payload["models"][name])
            self.optimizers[name].load_state_dict(payload["optimizers"][name])
            for state in self.optimizers[name].state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.backend.device)
        self.branch_modes = {str(name): str(value) for name, value in payload["branch_modes"].items()}
        self.cumulative = payload["cumulative"]
        self.current_rung = str(payload["current_rung"])
        self.active_branches = tuple(str(name) for name in payload["active_branches"])
        self.stage_step = int(payload["stage_step"])
        self.global_round = int(payload["global_round"])
        self.final_disposition = payload["final_disposition"]
        if not set(self.active_branches) <= set(self.models):
            raise RuntimeError("R5.1 checkpoint active branches are unavailable")
        if self.current_rung == "binary":
            binary = self.ledger.get("rungs", {}).get("binary", {})
            if isinstance(binary, dict) and bool(binary.get("passed")):
                if model_state_digest(self.models["binary-root"]) != binary.get(
                    "passed_state_digest"
                ):
                    raise RuntimeError(
                        "R5.1 durable Rung 1 evidence does not match checkpoint weights"
                    )
        torch.set_rng_state(payload["torch_rng_state"])
        self.elapsed_before_resume = float(payload["elapsed_seconds"])
        self.process_started = time.monotonic()
        self.last_checkpoint = str(path)


def latest_stage2_ladder_checkpoint(run_dir: Path) -> Path:
    path = run_dir / "checkpoints" / "latest.json"
    if not path.is_file():
        raise FileNotFoundError("R5.1 run has no latest checkpoint receipt")
    raw = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(str(raw["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"R5.1 checkpoint is missing: {checkpoint}")
    return checkpoint
