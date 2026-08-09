"""Training, causal evaluation, and one-shot evidence for Stage 2 R6."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .backend import resolve_backend
from .optim import DirectMLCompatibleAdamWCore
from .stage2_congruence_config import (
    R6_PACKET,
    Stage2CongruenceConfig,
)
from .stage2_congruence_data import (
    StateCongruenceData,
    counterfactual_labels,
    digest,
    index_model_input,
    partner_map_receipt,
    partner_source_indices,
    shortcut_canaries,
)
from .stage2_congruence_model import StateCongruenceModel
from .stage2_ladder_data import ADD_FIRST, SUB_FIRST, LadderGeneratedSplit
from .stage2_ladder_model import model_state_digest, parameter_count


BRANCH_ORDER = (
    "fixed-add-root",
    "fixed-add-teacher",
    "fixed-add-self-duplicate",
    "fixed-add-congruence-true",
    "fixed-add-mixed-counterfactual",
    "fixed-sub-root",
    "fixed-sub-teacher",
    "fixed-sub-self-duplicate",
    "fixed-sub-congruence-true",
    "fixed-sub-mixed-counterfactual",
)
MATCHED_MODES = {
    "self-duplicate",
    "congruence-true",
    "mixed-counterfactual",
}


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest().upper()


def _project_source_files(project_root: Path) -> tuple[Path, ...]:
    files = [
        project_root / "pyproject.toml",
        project_root / "scripts" / "run_stage2_congruence.py",
    ]
    files.extend(sorted((project_root / "src" / "dynamic_hierarchy").glob("*.py")))
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("R6 source snapshot input is missing")
    return tuple(files)


def _source_manifest_payload(project_root: Path) -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in _project_source_files(project_root)
    ]
    source_digest = digest({"packet": R6_PACKET, "files": files})
    return {
        "schema_version": 1,
        "packet": R6_PACKET,
        "source_root": str(project_root.resolve()),
        "files": files,
        "source_digest": source_digest,
    }


def load_or_create_source_snapshot(
    run_dir: Path, *, allow_create: bool
) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    snapshot_dir = run_dir / "snapshot"
    manifest_path = snapshot_dir / "source-manifest.json"
    if not manifest_path.is_file():
        if not allow_create:
            raise FileNotFoundError("R6 source snapshot manifest is required")
        unexpected = {item.name for item in run_dir.iterdir()} - {
            "frozen-config.json"
        }
        if unexpected:
            raise RuntimeError(
                "R6 fresh source snapshot found existing evidence: "
                f"{sorted(unexpected)}"
            )
        payload = _source_manifest_payload(project_root)
        temporary = run_dir / f".snapshot.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
        for item in payload["files"]:
            relative = Path(str(item["path"]))
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project_root / relative, destination)
        atomic_write_json(temporary / "source-manifest.json", payload)
        os.replace(temporary, snapshot_dir)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError("R6 source snapshot manifest is malformed")
    if raw.get("packet") != R6_PACKET:
        raise RuntimeError("R6 source snapshot packet changed")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("R6 source snapshot file list is malformed")
    current = _source_manifest_payload(project_root)
    if raw.get("source_root") != str(project_root.resolve()):
        raise RuntimeError("R6 source snapshot root changed")
    if files != current["files"]:
        raise RuntimeError("R6 current source differs from the frozen snapshot")
    if raw.get("source_digest") != current["source_digest"]:
        raise RuntimeError("R6 source snapshot digest changed")
    expected_snapshot_files = {str(item["path"]) for item in files}
    observed_snapshot_files = {
        path.relative_to(snapshot_dir).as_posix()
        for path in snapshot_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed_snapshot_files != expected_snapshot_files:
        raise RuntimeError("R6 source snapshot file set changed")
    hashes = {str(item["path"]): str(item["sha256"]) for item in files}
    for relative, expected_hash in hashes.items():
        if sha256_file(snapshot_dir / relative) != expected_hash:
            raise RuntimeError("R6 frozen source file hash changed")
    source_root = (project_root / "src" / "dynamic_hierarchy").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "dynamic_hierarchy" and not name.startswith("dynamic_hierarchy."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        imported = Path(module_file).resolve()
        if imported.parent != source_root:
            raise RuntimeError("R6 imported module resolved outside the frozen source root")
        relative = imported.relative_to(project_root).as_posix()
        if hashes.get(relative) != sha256_file(imported):
            raise RuntimeError("R6 imported module hash differs from the source snapshot")
    runner_relative = "scripts/run_stage2_congruence.py"
    runner_expected = (project_root / runner_relative).resolve()
    runner_modules = []
    imported_runner = sys.modules.get("scripts.run_stage2_congruence")
    if imported_runner is not None:
        runner_modules.append(imported_runner)
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file is not None and Path(main_file).name == runner_expected.name:
        runner_modules.append(main_module)
    for module in runner_modules:
        imported = Path(str(module.__file__)).resolve()
        if imported != runner_expected or hashes.get(runner_relative) != sha256_file(
            imported
        ):
            raise RuntimeError("R6 runner import differs from the source snapshot")
    return {
        "path": str(manifest_path),
        "manifest": raw,
        "source_digest": str(raw["source_digest"]),
    }


def load_run_instance_manifest(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise RuntimeError("R6 run-instance manifest is malformed")
    run_instance_id = raw.get("run_instance_id")
    if not isinstance(run_instance_id, str) or len(run_instance_id) != 32:
        raise RuntimeError("R6 run-instance ID is malformed")
    return raw


def load_or_create_run_instance(
    run_dir: Path,
    config_digest: str,
    partition_digest: str,
    source_snapshot_digest: str,
    *,
    allow_create: bool,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run-instance.json"
    if path.is_file():
        manifest = load_run_instance_manifest(path)
    else:
        if not allow_create:
            raise FileNotFoundError("R6 run-instance manifest is required")
        unexpected = {
            item.name for item in run_dir.iterdir()
        } - {"frozen-config.json", "snapshot"}
        if unexpected:
            raise RuntimeError(
                "R6 fresh run-instance creation found existing evidence: "
                f"{sorted(unexpected)}"
            )
        manifest = {
            "schema_version": 2,
            "packet": R6_PACKET,
            "run_instance_id": uuid.uuid4().hex,
            "run_dir": str(run_dir.resolve()),
            "config_digest": config_digest,
            "partition_digest": partition_digest,
            "source_snapshot_digest": source_snapshot_digest,
        }
        atomic_write_json(path, manifest)
    required = {
        "packet": R6_PACKET,
        "run_dir": str(run_dir.resolve()),
        "config_digest": config_digest,
        "partition_digest": partition_digest,
        "source_snapshot_digest": source_snapshot_digest,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("R6 run-instance manifest does not match this run")
    return {
        "path": str(path),
        "manifest": manifest,
        "manifest_digest": digest(manifest),
    }


def existing_initialization_identity(run_dir: Path) -> dict[str, str]:
    identity: dict[str, str] = {}
    try:
        source = load_or_create_source_snapshot(run_dir, allow_create=False)
    except (FileNotFoundError, RuntimeError, OSError, ValueError, TypeError):
        source = None
    if source is not None:
        identity["source_snapshot_digest"] = str(source["source_digest"])
    manifest_path = run_dir / "run-instance.json"
    try:
        manifest = load_run_instance_manifest(manifest_path)
    except (FileNotFoundError, RuntimeError, OSError, ValueError, TypeError):
        manifest = None
    if (
        manifest is not None
        and source is not None
        and manifest.get("source_snapshot_digest") == source["source_digest"]
    ):
        identity["run_instance_digest"] = digest(manifest)
    return identity


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


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def torch_tree_digest(value: Any) -> str:
    result = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(b"tensor\0")
            result.update(str(tensor.dtype).encode("ascii"))
            result.update(json.dumps(list(tensor.shape)).encode("ascii"))
            result.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b"dict\0")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(b"list\0" if isinstance(item, list) else b"tuple\0")
            for child in item:
                update(child)
        elif item is None:
            result.update(b"none\0")
        else:
            result.update(type(item).__name__.encode("ascii"))
            result.update(b"\0")
            result.update(repr(item).encode("ascii"))
            result.update(b"\0")

    update(value)
    return result.hexdigest()


def branch_query(branch: str) -> int:
    if branch.startswith("fixed-add-"):
        return ADD_FIRST
    if branch.startswith("fixed-sub-"):
        return SUB_FIRST
    raise ValueError(f"unknown R6 branch: {branch}")


def branch_mode(branch: str) -> str:
    prefix = "fixed-add-" if branch.startswith("fixed-add-") else "fixed-sub-"
    mode = branch.removeprefix(prefix)
    if mode not in {"root", "teacher", *MATCHED_MODES}:
        raise ValueError(f"unknown R6 branch mode: {branch}")
    return mode


def load_inherited_rung1(
    config: Stage2CongruenceConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    path = Path(config.inherited_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"R6 inherited checkpoint is missing: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != config.inherited_checkpoint_sha256:
        raise RuntimeError("R6 inherited checkpoint SHA256 changed")
    frozen_config_path = path.parent.parent / "frozen-config.json"
    if not frozen_config_path.is_file() or sha256_file(frozen_config_path) != config.inherited_frozen_config_sha256:
        raise RuntimeError("R6 inherited frozen config SHA256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version": 1,
        "packet": "DH-S2-R5.1",
        "global_round": 600,
        "config_digest": config.inherited_config_digest,
        "partition_digest": config.inherited_partition_digest,
        "rung1_state_digest": config.inherited_state_digest,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("R6 inherited checkpoint receipts changed")
    state = payload.get("rung1_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("R6 inherited checkpoint lacks rung1_state")
    cloned = {
        str(name): value.detach().cpu().clone()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    model = StateCongruenceModel(config.model)
    model.load_state_dict(cloned)
    observed_state_digest = model_state_digest(model)
    if observed_state_digest != config.inherited_state_digest:
        raise RuntimeError("R6 inherited Rung 1 state digest changed")
    return cloned, {
        "checkpoint": str(path),
        "checkpoint_sha256": observed_hash,
        "frozen_config": str(frozen_config_path),
        "frozen_config_sha256": sha256_file(frozen_config_path),
        "checkpoint_schema_version": payload["schema_version"],
        "checkpoint_packet": payload["packet"],
        "checkpoint_global_round": payload["global_round"],
        "checkpoint_config_digest": payload["config_digest"],
        "checkpoint_partition_digest": payload["partition_digest"],
        "state_key": "rung1_state",
        "state_digest": observed_state_digest,
    }


class Stage2CongruenceTrainer:
    def __init__(
        self,
        config: Stage2CongruenceConfig,
        run_dir: Path,
        *,
        allow_create_run_instance: bool = False,
        allow_create_source_snapshot: bool | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.run_dir = run_dir
        if allow_create_source_snapshot is None:
            allow_create_source_snapshot = allow_create_run_instance
        self.source_snapshot = load_or_create_source_snapshot(
            run_dir, allow_create=allow_create_source_snapshot
        )
        self.source_snapshot_digest = str(
            self.source_snapshot["source_digest"]
        )
        self.backend = resolve_backend(
            config.device, config.cpu_threads, deterministic=config.deterministic
        )
        torch.manual_seed(config.seed)
        self.data = StateCongruenceData(config.seed)
        self.config_digest = digest(config.to_dict())
        self.run_instance = load_or_create_run_instance(
            run_dir,
            self.config_digest,
            self.data.partition_digest,
            self.source_snapshot_digest,
            allow_create=allow_create_run_instance,
        )
        self.run_instance_digest = str(self.run_instance["manifest_digest"])
        self.ledger_path = run_dir / "r6-evaluation-ledger.json"
        self.ledger = self._load_or_initialize_ledger()
        reserve_state = self.ledger.get("true_reserve_state")
        if reserve_state not in {
            "unopened",
            "not_opened",
            "reserve_opened",
            "reserve_stranded",
            "complete",
        }:
            raise RuntimeError("R6 evaluation ledger reserve state is invalid")
        self.reserve_stranded = reserve_state == "reserve_stranded"
        self._pending_reserve_stranding = reserve_state == "reserve_opened"
        self.models: dict[str, StateCongruenceModel] = {}
        self.optimizers: dict[str, DirectMLCompatibleAdamWCore] = {}
        self.cumulative: dict[str, dict[str, Any]] = {}
        self.partner_counts = {
            branch: [0] * 34
            for branch in BRANCH_ORDER
            if branch_mode(branch) in MATCHED_MODES
        }
        self.partner_map_digests: dict[str, list[str | None]] = {
            branch: [None] * 34
            for branch in BRANCH_ORDER
            if branch_mode(branch) in MATCHED_MODES
        }
        self.source_use_counts: dict[str, list[int]] = {
            branch: [0] * 245
            for branch in BRANCH_ORDER
            if branch_mode(branch) in MATCHED_MODES
        }
        self.value_transition_counts: dict[str, list[list[int]]] = {
            branch: [[0] * 7 for _ in range(7)]
            for branch in BRANCH_ORDER
            if branch_mode(branch) in MATCHED_MODES
        }
        self.operation_counts: dict[str, dict[str, int]] = {
            branch: {
                "first_compositions": 0,
                "outer_compositions": 0,
                "readouts": 0,
                "ce_terms": 0,
            }
            for branch in BRANCH_ORDER
        }
        self.update_sequence: list[str] = []
        inherited, self.inherited_receipt = load_inherited_rung1(config)
        self.initialization_receipt = self._create_models(inherited)
        self.stage_step = 0
        self.global_round = 0
        self.process_started = time.monotonic()
        self.elapsed_before_resume = 0.0
        self.last_checkpoint: str | None = None
        self.research_disposition = self.ledger.get("research_disposition")
        self.execution_disposition = self.ledger.get("execution_disposition")
        self._validate_ledger_semantics()
        self.terminal = self.execution_disposition in {
            "completed",
            "implementation_invalid",
            "reserve_stranded",
        }

    def _load_or_initialize_ledger(self) -> dict[str, object]:
        if not self.ledger_path.is_file():
            return {
                "schema_version": 3,
                "packet": R6_PACKET,
                "config_digest": self.config_digest,
                "partition_digest": self.data.partition_digest,
                "source_snapshot_digest": self.source_snapshot_digest,
                "run_instance_digest": self.run_instance_digest,
                "validation_state": "unopened",
                "validation": None,
                "validation_digest": None,
                "validation_binding": None,
                "validation_disposition": None,
                "true_reserve_state": "unopened",
                "true_reserve": None,
                "true_reserve_digest": None,
                "true_reserve_binding": None,
                "execution_disposition": None,
                "research_disposition": None,
            }
        raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 3:
            raise RuntimeError("R6 evaluation ledger is malformed")
        if raw.get("packet") != R6_PACKET:
            raise RuntimeError("R6 evaluation ledger packet changed")
        if raw.get("config_digest") != self.config_digest:
            raise RuntimeError("R6 evaluation ledger config changed")
        if raw.get("partition_digest") != self.data.partition_digest:
            raise RuntimeError("R6 evaluation ledger partition changed")
        if raw.get("source_snapshot_digest") != self.source_snapshot_digest:
            raise RuntimeError("R6 evaluation ledger source snapshot changed")
        if raw.get("run_instance_digest") != self.run_instance_digest:
            raise RuntimeError("R6 evaluation ledger run instance changed")
        validation_state = raw.get("validation_state")
        if validation_state not in {"unopened", "complete"}:
            raise RuntimeError("R6 evaluation ledger validation state is invalid")
        if validation_state == "complete" and any(
            raw.get(key) is None
            for key in ("validation", "validation_digest", "validation_binding")
        ):
            raise RuntimeError("R6 completed validation evidence is incomplete")
        execution = raw.get("execution_disposition")
        research = raw.get("research_disposition")
        if execution not in {
            None,
            "completed",
            "implementation_invalid",
            "reserve_stranded",
        }:
            raise RuntimeError("R6 evaluation ledger execution state is invalid")
        if research is not None and execution != "completed":
            raise RuntimeError("R6 research disposition lacks completed execution")
        if execution == "implementation_invalid" and research is not None:
            raise RuntimeError("R6 invalid execution cannot carry a research result")
        if raw.get("true_reserve_state") == "complete" and any(
            raw.get(key) is None
            for key in (
                "true_reserve",
                "true_reserve_digest",
                "true_reserve_binding",
            )
        ):
            raise RuntimeError("R6 completed reserve evidence is incomplete")
        return raw

    @staticmethod
    def _empty_cumulative() -> dict[str, Any]:
        return {
            "optimizer_updates": 0,
            "examples": 0,
            "ordinary_correct": 0,
            "intervention_correct": 0,
            "ordinary_loss_sum": 0.0,
            "intervention_loss_sum": 0.0,
            "objective_loss_sum": 0.0,
            "forward_backward_seconds": 0.0,
            "last_gradient_norms": {},
        }

    def _create_models(
        self, inherited: dict[str, torch.Tensor]
    ) -> dict[str, object]:
        state_digests: dict[str, str] = {}
        parameter_counts: dict[str, int] = {}
        optimizer_empty: dict[str, bool] = {}
        cpu_models: dict[str, StateCongruenceModel] = {}
        for branch in BRANCH_ORDER:
            model = StateCongruenceModel(self.config.model)
            model.load_state_dict(
                {name: value.detach().clone() for name, value in inherited.items()}
            )
            cpu_models[branch] = model

        cpu_parameters = [
            parameter
            for model in cpu_models.values()
            for parameter in model.parameters()
        ]
        cpu_storage_pointers = [
            parameter.untyped_storage().data_ptr()
            for parameter in cpu_parameters
        ]
        no_parameter_storage_sharing = (
            len(set(cpu_storage_pointers)) == len(cpu_storage_pointers)
        )
        distinct_parameter_objects = (
            len({id(item) for item in cpu_parameters}) == len(cpu_parameters)
        )

        for branch, cpu_model in cpu_models.items():
            model = cpu_model.to(self.backend.device)
            optimizer = DirectMLCompatibleAdamWCore(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )
            self.models[branch] = model
            self.optimizers[branch] = optimizer
            self.cumulative[branch] = self._empty_cumulative()
            state_digests[branch] = model_state_digest(model)
            parameter_counts[branch] = parameter_count(model)
            optimizer_empty[branch] = not bool(optimizer.state)
        receipt = {
            "inherited_state_digest": self.config.inherited_state_digest,
            "branch_state_digests": state_digests,
            "all_state_digests_equal": set(state_digests.values())
            == {self.config.inherited_state_digest},
            "parameter_counts": parameter_counts,
            "all_parameter_counts_equal": len(set(parameter_counts.values())) == 1,
            "fresh_optimizer_state_empty": optimizer_empty,
            "all_optimizer_states_empty": all(optimizer_empty.values()),
            "distinct_parameter_objects": distinct_parameter_objects,
            "no_parameter_storage_sharing": no_parameter_storage_sharing,
            "branch_order": list(BRANCH_ORDER),
        }
        if not all(
            bool(receipt[key])
            for key in (
                "all_state_digests_equal",
                "all_parameter_counts_equal",
                "all_optimizer_states_empty",
                "distinct_parameter_objects",
                "no_parameter_storage_sharing",
            )
        ):
            raise RuntimeError("R6 branch initialization receipts failed")
        return receipt

    @property
    def is_complete(self) -> bool:
        return self.terminal

    @property
    def needs_gate(self) -> bool:
        return not self.terminal and self.stage_step >= self.config.steps

    def elapsed_seconds(self) -> float:
        return self.elapsed_before_resume + (time.monotonic() - self.process_started)

    def time_budget_exhausted(self) -> bool:
        return self.elapsed_seconds() >= self.config.time_budget_minutes * 60.0

    @staticmethod
    def _operation_receipt(mode: str) -> dict[str, int]:
        if mode == "teacher":
            return {
                "first_compositions": 0,
                "outer_compositions": 1,
                "readouts": 1,
                "ce_terms": 1,
            }
        if mode == "root":
            return {
                "first_compositions": 1,
                "outer_compositions": 1,
                "readouts": 1,
                "ce_terms": 1,
            }
        return {
            "first_compositions": 1,
            "outer_compositions": 2,
            "readouts": 2,
            "ce_terms": 2,
        }

    @staticmethod
    def _expected_partner_counts(steps: int) -> list[int]:
        return [steps // 34 + int(index < steps % 34) for index in range(34)]

    def _expected_value_transitions(
        self, branch: str, steps: int
    ) -> list[list[int]]:
        expected = [[0] * 7 for _ in range(7)]
        batch = self.data.batch(branch_query(branch), "train")
        mode = branch_mode(branch)
        for step in range(steps):
            receipt = partner_map_receipt(batch, mode, 1 + step % 34)
            transitions = receipt["value_transitions"]
            for target_value in range(7):
                for source_value in range(7):
                    expected[target_value][source_value] += int(
                        transitions[target_value][source_value]
                    )
        return expected

    def _training_receipt_checks(self, steps: int) -> dict[str, bool]:
        expected_sequence = list(BRANCH_ORDER) * steps
        expected_partner_counts = self._expected_partner_counts(steps)
        branch_budget = all(
            int(self.cumulative[branch]["optimizer_updates"]) == steps
            and int(self.cumulative[branch]["examples"]) == 245 * steps
            for branch in BRANCH_ORDER
        )
        operations = all(
            self.operation_counts[branch]
            == {
                key: value * steps
                for key, value in self._operation_receipt(
                    branch_mode(branch)
                ).items()
            }
            for branch in BRANCH_ORDER
        )
        partner_counts = all(
            counts == expected_partner_counts
            for counts in self.partner_counts.values()
        )
        source_use = all(
            counts == [steps] * 245 for counts in self.source_use_counts.values()
        )
        transitions = all(
            self.value_transition_counts[branch]
            == self._expected_value_transitions(branch, steps)
            for branch in self.value_transition_counts
        )
        map_digests = True
        map_cycles = True
        for branch, observed in self.partner_map_digests.items():
            batch = self.data.batch(branch_query(branch), "train")
            mode = branch_mode(branch)
            for index, count in enumerate(expected_partner_counts, start=1):
                expected = (
                    partner_map_receipt(batch, mode, index)["map_digest"]
                    if count
                    else None
                )
                map_digests = map_digests and observed[index - 1] == expected
                if count:
                    receipt = partner_map_receipt(batch, mode, index)
                    map_cycles = bool(
                        map_cycles
                        and sum(receipt["cycle_lengths"]) == 245
                        and receipt["source_use_counts"] == [1] * 245
                    )
        return {
            "stage_step_exact": self.stage_step == steps,
            "global_round_exact": self.global_round == steps,
            "branch_update_budget_exact": branch_budget,
            "measured_operation_counts_exact": operations,
            "partner_schedule_exact": partner_counts,
            "source_use_counts_exact": source_use,
            "value_transition_counts_exact": transitions,
            "partner_map_digests_exact": map_digests,
            "partner_map_cycles_exact": map_cycles,
            "branch_update_sequence_exact": self.update_sequence
            == expected_sequence,
            "calibration_306_updates_exact": (
                self.config.run_kind != "calibration_only"
                or self.config.steps == 306
            ),
            "calibration_nine_cycles_exact": (
                self.config.run_kind != "calibration_only"
                or steps != self.config.steps
                or all(counts == [9] * 34 for counts in self.partner_counts.values())
            ),
        }

    def _state_binding(self) -> dict[str, object]:
        binding: dict[str, object] = {
            "run_instance_digest": self.run_instance_digest,
            "source_snapshot_digest": self.source_snapshot_digest,
            "stage_step": self.stage_step,
            "global_round": self.global_round,
            "branch_state_digests": {
                branch: model_state_digest(self.models[branch])
                for branch in BRANCH_ORDER
            },
            "optimizer_state_digests": {
                branch: torch_tree_digest(self.optimizers[branch].state_dict())
                for branch in BRANCH_ORDER
            },
            "cumulative_digest": digest(self.cumulative),
            "partner_counts_digest": digest(self.partner_counts),
            "partner_map_digests_digest": digest(self.partner_map_digests),
            "source_use_counts_digest": digest(self.source_use_counts),
            "value_transition_counts_digest": digest(
                self.value_transition_counts
            ),
            "operation_counts_digest": digest(self.operation_counts),
            "update_sequence_digest": digest(self.update_sequence),
        }
        binding["binding_digest"] = digest(binding)
        return binding

    def _assert_ledger_binding(self, key: str) -> None:
        expected = self.ledger.get(key)
        if not isinstance(expected, dict) or expected != self._state_binding():
            raise RuntimeError("R6 evaluation ledger is bound to another model state")

    @staticmethod
    def _gradient_norms(model: StateCongruenceModel) -> dict[str, object]:
        groups = {
            "literal_embedding": tuple(model.literal_embedding.parameters()),
            "operator_embedding": tuple(model.operator_embedding.parameters()),
            "query_embedding": tuple(model.query_embedding.parameters()),
            "composer": tuple(model.composer.parameters()),
            "readout": tuple(model.readout.parameters()),
        }
        result: dict[str, object] = {}
        for name, parameters in groups.items():
            present = [parameter.grad for parameter in parameters if parameter.grad is not None]
            finite = all(bool(torch.isfinite(grad).all().detach().cpu().item()) for grad in present)
            squared = sum(
                float(grad.detach().float().square().sum().cpu().item())
                for grad in present
            )
            result[name] = {
                "present": bool(present),
                "finite": finite,
                "norm": math.sqrt(squared),
            }
        return result

    def train_step(self) -> dict[str, float]:
        if self.terminal or self.needs_gate:
            raise RuntimeError("R6 train_step called outside the frozen stage")
        if not (
            self.ledger.get("validation_state") == "unopened"
            and self.ledger.get("true_reserve_state") == "unopened"
            and self.ledger.get("execution_disposition") is None
            and self.ledger.get("research_disposition") is None
        ):
            raise RuntimeError("R6 training is forbidden after cohort state changes")
        schedule_index = 1 + (self.stage_step % 34)
        losses: dict[str, float] = {}
        for branch in BRANCH_ORDER:
            query = branch_query(branch)
            mode = branch_mode(branch)
            source_batch = self.data.batch(query, "train")
            batch = source_batch.to(self.backend.device)
            model = self.models[branch]
            optimizer = self.optimizers[branch]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            self.backend.synchronize(next(model.parameters()))
            started = time.perf_counter()
            intervention_loss: torch.Tensor | None = None
            intervention_correct = 0
            source_indices_cpu: torch.Tensor | None = None
            map_receipt: dict[str, object] | None = None
            if mode == "teacher":
                labels = batch.targets.intermediate_labels[:, 0]
                output = model(
                    batch.model_input, teacher_intermediate_labels=labels
                )
                ordinary_loss = F.cross_entropy(
                    output.ordinary_logits, batch.targets.final_labels
                )
                objective = ordinary_loss
            elif mode == "root":
                output = model(batch.model_input)
                ordinary_loss = F.cross_entropy(
                    output.ordinary_logits, batch.targets.final_labels
                )
                objective = ordinary_loss
            else:
                source_indices_cpu = partner_source_indices(
                    source_batch, mode, schedule_index
                )
                map_receipt = partner_map_receipt(
                    source_batch, mode, schedule_index
                )
                source_indices = source_indices_cpu.to(self.backend.device)
                output = model(
                    batch.model_input, source_indices=source_indices
                )
                if output.intervention_logits is None:
                    raise RuntimeError("R6 matched branch lacks intervention logits")
                ordinary_loss = F.cross_entropy(
                    output.ordinary_logits, batch.targets.final_labels
                )
                intervention_targets_cpu = source_batch.targets.final_labels
                if mode == "mixed-counterfactual":
                    source_values = source_batch.targets.intermediate_labels[
                        source_indices_cpu, 0
                    ]
                    intervention_targets_cpu = counterfactual_labels(
                        source_batch,
                        torch.arange(245),
                        source_values,
                    )
                intervention_targets = intervention_targets_cpu.to(
                    self.backend.device
                )
                intervention_loss = F.cross_entropy(
                    output.intervention_logits, intervention_targets
                )
                objective = (
                    ordinary_loss
                    + self.config.intervention_weight * intervention_loss
                ) / (1.0 + self.config.intervention_weight)
                intervention_correct = int(
                    (
                        output.intervention_logits.detach().argmax(dim=-1)
                        == intervention_targets
                    )
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
            objective.backward()
            gradient_norms = self._gradient_norms(model)
            if not all(
                bool(value["finite"])
                for value in gradient_norms.values()
                if isinstance(value, dict)
            ):
                raise RuntimeError("R6 produced a non-finite gradient")
            optimizer.step()
            self.backend.synchronize(next(model.parameters()))
            elapsed = time.perf_counter() - started
            ordinary_correct = int(
                (
                    output.ordinary_logits.detach().argmax(dim=-1)
                    == batch.targets.final_labels
                )
                .sum()
                .detach()
                .cpu()
                .item()
            )
            ordinary_value = self.backend.scalar(ordinary_loss)
            intervention_value = (
                self.backend.scalar(intervention_loss)
                if intervention_loss is not None
                else 0.0
            )
            objective_value = self.backend.scalar(objective)
            cumulative = self.cumulative[branch]
            cumulative["optimizer_updates"] += 1
            cumulative["examples"] += 245
            cumulative["ordinary_correct"] += ordinary_correct
            cumulative["intervention_correct"] += intervention_correct
            cumulative["ordinary_loss_sum"] += ordinary_value
            cumulative["intervention_loss_sum"] += intervention_value
            cumulative["objective_loss_sum"] += objective_value
            cumulative["forward_backward_seconds"] += elapsed
            cumulative["last_gradient_norms"] = gradient_norms
            ce_terms = 2 if intervention_loss is not None else 1
            observed_operations = {
                **output.operation_counts,
                "ce_terms": ce_terms,
            }
            for key, value in observed_operations.items():
                self.operation_counts[branch][key] += int(value)
            if source_indices_cpu is not None and map_receipt is not None:
                self.partner_counts[branch][schedule_index - 1] += 1
                observed_digest = str(map_receipt["map_digest"])
                prior_digest = self.partner_map_digests[branch][schedule_index - 1]
                if prior_digest not in {None, observed_digest}:
                    raise RuntimeError("R6 partner map changed during training")
                self.partner_map_digests[branch][schedule_index - 1] = observed_digest
                for source_index in source_indices_cpu.tolist():
                    self.source_use_counts[branch][source_index] += 1
                transitions = map_receipt["value_transitions"]
                for target_value in range(7):
                    for source_value in range(7):
                        self.value_transition_counts[branch][target_value][
                            source_value
                        ] += int(transitions[target_value][source_value])
            self.update_sequence.append(branch)
            losses[branch] = objective_value
            if self.config.yield_ms:
                time.sleep(self.config.yield_ms / 1000.0)
        self.stage_step += 1
        self.global_round += 1
        return losses

    @staticmethod
    def _metric(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
        predictions = logits.argmax(dim=-1)
        predictions_cpu = predictions.detach().cpu()
        labels_cpu = labels.detach().cpu()
        correct = int((predictions == labels).sum().detach().cpu().item())
        row_nll = F.cross_entropy(logits, labels, reduction="none")
        nll = [float(value) for value in row_nll.detach().cpu().tolist()]
        rows = int(labels.numel())
        return {
            "rows": rows,
            "correct": correct,
            "accuracy": correct / rows,
            "cross_entropy": sum(nll) / rows,
            "nll": nll,
            "predicted_classes": int(
                torch.unique(predictions_cpu).numel()
            ),
            "prediction_class_counts": [
                int((predictions_cpu == label).sum().item()) for label in range(7)
            ],
            "target_class_counts": [
                int((labels_cpu == label).sum().item()) for label in range(7)
            ],
            "predictions": predictions_cpu.tolist(),
        }

    def _evaluate_branch(
        self, branch: str, source_batch: LadderGeneratedSplit
    ) -> dict[str, object]:
        if source_batch.split not in {"validation", "reserve"}:
            raise ValueError("R6 evaluation accepts validation or reserve only")
        model = self.models[branch]
        mode = branch_mode(branch)
        batch = source_batch.to(self.backend.device)
        model.eval()
        with torch.no_grad():
            if mode == "teacher":
                source_states = model.literal_embedding(
                    batch.targets.intermediate_labels[:, 0]
                )
                ordinary_logits = model.outer_logits(
                    batch.model_input, source_states
                )
            else:
                source_states = model.first_states(batch.model_input)
                ordinary_logits = model.outer_logits(
                    batch.model_input, source_states
                )
            ordinary = self._metric(
                ordinary_logits, batch.targets.final_labels
            )
            rows = len(source_batch.targets.final_labels)
            target_cpu = torch.arange(rows).repeat_interleave(rows)
            source_cpu = torch.arange(rows).repeat(rows)
            target_input_cpu = index_model_input(
                source_batch.model_input, target_cpu
            )
            target_input = target_input_cpu.to(self.backend.device)
            source_indices = source_cpu.to(self.backend.device)
            all_logits = model.outer_logits(
                target_input, source_states[source_indices]
            )
            source_values_cpu = source_batch.targets.intermediate_labels[
                source_cpu, 0
            ]
            all_targets_cpu = counterfactual_labels(
                source_batch, target_cpu, source_values_cpu
            )
            all_targets = all_targets_cpu.to(self.backend.device)
            target_values_cpu = source_batch.targets.intermediate_labels[
                target_cpu, 0
            ]
            same_cpu = source_values_cpu == target_values_cpu
            nonself_cpu = same_cpu & (source_cpu != target_cpu)
            wrong_cpu = ~same_cpu

            def subset(mask_cpu: torch.Tensor) -> dict[str, object]:
                indices = torch.nonzero(mask_cpu, as_tuple=False).flatten().to(
                    self.backend.device
                )
                return self._metric(all_logits[indices], all_targets[indices])

            same = subset(same_cpu)
            nonself = subset(nonself_cpu)
            all_state = self._metric(all_logits, all_targets)
            wrong = subset(wrong_cpu)
            literal_logits = model.outer_logits(
                batch.model_input,
                model.literal_embedding(batch.targets.intermediate_labels[:, 0]),
            )
            literal = self._metric(literal_logits, batch.targets.final_labels)
            all_predictions = all_logits.argmax(dim=-1).detach().cpu()
            target_row_ids = [
                source_batch.row_hashes[index] for index in target_cpu.tolist()
            ]
            source_row_ids = [
                source_batch.row_hashes[index] for index in source_cpu.tolist()
            ]
            pair_ids = [
                digest(
                    {
                        "target_row_id": target_row_id,
                        "source_row_id": source_row_id,
                    }
                )
                for target_row_id, source_row_id in zip(
                    target_row_ids, source_row_ids, strict=True
                )
            ]
            target_family_ids = [
                source_batch.family_hashes[index] for index in target_cpu.tolist()
            ]
            source_family_ids = [
                source_batch.family_hashes[index] for index in source_cpu.tolist()
            ]
            target_answers_cpu = source_batch.targets.final_labels[target_cpu]
            values = source_batch.model_input.values
            query = branch_query(branch)
            ordinary_replay = (values[:, 0] + values[:, 1] - values[:, 2]) % 7
            source_rows = values[source_cpu]
            source_value_replay = (
                (source_rows[:, 0] + source_rows[:, 1]) % 7
                if query == ADD_FIRST
                else (source_rows[:, 1] - source_rows[:, 2]) % 7
            )
            target_rows = values[target_cpu]
            counterfactual_replay = (
                (source_value_replay - target_rows[:, 2]) % 7
                if query == ADD_FIRST
                else (target_rows[:, 0] + source_value_replay) % 7
            )
            exact_replay = {
                "ordinary_targets": torch.equal(
                    ordinary_replay, source_batch.targets.final_labels
                ),
                "source_values": torch.equal(
                    source_value_replay, source_values_cpu
                ),
                "counterfactual_targets": torch.equal(
                    counterfactual_replay, all_targets_cpu
                ),
                "pair_ids_unique": len(set(pair_ids)) == 2401,
                "denominators": bool(
                    ordinary["rows"] == 49
                    and same["rows"] == 343
                    and nonself["rows"] == 294
                    and all_state["rows"] == 2401
                    and wrong["rows"] == 2058
                ),
                "pair_direction": True,
            }
            errors = torch.nonzero(
                all_predictions != all_targets_cpu,
                as_tuple=False,
            ).flatten()
            error_pairs = [
                [
                    int(target_cpu[index]),
                    int(source_cpu[index]),
                    int(all_predictions[index]),
                    int(all_targets_cpu[index]),
                    pair_ids[index],
                ]
                for index in errors.tolist()
            ]
            matrix_digest = digest(
                {
                    "target_indices": target_cpu.tolist(),
                    "source_indices": source_cpu.tolist(),
                    "pair_ids": pair_ids,
                    "target_family_ids": target_family_ids,
                    "source_family_ids": source_family_ids,
                    "source_values": source_values_cpu.tolist(),
                    "target_values": target_values_cpu.tolist(),
                    "target_answers": target_answers_cpu.tolist(),
                    "predictions": all_predictions.tolist(),
                    "targets": all_targets_cpu.tolist(),
                }
            )
            pair_receipt = {
                "target_indices": target_cpu.tolist(),
                "source_indices": source_cpu.tolist(),
                "pair_ids": pair_ids,
                "target_row_ids": target_row_ids,
                "source_row_ids": source_row_ids,
                "target_family_ids": target_family_ids,
                "source_family_ids": source_family_ids,
                "source_values": source_values_cpu.tolist(),
                "target_values": target_values_cpu.tolist(),
                "target_answers": target_answers_cpu.tolist(),
                "counterfactual_answers": all_targets_cpu.tolist(),
                "predictions": all_predictions.tolist(),
            }
        evidence_passed = all(bool(value) for value in exact_replay.values())
        semantic = bool(
            evidence_passed
            and ordinary["correct"] == 49
            and ordinary["predicted_classes"] == self.config.required_predicted_classes
            and same["correct"] == 343
            and nonself["correct"] == 294
            and all_state["correct"] == 2401
            and wrong["correct"] == 2058
        )
        confidence = all(
            float(metrics["cross_entropy"]) <= self.config.max_cross_entropy
            for metrics in (ordinary, same, nonself, all_state, wrong)
        )
        result = {
            "branch": branch,
            "mode": mode,
            "query_id": branch_query(branch),
            "split": source_batch.split,
            "ordinary": ordinary,
            "same_value": same,
            "nonself_same_value": nonself,
            "all_state_counterfactual": all_state,
            "wrong_state_counterfactual": wrong,
            "literal_injection_report_only": literal,
            "semantic_accuracy_passed": semantic,
            "confidence_passed": confidence,
            "full_gate_passed": bool(semantic and confidence),
            "pair_order": "target-major; target outer context plus source state",
            "ordinary_row_ids": list(source_batch.row_hashes),
            "target_family_hashes": list(source_batch.family_hashes),
            "source_family_hashes": list(source_batch.family_hashes),
            "pair_receipt": pair_receipt,
            "exact_replay": exact_replay,
            "evidence_passed": evidence_passed,
            "matrix_digest": matrix_digest,
            "all_state_predictions": all_predictions.tolist(),
            "error_pairs": error_pairs,
            "operation_receipt": self._operation_receipt(mode),
        }
        result["finite"] = _all_finite(result)
        return result

    def _derive_branch_gate(
        self,
        metrics: dict[str, object],
        source_batch: LadderGeneratedSplit,
        max_cross_entropy: float,
        required_predicted_classes: int = 7,
    ) -> dict[str, bool]:
        if len(source_batch.targets.final_labels) != 49:
            raise RuntimeError("R6 frozen evaluation batch size changed")
        query_ids = source_batch.model_input.query_ids.tolist()
        if len(set(query_ids)) != 1 or metrics.get("query_id") != query_ids[0]:
            raise RuntimeError("R6 branch query disagrees with frozen evaluation data")
        if metrics.get("split") != source_batch.split:
            raise RuntimeError("R6 branch split disagrees with frozen evaluation data")

        rows = 49
        target_indices = torch.arange(rows).repeat_interleave(rows)
        source_indices = torch.arange(rows).repeat(rows)
        target_indices_list = target_indices.tolist()
        source_indices_list = source_indices.tolist()
        target_row_ids = [
            source_batch.row_hashes[index] for index in target_indices_list
        ]
        source_row_ids = [
            source_batch.row_hashes[index] for index in source_indices_list
        ]
        pair_ids = [
            digest(
                {
                    "target_row_id": target_row_id,
                    "source_row_id": source_row_id,
                }
            )
            for target_row_id, source_row_id in zip(
                target_row_ids, source_row_ids, strict=True
            )
        ]
        target_family_ids = [
            source_batch.family_hashes[index] for index in target_indices_list
        ]
        source_family_ids = [
            source_batch.family_hashes[index] for index in source_indices_list
        ]
        source_values_tensor = source_batch.targets.intermediate_labels[
            source_indices, 0
        ]
        target_values_tensor = source_batch.targets.intermediate_labels[
            target_indices, 0
        ]
        target_answers_tensor = source_batch.targets.final_labels[target_indices]
        counterfactual_tensor = counterfactual_labels(
            source_batch, target_indices, source_values_tensor
        )
        source_values = source_values_tensor.tolist()
        target_values = target_values_tensor.tolist()
        target_answers = target_answers_tensor.tolist()
        counterfactual_answers = counterfactual_tensor.tolist()
        expected_pair = {
            "target_indices": target_indices_list,
            "source_indices": source_indices_list,
            "pair_ids": pair_ids,
            "target_row_ids": target_row_ids,
            "source_row_ids": source_row_ids,
            "target_family_ids": target_family_ids,
            "source_family_ids": source_family_ids,
            "source_values": source_values,
            "target_values": target_values,
            "target_answers": target_answers,
            "counterfactual_answers": counterfactual_answers,
        }
        pair_receipt = metrics.get("pair_receipt")
        if not isinstance(pair_receipt, dict):
            raise RuntimeError("R6 branch pair receipt is malformed")
        if any(pair_receipt.get(key) != value for key, value in expected_pair.items()):
            raise RuntimeError("R6 branch pair receipt disagrees with frozen data")
        predictions_value = pair_receipt.get("predictions")
        if not isinstance(predictions_value, list) or len(predictions_value) != rows * rows:
            raise RuntimeError("R6 branch pair predictions are malformed")
        all_predictions = [int(value) for value in predictions_value]
        if any(value not in range(7) for value in all_predictions):
            raise RuntimeError("R6 branch pair prediction is outside the label space")

        values = source_batch.model_input.values
        ordinary_replay = (values[:, 0] + values[:, 1] - values[:, 2]) % 7
        source_rows = values[source_indices]
        if query_ids[0] == ADD_FIRST:
            source_value_replay = (source_rows[:, 0] + source_rows[:, 1]) % 7
            target_rows = values[target_indices]
            counterfactual_replay = (
                source_value_replay - target_rows[:, 2]
            ) % 7
        elif query_ids[0] == SUB_FIRST:
            source_value_replay = (source_rows[:, 1] - source_rows[:, 2]) % 7
            target_rows = values[target_indices]
            counterfactual_replay = (
                target_rows[:, 0] + source_value_replay
            ) % 7
        else:
            raise RuntimeError("R6 frozen evaluation query is invalid")
        exact_replay = {
            "ordinary_targets": torch.equal(
                ordinary_replay, source_batch.targets.final_labels
            ),
            "source_values": torch.equal(
                source_value_replay, source_values_tensor
            ),
            "counterfactual_targets": torch.equal(
                counterfactual_replay, counterfactual_tensor
            ),
            "pair_ids_unique": len(set(pair_ids)) == rows * rows,
            "denominators": True,
            "pair_direction": True,
        }
        if metrics.get("exact_replay") != exact_replay:
            raise RuntimeError("R6 branch exact replay disagrees with frozen data")
        evidence = all(exact_replay.values())
        if metrics.get("evidence_passed") is not evidence:
            raise RuntimeError("R6 branch evidence summary disagrees with replay")

        rebuilt_matrix_digest = digest(
            {
                "target_indices": target_indices_list,
                "source_indices": source_indices_list,
                "pair_ids": pair_ids,
                "target_family_ids": target_family_ids,
                "source_family_ids": source_family_ids,
                "source_values": source_values,
                "target_values": target_values,
                "target_answers": target_answers,
                "predictions": all_predictions,
                "targets": counterfactual_answers,
            }
        )
        if metrics.get("matrix_digest") != rebuilt_matrix_digest:
            raise RuntimeError("R6 branch intervention matrix digest changed")
        if metrics.get("pair_order") != (
            "target-major; target outer context plus source state"
        ):
            raise RuntimeError("R6 branch pair direction changed")
        if metrics.get("ordinary_row_ids") != list(source_batch.row_hashes):
            raise RuntimeError("R6 ordinary row identities changed")
        if metrics.get("target_family_hashes") != list(source_batch.family_hashes):
            raise RuntimeError("R6 target family identities changed")
        if metrics.get("source_family_hashes") != list(source_batch.family_hashes):
            raise RuntimeError("R6 source family identities changed")
        if metrics.get("all_state_predictions") != all_predictions:
            raise RuntimeError("R6 all-state predictions disagree with pair receipt")

        same_mask = [
            source == target
            for source, target in zip(source_values, target_values, strict=True)
        ]
        nonself_mask = [
            same and source_index != target_index
            for same, source_index, target_index in zip(
                same_mask,
                source_indices_list,
                target_indices_list,
                strict=True,
            )
        ]
        wrong_mask = [not same for same in same_mask]
        labels_by_name = {
            "ordinary": source_batch.targets.final_labels.tolist(),
            "same_value": [
                label
                for label, keep in zip(
                    counterfactual_answers, same_mask, strict=True
                )
                if keep
            ],
            "nonself_same_value": [
                label
                for label, keep in zip(
                    counterfactual_answers, nonself_mask, strict=True
                )
                if keep
            ],
            "all_state_counterfactual": counterfactual_answers,
            "wrong_state_counterfactual": [
                label
                for label, keep in zip(
                    counterfactual_answers, wrong_mask, strict=True
                )
                if keep
            ],
        }
        expected_predictions = {
            "same_value": [
                value
                for value, keep in zip(all_predictions, same_mask, strict=True)
                if keep
            ],
            "nonself_same_value": [
                value
                for value, keep in zip(
                    all_predictions, nonself_mask, strict=True
                )
                if keep
            ],
            "all_state_counterfactual": all_predictions,
            "wrong_state_counterfactual": [
                value
                for value, keep in zip(all_predictions, wrong_mask, strict=True)
                if keep
            ],
        }
        raw: dict[str, dict[str, object]] = {}
        for name, labels in labels_by_name.items():
            expected_rows = len(labels)
            item = metrics.get(name)
            if not isinstance(item, dict):
                raise RuntimeError(f"R6 branch metric {name} is malformed")
            if item.get("rows") != expected_rows:
                raise RuntimeError(f"R6 branch metric {name} denominator changed")
            predictions = item.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != len(labels):
                raise RuntimeError("R6 branch prediction vector is malformed")
            predictions = [int(value) for value in predictions]
            if any(value not in range(7) for value in predictions):
                raise RuntimeError("R6 branch prediction is outside the label space")
            if name in expected_predictions and predictions != expected_predictions[name]:
                raise RuntimeError("R6 subset predictions disagree with all-state rows")
            correct = sum(
                prediction == label
                for prediction, label in zip(predictions, labels, strict=True)
            )
            prediction_counts = [predictions.count(label) for label in range(7)]
            target_counts = [labels.count(label) for label in range(7)]
            nll = item.get("nll")
            if (
                not isinstance(nll, list)
                or len(nll) != expected_rows
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in nll
                )
            ):
                raise RuntimeError("R6 branch row NLL evidence is malformed")
            reconstructed_cross_entropy = sum(float(value) for value in nll) / expected_rows
            cross_entropy = item.get("cross_entropy")
            if (
                not isinstance(cross_entropy, (int, float))
                or not math.isfinite(float(cross_entropy))
                or float(cross_entropy) < 0.0
                or not math.isclose(
                    float(cross_entropy),
                    reconstructed_cross_entropy,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError("R6 branch cross-entropy disagrees with row NLL")
            accuracy = item.get("accuracy")
            if (
                not isinstance(accuracy, (int, float))
                or not math.isclose(
                    float(accuracy),
                    correct / expected_rows,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError("R6 branch accuracy disagrees with raw predictions")
            if (
                item.get("correct") != correct
                or item.get("prediction_class_counts") != prediction_counts
                or item.get("target_class_counts") != target_counts
                or item.get("predicted_classes")
                != sum(count > 0 for count in prediction_counts)
            ):
                raise RuntimeError("R6 branch raw metric summary changed")
            raw[name] = item
        expected_errors = [
            [
                target_indices_list[index],
                source_indices_list[index],
                prediction,
                counterfactual_answers[index],
                pair_ids[index],
            ]
            for index, prediction in enumerate(all_predictions)
            if prediction != counterfactual_answers[index]
        ]
        if metrics.get("error_pairs") != expected_errors:
            raise RuntimeError("R6 branch error-pair evidence changed")
        finite_payload = {
            key: value for key, value in metrics.items() if key != "finite"
        }
        if metrics.get("finite") is not _all_finite(finite_payload):
            raise RuntimeError("R6 branch finite summary disagrees with raw evidence")
        semantic = bool(
            evidence
            and raw["ordinary"]["correct"] == 49
            and raw["ordinary"]["predicted_classes"]
            == required_predicted_classes
            and raw["same_value"]["correct"] == 343
            and raw["nonself_same_value"]["correct"] == 294
            and raw["all_state_counterfactual"]["correct"] == 2401
            and raw["wrong_state_counterfactual"]["correct"] == 2058
        )
        confidence = all(
            float(raw[name]["cross_entropy"]) <= max_cross_entropy
            for name in labels_by_name
        )
        derived = {
            "semantic_accuracy_passed": semantic,
            "confidence_passed": confidence,
            "full_gate_passed": bool(semantic and confidence),
        }
        if any(metrics.get(key) is not value for key, value in derived.items()):
            raise RuntimeError("R6 branch gate summary disagrees with raw metrics")
        return derived

    def validation_decision(
        self,
        validation: dict[str, dict[str, object]],
        invariants_passed: bool,
        max_cross_entropy: float = 0.10,
    ) -> str:
        if not invariants_passed:
            return "implementation_invalid"
        if set(validation) != set(BRANCH_ORDER):
            raise RuntimeError("R6 validation branch set changed")
        derived = {
            branch: self._derive_branch_gate(
                validation[branch],
                self.data.batch(branch_query(branch), "validation"),
                max_cross_entropy,
            )
            for branch in BRANCH_ORDER
        }
        roots = (
            derived["fixed-add-root"],
            derived["fixed-sub-root"],
        )
        if any(bool(item["semantic_accuracy_passed"]) for item in roots):
            return "task_ceiling"
        true = (
            derived["fixed-add-congruence-true"],
            derived["fixed-sub-congruence-true"],
        )
        if not all(bool(item["full_gate_passed"]) for item in true):
            return "state_congruence_failed"
        self_controls = (
            derived["fixed-add-self-duplicate"],
            derived["fixed-sub-self-duplicate"],
        )
        if any(
            bool(item["semantic_accuracy_passed"]) for item in self_controls
        ):
            return "control_sufficient"
        mixed = (
            derived["fixed-add-mixed-counterfactual"],
            derived["fixed-sub-mixed-counterfactual"],
        )
        if any(bool(item["semantic_accuracy_passed"]) for item in mixed):
            return "valid_augmentation_non_specific"
        return "reserve_eligible"

    def _reserve_research_decision(
        self,
        reserve: dict[str, dict[str, object]],
        max_cross_entropy: float = 0.10,
    ) -> str:
        required = {
            "fixed-add-congruence-true",
            "fixed-sub-congruence-true",
        }
        if set(reserve) != required:
            raise RuntimeError("R6 reserve branch set is invalid")
        derived = {
            branch: self._derive_branch_gate(
                metrics,
                self.data.batch(
                    branch_query(branch), "reserve", materialize=False
                ),
                max_cross_entropy,
            )
            for branch, metrics in reserve.items()
        }
        return (
            "state_congruence_signal"
            if all(bool(item["full_gate_passed"]) for item in derived.values())
            else "state_congruence_failed"
        )

    def _validate_ledger_semantics(self) -> None:
        validation_state = self.ledger.get("validation_state")
        reserve_state = self.ledger.get("true_reserve_state")
        execution = self.ledger.get("execution_disposition")
        research = self.ledger.get("research_disposition")
        validation_disposition = self.ledger.get("validation_disposition")
        reserve = self.ledger.get("true_reserve")
        empty_reserve_evidence = all(
            self.ledger.get(key) is None
            for key in (
                "true_reserve",
                "true_reserve_digest",
                "true_reserve_binding",
            )
        )
        if validation_state == "unopened":
            if not (
                self.ledger.get("validation") is None
                and self.ledger.get("validation_digest") is None
                and self.ledger.get("validation_binding") is None
                and validation_disposition is None
                and reserve_state == "unopened"
                and empty_reserve_evidence
                and execution is None
                and research is None
            ):
                raise RuntimeError("R6 unopened ledger state combination is invalid")
            return
        validation = self.ledger.get("validation")
        invariants = self.ledger.get("invariants")
        if not isinstance(validation, dict) or not isinstance(invariants, dict):
            raise RuntimeError("R6 completed validation state lacks evidence")
        if self.ledger.get("validation_digest") != digest(validation):
            raise RuntimeError("R6 completed validation digest changed")
        validation_binding = self.ledger.get("validation_binding")
        if (
            not isinstance(validation_binding, dict)
            or validation_binding.get("run_instance_digest")
            != self.run_instance_digest
            or validation_binding.get("source_snapshot_digest")
            != self.source_snapshot_digest
        ):
            raise RuntimeError("R6 completed validation binding is malformed")
        expected_invariant_keys = set(self._invariants()) | {
            "all_validation_metrics_finite"
        }
        if (
            set(invariants) != expected_invariant_keys
            or any(type(value) is not bool for value in invariants.values())
        ):
            raise RuntimeError("R6 validation invariant schema changed")
        derived = self.validation_decision(
            validation,
            all(bool(value) for value in invariants.values()),
            self.config.max_cross_entropy,
        )
        if validation_disposition is None:
            if not (
                reserve_state == "unopened"
                and empty_reserve_evidence
                and execution is None
                and research is None
            ):
                raise RuntimeError("R6 pre-decision validation state is invalid")
            return
        if validation_disposition != derived:
            raise RuntimeError("R6 validation disposition does not match evidence")
        if derived == "implementation_invalid":
            expected = (
                reserve_state == "not_opened"
                and empty_reserve_evidence
                and execution == "implementation_invalid"
                and research is None
            )
        elif derived != "reserve_eligible":
            expected = (
                reserve_state == "not_opened"
                and empty_reserve_evidence
                and execution == "completed"
                and research == derived
            )
        elif self.config.run_kind == "smoke":
            expected = (
                reserve_state == "not_opened"
                and empty_reserve_evidence
                and execution == "completed"
                and research is None
                and self.ledger.get("smoke_reserve_disposition")
                == "eligible_but_forbidden_in_smoke"
            )
        elif reserve_state == "reserve_opened":
            expected = empty_reserve_evidence and execution is None and research is None
        elif reserve_state == "reserve_stranded":
            expected = (
                empty_reserve_evidence
                and execution == "reserve_stranded"
                and research is None
            )
        elif reserve_state == "complete":
            expected = (
                isinstance(reserve, dict)
                and self.ledger.get("true_reserve_digest") == digest(reserve)
                and isinstance(self.ledger.get("true_reserve_binding"), dict)
                and execution == "completed"
                and research
                == self._reserve_research_decision(
                    reserve, self.config.max_cross_entropy
                )
            )
        else:
            expected = False
        if not expected:
            raise RuntimeError("R6 terminal ledger state combination is invalid")

    def _strand_verified_open_reserve(self) -> None:
        if not self._pending_reserve_stranding:
            return
        if self.ledger.get("true_reserve_state") != "reserve_opened":
            raise RuntimeError("R6 pending reserve state changed before recovery")
        self._validate_ledger_semantics()
        self._assert_ledger_binding("validation_binding")
        validation = self.ledger.get("validation")
        invariants = self.ledger.get("invariants")
        if not isinstance(validation, dict) or not isinstance(invariants, dict):
            raise RuntimeError("R6 open reserve lacks complete validation evidence")
        recomputed_invariants = self._invariants()
        recomputed_invariants["all_validation_metrics_finite"] = all(
            isinstance(item, dict) and bool(item.get("finite"))
            for item in validation.values()
        )
        if invariants != recomputed_invariants:
            raise RuntimeError("R6 open reserve invariant receipt changed")
        replay = self.verify_completed_validation_replay()
        if not replay["matched"] or not replay["ledger_unchanged"]:
            raise RuntimeError("R6 open reserve validation failed exact replay")
        self.ledger["true_reserve_state"] = "reserve_stranded"
        self.ledger["execution_disposition"] = "reserve_stranded"
        self.ledger["research_disposition"] = None
        self._pending_reserve_stranding = False
        self.reserve_stranded = True
        self.execution_disposition = "reserve_stranded"
        self.research_disposition = None
        self.terminal = True
        self._validate_ledger_semantics()
        atomic_write_json(self.ledger_path, self.ledger)

    def _invariants(self) -> dict[str, bool]:
        matched_receipts = [
            self._operation_receipt(mode) for mode in sorted(MATCHED_MODES)
        ]
        train_hashes = self.data.family_hashes("train")
        return {
            "all_branches_present": set(self.models) == set(BRANCH_ORDER),
            "branch_order_exact": tuple(self.models) == BRANCH_ORDER,
            "train_full_batch_exact": len(train_hashes) == 245
            and tuple(sorted(train_hashes)) == train_hashes,
            "partition_digest_exact": self.data.partition_digest
            == self.config.partition_digest,
            "initialization_passed": all(
                bool(self.initialization_receipt[key])
                for key in (
                    "all_state_digests_equal",
                    "all_parameter_counts_equal",
                    "all_optimizer_states_empty",
                    "distinct_parameter_objects",
                    "no_parameter_storage_sharing",
                )
            ),
            "matched_operation_counts_equal": len(
                {json.dumps(item, sort_keys=True) for item in matched_receipts}
            )
            == 1,
            "validation_reserve_unmaterialized": not any(
                self.data.is_materialized(query, "reserve")
                for query in (ADD_FIRST, SUB_FIRST)
            ),
            "inherited_checkpoint_unchanged": sha256_file(
                Path(self.config.inherited_checkpoint)
            )
            == self.config.inherited_checkpoint_sha256,
            "inherited_frozen_config_unchanged": sha256_file(
                Path(self.config.inherited_checkpoint).parent.parent
                / "frozen-config.json"
            )
            == self.config.inherited_frozen_config_sha256,
            **self._training_receipt_checks(self.config.steps),
        }

    def _finish(
        self,
        execution_disposition: str,
        research_disposition: str | None,
    ) -> None:
        self.ledger["execution_disposition"] = execution_disposition
        self.ledger["research_disposition"] = research_disposition
        self.execution_disposition = execution_disposition
        self.research_disposition = research_disposition
        self.terminal = True
        self._validate_ledger_semantics()

    def _finish_implementation_invalid(self, reason: str) -> dict[str, object]:
        if self.ledger.get("true_reserve_state") == "unopened":
            self.ledger["true_reserve_state"] = "not_opened"
        self.ledger["implementation_error"] = reason
        self._finish("implementation_invalid", None)
        atomic_write_json(self.ledger_path, self.ledger)
        return self.ledger

    def run_gate(self) -> dict[str, object]:
        if not self.needs_gate:
            raise RuntimeError("R6 gate requested before 306/full smoke updates")
        if self.reserve_stranded:
            raise RuntimeError("R6 reserve is stranded")
        validation = self.ledger.get("validation")
        if self.ledger.get("validation_state") == "complete":
            if digest(validation) != self.ledger.get("validation_digest"):
                raise RuntimeError("R6 completed validation digest changed")
            self._assert_ledger_binding("validation_binding")
        else:
            computed: dict[str, dict[str, object]] = {}
            for branch in BRANCH_ORDER:
                batch = self.data.batch(branch_query(branch), "validation")
                computed[branch] = self._evaluate_branch(branch, batch)
            invariants = self._invariants()
            invariants["all_validation_metrics_finite"] = all(
                bool(item["finite"]) for item in computed.values()
            )
            validation = computed
            self.ledger["validation_state"] = "complete"
            self.ledger["validation"] = validation
            self.ledger["validation_digest"] = digest(validation)
            self.ledger["validation_binding"] = self._state_binding()
            self.ledger["invariants"] = invariants
            atomic_write_json(self.ledger_path, self.ledger)
        if not isinstance(validation, dict):
            raise RuntimeError("R6 validation cohort is malformed")
        invariants = self.ledger.get("invariants")
        if not isinstance(invariants, dict):
            raise RuntimeError("R6 validation invariants are missing")
        recomputed_invariants = self._invariants()
        recomputed_invariants["all_validation_metrics_finite"] = all(
            isinstance(item, dict) and bool(item.get("finite"))
            for item in validation.values()
        )
        if invariants != recomputed_invariants:
            return self._finish_implementation_invalid(
                "validation invariant receipt does not match current state"
            )
        decision = self.validation_decision(
            validation,
            all(bool(value) for value in invariants.values()),
            self.config.max_cross_entropy,
        )
        self.ledger["validation_disposition"] = decision
        if decision == "implementation_invalid":
            return self._finish_implementation_invalid(
                "one or more preregistered implementation invariants failed"
            )
        if decision != "reserve_eligible":
            self.ledger["true_reserve_state"] = "not_opened"
            self._finish("completed", decision)
            atomic_write_json(self.ledger_path, self.ledger)
            return self.ledger
        if self.config.run_kind == "smoke":
            self.ledger["true_reserve_state"] = "not_opened"
            self.ledger["smoke_reserve_disposition"] = (
                "eligible_but_forbidden_in_smoke"
            )
            self._finish("completed", None)
            atomic_write_json(self.ledger_path, self.ledger)
            return self.ledger
        if self.ledger.get("true_reserve_state") != "unopened":
            return self._finish_implementation_invalid(
                "calibration reserve can open only from unopened"
            )
        self.ledger["true_reserve_state"] = "reserve_opened"
        self._validate_ledger_semantics()
        atomic_write_json(self.ledger_path, self.ledger)
        reserve: dict[str, dict[str, object]] = {}
        for branch in (
            "fixed-add-congruence-true",
            "fixed-sub-congruence-true",
        ):
            batch = self.data.batch(branch_query(branch), "reserve")
            reserve[branch] = self._evaluate_branch(branch, batch)
        self.ledger["true_reserve"] = reserve
        self.ledger["true_reserve_digest"] = digest(reserve)
        self.ledger["true_reserve_binding"] = self._state_binding()
        self.ledger["true_reserve_state"] = "complete"
        passed = all(bool(item["full_gate_passed"]) for item in reserve.values())
        research = "state_congruence_signal" if passed else "state_congruence_failed"
        self._finish("completed", research)
        atomic_write_json(self.ledger_path, self.ledger)
        return self.ledger

    def verify_completed_validation_replay(self) -> dict[str, object]:
        if self.ledger.get("validation_state") != "complete":
            raise RuntimeError("R6 validation replay requires completed evidence")
        self._assert_ledger_binding("validation_binding")
        before = digest(self.ledger)
        replay = {
            branch: self._evaluate_branch(
                branch,
                self.data.batch(branch_query(branch), "validation"),
            )
            for branch in BRANCH_ORDER
        }
        replay_digest = digest(replay)
        expected_digest = self.ledger.get("validation_digest")
        after = digest(self.ledger)
        return {
            "matched": replay_digest == expected_digest,
            "expected_digest": expected_digest,
            "replay_digest": replay_digest,
            "ledger_unchanged": before == after,
        }

    def verify_completed_reserve_replay(self) -> dict[str, object]:
        if self.ledger.get("true_reserve_state") != "complete":
            raise RuntimeError("R6 reserve replay requires completed evidence")
        self._assert_ledger_binding("true_reserve_binding")
        before = digest(self.ledger)
        replay = {
            branch: self._evaluate_branch(
                branch,
                self.data.batch(branch_query(branch), "reserve"),
            )
            for branch in (
                "fixed-add-congruence-true",
                "fixed-sub-congruence-true",
            )
        }
        replay_digest = digest(replay)
        expected_digest = self.ledger.get("true_reserve_digest")
        after = digest(self.ledger)
        return {
            "matched": replay_digest == expected_digest,
            "expected_digest": expected_digest,
            "replay_digest": replay_digest,
            "ledger_unchanged": before == after,
            "semantics": (
                "read-only integrity replay of an already opened cohort; "
                "no training, selection, or second research decision"
            ),
        }

    def training_report(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for branch in BRANCH_ORDER:
            cumulative = self.cumulative[branch]
            updates = int(cumulative["optimizer_updates"])
            examples = int(cumulative["examples"])
            result[branch] = {
                **cumulative,
                "ordinary_accuracy": (
                    int(cumulative["ordinary_correct"]) / examples
                    if examples
                    else 0.0
                ),
                "intervention_accuracy": (
                    int(cumulative["intervention_correct"]) / examples
                    if examples and branch_mode(branch) in MATCHED_MODES
                    else None
                ),
                "mean_ordinary_loss": (
                    float(cumulative["ordinary_loss_sum"]) / updates
                    if updates
                    else None
                ),
                "mean_intervention_loss": (
                    float(cumulative["intervention_loss_sum"]) / updates
                    if updates and branch_mode(branch) in MATCHED_MODES
                    else None
                ),
                "mean_objective_loss": (
                    float(cumulative["objective_loss_sum"]) / updates
                    if updates
                    else None
                ),
                "mode": branch_mode(branch),
                "query_id": branch_query(branch),
                "train_rows_per_update": 245,
                "expected_operations_per_update": self._operation_receipt(
                    branch_mode(branch)
                ),
                "measured_operation_counts": self.operation_counts[branch],
                "parameter_count": parameter_count(self.models[branch]),
            }
        return result

    def partner_receipt_report(self) -> dict[str, object]:
        catalog = {}
        for branch in self.partner_counts:
            batch = self.data.batch(branch_query(branch), "train")
            mode = branch_mode(branch)
            catalog[branch] = [
                partner_map_receipt(batch, mode, index)
                for index in range(1, 35)
            ]
        return {
            "maps": 34,
            "target_cycles": 9 if self.config.steps == 306 else None,
            "counts": self.partner_counts,
            "map_digests": self.partner_map_digests,
            "source_use_counts": self.source_use_counts,
            "value_transition_counts": self.value_transition_counts,
            "map_catalog": catalog,
            "update_sequence": {
                "length": len(self.update_sequence),
                "digest": digest(self.update_sequence),
                "first_round": self.update_sequence[: len(BRANCH_ORDER)],
                "last_round": self.update_sequence[-len(BRANCH_ORDER) :],
            },
            "receipt_checks": self._training_receipt_checks(self.stage_step),
        }

    def result(self, execution_disposition: str) -> dict[str, object]:
        validation_canaries = {
            str(query): shortcut_canaries(
                self.data.batch(query, "train"),
                self.data.batch(query, "validation"),
            )
            for query in (ADD_FIRST, SUB_FIRST)
        }
        return {
            "schema_version": 3,
            "packet": R6_PACKET,
            "revision": self.config.revision,
            "phase": self.config.phase,
            "run_kind": self.config.run_kind,
            "execution_disposition": execution_disposition,
            "research_disposition": self.research_disposition,
            "config": self.config.to_dict(),
            "backend": self.backend.metadata(),
            "run_instance": self.run_instance,
            "source_snapshot": self.source_snapshot,
            "source_snapshot_digest": self.source_snapshot_digest,
            "elapsed_seconds": self.elapsed_seconds(),
            "global_round": self.global_round,
            "stage_step": self.stage_step,
            "training": self.training_report(),
            "partner_schedule": self.partner_receipt_report(),
            "evaluation_ledger": self.ledger,
            "data": self.data.partition_evidence(),
            "validation_canaries": validation_canaries,
            "reserve_canaries": {
                "preregistered_only_until_open": True,
                "query_only_correct": 7,
                "one_literal_correct": 7,
                "two_literal_correct": 0,
                "full_solver_correct": 49,
            },
            "inherited_state": self.inherited_receipt,
            "inherited_artifacts_after": {
                "checkpoint_sha256": sha256_file(
                    Path(self.config.inherited_checkpoint)
                ),
                "frozen_config_sha256": sha256_file(
                    Path(self.config.inherited_checkpoint).parent.parent
                    / "frozen-config.json"
                ),
            },
            "initialization": self.initialization_receipt,
            "recovery": {
                "semantics": "at-least-once training; cohort reserve fail-closed",
                "last_checkpoint": self.last_checkpoint,
                "reserve_stranded": self.reserve_stranded,
            },
            "claim_boundary": {
                "single_seed_candidate_claim": False,
                "learned_routing_trained": False,
                "continuous_phase_trained": False,
                "paired_query_trained": False,
                "state_congruence_signal_is_diagnostic_only": True,
            },
        }

    def status(self, state: str, detail: str | None = None) -> dict[str, object]:
        return {
            "schema_version": 3,
            "packet": R6_PACKET,
            "run_instance_digest": self.run_instance_digest,
            "source_snapshot_digest": self.source_snapshot_digest,
            "state": state,
            "detail": detail,
            "stage_step": self.stage_step,
            "target_steps": self.config.steps,
            "global_round": self.global_round,
            "elapsed_seconds": self.elapsed_seconds(),
            "execution_disposition": self.execution_disposition,
            "research_disposition": self.research_disposition,
            "reserve_state": self.ledger.get("true_reserve_state"),
        }

    def save_checkpoint(self, kind: str = "scheduled") -> Path:
        directory = self.run_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"r6-{self.global_round:08d}-{kind}-{uuid.uuid4().hex}.pt"
        )
        temporary = directory / f".{path.name}.{os.getpid()}.tmp"
        payload = {
            "schema_version": 3,
            "packet": R6_PACKET,
            "config": self.config.to_dict(),
            "config_digest": self.config_digest,
            "partition_digest": self.data.partition_digest,
            "source_snapshot_digest": self.source_snapshot_digest,
            "run_instance_digest": self.run_instance_digest,
            "stage_step": self.stage_step,
            "global_round": self.global_round,
            "execution_disposition": self.execution_disposition,
            "research_disposition": self.research_disposition,
            "model_names": BRANCH_ORDER,
            "models": {
                name: _cpu_tree(model.state_dict())
                for name, model in self.models.items()
            },
            "optimizers": {
                name: _cpu_tree(optimizer.state_dict())
                for name, optimizer in self.optimizers.items()
            },
            "cumulative": self.cumulative,
            "partner_counts": self.partner_counts,
            "partner_map_digests": self.partner_map_digests,
            "source_use_counts": self.source_use_counts,
            "value_transition_counts": self.value_transition_counts,
            "operation_counts": self.operation_counts,
            "update_sequence": self.update_sequence,
            "inherited_receipt": self.inherited_receipt,
            "initialization_receipt": self.initialization_receipt,
            "ledger_snapshot": self.ledger,
            "ledger_digest": digest(self.ledger),
            "state_binding": self._state_binding(),
            "torch_rng_state": torch.get_rng_state(),
            "elapsed_seconds": self.elapsed_seconds(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
        checkpoint_sha256 = sha256_file(path)
        atomic_write_json(
            directory / "latest.json",
            {
                "checkpoint": str(Path("checkpoints") / path.name),
                "checkpoint_sha256": checkpoint_sha256,
                "run_instance_digest": self.run_instance_digest,
                "source_snapshot_digest": self.source_snapshot_digest,
                "global_round": self.global_round,
                "kind": kind,
            },
        )
        self.last_checkpoint = str(path)
        return path

    @staticmethod
    def _ledger_extension_allowed(
        checkpoint: dict[str, object], current: dict[str, object]
    ) -> bool:
        for key in (
            "schema_version",
            "packet",
            "config_digest",
            "partition_digest",
            "source_snapshot_digest",
            "run_instance_digest",
        ):
            if checkpoint.get(key) != current.get(key):
                return False
        if checkpoint == current:
            return True
        checkpoint_validation = checkpoint.get("validation_state")
        current_validation = current.get("validation_state")
        if checkpoint_validation == "unopened":
            return current_validation == "complete"
        if checkpoint_validation != "complete" or current_validation != "complete":
            return False
        for key in (
            "validation",
            "validation_digest",
            "validation_binding",
            "invariants",
            "validation_disposition",
        ):
            if checkpoint.get(key) != current.get(key):
                return False
        transitions = {
            "unopened": {"not_opened", "reserve_opened", "reserve_stranded", "complete"},
            "reserve_opened": {"reserve_stranded", "complete"},
        }
        checkpoint_reserve = checkpoint.get("true_reserve_state")
        current_reserve = current.get("true_reserve_state")
        return current_reserve in transitions.get(str(checkpoint_reserve), set())

    def load_checkpoint(self, path: Path) -> None:
        expected_latest = latest_stage2_congruence_checkpoint(self.run_dir)
        if path.resolve() != expected_latest.resolve():
            raise RuntimeError("R6 checkpoint is not this run's verified latest file")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 3 or payload.get("packet") != R6_PACKET:
            raise RuntimeError("R6 checkpoint schema or packet changed")
        if payload.get("config_digest") != self.config_digest or payload.get("config") != self.config.to_dict():
            raise RuntimeError("R6 checkpoint config changed")
        if payload.get("partition_digest") != self.data.partition_digest:
            raise RuntimeError("R6 checkpoint partition changed")
        if payload.get("source_snapshot_digest") != self.source_snapshot_digest:
            raise RuntimeError("R6 checkpoint source snapshot changed")
        if payload.get("run_instance_digest") != self.run_instance_digest:
            raise RuntimeError("R6 checkpoint run instance changed")
        snapshot = payload.get("ledger_snapshot")
        if not isinstance(snapshot, dict) or digest(snapshot) != payload.get(
            "ledger_digest"
        ):
            raise RuntimeError("R6 checkpoint ledger receipt is malformed")
        ledger_extended = digest(self.ledger) != payload.get("ledger_digest")
        if tuple(payload.get("model_names", ())) != BRANCH_ORDER:
            raise RuntimeError("R6 checkpoint model order changed")
        if payload.get("inherited_receipt") != self.inherited_receipt or payload.get("initialization_receipt") != self.initialization_receipt:
            raise RuntimeError("R6 checkpoint initialization receipts changed")
        for name in BRANCH_ORDER:
            self.models[name].load_state_dict(payload["models"][name])
            self.optimizers[name].load_state_dict(payload["optimizers"][name])
            for state in self.optimizers[name].state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.backend.device)
        self.cumulative = payload["cumulative"]
        self.partner_counts = payload["partner_counts"]
        self.partner_map_digests = payload["partner_map_digests"]
        self.source_use_counts = payload["source_use_counts"]
        self.value_transition_counts = payload["value_transition_counts"]
        self.operation_counts = payload["operation_counts"]
        self.update_sequence = payload["update_sequence"]
        self.stage_step = int(payload["stage_step"])
        self.global_round = int(payload["global_round"])
        if not all(self._training_receipt_checks(self.stage_step).values()):
            raise RuntimeError("R6 checkpoint training receipts are inconsistent")
        if payload.get("state_binding") != self._state_binding():
            raise RuntimeError("R6 checkpoint model-state binding changed")
        if ledger_extended and not self._ledger_extension_allowed(
            snapshot, self.ledger
        ):
            raise RuntimeError("R6 checkpoint and ledger disagree")
        if self.ledger.get("validation_state") == "complete":
            if digest(self.ledger.get("validation")) != self.ledger.get(
                "validation_digest"
            ):
                raise RuntimeError("R6 validation ledger digest changed")
            self._assert_ledger_binding("validation_binding")
            recomputed_invariants = self._invariants()
            validation = self.ledger.get("validation")
            recomputed_invariants["all_validation_metrics_finite"] = all(
                isinstance(item, dict) and bool(item.get("finite"))
                for item in validation.values()
            )
            if recomputed_invariants != self.ledger.get("invariants"):
                raise RuntimeError("R6 validation invariant receipt changed")
            if ledger_extended and snapshot.get("validation_state") == "unopened":
                replay = self.verify_completed_validation_replay()
                if not replay["matched"] or not replay["ledger_unchanged"]:
                    raise RuntimeError("R6 extended validation failed exact replay")
        if self.ledger.get("true_reserve_state") == "complete":
            if digest(self.ledger.get("true_reserve")) != self.ledger.get(
                "true_reserve_digest"
            ):
                raise RuntimeError("R6 reserve ledger digest changed")
            self._assert_ledger_binding("true_reserve_binding")
            reserve_replay = self.verify_completed_reserve_replay()
            if (
                not reserve_replay["matched"]
                or not reserve_replay["ledger_unchanged"]
            ):
                raise RuntimeError("R6 completed reserve failed exact replay")
        self.execution_disposition = self.ledger.get("execution_disposition")
        self.research_disposition = self.ledger.get("research_disposition")
        self._validate_ledger_semantics()
        self.terminal = self.execution_disposition in {
            "completed",
            "implementation_invalid",
            "reserve_stranded",
        }
        torch.set_rng_state(payload["torch_rng_state"])
        self.elapsed_before_resume = float(payload["elapsed_seconds"])
        self.process_started = time.monotonic()
        self.last_checkpoint = str(path)
        self._strand_verified_open_reserve()


def latest_stage2_congruence_checkpoint(run_dir: Path) -> Path:
    manifest_path = run_dir / "run-instance.json"
    manifest = load_run_instance_manifest(manifest_path)
    run_instance_digest = digest(manifest)
    receipt = run_dir / "checkpoints" / "latest.json"
    if not receipt.is_file():
        raise FileNotFoundError("R6 run has no latest checkpoint receipt")
    raw = json.loads(receipt.read_text(encoding="utf-8"))
    if raw.get("run_instance_digest") != run_instance_digest:
        raise RuntimeError("R6 latest checkpoint run instance changed")
    if raw.get("source_snapshot_digest") != manifest.get(
        "source_snapshot_digest"
    ):
        raise RuntimeError("R6 latest checkpoint source snapshot changed")
    relative = Path(str(raw.get("checkpoint")))
    if relative.is_absolute():
        raise RuntimeError("R6 latest checkpoint path must be run-relative")
    path = (run_dir / relative).resolve()
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    if path.parent != checkpoint_dir:
        raise RuntimeError("R6 latest checkpoint escaped the run directory")
    if not path.is_file():
        raise FileNotFoundError(f"R6 checkpoint is missing: {path}")
    observed_sha256 = sha256_file(path)
    if raw.get("checkpoint_sha256") != observed_sha256:
        raise RuntimeError("R6 latest checkpoint SHA256 changed")
    return path
