"""Run formal Stage 1 confirmation seeds serially with fail-closed verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_hierarchy.stage1_config import (
    Stage1Config,
    load_stage1_config,
    stage1_config_digest,
    validated_experiment_compatibility_spec_digest,
    validated_experiment_spec_digest,
)
from dynamic_hierarchy.stage1_confirmation import run_completion_checks
from dynamic_hierarchy.stage1_integrity import (
    formal_seed_freshness,
    verify_result_manifests,
    verify_snapshot_manifest,
)
from dynamic_hierarchy.stage1_runtime import latest_checkpoint
from dynamic_hierarchy.run_lock import PerRunMutex
from scripts.stage1_worker import validate_candidate_prerequisite


TERMINAL_STATES = {"completed", "failed", "incomplete"}
LEGACY_INCOMPLETE_FINALIZATION_ERROR = (
    "RuntimeError: final evaluation must run before the learning gate"
)


class SequenceAbort(RuntimeError):
    """A fail-closed stop that prevents the next training seed from starting."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _candidate_expected(config: Stage1Config) -> dict[str, str]:
    return {
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
    }


def formal_result_checks(
    result: dict[str, Any],
    config: Stage1Config,
    expected_seed: int,
    run_dir: Path | None = None,
) -> dict[str, bool]:
    expected_config = replace(config, seed=expected_seed)
    expected_config.validate()
    expected_config_dict = json.loads(json.dumps(expected_config.to_dict()))
    result_config = result.get("config", {})
    final_evaluation = result.get("final_evaluation", {})
    overlap_audit = final_evaluation.get("overlap_audit", {})
    prerequisite = result.get("candidate_prerequisite", {})
    expected_prerequisite = _candidate_expected(config)
    completion = run_completion_checks(result)
    manifest_checks = (
        verify_result_manifests(result, run_dir)
        if run_dir is not None
        else {
            "manifest_run_directory_provided": False,
        }
    )
    return {
        **completion,
        **manifest_checks,
        "schema_version_3": (
            type(result.get("schema_version")) is int
            and result.get("schema_version") == 3
        ),
        "state_completed": result.get("state") == "completed",
        "target_reason": result.get("reason") == "target_steps_reached",
        "exact_steps": (
            type(result.get("global_step")) is int
            and type(result.get("target_steps")) is int
            and result.get("global_step") == config.optimizer_steps
            and result.get("target_steps") == config.optimizer_steps
        ),
        "eligible": result.get("run_eligible_for_aggregation") is True,
        "expected_training_seed": result_config.get("seed") == expected_seed,
        "exact_frozen_config": result_config == expected_config_dict,
        "formal_evaluation": result_config.get("formal_evaluation") is True,
        "config_digest": (
            result.get("config_digest")
            == stage1_config_digest(expected_config_dict)
        ),
        "self_spec_digest": (
            result.get("validated_experiment_spec_digest")
            == validated_experiment_spec_digest(expected_config)
        ),
        "compatibility_spec_digest": (
            result.get("validated_experiment_compatibility_spec_digest")
            == validated_experiment_compatibility_spec_digest(expected_config)
            == config.candidate_prerequisite_compatibility_spec_digest
        ),
        "candidate_prerequisite_required": (
            prerequisite.get("required") is True
        ),
        "candidate_prerequisite_passed": prerequisite.get("passed") is True,
        "candidate_prerequisite_pins": (
            prerequisite.get("expected") == expected_prerequisite
        ),
        "foundation_gate": (
            result.get("foundation_gate", {}).get("passed") is True
        ),
        "learning_gate": (
            result.get("learning_gate", {}).get("passed") is True
        ),
        "candidate_gate": (
            result.get("candidate_gate", {}).get("candidate_pass") is True
        ),
        "single_run_stage2_block": (
            result.get("candidate_gate", {}).get("stage2_unblocked") is False
        ),
        "formal_evaluation_kind": (
            final_evaluation.get("kind") == "formal_confirmation"
        ),
        "formal_evaluation_scale": (
            final_evaluation.get("examples_per_split_seed")
            == config.final_eval_examples_per_seed
        ),
        "formal_evaluation_seeds": (
            final_evaluation.get("evaluation_seeds")
            == list(config.eval_seeds)
        ),
        "content_disjoint": overlap_audit.get("all_content_disjoint") is True,
        "shape_rules_valid": (
            overlap_audit.get("all_shape_rules_valid") is True
        ),
        "formal_final_attempt_completed": (
            result.get("formal_final_attempt", {}).get("required") is True
            and result.get("formal_final_attempt", {}).get("state") == "completed"
        ),
    }


def verify_formal_result(
    result_path: Path,
    config: Stage1Config,
    expected_seed: int,
) -> dict[str, Any]:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SequenceAbort(
            f"cannot read terminal result for seed {expected_seed}: {error}"
        ) from error
    checks = formal_result_checks(
        result,
        config,
        expected_seed,
        result_path.resolve().parent,
    )
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise SequenceAbort(
            f"formal seed {expected_seed} failed terminal verification: {failed}"
        )
    return result


def _run_is_live(run_dir: Path) -> bool:
    pid_path = run_dir / "pid.json"
    if not pid_path.is_file():
        return False
    try:
        record = json.loads(pid_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    candidates = [record.get("worker_pid"), record.get("pid")]
    for value in candidates:
        if type(value) is not int or not psutil.pid_exists(value):
            continue
        try:
            command = " ".join(psutil.Process(value).cmdline())
        except (psutil.Error, OSError):
            continue
        if (
            "stage1_worker.py" in command
            and str(run_dir).lower() in command.lower()
        ):
            return True
    expected = str(run_dir.resolve()).lower()
    project = str(ROOT.resolve()).lower()
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
        except (psutil.Error, OSError):
            continue
        if (
            "stage1_worker.py" in command
            and project in command
            and expected in command
        ):
            return True
    return False


def _assert_no_other_project_worker(expected_run_dir: Path) -> None:
    expected = str(expected_run_dir.resolve()).lower()
    project = str(ROOT.resolve()).lower()
    blockers: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, OSError):
            continue
        lowered = command.lower()
        if (
            "stage1_worker.py" in lowered
            and project in lowered
            and expected not in lowered
        ):
            blockers.append(process.pid)
    if blockers:
        raise SequenceAbort(
            "another project Stage 1 worker is live; refusing parallel DirectML "
            f"confirmation (PIDs {sorted(blockers)})"
        )


def _wait_for_worker_exit(
    run_dir: Path,
    poll_seconds: float,
    timeout_seconds: float = 300.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while _run_is_live(run_dir):
        if time.monotonic() >= deadline:
            raise SequenceAbort(
                f"{run_dir} wrote a terminal result but its worker did not exit"
            )
        time.sleep(poll_seconds)


def _read_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SequenceAbort(f"cannot read result evidence {path}: {error}") from error
    if not isinstance(result, dict):
        raise SequenceAbort(f"result evidence is not an object: {path}")
    return result


def _recoverable_incomplete_checks(
    result: dict[str, Any],
    run_dir: Path,
    config: Stage1Config,
) -> dict[str, bool]:
    final_evaluation = result.get("final_evaluation")
    final_attempt = result.get("formal_final_attempt", {})
    try:
        checkpoint = latest_checkpoint(run_dir)
        checkpoint_present = checkpoint.is_file()
    except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError):
        checkpoint_present = False
    return {
        "state_incomplete": result.get("state") == "incomplete",
        "recoverable_reason": result.get("reason")
        in {"user_stop", "time_budget_reached"},
        "not_eligible": result.get("run_eligible_for_aggregation") is False,
        "step_before_target": (
            type(result.get("global_step")) is int
            and 0 <= result["global_step"] < config.optimizer_steps
        ),
        "formal_holdout_untouched": final_evaluation in ({}, None),
        "formal_final_not_started": (
            final_attempt.get("state") in {None, "not_started"}
        ),
        "checkpoint_present": checkpoint_present,
    }


def _checkpoint_step_checks(
    run_dir: Path,
    expected_step: int,
) -> dict[str, bool]:
    pointer_path = run_dir / "checkpoints" / "latest.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        checkpoint = latest_checkpoint(run_dir).resolve()
        checkpoint.relative_to((run_dir / "checkpoints").resolve())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        FileNotFoundError,
    ):
        return {
            "checkpoint_pointer_valid": False,
            "checkpoint_step_matches": False,
            "checkpoint_kind_recoverable": False,
            "checkpoint_name_matches_step": False,
        }
    return {
        "checkpoint_pointer_valid": checkpoint.is_file(),
        "checkpoint_step_matches": (
            type(pointer.get("global_step")) is int
            and pointer["global_step"] == expected_step
        ),
        "checkpoint_kind_recoverable": pointer.get("kind")
        in {"emergency", "incomplete"},
        "checkpoint_name_matches_step": (
            checkpoint.name == f"checkpoint-{expected_step:08d}.pt"
        ),
    }


def _legacy_failed_finalization_checks(
    result: dict[str, Any],
    run_dir: Path,
    config: Stage1Config,
    expected_seed: int,
) -> dict[str, bool]:
    expected_config = replace(config, seed=expected_seed)
    expected_config.validate()
    expected_config_dict = json.loads(json.dumps(expected_config.to_dict()))
    result_config = result.get("config", {})
    prerequisite = result.get("candidate_prerequisite", {})
    final_attempt = result.get("formal_final_attempt", {})
    candidate_gate = result.get("candidate_gate", {})
    metrics = result.get("metrics", {})
    models = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    step = result.get("global_step")
    expected_examples = (
        step * config.effective_batch_size if type(step) is int else -1
    )
    try:
        manifest_checks = verify_result_manifests(result, run_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        manifest_checks = {"manifest_verification": False}
    try:
        config_checks = {
            "config_digest": (
                result.get("config_digest")
                == stage1_config_digest(expected_config_dict)
            ),
            "experiment_spec_digest": (
                result.get("validated_experiment_spec_digest")
                == validated_experiment_spec_digest(expected_config)
            ),
            "compatibility_spec_digest": (
                result.get("validated_experiment_compatibility_spec_digest")
                == validated_experiment_compatibility_spec_digest(expected_config)
            ),
        }
    except (TypeError, ValueError):
        config_checks = {
            "config_digest": False,
            "experiment_spec_digest": False,
            "compatibility_spec_digest": False,
        }
    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    checkpoint_checks = (
        _checkpoint_step_checks(run_dir, step)
        if type(step) is int
        else _checkpoint_step_checks(run_dir, -1)
    )
    return {
        **manifest_checks,
        **config_checks,
        **checkpoint_checks,
        "legacy_schema_3": (
            type(result.get("schema_version")) is int
            and result.get("schema_version") == 3
        ),
        "legacy_failed_state": result.get("state") == "failed",
        "legacy_exact_error": (
            result.get("reason") == LEGACY_INCOMPLETE_FINALIZATION_ERROR
        ),
        "legacy_not_eligible": (
            "run_eligible_for_aggregation" in result
            and result.get("run_eligible_for_aggregation") is False
        ),
        "legacy_step_before_target": (
            type(step) is int
            and 0 <= step < config.optimizer_steps
            and result.get("target_steps") == config.optimizer_steps
        ),
        "legacy_formal_config_exact": result_config == expected_config_dict,
        "legacy_formal_config": (
            isinstance(result_config, dict)
            and result_config.get("formal_evaluation") is True
            and result_config.get("seed") == expected_seed
        ),
        "legacy_candidate_prerequisite": (
            isinstance(prerequisite, dict)
            and prerequisite.get("required") is True
            and prerequisite.get("passed") is True
            and prerequisite.get("expected") == _candidate_expected(config)
        ),
        "legacy_holdout_empty": result.get("final_evaluation") in ({}, None),
        "legacy_marker_not_started": (
            isinstance(final_attempt, dict)
            and set(final_attempt) == {"required", "state"}
            and final_attempt.get("required") is True
            and final_attempt.get("state") == "not_started"
            and not (run_dir / "formal-final-attempt.json").exists()
        ),
        "legacy_candidate_gate_fail_closed": (
            isinstance(candidate_gate, dict)
            and candidate_gate.get("run_complete") is False
            and candidate_gate.get("run_completion_reason")
            == LEGACY_INCOMPLETE_FINALIZATION_ERROR
            and candidate_gate.get("candidate_pass") is False
            and candidate_gate.get("stage2_unblocked") is False
        ),
        "legacy_metrics_incomplete": (
            isinstance(metrics, dict)
            and metrics.get("curriculum_position", {}).get("complete") is False
        ),
        "legacy_model_steps_match": all(
            isinstance(models.get(name), dict)
            and models[name].get("optimizer_updates") == step
            and models[name].get("examples") == expected_examples
            for name in ("A", "D_true", "D_sham")
        ),
        "legacy_failed_status_matches": (
            status.get("state") == "failed"
            and status.get("step") == step
            and status.get("error") == LEGACY_INCOMPLETE_FINALIZATION_ERROR
        ),
    }


def _requires_explicit_stop_resume(
    result: dict[str, Any],
    run_dir: Path,
    legacy_failed: bool,
) -> bool:
    return (
        (run_dir / "control" / "STOP").is_file()
        and (
            legacy_failed
            or result.get("reason") == "user_stop"
        )
    )


def _archive_incomplete_result(result_path: Path, result: dict[str, Any]) -> Path:
    evidence_dir = result_path.parent / "attempt-results"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(evidence_dir.glob("result-*.json"))) + 1
    reason = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in str(result.get("reason", "incomplete"))
    )[:96]
    destination = evidence_dir / f"result-{index:03d}-{reason}.json"
    if destination.exists():
        raise SequenceAbort(f"incomplete result archive collision: {destination}")
    source_digest = _sha256(result_path)
    os.replace(result_path, destination)
    if _sha256(destination) != source_digest:
        raise SequenceAbort(
            f"recoverable result archive digest changed: {destination}"
        )
    return destination


def _partial_shell_is_safe_to_rebuild(run_dir: Path) -> tuple[bool, list[str]]:
    run_dir = run_dir.resolve()
    runs_root = (ROOT / "runs").resolve()
    try:
        run_dir.relative_to(runs_root)
    except ValueError:
        return False, ["outside_project_runs"]
    if not run_dir.name.startswith("stage1-formal-"):
        return False, ["unexpected_run_name"]
    forbidden = [
        "pid.json",
        "status.json",
        "result.json",
        "formal-final-attempt.json",
        "checkpoints",
        "control",
        "attempt-results",
    ]
    evidence = [name for name in forbidden if (run_dir / name).exists()]
    for log_name in ("stdout.log", "stderr.log"):
        log_path = run_dir / log_name
        if log_path.is_file() and log_path.stat().st_size:
            evidence.append(f"nonempty_{log_name}")
    snapshot_root = run_dir / "snapshot"
    snapshot_checks, _ = verify_snapshot_manifest(snapshot_root)
    if not all(value is True for value in snapshot_checks.values()):
        evidence.append("invalid_snapshot")
    allowed = {"snapshot", ".launch.lock", "stdout.log", "stderr.log"}
    unexpected = sorted(
        path.name for path in run_dir.iterdir() if path.name not in allowed
    )
    evidence.extend(f"unexpected:{name}" for name in unexpected)
    return not evidence, evidence


def _rebuild_partial_shell(run_dir: Path) -> None:
    safe, evidence = _partial_shell_is_safe_to_rebuild(run_dir)
    if not safe:
        raise SequenceAbort(
            f"partial launch contains evidence and cannot be rebuilt: {evidence}"
        )
    resolved = run_dir.resolve()
    runs_root = (ROOT / "runs").resolve()
    resolved.relative_to(runs_root)
    shutil.rmtree(resolved)


def _run_launcher_command(
    command: list[str],
    receipt_path: Path,
    label: str,
) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = receipt_path.with_name(
        f"{receipt_path.stem}.launcher.stdout.log"
    )
    stderr_path = receipt_path.with_name(
        f"{receipt_path.stem}.launcher.stderr.log"
    )
    with (
        stdout_path.open("a", encoding="utf-8") as stdout_file,
        stderr_path.open("a", encoding="utf-8") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        try:
            return_code = process.wait(timeout=180)
        except subprocess.TimeoutExpired as error:
            raise SequenceAbort(
                f"{label} launcher did not exit within 180 seconds; "
                f"inspect {stdout_path} and {stderr_path}"
            ) from error
    if return_code != 0:
        stderr = stderr_path.read_text(encoding="utf-8").strip()
        stdout = stdout_path.read_text(encoding="utf-8").strip()
        raise SequenceAbort(
            f"{label} launch failed ({return_code}): "
            f"{stderr or stdout or 'no launcher output'}"
        )


def _launch(
    config_path: Path,
    candidate_path: Path,
    seed: int,
    run_dir: Path,
    receipt_path: Path,
) -> None:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "scripts" / "start_stage1.ps1"),
        "-Config",
        str(config_path.relative_to(ROOT)),
        "-TrainingSeed",
        str(seed),
        "-CandidateResult",
        str(candidate_path),
        "-NewRunDir",
        str(run_dir),
        "-LaunchReceipt",
        str(receipt_path),
    ]
    _run_launcher_command(command, receipt_path, f"seed {seed}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SequenceAbort(f"seed {seed} launch receipt is invalid") from error
    if Path(receipt.get("run_dir", "")).resolve() != run_dir.resolve():
        raise SequenceAbort(f"seed {seed} launch receipt run directory mismatch")


def _resume(run_dir: Path, receipt_path: Path) -> None:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "scripts" / "start_stage1.ps1"),
        "-ResumeRun",
        str(run_dir),
        "-LaunchReceipt",
        str(receipt_path),
    ]
    _run_launcher_command(command, receipt_path, f"resume {run_dir}")


def _wait_for_terminal(run_dir: Path, poll_seconds: float) -> Path:
    result_path = run_dir / "result.json"
    no_process_since: float | None = None
    while True:
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(poll_seconds)
                continue
            if result.get("state") in TERMINAL_STATES:
                return result_path
        status_path = run_dir / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {}
            if (
                status.get("state") in {"failed", "incomplete"}
                and not result_path.is_file()
            ):
                raise SequenceAbort(
                    f"{run_dir} reached terminal status without result.json"
                )
        if _run_is_live(run_dir):
            no_process_since = None
        else:
            no_process_since = no_process_since or time.monotonic()
            if time.monotonic() - no_process_since > 120:
                raise SequenceAbort(
                    f"{run_dir} has no live worker and no terminal result"
                )
        time.sleep(poll_seconds)


def _new_state(
    config_path: Path,
    config: Stage1Config,
    candidate_path: Path,
    aggregate_output: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "ready",
        "config_path": str(config_path.resolve()),
        "config_digest": stage1_config_digest(config.to_dict()),
        "compatibility_spec_digest": (
            validated_experiment_compatibility_spec_digest(config)
        ),
        "candidate_result_path": str(candidate_path.resolve()),
        "candidate_result_digest": _sha256(candidate_path),
        "training_seeds": list(config.confirmation_training_seeds),
        "aggregate_output": str(aggregate_output.resolve()),
        "runs": {},
    }


def _load_or_create_state(
    state_path: Path,
    config_path: Path,
    config: Stage1Config,
    candidate_path: Path,
    aggregate_output: Path,
) -> dict[str, Any]:
    expected = _new_state(
        config_path,
        config,
        candidate_path,
        aggregate_output,
    )
    if not state_path.is_file():
        _atomic_write_json(state_path, expected)
        return expected
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for field in (
        "schema_version",
        "config_path",
        "config_digest",
        "compatibility_spec_digest",
        "candidate_result_path",
        "candidate_result_digest",
        "training_seeds",
        "aggregate_output",
    ):
        if state.get(field) != expected.get(field):
            raise SequenceAbort(f"sequence state mismatch for {field}")
    if not isinstance(state.get("runs"), dict):
        raise SequenceAbort("sequence state runs must be an object")
    return state


def _run_sequence_locked(
    config_path: Path,
    state_path: Path,
    aggregate_output: Path,
    poll_seconds: float,
) -> int:
    config = load_stage1_config(config_path)
    if (
        config.device != "directml"
        or config.formal_evaluation is not True
        or config.requires_candidate_pass is not True
        or len(config.confirmation_training_seeds) != 8
        or 82421 in set(config.confirmation_training_seeds)
    ):
        raise SequenceAbort("config is not an eligible eight-seed DirectML formal plan")
    candidate_path = (ROOT / config.candidate_prerequisite_result_path).resolve()
    if not candidate_path.is_file():
        raise SequenceAbort(f"candidate result is missing: {candidate_path}")
    if _sha256(candidate_path).lower() != config.candidate_prerequisite_result_digest.lower():
        raise SequenceAbort("candidate result digest does not match the formal config")
    try:
        validate_candidate_prerequisite(config, candidate_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SequenceAbort(
            f"candidate prerequisite failed before sequence creation: {error}"
        ) from error
    seed_freshness = formal_seed_freshness(ROOT, config_path, config)
    if seed_freshness.get("passed") is not True:
        raise SequenceAbort(
            "formal seed freshness failed against historical nonformal evidence: "
            f"{seed_freshness.get('overlap')}"
        )

    state = _load_or_create_state(
        state_path,
        config_path,
        config,
        candidate_path,
        aggregate_output,
    )
    state["seed_freshness"] = seed_freshness
    _atomic_write_json(state_path, state)
    result_paths: list[Path] = []
    try:
        for index, seed in enumerate(config.confirmation_training_seeds):
            key = str(seed)
            entry = state["runs"].get(key)
            if entry is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                run_dir = (
                    ROOT
                    / "runs"
                    / f"stage1-formal-{index + 1:02d}-{seed}-{stamp}"
                )
                entry = {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "status": "planned",
                }
                state["runs"][key] = entry
                _atomic_write_json(state_path, state)
            run_dir = Path(entry["run_dir"]).resolve()
            result_path = run_dir / "result.json"
            while True:
                if result_path.is_file():
                    _wait_for_worker_exit(run_dir, poll_seconds)
                    observed_result = _read_result(result_path)
                    legacy_failed = False
                    if observed_result.get("state") == "completed":
                        verify_formal_result(result_path, config, seed)
                        break
                    if observed_result.get("state") == "failed":
                        legacy_checks = _legacy_failed_finalization_checks(
                            observed_result,
                            run_dir,
                            config,
                            seed,
                        )
                        if not all(
                            value is True for value in legacy_checks.values()
                        ):
                            failed_legacy = sorted(
                                name
                                for name, passed in legacy_checks.items()
                                if passed is not True
                            )
                            raise SequenceAbort(
                                f"formal seed {seed} worker failed and is not "
                                "the recoverable legacy finalization artifact: "
                                f"{failed_legacy}"
                            )
                        legacy_failed = True
                        recovery_checks = legacy_checks
                    else:
                        recovery_checks = _recoverable_incomplete_checks(
                            observed_result,
                            run_dir,
                            config,
                        )
                    if not all(value is True for value in recovery_checks.values()):
                        failed_recovery = sorted(
                            name
                            for name, passed in recovery_checks.items()
                            if passed is not True
                        )
                        raise SequenceAbort(
                            f"formal seed {seed} incomplete result is not safely "
                            f"recoverable: {failed_recovery}"
                        )
                    if _requires_explicit_stop_resume(
                        observed_result,
                        run_dir,
                        legacy_failed,
                    ):
                        entry["status"] = "awaiting_user_resume"
                        entry["recoverable_result"] = str(result_path)
                        entry["recoverable_result_digest"] = _sha256(result_path)
                        entry["legacy_failed_finalization"] = legacy_failed
                        state["state"] = "paused_recoverable"
                        state["failure"] = None
                        state["failure_class"] = None
                        _atomic_write_json(state_path, state)
                        return 3
                    archived = _archive_incomplete_result(
                        result_path,
                        observed_result,
                    )
                    entry.setdefault("incomplete_attempts", []).append(
                        {
                            "reason": observed_result.get("reason"),
                            "global_step": observed_result.get("global_step"),
                            "archived_result": str(archived),
                            "archived_result_digest": _sha256(archived),
                            "legacy_failed_finalization": legacy_failed,
                        }
                    )
                    entry["status"] = "resuming"
                    state["state"] = "running"
                    state["failure"] = None
                    state["failure_class"] = None
                    _atomic_write_json(state_path, state)

                receipt = state_path.with_name(
                    f"{state_path.stem}.seed-{seed}.receipt.json"
                )
                _assert_no_other_project_worker(run_dir)
                if run_dir.exists():
                    if _run_is_live(run_dir):
                        entry["status"] = "waiting"
                    elif (run_dir / "pid.json").is_file():
                        entry["status"] = "resuming"
                        _atomic_write_json(state_path, state)
                        _resume(run_dir, receipt)
                    else:
                        entry["status"] = "rebuilding_unstarted_partial_launch"
                        _atomic_write_json(state_path, state)
                        _rebuild_partial_shell(run_dir)
                        _launch(
                            config_path,
                            candidate_path,
                            seed,
                            run_dir,
                            receipt,
                        )
                else:
                    entry["status"] = "launching"
                    _atomic_write_json(state_path, state)
                    _launch(
                        config_path,
                        candidate_path,
                        seed,
                        run_dir,
                        receipt,
                    )
                entry["status"] = "waiting"
                _atomic_write_json(state_path, state)
                result_path = _wait_for_terminal(run_dir, poll_seconds)
                _wait_for_worker_exit(run_dir, poll_seconds)

            entry["status"] = "verified"
            entry["result_path"] = str(result_path)
            entry["result_digest"] = _sha256(result_path)
            _atomic_write_json(state_path, state)
            result_paths.append(result_path)
    except SequenceAbort as error:
        state["state"] = "failed_closed"
        state["failure"] = str(error)
        state["failure_class"] = (
            "worker_failed_nonrecoverable"
            if "worker failed" in str(error)
            else "integrity_or_recovery_failure"
        )
        _atomic_write_json(state_path, state)
        raise

    state["state"] = "aggregating"
    _atomic_write_json(state_path, state)
    aggregate_command = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_stage1_confirmation.py"),
        "--results",
        *(str(path) for path in result_paths),
        "--output",
        str(aggregate_output),
    ]
    completed = subprocess.run(
        aggregate_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        state["state"] = "aggregate_failed"
        state["failure"] = completed.stderr.strip() or completed.stdout.strip()
        _atomic_write_json(state_path, state)
        return completed.returncode
    state["state"] = "completed"
    state["aggregate_result"] = str(aggregate_output.resolve())
    _atomic_write_json(state_path, state)
    return 0


def run_sequence(
    config_path: Path,
    state_path: Path,
    aggregate_output: Path,
    poll_seconds: float,
) -> int:
    sequence_mutex = PerRunMutex(
        ROOT / "runs" / ".stage1-formal-confirmation-sequence"
    )
    try:
        sequence_mutex.acquire()
    except RuntimeError as error:
        raise SequenceAbort(
            "another formal confirmation coordinator is already active"
        ) from error
    try:
        return _run_sequence_locked(
            config_path,
            state_path,
            aggregate_output,
            poll_seconds,
        )
    finally:
        sequence_mutex.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "stage1-revised-literal-formal-confirmation-directml.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "runs" / "stage1-literal-formal-sequence.json",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=ROOT / "runs" / "stage1-literal-formal-confirmation.json",
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    try:
        return run_sequence(
            args.config.resolve(),
            args.state.resolve(),
            args.aggregate_output.resolve(),
            args.poll_seconds,
        )
    except SequenceAbort as error:
        print(f"fail-closed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "sequence monitor interrupted; the active worker was not stopped. "
            "Run the same command to resume monitoring.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
