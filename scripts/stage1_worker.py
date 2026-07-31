"""Long-running cooperative Stage 1 worker launched from a frozen snapshot."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import psutil

from dynamic_hierarchy.reporting import fallback_observability
from dynamic_hierarchy.process_registry import register_worker_pid
from dynamic_hierarchy.resource_guard import ResourceGuard, ResourceSample
from dynamic_hierarchy.run_lock import PerRunMutex
from dynamic_hierarchy.stage1_config import (
    load_stage1_config,
    stage1_config_digest,
    validated_experiment_compatibility_spec_digest,
    validated_experiment_spec_digest,
)
from dynamic_hierarchy.stage1_confirmation import run_completion_checks
from dynamic_hierarchy.stage1_integrity import (
    verify_result_manifests,
    verify_snapshot_manifest,
)
from dynamic_hierarchy.stage1_runtime import (
    Stage1Trainer,
    atomic_write_json,
    latest_checkpoint,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_run_outcome(
    reason: str,
    global_step: int,
    optimizer_steps: int,
) -> tuple[str, bool]:
    complete = reason == "target_steps_reached" and global_step == optimizer_steps
    return ("completed", True) if complete else ("incomplete", False)


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def finalize_training_attempt(
    trainer: Stage1Trainer,
    config,
    final_reason: str,
    run_dir: Path,
) -> tuple[str, bool, dict[str, object], dict[str, object]]:
    result_state, target_complete = classify_run_outcome(
        final_reason,
        trainer.global_step,
        config.optimizer_steps,
    )
    marker_path = run_dir / "formal-final-attempt.json"
    if not target_complete:
        if config.formal_evaluation and marker_path.exists():
            raise RuntimeError(
                "incomplete formal training found a final-attempt marker"
            )
        trainer.record_run_completion(False, final_reason)
        trainer.save_checkpoint("incomplete")
        return (
            result_state,
            False,
            {
                "passed": False,
                "state": "not_evaluated",
                "reason": "target_incomplete",
            },
            {
                "required": config.formal_evaluation is True,
                "state": "not_started",
                "reason": "target_incomplete",
            },
        )

    marker = _read_json_object(marker_path) if config.formal_evaluation else None
    if marker is not None:
        if (
            marker.get("state") in {"started", "completed"}
            and trainer.final_evaluation
            and trainer.gate_result
        ):
            learning_gate = trainer.learning_gate()
            trainer.record_run_completion(True, final_reason)
            trainer.save_checkpoint("final")
            marker = {
                **marker,
                "state": "completed",
                "completed_at": utc_now(),
                "recovered_without_holdout_reuse": True,
            }
            atomic_write_json(marker_path, marker)
            return result_state, True, learning_gate, marker
        raise RuntimeError(
            "formal final holdout was already attempted without a recoverable "
            "completed final checkpoint"
        )

    if config.formal_evaluation:
        marker = {
            "schema_version": 1,
            "state": "started",
            "started_at": utc_now(),
            "global_step": trainer.global_step,
            "target_steps": config.optimizer_steps,
            "evaluation_seeds": list(config.eval_seeds),
            "examples_per_split_seed": config.final_eval_examples_per_seed,
        }
        atomic_write_json(marker_path, marker)

    trainer.evaluate_final_gate()
    learning_gate = trainer.learning_gate()
    trainer.record_run_completion(True, final_reason)
    trainer.save_checkpoint("final")
    final_attempt = {
        **(marker or {"schema_version": 1}),
        "required": config.formal_evaluation is True,
        "state": "completed",
        "completed_at": utc_now(),
        "recovered_without_holdout_reuse": False,
    }
    if config.formal_evaluation:
        atomic_write_json(marker_path, final_attempt)
    return result_state, True, learning_gate, final_attempt


def validate_candidate_prerequisite(
    config,
    candidate_result: Path | None,
) -> dict[str, object]:
    if not config.requires_candidate_pass:
        return {"required": False, "passed": True}
    if candidate_result is None or not candidate_result.is_file():
        raise RuntimeError("formal literal launch requires --candidate-result")
    expected = {
        "config_digest": config.candidate_prerequisite_config_digest,
        "manifest_hash": config.candidate_prerequisite_manifest_hash,
        "snapshot_manifest_hash": config.candidate_prerequisite_snapshot_manifest_hash,
        "result_digest": config.candidate_prerequisite_result_digest,
        "experiment_spec_digest": (
            config.candidate_prerequisite_experiment_spec_digest
        ),
        "compatibility_spec_digest": (
            config.candidate_prerequisite_compatibility_spec_digest
        ),
    }
    if not all(expected.values()):
        raise RuntimeError(
            "formal config has no fully pinned candidate prerequisite evidence"
        )
    result_bytes = candidate_result.read_bytes()
    result = json.loads(result_bytes.decode("utf-8"))
    candidate_config = result.get("config", {})
    try:
        candidate_spec_digest = validated_experiment_spec_digest(candidate_config)
        candidate_compatibility_digest = (
            validated_experiment_compatibility_spec_digest(candidate_config)
        )
        formal_compatibility_digest = (
            validated_experiment_compatibility_spec_digest(config)
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "candidate prerequisite failed pinned completion, specification, "
            "source, or gate verification: invalid experiment config"
        ) from error
    completion = run_completion_checks(result)
    candidate_manifest_checks = verify_result_manifests(
        result,
        candidate_result.resolve().parent,
        require_embedded_snapshot_manifest=False,
    )
    checks = {
        **completion,
        **candidate_manifest_checks,
        "current_result_schema": int(result.get("schema_version", -1)) >= 3,
        "explicit_aggregation_eligibility": (
            "run_eligible_for_aggregation" in result
            and result.get("run_eligible_for_aggregation") is True
        ),
        "operand_mode": candidate_config.get("operand_mode") == config.operand_mode,
        "candidate_only": candidate_config.get("formal_evaluation") is False,
        "config_digest": stage1_config_digest(candidate_config)
        == expected["config_digest"],
        "manifest_hash": result.get("manifest", {}).get("manifest_hash")
        == expected["manifest_hash"],
        "snapshot_manifest_hash": result.get("snapshot_manifest_hash")
        == expected["snapshot_manifest_hash"],
        "result_digest": (
            hashlib.sha256(result_bytes).hexdigest().lower()
            == str(expected["result_digest"]).lower()
        ),
        "candidate_result_spec_digest": (
            result.get("validated_experiment_spec_digest")
            == candidate_spec_digest
        ),
        "candidate_spec_matches_pin": (
            candidate_spec_digest == expected["experiment_spec_digest"]
        ),
        "candidate_compatibility_spec_matches_pin": (
            candidate_compatibility_digest
            == expected["compatibility_spec_digest"]
        ),
        "formal_compatibility_spec_matches_pin": (
            formal_compatibility_digest
            == expected["compatibility_spec_digest"]
        ),
        "foundation_gate": result.get("foundation_gate", {}).get("passed") is True,
        "candidate_gate": (
            result.get("candidate_gate", {}).get("candidate_pass") is True
        ),
        "learning_gate": result.get("learning_gate", {}).get("passed") is True,
        "evaluation_scale": (
            int(result.get("final_evaluation", {}).get("examples_per_split_seed", -1))
            == int(candidate_config.get("final_eval_examples_per_seed", -2))
        ),
        "evaluation_seeds": (
            result.get("final_evaluation", {}).get("evaluation_seeds")
            == candidate_config.get("eval_seeds")
        ),
    }
    if not all(checks.values()):
        failed_checks = sorted(
            name for name, passed in checks.items() if passed is not True
        )
        raise RuntimeError(
            "candidate prerequisite failed pinned completion, specification, source, "
            f"or gate verification: {failed_checks}"
        )
    return {
        "required": True,
        "passed": True,
        "candidate_result": str(candidate_result.resolve()),
        "expected": expected,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launch-id")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--candidate-result", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_mutex = PerRunMutex(run_dir)
    try:
        run_mutex.acquire()
    except RuntimeError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 3
    atexit.register(run_mutex.release)
    source_root = Path(__file__).resolve().parents[1]
    snapshot_manifest = json.loads(
        (source_root / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    snapshot_checks, verified_snapshot_manifest = verify_snapshot_manifest(
        source_root,
        snapshot_manifest,
    )
    if (
        verified_snapshot_manifest is None
        or not all(check is True for check in snapshot_checks.values())
    ):
        failed = sorted(
            name for name, passed in snapshot_checks.items() if passed is not True
        )
        raise RuntimeError(f"frozen snapshot integrity verification failed: {failed}")
    snapshot_manifest = verified_snapshot_manifest
    config = load_stage1_config(args.config)
    candidate_prerequisite = validate_candidate_prerequisite(
        config,
        args.candidate_result,
    )
    if args.training_seed is not None:
        if args.training_seed not in config.confirmation_training_seeds:
            raise ValueError("training seed is not declared in confirmation_training_seeds")
        config = replace(config, seed=args.training_seed)
        config.validate()
    process_registry = register_worker_pid(run_dir, args.launch_id)
    status_path = run_dir / "status.json"
    runtime_warnings: list[str] = []
    started_at = utc_now()
    priority_status: dict[str, object]
    try:
        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            priority_status = {"requested": "BelowNormal", "applied": True}
        else:
            process.nice(10)
            priority_status = {"requested": "nice=10", "applied": True}
    except Exception as error:
        priority_status = {
            "requested": "BelowNormal",
            "applied": False,
            "error": f"{type(error).__name__}: {error}",
        }

    trainer: Stage1Trainer | None = None
    checkpoint_loaded_or_created = False
    last_resource_sample = ResourceSample(cpu_percent=0.0, available_ram_gb=0.0)
    last_resource_sample_at = 0.0
    last_heartbeat_at = 0.0
    last_checkpoint_at = 0.0
    pause_reason: str | None = None
    final_reason: str | None = None
    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    guard = ResourceGuard(
        config.cpu_pause_percent,
        config.cpu_resume_percent,
        config.ram_pause_gb,
        config.ram_resume_gb,
        config.pressure_samples,
        config.recovery_samples,
    )
    psutil.cpu_percent(interval=None)

    def write_status(state: str, current_model: str = "A+D paired") -> None:
        nonlocal last_heartbeat_at
        metrics = trainer.metrics() if trainer is not None else {}
        warnings_snapshot = list(runtime_warnings)
        atomic_write_json(
            status_path,
            {
                "schema_version": 2,
                "pid": os.getpid(),
                "process_registry": process_registry,
                "run_mutex": run_mutex.metadata(),
                "state": state,
                "started_at": started_at,
                "updated_at": utc_now(),
                "current_model": current_model,
                "step": trainer.global_step if trainer is not None else 0,
                "curriculum_position": (
                    trainer.curriculum_position() if trainer is not None else None
                ),
                "target_steps": config.optimizer_steps,
                "time_budget_minutes": config.time_budget_minutes,
                "elapsed_seconds": trainer.elapsed_seconds() if trainer is not None else 0.0,
                "latest_loss": trainer.latest_loss if trainer is not None else {"A": None, "D": None},
                "latest_checkpoint": trainer.last_checkpoint if trainer is not None else None,
                "checkpoint_recovery": trainer.recovery_state() if trainer is not None else None,
                "pause_reason": pause_reason,
                "resource_sample": last_resource_sample.to_dict(),
                "resource_policy": {
                    "cpu_pause_percent": config.cpu_pause_percent,
                    "cpu_resume_percent": config.cpu_resume_percent,
                    "ram_pause_gb": config.ram_pause_gb,
                    "ram_resume_gb": config.ram_resume_gb,
                    "gpu_memory_quota": "unavailable: DirectML exposes no hard VRAM quota in this runner",
                },
                "device": trainer.backend.metadata() if trainer is not None else {"backend": config.device},
                "priority": priority_status,
                "manifest": trainer.source_manifest if trainer is not None else None,
                "snapshot_manifest_hash": snapshot_manifest["manifest_hash"],
                "checkpoint_resume_source": str(latest_checkpoint(run_dir)) if args.resume else None,
                "metrics": metrics,
                "runtime_warnings": warnings_snapshot,
                "fallback_observability": fallback_observability(config.device, warnings_snapshot),
                "final_reason": final_reason,
            },
        )
        last_heartbeat_at = time.monotonic()

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            trainer = Stage1Trainer(
                config=config,
                run_dir=run_dir,
                source_root=source_root,
                snapshot_manifest_hash=str(snapshot_manifest["manifest_hash"]),
            )
            if args.resume:
                trainer.load_checkpoint(latest_checkpoint(run_dir))
            else:
                trainer.evaluate()
                trainer.save_checkpoint("bootstrap")
            checkpoint_loaded_or_created = True
            last_checkpoint_at = time.monotonic()
            last_resource_sample = guard.sample()
            last_resource_sample_at = time.monotonic()
            write_status("running")

            while trainer.global_step < config.optimizer_steps:
                runtime_warnings[:] = [str(item.message) for item in captured]
                if (
                    trainer.session_elapsed_seconds()
                    >= config.time_budget_minutes * 60.0
                ):
                    final_reason = "time_budget_reached"
                    break
                if (control_dir / "STOP").is_file():
                    final_reason = "user_stop"
                    break
                if (control_dir / "RESUME").is_file():
                    (control_dir / "RESUME").unlink(missing_ok=True)
                    (control_dir / "PAUSE").unlink(missing_ok=True)

                now = time.monotonic()
                if now - last_resource_sample_at >= config.resource_sample_seconds:
                    last_resource_sample = guard.sample()
                    guard.observe(last_resource_sample)
                    last_resource_sample_at = now
                manual_pause = (control_dir / "PAUSE").is_file()
                pause_reason = "manual PAUSE control file" if manual_pause else guard.pause_reason
                if manual_pause or guard.paused:
                    write_status("paused")
                    time.sleep(min(1.0, config.resource_sample_seconds))
                    continue

                pause_reason = None
                trainer.train_pair()
                boundary_evaluation = trainer.evaluate_pending_stage_boundary()
                now = time.monotonic()
                checkpoint_due = (
                    boundary_evaluation is not None
                    or
                    trainer.global_step % config.checkpoint_steps == 0
                    or now - last_checkpoint_at >= config.checkpoint_minutes * 60.0
                )
                if trainer.global_step % config.eval_interval_steps == 0:
                    trainer.evaluate()
                if checkpoint_due:
                    trainer.save_checkpoint()
                    last_checkpoint_at = time.monotonic()
                if now - last_heartbeat_at >= config.heartbeat_seconds:
                    write_status("running")

            if final_reason is None:
                final_reason = "target_steps_reached"
            if trainer.global_step == config.optimizer_steps:
                write_status("evaluating_final_gate")
            result_state, target_complete, learning_gate, formal_final_attempt = (
                finalize_training_attempt(
                    trainer,
                    config,
                    final_reason,
                    run_dir,
                )
            )
            runtime_warnings[:] = [str(item.message) for item in captured]
            result = {
                "schema_version": 3,
                "state": result_state,
                "reason": final_reason,
                "global_step": trainer.global_step,
                "target_steps": config.optimizer_steps,
                "run_eligible_for_aggregation": target_complete,
                "candidate_prerequisite": candidate_prerequisite,
                "config": config.to_dict(),
                "config_digest": stage1_config_digest(config.to_dict()),
                "validated_experiment_spec_digest": (
                    validated_experiment_spec_digest(config)
                ),
                "validated_experiment_compatibility_spec_digest": (
                    validated_experiment_compatibility_spec_digest(config)
                ),
                "metrics": trainer.metrics(),
                "final_evaluation": trainer.final_evaluation,
                "candidate_gate": trainer.gate_result,
                "stage_boundary_evaluations": trainer.stage_boundary_evaluations,
                "foundation_gate": trainer.foundation_gate_result,
                "learning_gate": learning_gate,
                "checkpoint_recovery": trainer.recovery_state(),
                "runtime_warnings": runtime_warnings,
                "fallback_observability": fallback_observability(
                    config.device,
                    runtime_warnings,
                ),
                "manifest": trainer.source_manifest,
                "snapshot_manifest": snapshot_manifest,
                "snapshot_manifest_hash": snapshot_manifest["manifest_hash"],
                "formal_final_attempt": formal_final_attempt,
            }
            result["run_completion_checks"] = run_completion_checks(result)
            if not all(result["run_completion_checks"].values()):
                result["state"] = "incomplete"
                result["run_eligible_for_aggregation"] = False
                trainer.record_run_completion(False, "run_completion_integrity_failed")
                result["candidate_gate"] = trainer.gate_result
                result["metrics"] = trainer.metrics()
            atomic_write_json(run_dir / "result.json", result)
            write_status(result["state"])
        return 0 if result["run_eligible_for_aggregation"] else 2
    except Exception as error:
        final_reason = f"{type(error).__name__}: {error}"
        if trainer is not None and checkpoint_loaded_or_created:
            trainer.record_run_completion(False, final_reason)
            try:
                trainer.save_checkpoint("emergency")
            except Exception as checkpoint_error:
                final_reason += f"; emergency checkpoint failed: {checkpoint_error}"
        if trainer is not None:
            try:
                atomic_write_json(
                    run_dir / "result.json",
                    {
                        "schema_version": 3,
                        "state": "failed",
                        "reason": final_reason,
                        "global_step": trainer.global_step,
                        "target_steps": config.optimizer_steps,
                        "run_eligible_for_aggregation": False,
                        "candidate_prerequisite": candidate_prerequisite,
                        "config": config.to_dict(),
                        "config_digest": stage1_config_digest(config.to_dict()),
                        "validated_experiment_spec_digest": (
                            validated_experiment_spec_digest(config)
                        ),
                        "validated_experiment_compatibility_spec_digest": (
                            validated_experiment_compatibility_spec_digest(config)
                        ),
                        "metrics": trainer.metrics(),
                        "candidate_gate": trainer.gate_result,
                        "manifest": trainer.source_manifest,
                        "snapshot_manifest": snapshot_manifest,
                        "snapshot_manifest_hash": snapshot_manifest["manifest_hash"],
                        "formal_final_attempt": (
                            _read_json_object(
                                run_dir / "formal-final-attempt.json"
                            )
                            or {
                                "required": config.formal_evaluation is True,
                                "state": "not_started",
                            }
                        ),
                    },
                )
            except Exception as result_error:
                final_reason += f"; failed result write failed: {result_error}"
        atomic_write_json(
            status_path,
            {
                "schema_version": 2,
                "pid": os.getpid(),
                "state": "failed",
                "started_at": started_at,
                "updated_at": utc_now(),
                "step": trainer.global_step if trainer is not None else 0,
                "latest_checkpoint": trainer.last_checkpoint if trainer is not None else None,
                "error": final_reason,
                "traceback": traceback.format_exc(),
                "runtime_warnings": runtime_warnings,
                "fallback_observability": fallback_observability(config.device, runtime_warnings),
                "snapshot_manifest_hash": snapshot_manifest["manifest_hash"],
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
