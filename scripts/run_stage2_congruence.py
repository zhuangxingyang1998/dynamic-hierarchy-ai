"""Run bounded Stage 2 R6 state-congruence construction or calibration."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dynamic_hierarchy.provenance import runtime_provenance
from dynamic_hierarchy.reporting import fallback_observability
from dynamic_hierarchy.resource_guard import ResourceGuard
from dynamic_hierarchy.run_lock import PerRunMutex
from dynamic_hierarchy.stage2_congruence_config import (
    R6_PACKET,
    load_stage2_congruence_config,
)
from dynamic_hierarchy.stage2_congruence_runtime import (
    Stage2CongruenceTrainer,
    atomic_write_json,
    existing_initialization_identity,
    latest_stage2_congruence_checkpoint,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_dir(config_path: Path, output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"stage2-r6-{config_path.stem}-{stamp}"


def _freeze_config(run_dir: Path, config: dict[str, object]) -> None:
    path = run_dir / "frozen-config.json"
    canonical = json.loads(json.dumps(config))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != canonical:
            raise RuntimeError("run directory contains a different frozen R6 config")
        return
    atomic_write_json(path, canonical)


def _validate_run_dir(run_dir: Path, run_kind: str, canonical: str) -> None:
    if run_kind == "calibration_only" and run_dir.resolve() != Path(canonical).resolve():
        raise ValueError("R6 calibration_only requires its canonical run directory")


def _validate_run_lifecycle(run_dir: Path, resume: bool) -> None:
    entries = tuple(run_dir.iterdir())
    if (run_dir / "result.json").is_file():
        raise FileExistsError(f"R6 completed result is immutable: {run_dir}")
    if entries and not resume:
        raise FileExistsError(
            f"R6 nonempty run directory requires --resume: {run_dir}"
        )
    if not entries and resume:
        raise FileNotFoundError(f"R6 cannot resume an empty run directory: {run_dir}")
    if resume and not (run_dir / "frozen-config.json").is_file():
        raise FileNotFoundError("R6 resumed run lacks frozen-config.json")


def _initialization_recovery_permissions(run_dir: Path) -> tuple[bool, bool]:
    instance_exists = (run_dir / "run-instance.json").is_file()
    snapshot_exists = (run_dir / "snapshot" / "source-manifest.json").is_file()
    if instance_exists:
        if not snapshot_exists:
            raise RuntimeError("R6 run instance exists without a source snapshot")
        return False, False
    temporary_pattern = re.compile(r"^\.snapshot\.[0-9a-f]{32}\.tmp$")
    for item in tuple(run_dir.iterdir()):
        if temporary_pattern.fullmatch(item.name):
            if not item.is_dir() or item.parent.resolve() != run_dir.resolve():
                raise RuntimeError("R6 source snapshot temporary path is invalid")
            shutil.rmtree(item)
    allowed = {
        "frozen-config.json",
        "snapshot",
        "run-instance.json",
        "checkpoints",
    }
    unexpected = {item.name for item in run_dir.iterdir()} - allowed
    if unexpected:
        raise RuntimeError(
            "R6 initialization recovery found existing evidence: "
            f"{sorted(unexpected)}"
        )
    snapshot_exists = (run_dir / "snapshot" / "source-manifest.json").is_file()
    checkpoints_exists = (run_dir / "checkpoints").exists()
    if checkpoints_exists and not instance_exists:
        raise RuntimeError("R6 checkpoints exist without a run instance")
    return not snapshot_exists, not instance_exists


def _exit_code(execution_disposition: str) -> int:
    return 0 if execution_disposition == "completed" else 2


def _result_output_path(
    run_dir: Path, execution_disposition: str, global_round: int
) -> Path:
    if execution_disposition != "calibration_incomplete":
        return run_dir / "result.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        run_dir
        / "attempt-results"
        / f"r6-{global_round:08d}-{stamp}.json"
    )


def _can_recover_missing_initial_checkpoint(run_dir: Path) -> bool:
    allowed_root = {
        "frozen-config.json",
        "snapshot",
        "run-instance.json",
        "checkpoints",
    }
    if {item.name for item in run_dir.iterdir()} - allowed_root:
        return False
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return True
    if not checkpoint_dir.is_dir() or (checkpoint_dir / "latest.json").exists():
        return False
    return all(
        item.is_file() and (item.name.endswith(".pt") or item.name.endswith(".tmp"))
        for item in checkpoint_dir.iterdir()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_stage2_congruence_config(args.config)
    run_dir = args.run_dir or (
        Path(config.canonical_run_dir)
        if config.run_kind == "calibration_only"
        else _default_run_dir(args.config, args.output_root)
    )
    _validate_run_dir(run_dir, config.run_kind, config.canonical_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    final_result_path = run_dir / "result.json"
    stop_path = run_dir / "STOP"
    guard = ResourceGuard(
        config.cpu_pause_percent,
        config.cpu_resume_percent,
        config.ram_pause_gb,
        config.ram_resume_gb,
        config.pressure_samples,
        config.recovery_samples,
    )

    with PerRunMutex(run_dir) as mutex, warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _validate_run_lifecycle(run_dir, args.resume)
        _freeze_config(run_dir, config.to_dict())
        allow_source_create, allow_instance_create = (
            _initialization_recovery_permissions(run_dir)
        )
        trainer: Stage2CongruenceTrainer | None = None
        try:
            trainer = Stage2CongruenceTrainer(
                config,
                run_dir,
                allow_create_run_instance=allow_instance_create,
                allow_create_source_snapshot=allow_source_create,
            )
            if args.resume:
                try:
                    checkpoint = latest_stage2_congruence_checkpoint(run_dir)
                except FileNotFoundError:
                    if not _can_recover_missing_initial_checkpoint(run_dir):
                        raise
                    trainer.save_checkpoint("initial-recovery")
                else:
                    trainer.load_checkpoint(checkpoint)
            else:
                trainer.save_checkpoint("initial")
            terminal_at_start = trainer.is_complete
            atomic_write_json(
                status_path,
                {**trainer.status("running"), "updated_at": utc_now(), "mutex": mutex.metadata()},
            )
            final_reason = "terminal_gate_reached"
            while not trainer.is_complete:
                if stop_path.exists():
                    final_reason = "cooperative_stop_requested"
                    break
                if trainer.time_budget_exhausted():
                    final_reason = "time_budget_exhausted"
                    break
                if trainer.needs_gate:
                    trainer.save_checkpoint("pre-gate")
                    trainer.run_gate()
                    trainer.save_checkpoint("post-gate")
                    continue
                sample = guard.sample()
                if guard.observe(sample):
                    atomic_write_json(
                        status_path,
                        {
                            **trainer.status("resource_paused", guard.pause_reason),
                            "updated_at": utc_now(),
                            "resource_sample": sample.to_dict(),
                            "mutex": mutex.metadata(),
                        },
                    )
                    time.sleep(2.0)
                    continue
                losses = trainer.train_step()
                if trainer.global_round % config.checkpoint_steps == 0:
                    trainer.save_checkpoint()
                atomic_write_json(
                    status_path,
                    {
                        **trainer.status("running"),
                        "updated_at": utc_now(),
                        "latest_losses": losses,
                        "resource_sample": sample.to_dict(),
                        "mutex": mutex.metadata(),
                    },
                )
            if trainer.is_complete:
                execution_disposition = str(trainer.execution_disposition)
                if execution_disposition == "reserve_stranded":
                    final_reason = "reserve_stranded"
                elif execution_disposition == "implementation_invalid":
                    final_reason = "implementation_invalid"
                    if not terminal_at_start:
                        trainer.save_checkpoint("implementation-invalid")
                elif execution_disposition == "completed":
                    if not terminal_at_start:
                        trainer.save_checkpoint("final")
                else:
                    raise RuntimeError("R6 terminal execution disposition is invalid")
            elif trainer.reserve_stranded:
                execution_disposition = "reserve_stranded"
                final_reason = "reserve_stranded"
            else:
                execution_disposition = "calibration_incomplete"
                trainer.save_checkpoint("incomplete")
            runtime_warnings = [str(item.message) for item in captured]
            result = trainer.result(execution_disposition)
            result.update(
                {
                    "final_reason": final_reason,
                    "runtime_warnings": runtime_warnings,
                    "fallback_observability": fallback_observability(
                        config.device, runtime_warnings
                    ),
                    "provenance": runtime_provenance(
                        config.cpu_threads, trainer.backend.metadata()
                    ),
                    "completed_at": utc_now(),
                }
            )
            result_path = _result_output_path(
                run_dir, execution_disposition, trainer.global_round
            )
            if result_path == final_result_path and final_result_path.exists():
                raise FileExistsError("R6 terminal result became immutable during execution")
            atomic_write_json(result_path, result)
            atomic_write_json(
                status_path,
                {
                    **trainer.status(
                        "completed"
                        if execution_disposition == "completed"
                        else execution_disposition,
                        final_reason,
                    ),
                    "updated_at": utc_now(),
                    "execution_disposition": execution_disposition,
                    "result": str(result_path),
                    "mutex": mutex.metadata(),
                },
            )
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "result": str(result_path),
                        "execution_disposition": execution_disposition,
                        "research_disposition": trainer.research_disposition,
                        "final_reason": final_reason,
                        "global_round": trainer.global_round,
                    },
                    indent=2,
                )
            )
            return _exit_code(execution_disposition)
        except Exception as error:
            failure = {
                "schema_version": 3,
                "packet": R6_PACKET,
                "revision": config.revision,
                "phase": config.phase,
                "state": "implementation_invalid",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "updated_at": utc_now(),
            }
            if trainer is not None:
                failure["global_round"] = trainer.global_round
                failure["elapsed_seconds"] = trainer.elapsed_seconds()
                failure["reserve_state"] = trainer.ledger.get("true_reserve_state")
                failure["run_instance_digest"] = trainer.run_instance_digest
                failure["source_snapshot_digest"] = (
                    trainer.source_snapshot_digest
                )
            else:
                failure.update(existing_initialization_identity(run_dir))
            atomic_write_json(run_dir / "failure.json", failure)
            atomic_write_json(status_path, failure)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
