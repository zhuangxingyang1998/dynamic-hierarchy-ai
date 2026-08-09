"""Run a bounded Stage 2 R2 smoke or calibration-only experiment."""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dynamic_hierarchy.provenance import runtime_provenance
from dynamic_hierarchy.reporting import fallback_observability
from dynamic_hierarchy.resource_guard import ResourceGuard
from dynamic_hierarchy.run_lock import PerRunMutex
from dynamic_hierarchy.stage2_config import load_stage2_config
from dynamic_hierarchy.stage2_runtime import (
    Stage2Trainer,
    atomic_write_json,
    latest_stage2_checkpoint,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_dir(config_path: Path, output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"stage2-{config_path.stem}-{stamp}"


def _freeze_config(run_dir: Path, config: dict[str, object]) -> None:
    path = run_dir / "frozen-config.json"
    canonical = json.loads(json.dumps(config))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != canonical:
            raise RuntimeError("run directory contains a different frozen Stage 2 config")
        return
    atomic_write_json(path, canonical)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reevaluate-only", action="store_true")
    args = parser.parse_args()
    if args.reevaluate_only and not args.resume:
        parser.error("--reevaluate-only requires --resume")

    config = load_stage2_config(args.config)
    run_dir = args.run_dir or _default_run_dir(args.config, args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_config(run_dir, config.to_dict())
    status_path = run_dir / "status.json"
    result_path = run_dir / (
        "result-reevaluated.json" if args.reevaluate_only else "result.json"
    )
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
        trainer: Stage2Trainer | None = None
        try:
            trainer = Stage2Trainer(config, run_dir)
            if args.resume:
                trainer.load_checkpoint(latest_stage2_checkpoint(run_dir))
            if args.reevaluate_only and trainer.global_step != config.optimizer_steps:
                raise RuntimeError("evaluation-only recovery requires a completed training checkpoint")
            atomic_write_json(
                status_path,
                {
                    **trainer.status("running"),
                    "updated_at": utc_now(),
                    "mutex": mutex.metadata(),
                },
            )
            final_reason = (
                "evaluation_only_recovery"
                if args.reevaluate_only
                else "target_steps_reached"
            )
            while not args.reevaluate_only and trainer.global_step < config.optimizer_steps:
                if stop_path.exists():
                    final_reason = "cooperative_stop_requested"
                    break
                if trainer.time_budget_exhausted():
                    final_reason = "time_budget_exhausted"
                    break
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
                if trainer.global_step % config.checkpoint_steps == 0:
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

            if final_reason in {"target_steps_reached", "evaluation_only_recovery"}:
                trainer.evaluate()
                if not args.reevaluate_only:
                    trainer.save_checkpoint("final")
                disposition = (
                    "calibration_inconclusive"
                    if config.run_kind == "calibration_only"
                    else "smoke_completed"
                )
            else:
                trainer.save_checkpoint("incomplete")
                disposition = (
                    "calibration_inconclusive"
                    if config.run_kind == "calibration_only"
                    else "smoke_incomplete"
                )
            runtime_warnings = [str(item.message) for item in captured]
            result = trainer.result(disposition)
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
            if args.reevaluate_only:
                result["evaluation_recovery"] = {
                    "kind": "evaluation_only_after_structure_telemetry_fix",
                    "training_checkpoint": trainer.last_checkpoint,
                    "optimizer_updates_per_model": trainer.global_step,
                    "training_weights_updated": False,
                    "original_result": str(run_dir / "result.json"),
                    "note": (
                        "The original result used a case-mismatched MERGE trace filter. "
                        "Task metrics and weights were retained; structure metrics were recomputed."
                    ),
                }
            atomic_write_json(result_path, result)
            atomic_write_json(
                status_path,
                {
                    **trainer.status("completed", final_reason),
                    "updated_at": utc_now(),
                    "disposition": disposition,
                    "result": str(result_path),
                    "mutex": mutex.metadata(),
                },
            )
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "result": str(result_path),
                        "disposition": disposition,
                        "final_reason": final_reason,
                        "global_step": trainer.global_step,
                    },
                    indent=2,
                )
            )
            return (
                0
                if final_reason in {"target_steps_reached", "evaluation_only_recovery"}
                else 2
            )
        except Exception as error:
            failure = {
                "schema_version": 1,
                "packet": "DH-S2-R2",
                "state": "implementation_invalid",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "updated_at": utc_now(),
            }
            if trainer is not None:
                failure["global_step"] = trainer.global_step
                failure["elapsed_seconds"] = trainer.elapsed_seconds()
            atomic_write_json(run_dir / "failure.json", failure)
            atomic_write_json(status_path, failure)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
