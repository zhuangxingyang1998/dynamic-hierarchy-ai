"""Run the isolated canonical-snapshot Stage 1 formal campaign v2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_hierarchy.provenance import source_manifest
from dynamic_hierarchy.run_lock import PerRunMutex
from dynamic_hierarchy.stage1_campaign import (
    CAMPAIGN_MANIFEST_NAME,
    CAMPAIGN_VERSION,
    CANONICAL_SNAPSHOT_NAME,
    campaign_seed_freshness,
    create_campaign_package,
    materialize_campaign_run,
    verify_campaign_package,
)
from dynamic_hierarchy.stage1_config import (
    Stage1Config,
    load_stage1_config,
    stage1_config_digest,
)
from dynamic_hierarchy.stage1_integrity import verify_snapshot_manifest
from scripts.run_stage1_confirmation_sequence import (
    SequenceAbort,
    _archive_incomplete_result,
    _assert_no_other_project_worker,
    _atomic_write_json,
    _read_result,
    _recoverable_incomplete_checks,
    _resume,
    _run_is_live,
    _run_launcher_command,
    _sha256,
    _wait_for_terminal,
    _wait_for_worker_exit,
    verify_formal_result,
)
from scripts.stage1_worker import validate_candidate_prerequisite


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "stage1-revised-literal-formal-confirmation-v2-directml.json"
)
DEFAULT_CAMPAIGN_ROOT = ROOT / "runs" / "stage1-literal-formal-v2-campaign"
DEFAULT_STATE = ROOT / "runs" / "stage1-literal-formal-v2-sequence.json"
DEFAULT_AGGREGATE = (
    ROOT / "runs" / "stage1-literal-formal-v2-confirmation.json"
)
LEGACY_STATE = ROOT / "runs" / "stage1-literal-formal-sequence.json"
RUN_PREFIX = "stage1-formal-v2-"
STATE_SCHEMA_VERSION = 2


def _eligible_config(config: Stage1Config) -> bool:
    return (
        config.device == "directml"
        and config.formal_evaluation is True
        and config.requires_candidate_pass is True
        and config.optimizer_steps == 8000
        and config.final_eval_examples_per_seed == 10010
        and len(config.confirmation_training_seeds) == 8
        and len(config.eval_seeds) == 3
        and config.minimum_confirmation_training_seeds == 8
        and len(
            {
                *config.confirmation_training_seeds,
                *config.eval_seeds,
                config.foundation_eval_seed,
            }
        )
        == 12
    )


def _campaign_config_path(
    campaign_root: Path,
    manifest: dict[str, Any],
) -> Path:
    relative = manifest.get("formal_config_path")
    if not isinstance(relative, str) or not relative:
        raise SequenceAbort("campaign formal config path is missing")
    snapshot_root = campaign_root.resolve() / CANONICAL_SNAPSHOT_NAME
    path = (snapshot_root / relative).resolve()
    try:
        path.relative_to(snapshot_root)
    except ValueError as error:
        raise SequenceAbort("campaign formal config escapes snapshot") from error
    if not path.is_file():
        raise SequenceAbort(f"campaign formal config is missing: {path}")
    return path


def _verify_campaign(
    campaign_root: Path,
    config: Stage1Config | None = None,
) -> dict[str, Any]:
    checks, manifest = verify_campaign_package(campaign_root, config)
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed or manifest is None:
        raise SequenceAbort(f"canonical campaign verification failed: {failed}")
    return manifest


def _candidate_path(config: Stage1Config) -> Path:
    path = (ROOT / config.candidate_prerequisite_result_path).resolve()
    if not path.is_file():
        raise SequenceAbort(f"candidate result is missing: {path}")
    if _sha256(path).lower() != (
        config.candidate_prerequisite_result_digest.lower()
    ):
        raise SequenceAbort("candidate result digest does not match v2 config")
    return path


def _verify_candidate(config: Stage1Config, path: Path) -> dict[str, object]:
    try:
        verification = validate_candidate_prerequisite(config, path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SequenceAbort(
            f"candidate prerequisite failed for campaign v2: {error}"
        ) from error
    if verification.get("passed") is not True:
        raise SequenceAbort("candidate prerequisite did not pass for campaign v2")
    return verification


def _state_payload(
    campaign_root: Path,
    manifest: dict[str, Any],
    config: Stage1Config,
    candidate_path: Path,
    aggregate_output: Path,
    freshness: dict[str, object],
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "queue_version": CAMPAIGN_VERSION,
        "state": "ready",
        "legacy_queue_policy": "excluded_from_formal_statistics",
        "campaign_root": str(campaign_root.resolve()),
        "campaign_manifest_path": str(
            (campaign_root / CAMPAIGN_MANIFEST_NAME).resolve()
        ),
        "campaign_manifest_hash": manifest["manifest_hash"],
        "campaign_snapshot_manifest_hash": manifest["snapshot_manifest_hash"],
        "campaign_source_manifest_hash": manifest["source_manifest_hash"],
        "environment_receipt_digest": manifest["environment_receipt_digest"],
        "candidate_identity_digest": manifest["candidate_identity_digest"],
        "config_digest": stage1_config_digest(config.to_dict()),
        "candidate_result_path": str(candidate_path.resolve()),
        "candidate_result_digest": _sha256(candidate_path),
        "training_seeds": list(config.confirmation_training_seeds),
        "evaluation_seeds": list(config.eval_seeds),
        "foundation_seed": config.foundation_eval_seed,
        "aggregate_output": str(aggregate_output.resolve()),
        "seed_freshness": freshness,
        "runs": {},
    }


def _load_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SequenceAbort(f"cannot read campaign v2 state: {error}") from error
    if not isinstance(state, dict):
        raise SequenceAbort("campaign v2 state is not an object")
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("queue_version") != CAMPAIGN_VERSION
        or state.get("legacy_queue_policy")
        != "excluded_from_formal_statistics"
        or not isinstance(state.get("runs"), dict)
    ):
        raise SequenceAbort("state is not an isolated campaign v2 state")
    return state


def _state_run_roots(state: dict[str, Any]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for entry in state.get("runs", {}).values():
        if not isinstance(entry, dict):
            raise SequenceAbort("campaign state run entry is invalid")
        run_dir = Path(str(entry.get("run_dir", ""))).resolve()
        if not run_dir.name.startswith(RUN_PREFIX):
            raise SequenceAbort("campaign state contains a non-v2 run")
        roots.append(run_dir)
    return tuple(roots)


def prepare_campaign(
    config_path: Path,
    campaign_root: Path,
    state_path: Path,
    aggregate_output: Path,
) -> tuple[dict[str, Any], Stage1Config, dict[str, Any], Path]:
    config_path = config_path.resolve()
    campaign_root = campaign_root.resolve()
    state_path = state_path.resolve()
    aggregate_output = aggregate_output.resolve()
    if state_path == LEGACY_STATE.resolve():
        raise SequenceAbort("campaign v2 refuses the legacy sequence state path")
    if campaign_root == LEGACY_STATE.resolve().parent:
        raise SequenceAbort("campaign v2 root must be separate from legacy state")

    if state_path.is_file():
        state = _load_state(state_path)
        if Path(state["campaign_root"]).resolve() != campaign_root:
            raise SequenceAbort("campaign root does not match v2 state")
        manifest = _verify_campaign(campaign_root)
        frozen_config_path = _campaign_config_path(campaign_root, manifest)
        config = load_stage1_config(frozen_config_path)
        manifest = _verify_campaign(campaign_root, config)
        candidate_path = Path(state["candidate_result_path"]).resolve()
    else:
        state = {}
        if campaign_root.exists():
            manifest = _verify_campaign(campaign_root)
            frozen_config_path = _campaign_config_path(campaign_root, manifest)
            config = load_stage1_config(frozen_config_path)
            manifest = _verify_campaign(campaign_root, config)
            candidate_path = _candidate_path(config)
        else:
            config = load_stage1_config(config_path)
            if not _eligible_config(config):
                raise SequenceAbort(
                    "config is not an eligible canonical eight-seed formal plan"
                )
            candidate_path = _candidate_path(config)
            candidate_verification = _verify_candidate(config, candidate_path)
            initial_freshness = campaign_seed_freshness(
                ROOT,
                config_path,
                config,
                excluded_files=(state_path, aggregate_output),
            )
            if initial_freshness.get("passed") is not True:
                raise SequenceAbort(
                    "campaign v2 seeds are not fresh: "
                    f"{initial_freshness.get('overlap')}"
                )
            try:
                create_campaign_package(
                    ROOT,
                    campaign_root,
                    config_path,
                    config,
                    candidate_path,
                    candidate_verification,
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise SequenceAbort(
                    f"cannot create canonical campaign: {error}"
                ) from error
            manifest = _verify_campaign(campaign_root, config)

    if not _eligible_config(config):
        raise SequenceAbort("frozen campaign config is not formal-v2 eligible")
    candidate_path = _candidate_path(config)
    _verify_candidate(config, candidate_path)
    excluded_roots = (campaign_root, *_state_run_roots(state))
    freshness = campaign_seed_freshness(
        ROOT,
        config_path,
        config,
        excluded_roots=excluded_roots,
        excluded_files=(state_path, aggregate_output),
    )
    if freshness.get("passed") is not True:
        raise SequenceAbort(
            "campaign v2 seed freshness failed: "
            f"{freshness.get('overlap')}; "
            f"unreadable={freshness.get('unreadable_evidence')}"
        )
    expected_state = _state_payload(
        campaign_root,
        manifest,
        config,
        candidate_path,
        aggregate_output,
        freshness,
    )
    if not state:
        state = expected_state
        _atomic_write_json(state_path, state)
    else:
        for field in (
            "schema_version",
            "queue_version",
            "legacy_queue_policy",
            "campaign_root",
            "campaign_manifest_path",
            "campaign_manifest_hash",
            "campaign_snapshot_manifest_hash",
            "campaign_source_manifest_hash",
            "environment_receipt_digest",
            "candidate_identity_digest",
            "config_digest",
            "candidate_result_path",
            "candidate_result_digest",
            "training_seeds",
            "evaluation_seeds",
            "foundation_seed",
            "aggregate_output",
        ):
            if state.get(field) != expected_state.get(field):
                raise SequenceAbort(f"campaign v2 state mismatch for {field}")
        state["seed_freshness"] = freshness
        _atomic_write_json(state_path, state)
    return state, config, manifest, candidate_path


def _materialized_run_checks(
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, bool]:
    snapshot_root = run_dir / "snapshot"
    snapshot_checks, snapshot_manifest = verify_snapshot_manifest(snapshot_root)
    try:
        source = source_manifest(snapshot_root)
        receipt = json.loads(
            (run_dir / "campaign-receipt.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError):
        source = {}
        receipt = {}
    return {
        **snapshot_checks,
        "materialized_snapshot_pin": (
            isinstance(snapshot_manifest, dict)
            and snapshot_manifest.get("manifest_hash")
            == manifest["snapshot_manifest_hash"]
        ),
        "materialized_source_pin": (
            source.get("manifest_hash") == manifest["source_manifest_hash"]
        ),
        "materialized_campaign_receipt": (
            receipt.get("schema_version") == 1
            and receipt.get("campaign_version") == CAMPAIGN_VERSION
            and receipt.get("campaign_manifest_hash") == manifest["manifest_hash"]
            and receipt.get("snapshot_manifest_hash")
            == manifest["snapshot_manifest_hash"]
            and receipt.get("source_manifest_hash")
            == manifest["source_manifest_hash"]
        ),
    }


def _verify_materialized_run(
    run_dir: Path,
    manifest: dict[str, Any],
) -> None:
    checks = _materialized_run_checks(run_dir, manifest)
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise SequenceAbort(
            f"materialized campaign run failed pins: {failed}"
        )


def _user_stop_requires_resume(
    result: dict[str, Any],
    run_dir: Path,
) -> bool:
    return (
        result.get("state") == "incomplete"
        and result.get("reason") == "user_stop"
        and (run_dir / "control" / "STOP").is_file()
    )


def _launch_prepared(
    config_relative_path: str,
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
        config_relative_path,
        "-TrainingSeed",
        str(seed),
        "-CandidateResult",
        str(candidate_path),
        "-PreparedRunDir",
        str(run_dir),
        "-LaunchReceipt",
        str(receipt_path),
    ]
    _run_launcher_command(command, receipt_path, f"campaign-v2 seed {seed}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SequenceAbort(f"seed {seed} launch receipt is invalid") from error
    if (
        Path(str(receipt.get("run_dir", ""))).resolve() != run_dir.resolve()
        or receipt.get("training_seed") != seed
        or receipt.get("prepared_campaign_run") is not True
    ):
        raise SequenceAbort(f"seed {seed} prepared launch receipt mismatch")


def verify_campaign_result(
    result_path: Path,
    config: Stage1Config,
    seed: int,
    campaign_root: Path,
) -> dict[str, Any]:
    manifest = _verify_campaign(campaign_root, config)
    result = verify_formal_result(result_path, config, seed)
    run_dir = result_path.resolve().parent
    _verify_materialized_run(run_dir, manifest)
    checks = {
        "campaign_source_pin": (
            result.get("manifest", {}).get("manifest_hash")
            == manifest["source_manifest_hash"]
        ),
        "campaign_snapshot_pin": (
            result.get("snapshot_manifest_hash")
            == manifest["snapshot_manifest_hash"]
        ),
        "campaign_embedded_snapshot_pin": (
            result.get("snapshot_manifest", {}).get("manifest_hash")
            == manifest["snapshot_manifest_hash"]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise SequenceAbort(
            f"formal campaign seed {seed} failed campaign pins: {failed}"
        )
    return result


def _run_campaign(
    config_path: Path,
    campaign_root: Path,
    state_path: Path,
    aggregate_output: Path,
    poll_seconds: float,
    prepare_only: bool,
) -> int:
    state, config, manifest, candidate_path = prepare_campaign(
        config_path,
        campaign_root,
        state_path,
        aggregate_output,
    )
    if prepare_only:
        return 0
    result_paths: list[Path] = []
    try:
        for index, seed in enumerate(config.confirmation_training_seeds):
            manifest = _verify_campaign(campaign_root, config)
            key = str(seed)
            entry = state["runs"].get(key)
            if entry is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                run_dir = (
                    ROOT
                    / "runs"
                    / f"{RUN_PREFIX}{index + 1:02d}-{seed}-{stamp}"
                )
                entry = {
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "status": "planned",
                    "campaign_manifest_hash": manifest["manifest_hash"],
                }
                state["runs"][key] = entry
                _atomic_write_json(state_path, state)
            run_dir = Path(entry["run_dir"]).resolve()
            if not run_dir.name.startswith(RUN_PREFIX):
                raise SequenceAbort("v2 state attempted to use a legacy run")
            result_path = run_dir / "result.json"
            while True:
                manifest = _verify_campaign(campaign_root, config)
                if result_path.is_file():
                    _wait_for_worker_exit(run_dir, poll_seconds)
                    observed = _read_result(result_path)
                    if observed.get("state") == "completed":
                        verify_campaign_result(
                            result_path,
                            config,
                            seed,
                            campaign_root,
                        )
                        break
                    if observed.get("state") == "failed":
                        raise SequenceAbort(
                            f"campaign-v2 seed {seed} worker failed: "
                            f"{observed.get('reason')}"
                        )
                    recovery_checks = _recoverable_incomplete_checks(
                        observed,
                        run_dir,
                        config,
                    )
                    if not all(
                        value is True for value in recovery_checks.values()
                    ):
                        failed = sorted(
                            name
                            for name, passed in recovery_checks.items()
                            if passed is not True
                        )
                        raise SequenceAbort(
                            f"campaign-v2 seed {seed} is not recoverable: {failed}"
                        )
                    if _user_stop_requires_resume(observed, run_dir):
                        entry["status"] = "awaiting_user_resume"
                        entry["recoverable_result"] = str(result_path)
                        entry["recoverable_result_digest"] = _sha256(result_path)
                        state["state"] = "paused_recoverable"
                        state["failure"] = None
                        _atomic_write_json(state_path, state)
                        return 3
                    archived = _archive_incomplete_result(
                        result_path,
                        observed,
                    )
                    entry.setdefault("incomplete_attempts", []).append(
                        {
                            "reason": observed.get("reason"),
                            "global_step": observed.get("global_step"),
                            "archived_result": str(archived),
                            "archived_result_digest": _sha256(archived),
                        }
                    )
                    entry["status"] = "resuming"
                    state["state"] = "running"
                    state["failure"] = None
                    _atomic_write_json(state_path, state)

                receipt_path = state_path.with_name(
                    f"{state_path.stem}.seed-{seed}.receipt.json"
                )
                _assert_no_other_project_worker(run_dir)
                if run_dir.exists():
                    _verify_materialized_run(run_dir, manifest)
                    if _run_is_live(run_dir):
                        entry["status"] = "waiting"
                    elif (run_dir / "pid.json").is_file():
                        entry["status"] = "resuming"
                        _atomic_write_json(state_path, state)
                        _resume(run_dir, receipt_path)
                    else:
                        allowed = {"snapshot", "campaign-receipt.json"}
                        if {path.name for path in run_dir.iterdir()} != allowed:
                            raise SequenceAbort(
                                "prepared v2 run has partial launch evidence "
                                "without pid.json"
                            )
                        entry["status"] = "launching_prepared"
                        _atomic_write_json(state_path, state)
                        _launch_prepared(
                            manifest["formal_config_path"],
                            candidate_path,
                            seed,
                            run_dir,
                            receipt_path,
                        )
                else:
                    entry["status"] = "materializing"
                    _atomic_write_json(state_path, state)
                    try:
                        materialize_campaign_run(
                            campaign_root,
                            run_dir,
                            config,
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        raise SequenceAbort(
                            f"cannot materialize seed {seed}: {error}"
                        ) from error
                    _verify_materialized_run(run_dir, manifest)
                    entry["status"] = "launching_prepared"
                    _atomic_write_json(state_path, state)
                    _launch_prepared(
                        manifest["formal_config_path"],
                        candidate_path,
                        seed,
                        run_dir,
                        receipt_path,
                    )
                entry["status"] = "waiting"
                state["state"] = "running"
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
        _atomic_write_json(state_path, state)
        raise

    if len(result_paths) != 8:
        raise SequenceAbort("campaign v2 cannot aggregate without eight results")
    state["state"] = "aggregating"
    _atomic_write_json(state_path, state)
    snapshot_root = campaign_root / CANONICAL_SNAPSHOT_NAME
    aggregate_command = [
        sys.executable,
        str(snapshot_root / "scripts" / "aggregate_stage1_confirmation.py"),
        "--results",
        *(str(path) for path in result_paths),
        "--output",
        str(aggregate_output),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(snapshot_root / "src")
    completed = subprocess.run(
        aggregate_command,
        cwd=snapshot_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        state["state"] = "aggregate_failed"
        state["failure"] = completed.stderr.strip() or completed.stdout.strip()
        _atomic_write_json(state_path, state)
        return completed.returncode
    aggregate = json.loads(aggregate_output.read_text(encoding="utf-8"))
    state["state"] = "completed"
    state["aggregate_result"] = str(aggregate_output)
    state["stage2_unblocked"] = aggregate.get("stage2_unblocked") is True
    _atomic_write_json(state_path, state)
    return 0


def run_campaign(
    config_path: Path,
    campaign_root: Path,
    state_path: Path,
    aggregate_output: Path,
    poll_seconds: float,
    prepare_only: bool = False,
) -> int:
    mutex = PerRunMutex(ROOT / "runs" / ".stage1-formal-v2-sequence")
    try:
        mutex.acquire()
    except RuntimeError as error:
        raise SequenceAbort(
            "another formal campaign v2 coordinator is active"
        ) from error
    try:
        return _run_campaign(
            config_path.resolve(),
            campaign_root.resolve(),
            state_path.resolve(),
            aggregate_output.resolve(),
            poll_seconds,
            prepare_only,
        )
    finally:
        mutex.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=DEFAULT_AGGREGATE,
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze and verify campaign state without creating or launching runs.",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    try:
        return run_campaign(
            args.config,
            args.campaign_root,
            args.state,
            args.aggregate_output,
            args.poll_seconds,
            args.prepare_only,
        )
    except SequenceAbort as error:
        print(f"fail-closed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "campaign v2 monitor interrupted; active worker was not stopped. "
            "Run the same command to resume.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
