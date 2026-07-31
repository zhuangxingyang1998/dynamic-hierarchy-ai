"""Canonical immutable campaign packages for formal Stage 1 confirmation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .provenance import source_manifest
from .snapshot import create_snapshot, write_snapshot_manifest
from .stage1_config import (
    Stage1Config,
    stage1_config_digest,
    validated_experiment_compatibility_spec_digest,
    validated_experiment_spec_digest,
)
from .stage1_integrity import SHA256_PATTERN, verify_snapshot_manifest


CAMPAIGN_VERSION = "literal-formal-confirmation-v2"
CANONICAL_SNAPSHOT_NAME = "canonical-snapshot"
CAMPAIGN_MANIFEST_NAME = "campaign-manifest.json"
ENVIRONMENT_RECEIPT_RELATIVE = Path("campaign/environment-receipt.json")
CANDIDATE_IDENTITY_RELATIVE = Path("campaign/candidate-identity.json")


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _candidate_pins(config: Stage1Config) -> dict[str, str]:
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


def _windows_hardware_identity() -> dict[str, object]:
    if platform.system() != "Windows":
        return {"status": "not_windows"}
    script = """
$ErrorActionPreference = "Stop"
$computer = Get-CimInstance Win32_ComputerSystem |
    Select-Object Manufacturer, Model, SystemType
$controllers = @(
    Get-CimInstance Win32_VideoController |
        Select-Object Name, DriverVersion, PNPDeviceID,
            AdapterCompatibility, VideoProcessor |
        Sort-Object PNPDeviceID
)
[ordered]@{
    computer = $computer
    video_controllers = $controllers
} | ConvertTo-Json -Depth 4 -Compress
""".strip()
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            creationflags=creation_flags,
        )
        if completed.returncode != 0:
            return {"status": "unavailable"}
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"status": "unavailable"}
    return {"status": "captured", **payload}


def _environment_identity(project_root: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    packages: dict[str, str | None] = {}
    for name in ("torch", "torch-directml", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock_hashes = {
        name: _file_digest(project_root / name)
        for name in ("requirements-cpu.lock", "requirements-directml.lock")
        if (project_root / name).is_file()
    }
    directml_python = (
        project_root / ".venv-directml" / "Scripts" / "python.exe"
    ).resolve()
    return {
        "project_root": str(project_root),
        "platform": platform.platform(),
        "coordinator_python": {
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
        },
        "directml_python": {
            "path": str(directml_python),
            "present": directml_python.is_file(),
            "digest": (
                _file_digest(directml_python)
                if directml_python.is_file()
                else None
            ),
        },
        "packages": packages,
        "lock_hashes": lock_hashes,
        "hardware": _windows_hardware_identity(),
    }


def _environment_receipt(project_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "identity": _environment_identity(project_root),
    }


def _candidate_identity(
    project_root: Path,
    config: Stage1Config,
    candidate_result: Path,
    candidate_verification: dict[str, object],
) -> dict[str, object]:
    try:
        relative_candidate = candidate_result.resolve().relative_to(
            project_root.resolve()
        )
        candidate_path = relative_candidate.as_posix()
    except ValueError:
        candidate_path = str(candidate_result.resolve())
    return {
        "schema_version": 1,
        "candidate_result_path": candidate_path,
        "candidate_result_digest": _file_digest(candidate_result),
        "pins": _candidate_pins(config),
        "verification_passed": candidate_verification.get("passed") is True,
        "verification_checks": candidate_verification.get("checks", {}),
    }


def _campaign_manifest_payload(
    snapshot_root: Path,
    config_relative_path: str,
    config: Stage1Config,
    candidate_identity: dict[str, object],
    campaign_version: str,
) -> dict[str, object]:
    snapshot_manifest = json.loads(
        (snapshot_root / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    source = source_manifest(snapshot_root)
    environment_path = snapshot_root / ENVIRONMENT_RECEIPT_RELATIVE
    candidate_path = snapshot_root / CANDIDATE_IDENTITY_RELATIVE
    return {
        "schema_version": 1,
        "campaign_version": campaign_version,
        "formal_config_path": config_relative_path,
        "config_digest": stage1_config_digest(config.to_dict()),
        "validated_experiment_spec_digest": (
            validated_experiment_spec_digest(config)
        ),
        "validated_experiment_compatibility_spec_digest": (
            validated_experiment_compatibility_spec_digest(config)
        ),
        "snapshot_manifest_hash": snapshot_manifest["manifest_hash"],
        "source_manifest_hash": source["manifest_hash"],
        "environment_receipt_digest": _file_digest(environment_path),
        "candidate_identity_digest": _file_digest(candidate_path),
        "candidate_result_digest": candidate_identity[
            "candidate_result_digest"
        ],
        "candidate_pins": _candidate_pins(config),
        "training_seeds": list(config.confirmation_training_seeds),
        "evaluation_seeds": list(config.eval_seeds),
        "foundation_seed": config.foundation_eval_seed,
    }


def create_campaign_package(
    project_root: Path,
    campaign_root: Path,
    config_path: Path,
    config: Stage1Config,
    candidate_result: Path,
    candidate_verification: dict[str, object],
    *,
    campaign_version: str = CAMPAIGN_VERSION,
    verify_live_environment: bool = False,
) -> dict[str, object]:
    project_root = project_root.resolve()
    campaign_root = campaign_root.resolve()
    config_path = config_path.resolve()
    if campaign_root.exists():
        raise FileExistsError(f"campaign root already exists: {campaign_root}")
    if not campaign_version:
        raise ValueError("campaign version must be non-empty")
    if candidate_verification.get("passed") is not True:
        raise RuntimeError("candidate verification did not pass")
    try:
        config_relative = config_path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise ValueError("formal config must be inside the project") from error
    temporary = campaign_root.with_name(
        f"{campaign_root.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        raise FileExistsError(f"campaign temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        snapshot_root = temporary / CANONICAL_SNAPSHOT_NAME
        create_snapshot(project_root, snapshot_root)
        environment = _environment_receipt(project_root)
        candidate_identity = _candidate_identity(
            project_root,
            config,
            candidate_result,
            candidate_verification,
        )
        _atomic_write_json(
            snapshot_root / ENVIRONMENT_RECEIPT_RELATIVE,
            environment,
        )
        _atomic_write_json(
            snapshot_root / CANDIDATE_IDENTITY_RELATIVE,
            candidate_identity,
        )
        write_snapshot_manifest(snapshot_root)
        payload = _campaign_manifest_payload(
            snapshot_root,
            config_relative,
            config,
            candidate_identity,
            campaign_version,
        )
        manifest = {**payload, "manifest_hash": _json_digest(payload)}
        _atomic_write_json(temporary / CAMPAIGN_MANIFEST_NAME, manifest)
        checks, _ = verify_campaign_package(
            temporary,
            config,
            expected_campaign_version=campaign_version,
            verify_live_environment=verify_live_environment,
        )
        failed = sorted(name for name, passed in checks.items() if passed is not True)
        if failed:
            raise RuntimeError(
                f"new canonical campaign failed verification: {failed}"
            )
        os.replace(temporary, campaign_root)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_campaign_package(
    campaign_root: Path,
    expected_config: Stage1Config | None = None,
    *,
    expected_campaign_version: str = CAMPAIGN_VERSION,
    verify_live_environment: bool = False,
) -> tuple[dict[str, bool], dict[str, Any] | None]:
    campaign_root = campaign_root.resolve()
    snapshot_root = campaign_root / CANONICAL_SNAPSHOT_NAME
    manifest_path = campaign_root / CAMPAIGN_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = None
    try:
        environment = json.loads(
            (snapshot_root / ENVIRONMENT_RECEIPT_RELATIVE).read_text(
                encoding="utf-8"
            )
        )
        candidate_identity = json.loads(
            (snapshot_root / CANDIDATE_IDENTITY_RELATIVE).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        environment = None
        candidate_identity = None
    live_environment_matches = not verify_live_environment
    if verify_live_environment and isinstance(environment, dict):
        identity = environment.get("identity")
        if isinstance(identity, dict) and isinstance(
            identity.get("project_root"), str
        ):
            try:
                live_environment_matches = identity == _environment_identity(
                    Path(identity["project_root"])
                )
            except (OSError, ValueError):
                live_environment_matches = False
    snapshot_checks, snapshot_manifest = verify_snapshot_manifest(snapshot_root)
    try:
        source = source_manifest(snapshot_root)
    except (OSError, ValueError):
        source = None
    expected_entries = {CANONICAL_SNAPSHOT_NAME, CAMPAIGN_MANIFEST_NAME}
    try:
        root_entries_match = {
            path.name for path in campaign_root.iterdir()
        } == expected_entries
    except OSError:
        root_entries_match = False
    manifest_format = (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 1
        and manifest.get("campaign_version") == expected_campaign_version
        and isinstance(manifest.get("manifest_hash"), str)
        and SHA256_PATTERN.fullmatch(manifest["manifest_hash"]) is not None
    )
    manifest_digest_matches = False
    if manifest_format:
        payload = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_hash"
        }
        manifest_digest_matches = _json_digest(payload) == manifest["manifest_hash"]
    config_record: object = None
    if isinstance(manifest, dict):
        relative_config = manifest.get("formal_config_path")
        if isinstance(relative_config, str):
            try:
                config_record = json.loads(
                    (snapshot_root / relative_config).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                config_record = None
    checks = {
        **snapshot_checks,
        "campaign_root_file_set": root_entries_match,
        "campaign_manifest_format": manifest_format,
        "campaign_manifest_digest": manifest_digest_matches,
        "environment_receipt_present": (
            isinstance(environment, dict)
            and environment.get("schema_version") == 1
        ),
        "environment_identity_present": (
            not verify_live_environment
            or (
                isinstance(environment, dict)
                and isinstance(environment.get("identity"), dict)
            )
        ),
        "live_environment_identity": live_environment_matches,
        "candidate_identity_present": (
            isinstance(candidate_identity, dict)
            and candidate_identity.get("schema_version") == 1
            and candidate_identity.get("verification_passed") is True
        ),
        "environment_in_snapshot_manifest": (
            isinstance(snapshot_manifest, dict)
            and ENVIRONMENT_RECEIPT_RELATIVE.as_posix()
            in snapshot_manifest.get("files", {})
        ),
        "candidate_identity_in_snapshot_manifest": (
            isinstance(snapshot_manifest, dict)
            and CANDIDATE_IDENTITY_RELATIVE.as_posix()
            in snapshot_manifest.get("files", {})
        ),
        "snapshot_hash_matches_campaign": (
            isinstance(manifest, dict)
            and isinstance(snapshot_manifest, dict)
            and manifest.get("snapshot_manifest_hash")
            == snapshot_manifest.get("manifest_hash")
        ),
        "source_hash_matches_campaign": (
            isinstance(manifest, dict)
            and isinstance(source, dict)
            and manifest.get("source_manifest_hash")
            == source.get("manifest_hash")
        ),
        "environment_digest_matches": (
            isinstance(manifest, dict)
            and isinstance(environment, dict)
            and manifest.get("environment_receipt_digest")
            == _file_digest(snapshot_root / ENVIRONMENT_RECEIPT_RELATIVE)
        ),
        "candidate_identity_digest_matches": (
            isinstance(manifest, dict)
            and isinstance(candidate_identity, dict)
            and manifest.get("candidate_identity_digest")
            == _file_digest(snapshot_root / CANDIDATE_IDENTITY_RELATIVE)
        ),
        "candidate_pins_match_identity": (
            isinstance(manifest, dict)
            and isinstance(candidate_identity, dict)
            and manifest.get("candidate_pins") == candidate_identity.get("pins")
            and manifest.get("candidate_result_digest")
            == candidate_identity.get("candidate_result_digest")
        ),
        "frozen_config_present": isinstance(config_record, dict),
        "frozen_config_digest": (
            isinstance(manifest, dict)
            and isinstance(config_record, dict)
            and manifest.get("config_digest")
            == stage1_config_digest(config_record)
        ),
        "frozen_seed_registry": (
            isinstance(manifest, dict)
            and isinstance(config_record, dict)
            and manifest.get("training_seeds")
            == config_record.get("confirmation_training_seeds")
            and manifest.get("evaluation_seeds") == config_record.get("eval_seeds")
            and manifest.get("foundation_seed")
            == config_record.get("foundation_eval_seed")
        ),
        "expected_config_matches": (
            expected_config is None
            or (
                isinstance(config_record, dict)
                and config_record
                == json.loads(json.dumps(expected_config.to_dict()))
                and isinstance(manifest, dict)
                and manifest.get("validated_experiment_spec_digest")
                == validated_experiment_spec_digest(expected_config)
                and manifest.get(
                    "validated_experiment_compatibility_spec_digest"
                )
                == validated_experiment_compatibility_spec_digest(
                    expected_config
                )
            )
        ),
    }
    return checks, manifest if isinstance(manifest, dict) else None


def materialize_campaign_run(
    campaign_root: Path,
    run_dir: Path,
    expected_config: Stage1Config,
    *,
    expected_campaign_version: str = CAMPAIGN_VERSION,
    verify_live_environment: bool = False,
) -> dict[str, object]:
    checks, manifest = verify_campaign_package(
        campaign_root,
        expected_config,
        expected_campaign_version=expected_campaign_version,
        verify_live_environment=verify_live_environment,
    )
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed or manifest is None:
        raise RuntimeError(f"canonical campaign integrity failed: {failed}")
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    temporary = run_dir.with_name(f"{run_dir.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"run temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(
            campaign_root.resolve() / CANONICAL_SNAPSHOT_NAME,
            temporary / "snapshot",
        )
        snapshot_checks, snapshot_manifest = verify_snapshot_manifest(
            temporary / "snapshot"
        )
        source = source_manifest(temporary / "snapshot")
        materialization_checks = {
            **snapshot_checks,
            "snapshot_pin": (
                isinstance(snapshot_manifest, dict)
                and snapshot_manifest.get("manifest_hash")
                == manifest["snapshot_manifest_hash"]
            ),
            "source_pin": (
                source.get("manifest_hash") == manifest["source_manifest_hash"]
            ),
        }
        failed_materialization = sorted(
            name
            for name, passed in materialization_checks.items()
            if passed is not True
        )
        if failed_materialization:
            raise RuntimeError(
                "materialized run snapshot failed campaign pins: "
                f"{failed_materialization}"
            )
        receipt = {
            "schema_version": 1,
            "campaign_version": manifest["campaign_version"],
            "campaign_manifest_hash": manifest["manifest_hash"],
            "snapshot_manifest_hash": manifest["snapshot_manifest_hash"],
            "source_manifest_hash": manifest["source_manifest_hash"],
        }
        _atomic_write_json(temporary / "campaign-receipt.json", receipt)
        os.replace(temporary, run_dir)
        return receipt
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def campaign_seed_freshness(
    project_root: Path,
    config_path: Path,
    config: Stage1Config,
    *,
    excluded_roots: tuple[Path, ...] = (),
    excluded_files: tuple[Path, ...] = (),
) -> dict[str, object]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    excluded_root_paths = tuple(path.resolve() for path in excluded_roots)
    excluded_file_paths = {path.resolve() for path in excluded_files}
    registered = {
        *config.confirmation_training_seeds,
        *config.eval_seeds,
        config.foundation_eval_seed,
    }
    expected_count = (
        len(config.confirmation_training_seeds) + len(config.eval_seeds) + 1
    )
    historical: set[int] = set()
    evidence: list[str] = []
    unreadable: list[str] = []

    def excluded(path: Path) -> bool:
        resolved = path.resolve()
        if resolved == config_path or resolved in excluded_file_paths:
            return True
        for root in excluded_root_paths:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def collect(value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            if key.endswith("seeds"):
                historical.update(
                    item for item in value if type(item) is int
                )
            else:
                for child in value:
                    collect(child, key)
        elif type(value) is int and (
            key == "seed" or key.endswith("_seed")
        ):
            historical.add(value)

    paths = set((project_root / "configs").glob("*.json"))
    runs_root = project_root / "runs"
    for pattern in (
        "**/result.json",
        "**/pid.json",
        "*sequence*.json",
        "**/snapshot/configs/*.json",
    ):
        paths.update(runs_root.glob(pattern))
    for path in sorted(paths):
        if excluded(path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unreadable.append(str(path.resolve()))
            continue
        before = set(historical)
        collect(payload)
        if historical != before:
            evidence.append(str(path.resolve()))
    overlap = sorted(registered & historical)
    evidence_digest = hashlib.sha256(
        "\n".join(sorted(evidence)).encode("utf-8")
    ).hexdigest()
    return {
        "passed": (
            len(registered) == expected_count
            and not overlap
            and not unreadable
        ),
        "registered_seeds": sorted(registered),
        "registered_seed_count": len(registered),
        "expected_seed_count": expected_count,
        "historical_seed_count": len(historical),
        "overlap": overlap,
        "evidence_file_count": len(evidence),
        "evidence_path_digest": evidence_digest,
        "unreadable_evidence": unreadable,
    }
