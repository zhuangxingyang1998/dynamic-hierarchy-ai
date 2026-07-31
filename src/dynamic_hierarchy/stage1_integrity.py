"""Fail-closed integrity and seed-registry checks for formal Stage 1."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .provenance import source_manifest
from .snapshot import snapshot_sources


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_manifest_hash(files: dict[str, str]) -> str:
    encoded = "\n".join(
        f"{name} {digest}" for name, digest in sorted(files.items())
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_format_valid(manifest: object) -> bool:
    if not isinstance(manifest, dict):
        return False
    if manifest.get("algorithm") != "sha256":
        return False
    manifest_hash = manifest.get("manifest_hash")
    files = manifest.get("files")
    if (
        not isinstance(manifest_hash, str)
        or SHA256_PATTERN.fullmatch(manifest_hash) is None
        or not isinstance(files, dict)
        or not files
    ):
        return False
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            return False
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            return False
    return _canonical_manifest_hash(files) == manifest_hash


def verify_snapshot_manifest(
    snapshot_root: Path,
    declared_manifest: object | None = None,
) -> tuple[dict[str, bool], dict[str, Any] | None]:
    snapshot_root = snapshot_root.resolve()
    manifest_path = snapshot_root / "snapshot-manifest.json"
    try:
        disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        disk_manifest = None
    format_valid = _manifest_format_valid(disk_manifest)
    declared_matches_disk = (
        declared_manifest == disk_manifest if declared_manifest is not None else True
    )
    files_match = False
    file_set_matches = False
    if format_valid and isinstance(disk_manifest, dict):
        declared_files = disk_manifest["files"]
        files_match = True
        for name, expected_digest in declared_files.items():
            candidate = (snapshot_root / Path(*PurePosixPath(name).parts)).resolve()
            try:
                candidate.relative_to(snapshot_root)
            except ValueError:
                files_match = False
                break
            if (
                not candidate.is_file()
                or hashlib.sha256(candidate.read_bytes()).hexdigest()
                != expected_digest
            ):
                files_match = False
                break
        try:
            observed_files = {
                path.relative_to(snapshot_root).as_posix()
                for path in snapshot_sources(snapshot_root)
            }
            file_set_matches = observed_files == set(declared_files)
        except (OSError, ValueError):
            file_set_matches = False
    checks = {
        "snapshot_manifest_present_and_formatted": format_valid,
        "snapshot_manifest_declared_matches_disk": declared_matches_disk,
        "snapshot_manifest_file_hashes_match": files_match,
        "snapshot_manifest_file_set_matches": file_set_matches,
    }
    return checks, disk_manifest if isinstance(disk_manifest, dict) else None


def verify_result_manifests(
    result: dict[str, Any],
    run_dir: Path,
    *,
    require_embedded_snapshot_manifest: bool = True,
) -> dict[str, bool]:
    snapshot_root = run_dir.resolve() / "snapshot"
    embedded_snapshot = result.get("snapshot_manifest")
    snapshot_checks, disk_snapshot = verify_snapshot_manifest(
        snapshot_root,
        embedded_snapshot if require_embedded_snapshot_manifest else None,
    )
    declared_source = result.get("manifest")
    source_format_valid = _manifest_format_valid(declared_source)
    try:
        recomputed_source = source_manifest(snapshot_root)
    except (OSError, ValueError):
        recomputed_source = None
    snapshot_hash = result.get("snapshot_manifest_hash")
    return {
        **snapshot_checks,
        "embedded_snapshot_manifest_required": (
            isinstance(embedded_snapshot, dict)
            if require_embedded_snapshot_manifest
            else True
        ),
        "snapshot_manifest_hash_declared": (
            isinstance(snapshot_hash, str)
            and SHA256_PATTERN.fullmatch(snapshot_hash) is not None
        ),
        "snapshot_manifest_hash_matches_disk": (
            isinstance(disk_snapshot, dict)
            and snapshot_hash == disk_snapshot.get("manifest_hash")
        ),
        "source_manifest_present_and_formatted": source_format_valid,
        "source_manifest_recomputed_from_snapshot": (
            source_format_valid and declared_source == recomputed_source
        ),
    }


def _collect_seed_values(value: object, observed: set[int], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _collect_seed_values(child, observed, str(child_key))
    elif isinstance(value, list):
        if key.endswith("seeds"):
            observed.update(item for item in value if type(item) is int)
        else:
            for child in value:
                _collect_seed_values(child, observed, key)
    elif type(value) is int and (key == "seed" or key.endswith("_seed")):
        observed.add(value)


def formal_seed_freshness(
    project_root: Path,
    formal_config_path: Path,
    formal_config: object,
) -> dict[str, object]:
    root = project_root.resolve()
    current_path = formal_config_path.resolve()
    registered = {
        *getattr(formal_config, "confirmation_training_seeds"),
        *getattr(formal_config, "eval_seeds"),
        getattr(formal_config, "foundation_eval_seed"),
    }
    expected_count = (
        len(getattr(formal_config, "confirmation_training_seeds"))
        + len(getattr(formal_config, "eval_seeds"))
        + 1
    )
    historical: set[int] = set()
    evidence_files: list[str] = []

    def inspect(path: Path, *, require_nonformal: bool) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        config = payload.get("config", payload) if isinstance(payload, dict) else {}
        if (
            require_nonformal
            and isinstance(config, dict)
            and config.get("formal_evaluation") is True
        ):
            return
        before = len(historical)
        _collect_seed_values(config, historical)
        if len(historical) != before:
            evidence_files.append(str(path.resolve()))

    for path in sorted((root / "configs").glob("*.json")):
        if path.resolve() != current_path:
            inspect(path, require_nonformal=True)
    for result_path in sorted((root / "runs").glob("*/result.json")):
        inspect(result_path, require_nonformal=True)
    for pid_path in sorted((root / "runs").glob("*/pid.json")):
        try:
            record = json.loads(pid_path.read_text(encoding="utf-8"))
            selected_config = Path(str(record.get("config", ""))).resolve()
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if selected_config.is_file():
            inspect(selected_config, require_nonformal=True)

    overlap = sorted(registered & historical)
    evidence_digest = hashlib.sha256(
        "\n".join(sorted(evidence_files)).encode("utf-8")
    ).hexdigest()
    return {
        "passed": len(registered) == expected_count and not overlap,
        "registered_seeds": sorted(registered),
        "registered_seed_count": len(registered),
        "expected_seed_count": expected_count,
        "historical_nonformal_seed_count": len(historical),
        "overlap": overlap,
        "evidence_file_count": len(evidence_files),
        "evidence_path_digest": evidence_digest,
    }
