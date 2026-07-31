"""Frozen Stage 1 source snapshot creation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "requirements-cpu.lock",
    "requirements-directml.lock",
}
EXPLICIT_DOCS: set[str] = set()
SOURCE_PATTERNS = (
    ("configs", "*.json"),
    ("campaign", "*.json"),
    ("scripts", "*.py"),
    ("scripts", "*.ps1"),
    ("src/dynamic_hierarchy", "*.py"),
    ("tests", "*.py"),
    ("docs", "*.md"),
)


def snapshot_sources(project_root: Path) -> list[Path]:
    paths = [project_root / name for name in ROOT_FILES if (project_root / name).is_file()]
    paths.extend(
        project_root / name
        for name in EXPLICIT_DOCS
        if (project_root / name).is_file()
    )
    for directory, pattern in SOURCE_PATTERNS:
        source_directory = project_root / directory
        if source_directory.is_dir():
            paths.extend(path for path in source_directory.glob(pattern) if path.is_file())
    return sorted(set(paths))


def write_snapshot_manifest(snapshot_root: Path) -> dict[str, object]:
    files: dict[str, str] = {}
    for source in snapshot_sources(snapshot_root):
        relative = source.relative_to(snapshot_root)
        files[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    encoded = "\n".join(
        f"{name} {digest}" for name, digest in sorted(files.items())
    ).encode("utf-8")
    manifest = {
        "algorithm": "sha256",
        "manifest_hash": hashlib.sha256(encoded).hexdigest(),
        "files": dict(sorted(files.items())),
    }
    (snapshot_root / "snapshot-manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def create_snapshot(project_root: Path, snapshot_root: Path) -> dict[str, object]:
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for source in snapshot_sources(project_root):
        relative = source.relative_to(project_root)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return write_snapshot_manifest(snapshot_root)
