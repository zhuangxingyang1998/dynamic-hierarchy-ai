"""Runtime and source provenance for reproducible, inspectable CPU runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

import numpy
import torch


ROOT_SOURCE_FILES = {"pyproject.toml", "requirements-cpu.lock", "requirements-directml.lock"}


def source_manifest(project_root: Path | None = None) -> dict[str, object]:
    root = project_root or Path(__file__).resolve().parents[2]
    files: dict[str, str] = {}
    for path in intentional_source_files(root):
        relative_path = path.relative_to(root)
        files[relative_path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = "\n".join(f"{name} {digest}" for name, digest in files.items()).encode("utf-8")
    return {"algorithm": "sha256", "manifest_hash": hashlib.sha256(encoded).hexdigest(), "files": files}


def intentional_source_files(root: Path) -> list[Path]:
    """Return only authored inputs that define this project's executable state."""
    paths = [root / name for name in ROOT_SOURCE_FILES if (root / name).is_file()]
    for directory, pattern in (
        ("configs", "*.json"),
        ("campaign", "*.json"),
        ("scripts", "*.py"),
        ("scripts", "*.ps1"),
        ("src/dynamic_hierarchy", "*.py"),
        ("tests", "*.py"),
    ):
        source_directory = root / directory
        if source_directory.is_dir():
            paths.extend(path for path in source_directory.glob(pattern) if path.is_file())
    return sorted(paths)


def runtime_provenance(cpu_threads: int, backend: dict[str, str]) -> dict[str, object]:
    distributions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "libraries": {"torch": torch.__version__, "numpy": numpy.__version__},
        "backend": backend,
        "dependencies": dict(sorted(distributions.items(), key=lambda item: item[0].lower())),
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "configured_threads": cpu_threads,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "source_manifest": source_manifest(),
    }
