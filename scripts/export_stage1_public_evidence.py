"""Export the completed Stage 1 campaign without local machine paths."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "evidence" / "stage1-formal-v4"
SEQUENCE_PATH = RUNS_ROOT / "stage1-literal-formal-v4-sequence.json"
AGGREGATE_PATH = RUNS_ROOT / "stage1-literal-formal-v4-confirmation.json"
CAMPAIGN_MANIFEST_PATH = (
    RUNS_ROOT
    / "stage1-literal-formal-v4-campaign"
    / "campaign-manifest.json"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def project_relative_path(value: str) -> str | None:
    if not value or ":\\" not in value:
        return None
    try:
        relative = PureWindowsPath(value).relative_to(
            PureWindowsPath(str(PROJECT_ROOT))
        )
    except ValueError:
        return None
    return relative.as_posix()


def sanitize_project_paths(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        transformed: dict[str, Any] = {}
        replacements = 0
        for key, child in value.items():
            sanitized, child_replacements = sanitize_project_paths(child)
            transformed[key] = sanitized
            replacements += child_replacements
        return transformed, replacements
    if isinstance(value, list):
        transformed_list: list[Any] = []
        replacements = 0
        for child in value:
            sanitized, child_replacements = sanitize_project_paths(child)
            transformed_list.append(sanitized)
            replacements += child_replacements
        return transformed_list, replacements
    if isinstance(value, str):
        relative = project_relative_path(value)
        if relative is not None:
            return relative, 1
    return value, 0


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    if sequence.get("state") != "completed":
        raise RuntimeError("campaign v4 must be completed before export")
    runs = sequence.get("runs")
    if not isinstance(runs, dict) or len(runs) != 8:
        raise RuntimeError("campaign v4 must contain exactly eight runs")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_root = OUTPUT_ROOT / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    expected_result_names = {f"{seed}.json" for seed in runs}
    unexpected_results = sorted(
        path.name
        for path in result_root.glob("*.json")
        if path.name not in expected_result_names
    )
    if unexpected_results:
        raise RuntimeError(
            f"unexpected published result files: {unexpected_results}"
        )

    aggregate_output = OUTPUT_ROOT / "aggregate.json"
    manifest_output = OUTPUT_ROOT / "campaign-manifest.json"
    shutil.copyfile(AGGREGATE_PATH, aggregate_output)
    shutil.copyfile(CAMPAIGN_MANIFEST_PATH, manifest_output)

    result_entries: list[dict[str, object]] = []
    source_hashes_before: dict[Path, str] = {}
    for seed_text, entry in sorted(runs.items(), key=lambda item: int(item[0])):
        if not isinstance(entry, dict) or entry.get("status") != "verified":
            raise RuntimeError(f"seed {seed_text} is not verified")
        run_name = PureWindowsPath(str(entry["run_dir"])).name
        source_path = RUNS_ROOT / run_name / "result.json"
        source_hash = file_sha256(source_path)
        source_hashes_before[source_path] = source_hash

        payload = json.loads(source_path.read_text(encoding="utf-8"))
        sanitized, replacements = sanitize_project_paths(payload)
        if replacements < 1:
            raise RuntimeError(
                f"seed {seed_text} contained no project-root path to sanitize"
            )
        published_path = result_root / f"{seed_text}.json"
        write_json(published_path, sanitized)
        published_text = published_path.read_text(encoding="utf-8")
        if str(PROJECT_ROOT).casefold() in published_text.casefold():
            raise RuntimeError(
                f"seed {seed_text} still contains the local project root"
            )
        result_entries.append(
            {
                "training_seed": int(seed_text),
                "source_path": f"runs/{run_name}/result.json",
                "source_sha256": source_hash,
                "published_path": f"results/{seed_text}.json",
                "published_sha256": file_sha256(published_path),
                "project_path_replacements": replacements,
            }
        )

    changed_sources = sorted(
        str(path)
        for path, digest in source_hashes_before.items()
        if file_sha256(path) != digest
    )
    if changed_sources:
        raise RuntimeError(f"source results changed during export: {changed_sources}")

    index = {
        "schema_version": 1,
        "campaign": "literal-formal-confirmation-v4",
        "decision": "formal_confirmation_passed",
        "stage2_unblocked": True,
        "publication_transform": (
            "absolute paths under the local project root are converted to "
            "repository-relative POSIX paths; scientific fields are unchanged"
        ),
        "aggregate": {
            "path": "aggregate.json",
            "sha256": file_sha256(aggregate_output),
            "byte_identical_to_source": (
                aggregate_output.read_bytes() == AGGREGATE_PATH.read_bytes()
            ),
        },
        "campaign_manifest": {
            "path": "campaign-manifest.json",
            "sha256": file_sha256(manifest_output),
            "byte_identical_to_source": (
                manifest_output.read_bytes()
                == CAMPAIGN_MANIFEST_PATH.read_bytes()
            ),
        },
        "results": result_entries,
    }
    write_json(OUTPUT_ROOT / "publication-index.json", index)
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
