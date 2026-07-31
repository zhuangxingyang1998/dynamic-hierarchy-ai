"""Check all repository text sources, including untracked files, for trailing whitespace."""

from __future__ import annotations

from pathlib import Path


TEXT_NAMES = {".gitignore"}
TEXT_SUFFIXES = {".json", ".lock", ".md", ".ps1", ".py", ".toml"}
EXCLUDED_PARTS = {".git", ".venv", ".venv-directml", "__pycache__", "data", "runs"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.name not in TEXT_NAMES and path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        checked += 1
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.rstrip(" \t") != line:
                violations.append(f"{path.relative_to(root)}:{line_number}")
    if violations:
        print("Trailing whitespace found:\n" + "\n".join(violations))
        return 1
    print(f"Checked {checked} project text files: no trailing whitespace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
