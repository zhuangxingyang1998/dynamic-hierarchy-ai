"""Launcher/worker PID registration for Windows virtual-environment redirects."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .stage1_runtime import atomic_write_json


def register_worker_pid(
    run_dir: Path,
    launch_id: str | None,
    *,
    worker_pid: int | None = None,
    attempts: int = 100,
    delay_seconds: float = 0.1,
) -> dict[str, object]:
    actual_worker_pid = worker_pid if worker_pid is not None else os.getpid()
    if launch_id is None:
        return {
            "registered": False,
            "detail": "no launch_id supplied; foreground execution does not update pid.json",
        }
    pid_path = run_dir / "pid.json"
    for _ in range(attempts):
        try:
            record = json.loads(pid_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(delay_seconds)
            continue
        if record.get("launch_id") != launch_id:
            time.sleep(delay_seconds)
            continue
        launcher_pid = record.get("launcher_pid", record.get("pid"))
        record.update(
            {
                "pid": actual_worker_pid,
                "launcher_pid": launcher_pid,
                "worker_pid": actual_worker_pid,
                "pid_role": "worker",
                "record_state": "running",
                "worker_started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(pid_path, record)
        return {
            "registered": True,
            "launcher_pid": launcher_pid,
            "worker_pid": actual_worker_pid,
            "launch_id": launch_id,
        }
    return {
        "registered": False,
        "detail": "launcher pid record with matching launch_id was not observed within the registration window",
        "worker_pid": actual_worker_pid,
        "launch_id": launch_id,
    }
