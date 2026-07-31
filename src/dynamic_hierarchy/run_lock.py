"""Per-run Windows mutex that follows the actual worker lifetime."""

from __future__ import annotations

import ctypes
import hashlib
import os
from ctypes import wintypes
from pathlib import Path


WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102


class PerRunMutex:
    def __init__(self, run_dir: Path) -> None:
        digest = hashlib.sha256(str(run_dir.resolve()).lower().encode("utf-8")).hexdigest()
        self.name = f"Local\\DynamicHierarchyStage1-{digest}"
        self._handle: int | None = None
        self._owned = False

    def acquire(self) -> None:
        if os.name != "nt":
            raise RuntimeError("PerRunMutex currently supports only the Windows Stage 1 runner")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateMutexW(None, False, self.name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        result = kernel32.WaitForSingleObject(handle, 0)
        if result in (WAIT_OBJECT_0, WAIT_ABANDONED):
            self._handle = int(handle)
            self._owned = True
            return
        kernel32.CloseHandle(handle)
        if result == WAIT_TIMEOUT:
            raise RuntimeError(f"another Stage 1 worker holds run mutex {self.name}")
        raise RuntimeError(f"unexpected WaitForSingleObject result: {result}")

    def release(self) -> None:
        if self._handle is None:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = wintypes.HANDLE(self._handle)
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        if self._owned:
            kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)
        self._handle = None
        self._owned = False

    def __enter__(self) -> "PerRunMutex":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "Windows named mutex",
            "name": self.name,
            "owned": self._owned,
        }
