"""Report the actual Python/Torch device state and the audited VRAM caveat."""

from __future__ import annotations

import platform
import sys


AUDITED_QW_MEMORY_BYTES = 8_539_602_944


def _registry_qw_memory_values() -> list[int]:
    try:
        import winreg
    except ImportError:
        return []
    values: list[int] = []
    root_path = r"SYSTEM\CurrentControlSet\Control\Video"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_path) as root:
            index = 0
            while True:
                try:
                    adapter = winreg.EnumKey(root, index)
                    index += 1
                except OSError:
                    break
                for child in ("0000", "0001"):
                    try:
                        with winreg.OpenKey(root, adapter + "\\" + child) as key:
                            raw, _ = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")
                            if isinstance(raw, bytes):
                                raw = int.from_bytes(raw[:8], "little", signed=False)
                            values.append(int(raw))
                    except OSError:
                        continue
    except OSError:
        return []
    return values


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Audited VRAM (HardwareInformation.qwMemorySize): {AUDITED_QW_MEMORY_BYTES} bytes / {AUDITED_QW_MEMORY_BYTES / 1024**3:.2f} GiB (record as 8GB)")
    print("Legacy 32-bit MemorySize near 4GB is a truncated value and is intentionally not used.")
    observed = _registry_qw_memory_values()
    print(f"Registry qword values observed: {observed or 'unavailable'}")
    try:
        import torch
    except ImportError:
        print("Torch: unavailable (install CPU PyTorch in a Python 3.12 project virtual environment)")
        print("Device: CPU path available; AMD ROCm GPU path blocked on this Windows 10 installation.")
        return 0
    print(f"Torch: {torch.__version__}")
    print(f"CUDA/ROCm device available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device 0: {torch.cuda.get_device_name(0)}")
        print(f"Device 0 total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
    else:
        print("Device: CPU (no usable Torch GPU backend detected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
