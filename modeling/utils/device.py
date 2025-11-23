#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
import torch


@dataclass
class DeviceContext:
    device: torch.device
    is_cuda: bool
    use_amp: bool
    pin_memory: bool
    non_blocking: bool


def _env_disables_cuda() -> bool:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    # Treat explicit empty string or -1 as CUDA disabled
    return v == "" or v == "-1"


def _safe_cuda_available() -> bool:
    """Return True only if CUDA can be initialized safely.

    This guards against environments with outdated or broken NVIDIA drivers where
    importing or probing CUDA can cause crashes.
    """
    try:
        if _env_disables_cuda():
            return False
        # torch.cuda.is_available itself can raise in some broken setups
        if not torch.cuda.is_available():
            return False
        # Access a lightweight property to validate runtime/driver health
        _ = torch.cuda.get_device_properties(0)
        return True
    except Exception:
        return False


essentially_cuda_aliases = {"cuda", "gpu"}


def get_device(prefer: str = "auto", cuda_device: int | None = None) -> DeviceContext:
    """Resolve a safe device to use.

    prefer: one of {"auto", "cpu", "cuda"}
    cuda_device: optional CUDA device index
    """
    prefer = (prefer or "auto").lower()

    # Force CPU path
    if prefer == "cpu":
        dev = torch.device("cpu")
        return DeviceContext(dev, False, False, False, False)

    # Try CUDA if requested or in auto mode
    if prefer in essentially_cuda_aliases or prefer == "auto":
        if _safe_cuda_available():
            try:
                if cuda_device is not None:
                    torch.cuda.set_device(cuda_device)
                    dev = torch.device(f"cuda:{cuda_device}")
                else:
                    dev = torch.device("cuda")
                # Light validation that the device is usable
                _ = torch.cuda.current_device()
                return DeviceContext(dev, True, True, True, True)
            except Exception:
                # Fall back to CPU on any CUDA error
                pass

    # Fallback: CPU
    dev = torch.device("cpu")
    return DeviceContext(dev, False, False, False, False)
