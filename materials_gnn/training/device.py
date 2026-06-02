"""Device and runtime helpers for CPU/CUDA training."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU.

    Keeping this logic in one place makes examples and future launchers consistent. The
    package remains CPU-first: CUDA is used opportunistically when PyTorch reports that it
    is available.
    """

    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def is_cuda_device(device: str | torch.device) -> bool:
    return resolve_device(device).type == "cuda"


def dataloader_device_kwargs(device: str | torch.device, *, num_workers: int = 0) -> dict[str, Any]:
    """Return DataLoader kwargs that help host-to-GPU transfer.

    ``pin_memory=True`` can make CPU-to-CUDA tensor copies faster. Persistent workers are
    useful only when workers are enabled.
    """

    cuda = is_cuda_device(device)
    kwargs: dict[str, Any] = {"pin_memory": cuda}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def move_to_device(value: Any, device: str | torch.device, *, non_blocking: bool = True) -> Any:
    """Recursively move tensors in a batch-like object to the selected device."""

    resolved = resolve_device(device)
    if isinstance(value, Tensor):
        return value.to(resolved, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move_to_device(item, resolved, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, resolved, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, resolved, non_blocking=non_blocking) for item in value)
    return value


def describe_device(device: str | torch.device = "auto") -> str:
    """Return a human-readable device description for logs."""

    resolved = resolve_device(device)
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        return f"cuda:{index} ({name})"
    return str(resolved)


def set_float32_matmul_precision(precision: str | None) -> None:
    """Optionally set PyTorch float32 matmul precision.

    Accepted values are those accepted by ``torch.set_float32_matmul_precision`` such as
    ``highest``, ``high``, and ``medium``. ``None`` leaves the PyTorch default untouched.
    """

    if precision is not None:
        torch.set_float32_matmul_precision(precision)
