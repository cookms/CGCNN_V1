from __future__ import annotations

import torch

from materials_gnn.training import dataloader_device_kwargs, move_batch_to_device, resolve_device


def test_resolve_device_auto_returns_torch_device() -> None:
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda"}


def test_move_batch_to_device_moves_nested_tensors() -> None:
    batch = {
        "x": torch.ones(2),
        "nested": {"y": torch.zeros(1)},
        "ids": ["a", "b"],
    }
    moved = move_batch_to_device(batch, "cpu")
    assert moved["x"].device.type == "cpu"
    assert moved["nested"]["y"].device.type == "cpu"
    assert moved["ids"] == ["a", "b"]


def test_dataloader_device_kwargs_cpu() -> None:
    kwargs = dataloader_device_kwargs("cpu", num_workers=0)
    assert kwargs == {"pin_memory": False}
