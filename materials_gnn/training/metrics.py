"""Regression metrics for materials property prediction."""

from __future__ import annotations

import torch
from torch import Tensor


def _to_1d(values: Tensor | list[float]) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float32).view(-1)


def mae(y_true: Tensor | list[float], y_pred: Tensor | list[float]) -> float:
    """Mean absolute error."""

    true = _to_1d(y_true)
    pred = _to_1d(y_pred)
    return float(torch.mean(torch.abs(pred - true)).item())


def rmse(y_true: Tensor | list[float], y_pred: Tensor | list[float]) -> float:
    """Root mean squared error."""

    true = _to_1d(y_true)
    pred = _to_1d(y_pred)
    return float(torch.sqrt(torch.mean((pred - true).pow(2))).item())


def r2_score(y_true: Tensor | list[float], y_pred: Tensor | list[float]) -> float:
    """Coefficient of determination."""

    true = _to_1d(y_true)
    pred = _to_1d(y_pred)
    ss_res = torch.sum((true - pred).pow(2))
    ss_tot = torch.sum((true - true.mean()).pow(2))
    if ss_tot <= 0:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())
