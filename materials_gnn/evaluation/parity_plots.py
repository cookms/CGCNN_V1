"""Plotting and prediction export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch import Tensor


def _to_tensor(values: Tensor | Iterable[float]) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float32).view(-1)


def parity_plot(
    y_true: Tensor | Iterable[float],
    y_pred: Tensor | Iterable[float],
    *,
    output_path: str | Path | None = None,
    title: str = "Parity plot",
    show: bool = False,
) -> None:
    """Create a predicted-vs-true parity plot."""

    import matplotlib.pyplot as plt

    true = _to_tensor(y_true).numpy()
    pred = _to_tensor(y_pred).numpy()
    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True target")
    ax.set_ylabel("Predicted target")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def residual_plot(
    y_true: Tensor | Iterable[float],
    y_pred: Tensor | Iterable[float],
    *,
    output_path: str | Path | None = None,
    title: str = "Residual plot",
    show: bool = False,
) -> None:
    """Create residual-vs-prediction plot."""

    import matplotlib.pyplot as plt

    true = _to_tensor(y_true).numpy()
    pred = _to_tensor(y_pred).numpy()
    residual = pred - true

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, residual, alpha=0.7)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Prediction residual")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def export_predictions_csv(
    material_ids: list[str],
    y_true: Tensor | Iterable[float],
    y_pred: Tensor | Iterable[float],
    output_path: str | Path,
) -> Path:
    """Write predictions and residuals to a CSV file."""

    true = _to_tensor(y_true).numpy()
    pred = _to_tensor(y_pred).numpy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "material_id": material_ids,
            "y_true": true,
            "y_pred": pred,
            "residual": pred - true,
        }
    ).to_csv(output, index=False)
    return output
