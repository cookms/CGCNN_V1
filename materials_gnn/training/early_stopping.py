"""Configurable early stopping for validation metrics.

The controller deliberately has no knowledge of models, optimizers, or checkpoints.
Training loops feed it one raw validation metric per epoch and use its public state
to decide when to stop and which raw-metric checkpoint to retain.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from typing import Any, Literal


MetricMode = Literal["min", "max"]

_MINIMIZED_METRICS = {"loss", "mae", "mse", "rmse"}
_MAXIMIZED_METRICS = {"accuracy", "auc", "f1", "precision", "r2", "recall"}


def infer_metric_mode(monitor: str) -> MetricMode:
    """Infer whether a familiar validation metric is minimized or maximized."""

    metric = monitor.lower()
    if metric.startswith("val_"):
        metric = metric[4:]
    if metric in _MINIMIZED_METRICS:
        return "min"
    if metric in _MAXIMIZED_METRICS:
        return "max"
    raise ValueError(
        f"Cannot infer early-stopping mode for {monitor!r}; "
        "set --early-stopping-mode to 'min' or 'max'."
    )


class EarlyStopping:
    """Track a validation metric and detect a meaningful plateau.

    ``best_metric`` and ``best_epoch`` describe the (possibly smoothed) decision
    metric. ``best_raw_metric`` and ``best_raw_epoch`` independently track the
    strict raw optimum so checkpoint selection is never blurred by smoothing.

    In adaptive mode, each meaningful decision-metric improvement may grow the
    patience to ``ceil(adaptive_factor * epoch)``. Patience never shrinks below
    its initial value and never exceeds ``max_patience``.
    """

    def __init__(
        self,
        *,
        monitor: str = "val_mae",
        mode: MetricMode = "min",
        patience: int = 25,
        min_delta: float = 1e-4,
        warmup: int = 0,
        adaptive: bool = False,
        adaptive_factor: float = 0.25,
        max_patience: int | None = None,
        smoothing: int = 1,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if adaptive_factor <= 0:
            raise ValueError("adaptive_factor must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        resolved_max_patience = patience if max_patience is None else max_patience
        if resolved_max_patience < patience:
            raise ValueError("max_patience must be greater than or equal to patience")

        self.monitor = monitor
        self.mode = mode
        self.initial_patience = patience
        self.min_delta = float(min_delta)
        self.warmup = warmup
        self.adaptive = adaptive
        self.adaptive_factor = float(adaptive_factor)
        self.max_patience = resolved_max_patience
        self.smoothing = smoothing

        self.current_patience = patience
        self.epochs_since_improvement = 0
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.best_raw_metric: float | None = None
        self.best_raw_epoch: int | None = None
        self.early_stopping_metric: float | None = None
        self.raw_improved = False
        self.meaningful_improvement = False
        self.should_stop = False
        self._recent_metrics: deque[float] = deque(maxlen=smoothing)

    def _is_better(self, current: float, best: float | None, min_delta: float) -> bool:
        if best is None:
            return True
        if self.mode == "min":
            return current < best - min_delta
        return current > best + min_delta

    def step(self, raw_metric: float, epoch: int) -> bool:
        """Consume one epoch's raw metric and return whether training should stop."""

        raw_metric = float(raw_metric)
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        if not math.isfinite(raw_metric):
            raise ValueError(f"{self.monitor} must be finite, got {raw_metric}")

        self._recent_metrics.append(raw_metric)
        decision_metric = sum(self._recent_metrics) / len(self._recent_metrics)
        self.early_stopping_metric = decision_metric

        self.raw_improved = self._is_better(raw_metric, self.best_raw_metric, 0.0)
        if self.raw_improved:
            self.best_raw_metric = raw_metric
            self.best_raw_epoch = epoch

        self.meaningful_improvement = self._is_better(
            decision_metric, self.best_metric, self.min_delta
        )
        if self.meaningful_improvement:
            self.best_metric = decision_metric
            self.best_epoch = epoch
            self.epochs_since_improvement = 0
            if self.adaptive:
                adaptive_patience = math.ceil(self.adaptive_factor * epoch)
                self.current_patience = min(
                    self.max_patience,
                    max(self.current_patience, self.initial_patience, adaptive_patience),
                )
        else:
            self.epochs_since_improvement += 1

        self.should_stop = bool(
            epoch >= self.warmup
            and self.epochs_since_improvement >= self.current_patience
        )
        return self.should_stop

    def history_fields(self) -> dict[str, Any]:
        """Return JSON-compatible fields for the current epoch history record."""

        return {
            "early_stopping_metric": self.early_stopping_metric,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_raw_metric": self.best_raw_metric,
            "best_raw_epoch": self.best_raw_epoch,
            "epochs_since_improvement": self.epochs_since_improvement,
            "current_patience": self.current_patience,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return configuration and final state suitable for a checkpoint."""

        return {
            "enabled": True,
            "monitor": self.monitor,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "initial_patience": self.initial_patience,
            "current_patience": self.current_patience,
            "max_patience": self.max_patience,
            "adaptive": self.adaptive,
            "adaptive_factor": self.adaptive_factor,
            "warmup": self.warmup,
            "smoothing": self.smoothing,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_validation_metric": self.best_raw_metric,
            "best_validation_epoch": self.best_raw_epoch,
            "epochs_since_improvement": self.epochs_since_improvement,
            "should_stop": self.should_stop,
        }


def add_early_stopping_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared early-stopping CLI contract to a training parser."""

    group = parser.add_argument_group("adaptive early stopping")
    group.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable validation-based early stopping",
    )
    group.add_argument(
        "--early-stopping-monitor",
        default="val_mae",
        help="Validation history field to monitor",
    )
    group.add_argument(
        "--early-stopping-mode",
        choices=["auto", "min", "max"],
        default="auto",
        help="Metric direction; auto infers common metrics such as MAE, RMSE, and R2",
    )
    group.add_argument("--early-stopping-patience", type=int, default=25)
    group.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    group.add_argument("--early-stopping-warmup", type=int, default=0)
    group.add_argument("--early-stopping-adaptive", action="store_true")
    group.add_argument("--early-stopping-factor", type=float, default=0.25)
    group.add_argument("--early-stopping-max-patience", type=int, default=75)
    group.add_argument("--early-stopping-smoothing", type=int, default=1)


def early_stopping_from_args(args: argparse.Namespace) -> EarlyStopping | None:
    """Build an enabled controller from parsed shared CLI arguments."""

    if not args.early_stopping:
        return None
    mode = (
        infer_metric_mode(args.early_stopping_monitor)
        if args.early_stopping_mode == "auto"
        else args.early_stopping_mode
    )
    return EarlyStopping(
        monitor=args.early_stopping_monitor,
        mode=mode,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        warmup=args.early_stopping_warmup,
        adaptive=args.early_stopping_adaptive,
        adaptive_factor=args.early_stopping_factor,
        max_patience=args.early_stopping_max_patience,
        smoothing=args.early_stopping_smoothing,
    )
