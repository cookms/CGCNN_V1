from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from materials_gnn.training import trainer
from materials_gnn.training.early_stopping import EarlyStopping, infer_metric_mode


def test_fixed_patience_stops_after_consecutive_non_improvements() -> None:
    stopping = EarlyStopping(patience=2)

    assert stopping.step(1.0, 1) is False
    assert stopping.step(1.1, 2) is False
    assert stopping.step(1.2, 3) is True
    assert stopping.epochs_since_improvement == 2


def test_meaningful_improvement_resets_counter() -> None:
    stopping = EarlyStopping(patience=3, min_delta=0.01)

    stopping.step(1.0, 1)
    stopping.step(1.1, 2)
    assert stopping.epochs_since_improvement == 1
    stopping.step(0.98, 3)

    assert stopping.epochs_since_improvement == 0
    assert stopping.best_epoch == 3


def test_min_delta_suppresses_insignificant_decision_improvement() -> None:
    stopping = EarlyStopping(patience=3, min_delta=0.1)

    stopping.step(1.0, 1)
    stopping.step(0.95, 2)

    assert stopping.best_metric == pytest.approx(1.0)
    assert stopping.epochs_since_improvement == 1
    assert stopping.best_raw_metric == pytest.approx(0.95)
    assert stopping.best_raw_epoch == 2


def test_mode_min_tracks_lower_values() -> None:
    stopping = EarlyStopping(mode="min", patience=2)
    for epoch, value in enumerate([3.0, 2.0, 2.5], start=1):
        stopping.step(value, epoch)

    assert stopping.best_metric == pytest.approx(2.0)
    assert stopping.best_epoch == 2


def test_mode_max_tracks_higher_values_and_can_be_inferred() -> None:
    stopping = EarlyStopping(mode="max", monitor="val_r2", patience=2)
    for epoch, value in enumerate([0.1, 0.4, 0.3], start=1):
        stopping.step(value, epoch)

    assert stopping.best_raw_metric == pytest.approx(0.4)
    assert stopping.best_raw_epoch == 2
    assert infer_metric_mode("val_r2") == "max"
    assert infer_metric_mode("val_rmse") == "min"


def test_warmup_prevents_premature_stopping() -> None:
    stopping = EarlyStopping(patience=1, warmup=4)

    stopping.step(1.0, 1)
    assert stopping.step(1.1, 2) is False
    assert stopping.step(1.2, 3) is False
    assert stopping.step(1.3, 4) is True


def test_adaptive_patience_grows_without_exceeding_cap() -> None:
    stopping = EarlyStopping(
        patience=2,
        adaptive=True,
        adaptive_factor=0.5,
        max_patience=5,
    )

    for epoch in range(1, 9):
        stopping.step(20.0 - epoch, epoch)
    assert stopping.current_patience == 4

    for epoch in range(9, 21):
        stopping.step(20.0 - epoch, epoch)
    assert stopping.current_patience == 5


def test_smoothing_uses_rolling_mean_but_raw_best_stays_raw() -> None:
    stopping = EarlyStopping(patience=3, smoothing=2)

    stopping.step(10.0, 1)
    stopping.step(8.0, 2)
    stopping.step(12.0, 3)

    assert stopping.early_stopping_metric == pytest.approx(10.0)
    assert stopping.best_metric == pytest.approx(9.0)
    assert stopping.best_epoch == 2
    assert stopping.best_raw_metric == pytest.approx(8.0)
    assert stopping.best_raw_epoch == 2


def _patch_tiny_training(monkeypatch: pytest.MonkeyPatch, metrics: list[float]) -> None:
    epoch_index = 0

    def fake_train_one_epoch(model: nn.Module, *_args, **_kwargs) -> float:
        nonlocal epoch_index
        epoch_index += 1
        with torch.no_grad():
            model.weight.fill_(float(epoch_index))
        return float(epoch_index)

    values = iter(metrics)

    def fake_evaluate_model(*_args, **_kwargs):
        value = next(values)
        return {
            "loss": value,
            "mae": value,
            "rmse": value,
            "r2": -value,
            "y_true": torch.empty(0),
            "y_pred": torch.empty(0),
            "material_id": [],
        }

    monkeypatch.setattr(trainer, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(trainer, "evaluate_model", fake_evaluate_model)


def test_disabled_early_stopping_preserves_fixed_epoch_training(monkeypatch) -> None:
    _patch_tiny_training(monkeypatch, [1.0, 1.1, 1.2, 1.3])
    model = nn.Linear(1, 1, bias=False)

    history = trainer.train_model(
        model,
        [None],
        [None],
        epochs=4,
        device="cpu",
        early_stopping=None,
        verbose=False,
    )

    assert len(history) == 4
    assert "early_stopping_metric" not in history[-1]
    assert model.weight.item() == pytest.approx(4.0)


def test_training_stops_restores_raw_best_and_records_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_training(monkeypatch, [1.0, 1.1, 1.2, 1.3])
    model = nn.Linear(1, 1, bias=False)
    best_path = tmp_path / "best.pt"
    final_path = tmp_path / "final.pt"
    stopping = EarlyStopping(patience=2, smoothing=2)

    history = trainer.train_model(
        model,
        [None],
        [None],
        epochs=10,
        device="cpu",
        checkpoint_path=best_path,
        final_checkpoint_path=final_path,
        early_stopping=stopping,
        verbose=False,
    )

    assert len(history) == 3
    assert history[-1]["early_stopping_metric"] == pytest.approx(1.15)
    assert model.weight.item() == pytest.approx(1.0)

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    assert best["epoch"] == 1
    assert best["best_validation_metric"] == pytest.approx(1.0)
    assert best["early_stopping"]["smoothing"] == 2
    assert len(best["history"]) == 3
    assert best["model_state_dict"]["weight"].item() == pytest.approx(1.0)
    assert final["epoch"] == 3
    assert final["model_state_dict"]["weight"].item() == pytest.approx(3.0)
