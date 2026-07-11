"""Minimal supervised training loop for scalar materials properties."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from materials_gnn.data.transforms import TargetNormalizer
from materials_gnn.training.device import describe_device, move_to_device, resolve_device, set_float32_matmul_precision
from materials_gnn.training.metrics import mae, r2_score, rmse


class NonFiniteTrainingError(RuntimeError):
    """Raised when NaN/Inf values are detected during training or evaluation."""


def _tensor_finite_summary(name: str, value: Tensor) -> str | None:
    """Return a compact non-finite summary for a tensor, or None when finite."""

    if not value.is_floating_point():
        return None
    detached = value.detach()
    finite = torch.isfinite(detached)
    if bool(finite.all().item()):
        return None
    total = detached.numel()
    nan_count = int(torch.isnan(detached).sum().item())
    posinf_count = int(torch.isposinf(detached).sum().item())
    neginf_count = int(torch.isneginf(detached).sum().item())
    finite_values = detached[finite]
    if finite_values.numel():
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        finite_range = f", finite_range=[{min_value:.4g}, {max_value:.4g}]"
    else:
        finite_range = ""
    return (
        f"{name}: shape={tuple(detached.shape)} total={total} "
        f"nan={nan_count} +inf={posinf_count} -inf={neginf_count}{finite_range}"
    )


def _batch_materials(batch: dict[str, Any]) -> str:
    ids = batch.get("material_id", [])
    if not ids:
        return "<unknown>"
    preview = ", ".join(str(mid) for mid in ids[:8])
    if len(ids) > 8:
        preview += f", ... ({len(ids)} total)"
    return preview


def _assert_finite_batch(batch: dict[str, Any], *, context: str) -> None:
    problems: list[str] = []
    for key, value in batch.items():
        if isinstance(value, Tensor):
            summary = _tensor_finite_summary(key, value)
            if summary is not None:
                problems.append(summary)
    if problems:
        raise NonFiniteTrainingError(
            f"Non-finite tensor(s) in {context} batch for material_id(s): "
            f"{_batch_materials(batch)}\n" + "\n".join(problems)
        )


def _assert_finite_tensor(name: str, value: Tensor, *, batch: dict[str, Any], context: str) -> None:
    summary = _tensor_finite_summary(name, value)
    if summary is not None:
        raise NonFiniteTrainingError(
            f"Non-finite {name} in {context} batch for material_id(s): "
            f"{_batch_materials(batch)}\n{summary}"
        )


def _assert_finite_parameters(model: nn.Module, *, context: str) -> None:
    for name, parameter in model.named_parameters():
        summary = _tensor_finite_summary(name, parameter)
        if summary is not None:
            raise NonFiniteTrainingError(f"Non-finite model parameter after {context}:\n{summary}")


def _assert_finite_gradients(model: nn.Module, *, context: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        summary = _tensor_finite_summary(f"{name}.grad", parameter.grad)
        if summary is not None:
            raise NonFiniteTrainingError(f"Non-finite gradient during {context}:\n{summary}")


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device | str,
    *,
    non_blocking: bool = True,
) -> dict[str, Any]:
    """Move tensor fields in a graph batch to a device.

    Graph tensors are flat in this prototype, but recursive movement keeps the trainer
    ready for future nested equivariant features or multi-task labels.
    """

    return move_to_device(batch, device, non_blocking=non_blocking)


def _maybe_inverse(values: Tensor, normalizer: TargetNormalizer | None) -> Tensor:
    if normalizer is None:
        return values.detach().cpu()
    return normalizer.inverse_transform(values.detach().cpu())


def _amp_enabled(device: torch.device, mixed_precision: bool) -> bool:
    return bool(mixed_precision and device.type == "cuda")


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - compatibility with older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _loader_num_samples(loader: DataLoader) -> int | None:
    dataset = getattr(loader, "dataset", None)
    return _safe_len(dataset) if dataset is not None else None


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str = "auto",
    loss_fn: nn.Module | None = None,
    target_normalizer: TargetNormalizer | None = None,
    mixed_precision: bool = False,
    non_blocking: bool = True,
    check_finite: bool = True,
) -> dict[str, Any]:
    """Evaluate a model and return loss plus raw-scale metrics."""

    resolved_device = resolve_device(device)
    model.eval()
    model.to(resolved_device)
    loss_fn = loss_fn or nn.MSELoss()
    amp = _amp_enabled(resolved_device, mixed_precision)
    losses: list[float] = []
    all_pred: list[Tensor] = []
    all_true: list[Tensor] = []
    material_ids: list[str] = []

    for batch in loader:
        batch = move_batch_to_device(batch, resolved_device, non_blocking=non_blocking)
        if check_finite:
            _assert_finite_batch(batch, context="evaluation")
        y = batch["y"].float().view(-1)
        with _autocast_context(resolved_device, amp):
            pred = model(batch).view(-1)
            loss = loss_fn(pred, y)
        if check_finite:
            _assert_finite_tensor("prediction", pred, batch=batch, context="evaluation")
            _assert_finite_tensor("loss", loss, batch=batch, context="evaluation")
        losses.append(float(loss.item()))
        all_pred.append(_maybe_inverse(pred, target_normalizer))
        all_true.append(_maybe_inverse(y, target_normalizer))
        material_ids.extend(batch.get("material_id", []))

    if all_pred:
        y_pred = torch.cat(all_pred).view(-1)
        y_true = torch.cat(all_true).view(-1)
        mean_loss = float(sum(losses) / max(len(losses), 1))
        return {
            "loss": mean_loss,
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "material_id": material_ids,
        }

    return {
        "loss": float("nan"),
        "mae": float("nan"),
        "rmse": float("nan"),
        "r2": float("nan"),
        "y_true": torch.empty(0),
        "y_pred": torch.empty(0),
        "material_id": material_ids,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "auto",
    loss_fn: nn.Module | None = None,
    grad_clip_norm: float | None = None,
    mixed_precision: bool = False,
    scaler: Any | None = None,
    non_blocking: bool = True,
    check_finite: bool = True,
    detect_anomaly: bool = False,
) -> float:
    """Train for one epoch and return mean normalized-scale loss."""

    resolved_device = resolve_device(device)
    model.train()
    loss_fn = loss_fn or nn.MSELoss()
    amp = _amp_enabled(resolved_device, mixed_precision)
    if scaler is None:
        scaler = _make_grad_scaler(amp)
    losses: list[float] = []

    if check_finite:
        _assert_finite_parameters(model, context="epoch start")

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, resolved_device, non_blocking=non_blocking)
        if check_finite:
            _assert_finite_batch(batch, context=f"training batch {batch_idx}")
        y = batch["y"].float().view(-1)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(resolved_device, amp):
            pred = model(batch).view(-1)
            loss = loss_fn(pred, y)

        if check_finite:
            _assert_finite_tensor("prediction", pred, batch=batch, context=f"training batch {batch_idx}")
            _assert_finite_tensor("loss", loss, batch=batch, context=f"training batch {batch_idx}")

        anomaly_context = torch.autograd.detect_anomaly(check_nan=True) if detect_anomaly else nullcontext()
        with anomaly_context:
            if amp and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=check_finite)
                elif check_finite:
                    scaler.unscale_(optimizer)
                    _assert_finite_gradients(model, context=f"training batch {batch_idx}")
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=check_finite)
                elif check_finite:
                    _assert_finite_gradients(model, context=f"training batch {batch_idx}")
                optimizer.step()

        if check_finite:
            _assert_finite_parameters(model, context=f"optimizer step on training batch {batch_idx}")
        losses.append(float(loss.item()))

    return float(sum(losses) / max(len(losses), 1))


def _make_checkpoint_payload(
    model: nn.Module,
    *,
    epoch: int,
    target_normalizer: TargetNormalizer | None = None,
    val_mae: float | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a checkpoint dictionary with enough metadata for inference scripts.

    Older checkpoints only contained the model weights and target normalizer. The
    metadata field is intentionally optional and JSON-like so example training
    scripts can store model, graph, and preprocessing settings without coupling the
    trainer to a specific model class.
    """

    payload: dict[str, Any] = {
        "checkpoint_schema_version": 2,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "target_normalizer": target_normalizer.state_dict() if target_normalizer is not None else None,
    }
    if val_mae is not None:
        payload["val_mae"] = val_mae
    if checkpoint_metadata is not None:
        payload["metadata"] = dict(checkpoint_metadata)
    if history is not None:
        payload["history"] = history
    return payload


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device | str = "auto",
    loss_fn: nn.Module | None = None,
    target_normalizer: TargetNormalizer | None = None,
    checkpoint_path: str | Path | None = None,
    final_checkpoint_path: str | Path | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    grad_clip_norm: float | None = 5.0,
    mixed_precision: bool = False,
    non_blocking: bool = True,
    matmul_precision: str | None = None,
    verbose: bool = True,
    check_finite: bool = True,
    detect_anomaly: bool = False,
) -> list[dict[str, Any]]:
    """Fit a model with AdamW and optional validation checkpointing.

    Targets are usually normalized for optimization. Validation metrics are reported on the
    original target scale when ``target_normalizer`` is supplied. ``device='auto'`` uses
    CUDA when available and otherwise falls back to CPU. Mixed precision is enabled only
    on CUDA because CPU autocast is less relevant for this prototype. When provided,
    ``checkpoint_metadata`` is copied into both best and final checkpoints so inference
    can reconstruct the original model and graph settings.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    resolved_device = resolve_device(device)
    set_float32_matmul_precision(matmul_precision)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn or nn.MSELoss()
    scaler = _make_grad_scaler(_amp_enabled(resolved_device, mixed_precision))
    history: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    train_samples = _loader_num_samples(train_loader)
    train_batches = _safe_len(train_loader)
    training_start = time.perf_counter()

    if verbose:
        print(f"training device: {describe_device(resolved_device)}")

    for epoch in range(1, epochs + 1):
        _sync_if_cuda(resolved_device)
        epoch_start = time.perf_counter()
        train_start = epoch_start
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=resolved_device,
            loss_fn=loss_fn,
            grad_clip_norm=grad_clip_norm,
            mixed_precision=mixed_precision,
            scaler=scaler,
            non_blocking=non_blocking,
            check_finite=check_finite,
            detect_anomaly=detect_anomaly,
        )
        _sync_if_cuda(resolved_device)
        train_seconds = time.perf_counter() - train_start
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
        }
        if train_samples is not None:
            record["train_samples"] = train_samples
            record["train_samples_per_second"] = train_samples / train_seconds if train_seconds > 0 else float("inf")
        if train_batches is not None:
            record["train_batches"] = train_batches
            record["train_batches_per_second"] = train_batches / train_seconds if train_seconds > 0 else float("inf")

        if val_loader is not None:
            _sync_if_cuda(resolved_device)
            val_start = time.perf_counter()
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=resolved_device,
                loss_fn=loss_fn,
                target_normalizer=target_normalizer,
                mixed_precision=mixed_precision,
                non_blocking=non_blocking,
                check_finite=check_finite,
            )
            _sync_if_cuda(resolved_device)
            record["val_seconds"] = time.perf_counter() - val_start
            record.update({f"val_{key}": value for key, value in val_metrics.items() if key not in {"y_true", "y_pred", "material_id"}})
            _sync_if_cuda(resolved_device)
            record["epoch_seconds"] = time.perf_counter() - epoch_start
            record["elapsed_seconds"] = time.perf_counter() - training_start

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = float(val_metrics["mae"])
                if checkpoint_path is not None:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        _make_checkpoint_payload(
                            model,
                            epoch=epoch,
                            target_normalizer=target_normalizer,
                            val_mae=best_val_mae,
                            checkpoint_metadata=checkpoint_metadata,
                            history=history + [record],
                        ),
                        checkpoint_path,
                    )

        if "epoch_seconds" not in record:
            _sync_if_cuda(resolved_device)
            record["epoch_seconds"] = time.perf_counter() - epoch_start
            record["elapsed_seconds"] = time.perf_counter() - training_start
        history.append(record)
        if verbose:
            msg = f"epoch={epoch:03d} train_loss={train_loss:.5f}"
            if val_loader is not None:
                msg += (
                    f" val_mae={record['val_mae']:.5f}"
                    f" val_rmse={record['val_rmse']:.5f}"
                    f" val_r2={record['val_r2']:.4f}"
                )
            msg += f" time={record['epoch_seconds']:.2f}s"
            if "train_samples_per_second" in record:
                msg += f" train_samples/s={record['train_samples_per_second']:.2f}"
            print(msg)

    if final_checkpoint_path is not None:
        Path(final_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            _make_checkpoint_payload(
                model,
                epoch=history[-1]["epoch"] if history else 0,
                target_normalizer=target_normalizer,
                checkpoint_metadata=checkpoint_metadata,
                history=history,
            ),
            final_checkpoint_path,
        )

    return history
