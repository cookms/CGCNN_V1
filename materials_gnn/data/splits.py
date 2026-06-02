"""Dataset splitting helpers."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset, Subset


def train_val_test_split_indices(
    n: int,
    *,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Create reproducible train/validation/test index splits."""

    if n < 0:
        raise ValueError("n must be non-negative")
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(round(train_size * n))
    n_val = int(round(val_size * n))
    # Ensure all examples are assigned despite rounding.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train : n_train + n_val].tolist()
    test_idx = indices[n_train + n_val :].tolist()
    return train_idx, val_idx, test_idx


def split_dataset(
    dataset: Dataset,
    *,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[Subset, Subset, Subset, tuple[list[int], list[int], list[int]]]:
    """Split a PyTorch dataset and return subsets plus the raw index lists."""

    train_idx, val_idx, test_idx = train_val_test_split_indices(
        len(dataset),
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        shuffle=shuffle,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx), (
        train_idx,
        val_idx,
        test_idx,
    )
