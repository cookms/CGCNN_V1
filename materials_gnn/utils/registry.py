"""Tiny registry for research extension points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name-to-object registry.

    Registries make it easy to swap graph builders, featurizers, layers, pooling modules,
    or metrics from a config file without hard-coding every choice in a training script.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            if name in self._items:
                raise KeyError(f"{name!r} is already registered in {self.name}")
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} entry {name!r}. Available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._items)
