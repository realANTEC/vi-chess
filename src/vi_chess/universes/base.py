"""Universe = strategic philosophy. An eval + (optional) move-ordering bias."""

from __future__ import annotations

from typing import Callable

import chess

from vi_chess.core.ordering import order_moves as default_order


class Universe:
    """Override `evaluate`. Optionally override `order_moves`."""

    name: str = "unnamed"

    def evaluate(self, board: chess.Board) -> int:
        """Centipawn score from side-to-move's perspective. Positive = good for STM."""
        raise NotImplementedError

    def order_moves(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        return default_order(board, moves)


_REGISTRY: dict[str, Callable[[], Universe]] = {}


def register(name: str) -> Callable[[type[Universe]], type[Universe]]:
    """Class decorator. Registers a universe under the given name."""

    def decorator(cls: type[Universe]) -> type[Universe]:
        if name in _REGISTRY:
            raise ValueError(f"Universe {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get(name: str) -> Universe:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown universe: {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def names() -> list[str]:
    return sorted(_REGISTRY)
