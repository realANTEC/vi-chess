"""Universe = strategic philosophy. An eval + (optional) move-ordering bias."""

from __future__ import annotations

from typing import Callable

import chess

from vi_chess.core.ordering import order_moves as default_order


class Universe:
    """Override `evaluate`. Optionally override `order_moves`.

    ``playing_as`` is set by the arena before each game and stays stable for the
    duration of that game. Most universes ignore it; universes whose stylistic
    preferences need to survive negamax sign-flips (e.g. chaos's complexity
    bonus) read it to attribute the preference to their own color instead of
    side-to-move. ``None`` means "unset" (legacy / tests) — eval should fall
    back to STM-relative behavior.
    """

    name: str = "unnamed"
    playing_as: chess.Color | None = None

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
