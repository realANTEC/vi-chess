"""Player abstraction: anything that maps (board, max_nodes) -> (move, score).

Wraps Universe (single-eval) and the two Multiverse architectures behind a
uniform interface so the arena can be polymorphic. Players return both the
chosen move and a self-reported eval (cp, from the mover's perspective) so
the arena can apply resign and adjudication rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import chess

from vi_chess.core.search import search
from vi_chess.multiverse.independent import IndependentMultiverse
from vi_chess.multiverse.shared import SharedTreeMultiverse
from vi_chess.universes.base import Universe


@dataclass
class PlayResult:
    move: chess.Move
    score: int  # centipawns from the mover's perspective


@runtime_checkable
class Player(Protocol):
    name: str

    def play(self, board: chess.Board, max_nodes: int) -> PlayResult: ...

    def set_playing_as(self, color: chess.Color) -> None: ...


@dataclass
class SingleUniversePlayer:
    universe: Universe
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or f"solo:{self.universe.name}"

    def set_playing_as(self, color: chess.Color) -> None:
        self.universe.playing_as = color

    def play(self, board: chess.Board, max_nodes: int) -> PlayResult:
        result = search(self.universe, board, max_nodes)
        if result.best_move is None:
            raise RuntimeError(f"{self.name} produced no move on {board.fen()}")
        return PlayResult(move=result.best_move, score=result.score)


@dataclass
class MultiversePlayer:
    multiverse: SharedTreeMultiverse | IndependentMultiverse
    label: str

    @property
    def name(self) -> str:
        return self.label

    def set_playing_as(self, color: chess.Color) -> None:
        # Color-aware universes (e.g. chaos) need their own color set; the others ignore it.
        for u in self.multiverse.universes:
            u.playing_as = color

    def play(self, board: chess.Board, max_nodes: int) -> PlayResult:
        result = self.multiverse.search(board, max_nodes)
        if result.best_move is None:
            raise RuntimeError(f"{self.name} produced no move on {board.fen()}")
        return PlayResult(move=result.best_move, score=result.score)
