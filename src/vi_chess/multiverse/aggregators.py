"""Aggregators combine multiple universes' outputs into one decision.

Two distinct aggregation surfaces:

- ``ScoreAggregator.combine(scores)`` is called inside shared-tree search to
  fold N per-universe leaf scores into one number that drives alpha-beta.
- ``MoveAggregator.pick(results)`` is called by independent-search after each
  universe has produced its own ``SearchResult``; it chooses the final move.

We deliberately keep these protocols separate — a learned shared-tree
aggregator that operates on score vectors is a different beast from a learned
move picker that operates on per-universe PVs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import chess

from vi_chess.core.search import SearchResult


@runtime_checkable
class ScoreAggregator(Protocol):
    name: str

    def combine(self, scores: list[int], board: chess.Board) -> int: ...


@runtime_checkable
class MoveAggregator(Protocol):
    name: str

    def pick(self, results: list[SearchResult]) -> chess.Move: ...


def uniform_weights(n: int) -> list[float]:
    return [1.0 / n] * n


@dataclass
class WeightedSum:
    """Linear combination of universe scores. Used by shared-tree.

    The ``board`` argument is accepted for protocol uniformity with
    ``LearnedAggregator`` (which conditions on position features). WeightedSum
    ignores it.
    """

    weights: list[float]
    name: str = "weighted_sum"

    def combine(self, scores: list[int], board: chess.Board) -> int:
        if len(scores) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} scores, got {len(scores)}")
        return round(sum(w * s for w, s in zip(self.weights, scores)))


@dataclass
class Vote:
    """Each universe votes its top move. Plurality wins; tie-break by score sum."""

    name: str = "vote"

    def pick(self, results: list[SearchResult]) -> chess.Move:
        candidates = [r for r in results if r.best_move is not None]
        if not candidates:
            raise ValueError("no universe returned a legal move")

        votes: Counter[chess.Move] = Counter(r.best_move for r in candidates)
        max_votes = max(votes.values())
        tied = [m for m, v in votes.items() if v == max_votes]
        if len(tied) == 1:
            return tied[0]

        def score_sum(move: chess.Move) -> int:
            return sum(r.score for r in candidates if r.best_move == move)

        return max(tied, key=score_sum)


@dataclass
class Best:
    """Trust the most confident universe — pick the move from the highest-scoring search."""

    name: str = "best"

    def pick(self, results: list[SearchResult]) -> chess.Move:
        candidates = [r for r in results if r.best_move is not None]
        if not candidates:
            raise ValueError("no universe returned a legal move")
        winner = max(candidates, key=lambda r: r.score)
        assert winner.best_move is not None  # narrowed by filter
        return winner.best_move
