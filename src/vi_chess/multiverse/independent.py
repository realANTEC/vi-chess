"""IndependentMultiverse: N separate alpha-beta searches, MoveAggregator picks the final move.

Each universe runs its own search tree with its own move ordering, gets
``max_nodes // N`` budget, and produces a ``SearchResult``. The MoveAggregator
then collapses the N results into one final move.

This is closest to the user's original spec: each universe is its own
strategic reality, and the final decision is a negotiation across the
parallel realities.

Cost note: total nodes across universes ≈ max_nodes (modulo budget rounding),
so this is compute-comparable to a single-universe search at max_nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import chess

from vi_chess.core.search import SearchResult, search
from vi_chess.multiverse.aggregators import MoveAggregator
from vi_chess.universes.base import Universe


@dataclass
class IndependentSearchResult(SearchResult):
    """SearchResult enriched with per-universe diagnostics."""

    per_universe: list[SearchResult] | None = None


@dataclass
class IndependentMultiverse:
    universes: Sequence[Universe]
    aggregator: MoveAggregator

    def __post_init__(self) -> None:
        if not self.universes:
            raise ValueError("at least one universe required")

    def search(
        self,
        board: chess.Board,
        max_nodes: int,
        max_depth: int = 64,
    ) -> IndependentSearchResult:
        n = len(self.universes)
        budget_each = max(1, max_nodes // n)

        results: list[SearchResult] = []
        for universe in self.universes:
            board_copy = board.copy(stack=False)
            results.append(search(universe, board_copy, budget_each, max_depth=max_depth))

        chosen_move = self.aggregator.pick(results)

        # Report the depth of the deepest universe that voted for the chosen move (if any)
        chosen_depths = [r.depth_reached for r in results if r.best_move == chosen_move]
        chosen_scores = [r.score for r in results if r.best_move == chosen_move]
        depth_reached = max(chosen_depths) if chosen_depths else max(r.depth_reached for r in results)
        score = round(sum(chosen_scores) / len(chosen_scores)) if chosen_scores else 0
        nodes_searched = sum(r.nodes_searched for r in results)

        return IndependentSearchResult(
            best_move=chosen_move,
            score=score,
            depth_reached=depth_reached,
            nodes_searched=nodes_searched,
            per_universe=results,
        )
