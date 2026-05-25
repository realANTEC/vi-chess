"""SharedTreeMultiverse: one alpha-beta tree, N evaluators at the leaves.

Move ordering is fixed to default MVV-LVA — per-universe ordering bias makes
no sense when only one tree is being walked. The diversity lives in the
leaf eval, which is the aggregator-of-N output.

Fairness: each leaf eval costs N node-budget ticks, so a budget of B nodes
corresponds to roughly the same total CPU work as a single-universe search
at budget B. This is the load-bearing property that makes the experiment fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import chess

from vi_chess.core.ordering import order_moves as default_order
from vi_chess.core.search import SearchResult, search_with_callables
from vi_chess.multiverse.aggregators import ScoreAggregator
from vi_chess.universes.base import Universe


@dataclass
class SharedTreeMultiverse:
    universes: Sequence[Universe]
    aggregator: ScoreAggregator

    def __post_init__(self) -> None:
        if not self.universes:
            raise ValueError("at least one universe required")

    def search(
        self,
        board: chess.Board,
        max_nodes: int,
        max_depth: int = 64,
    ) -> SearchResult:
        universes = self.universes
        aggregator = self.aggregator

        def evaluate(pos: chess.Board) -> int:
            return aggregator.combine([u.evaluate(pos) for u in universes])

        return search_with_callables(
            evaluate=evaluate,
            order_moves=default_order,
            board=board,
            max_nodes=max_nodes,
            eval_cost=len(universes),
            max_depth=max_depth,
        )
