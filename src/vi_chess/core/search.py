"""Iterative-deepening alpha-beta with a hard node budget.

The node budget is the load-bearing primitive for the multiverse experiment:
it lets us give an N-universe ensemble exactly the same total cost as a
single-universe baseline, so any Elo difference is attributable to the
architecture rather than extra compute.

Cost accounting: every ``_tick()`` is one unit of work. Each entry into
``negamax`` or ``quiesce`` ticks once (covers movegen + position state).
Each call to the evaluator ticks ``eval_cost`` times (1 for a single
universe, N for a shared-tree multiverse that runs N evaluators per leaf).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import chess

from vi_chess.universes.base import Universe

MATE_SCORE = 30000
MATE_BOUND = MATE_SCORE - 1000
INFINITY = MATE_SCORE + 1

EvaluateFn = Callable[[chess.Board], int]
OrderFn = Callable[[chess.Board, list[chess.Move]], list[chess.Move]]


@dataclass
class SearchResult:
    best_move: chess.Move | None
    score: int  # centipawns, root side-to-move perspective
    depth_reached: int
    nodes_searched: int


class _NodeBudgetExhausted(Exception):
    """Raised when nodes >= budget; caller returns best-completed-iteration result."""


class _Searcher:
    def __init__(
        self,
        evaluate: EvaluateFn,
        order_moves: OrderFn,
        max_nodes: int,
        eval_cost: int,
    ) -> None:
        self.evaluate = evaluate
        self.order_moves = order_moves
        self.max_nodes = max_nodes
        self.eval_cost = eval_cost
        self.nodes = 0

    def _tick(self, cost: int = 1) -> None:
        self.nodes += cost
        if self.nodes >= self.max_nodes:
            raise _NodeBudgetExhausted

    def _eval(self, board: chess.Board) -> int:
        score = self.evaluate(board)
        self._tick(self.eval_cost)
        return score

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._tick()

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        # Skip can_claim_* checks in q-search: expensive (stack scan) and rare here.
        # negamax catches them at depth > 0 which is where they usually fire anyway.

        stand_pat = self._eval(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if board.is_check():
            moves = list(board.legal_moves)
        else:
            moves = [m for m in board.legal_moves if board.is_capture(m)]
        moves = self.order_moves(board, moves)

        for move in moves:
            board.push(move)
            try:
                score = -self.quiesce(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_fifty_moves()
            or board.can_claim_threefold_repetition()
        ):
            return 0

        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        moves = self.order_moves(board, list(board.legal_moves))
        if not moves:
            return 0  # terminal cases handled above

        best = -INFINITY
        for move in moves:
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
        return best

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move | None, int]:
        moves = self.order_moves(board, list(board.legal_moves))
        if not moves:
            return None, 0

        best_move = moves[0]
        best_score = -INFINITY
        alpha = -INFINITY
        for move in moves:
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -INFINITY, -alpha, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
                if best_score > alpha:
                    alpha = best_score
        return best_move, best_score


def search_with_callables(
    evaluate: EvaluateFn,
    order_moves: OrderFn,
    board: chess.Board,
    max_nodes: int,
    eval_cost: int = 1,
    max_depth: int = 64,
) -> SearchResult:
    """Low-level search entry point. Used by both single-universe and shared-tree multiverse."""
    searcher = _Searcher(evaluate, order_moves, max_nodes, eval_cost)
    legal = list(board.legal_moves)
    best_move: chess.Move | None = legal[0] if legal else None
    best_score = 0
    depth_reached = 0

    for depth in range(1, max_depth + 1):
        try:
            move, score = searcher.search_root(board, depth)
        except _NodeBudgetExhausted:
            break
        if move is not None:
            best_move = move
            best_score = score
            depth_reached = depth
        if abs(score) >= MATE_BOUND:
            break

    return SearchResult(
        best_move=best_move,
        score=best_score,
        depth_reached=depth_reached,
        nodes_searched=searcher.nodes,
    )


def search(universe: Universe, board: chess.Board, max_nodes: int, max_depth: int = 64) -> SearchResult:
    """High-level: single-universe search."""
    return search_with_callables(
        evaluate=universe.evaluate,
        order_moves=universe.order_moves,
        board=board,
        max_nodes=max_nodes,
        eval_cost=1,
        max_depth=max_depth,
    )
