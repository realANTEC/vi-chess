"""Sanity tests for both multiverse architectures."""

from __future__ import annotations

import chess
import pytest

from vi_chess.core.search import MATE_BOUND
from vi_chess.multiverse import (
    IndependentMultiverse,
    SharedTreeMultiverse,
    Vote,
    WeightedSum,
    uniform_weights,
)
from vi_chess.universes import get, names

ALL = ["aggression", "balanced", "chaos", "endgame_purist",
       "material_greedy", "mobility", "structural"]


@pytest.fixture
def all_universes():
    return [get(n) for n in ALL]


@pytest.fixture
def shared(all_universes):
    return SharedTreeMultiverse(
        universes=all_universes,
        aggregator=WeightedSum(uniform_weights(len(all_universes))),
    )


@pytest.fixture
def independent(all_universes):
    return IndependentMultiverse(universes=all_universes, aggregator=Vote())


def test_shared_returns_legal_move(shared) -> None:
    board = chess.Board()
    result = shared.search(board, max_nodes=3_000)
    assert result.best_move in board.legal_moves


def test_shared_respects_budget(shared) -> None:
    board = chess.Board()
    budget = 5_000
    result = shared.search(board, max_nodes=budget)
    # Eval ticks are batched (eval_cost=N per call), so we may overshoot by up to N.
    n = len(shared.universes)
    assert result.nodes_searched <= budget + n


def test_shared_finds_mate_in_one(shared) -> None:
    board = chess.Board("7k/8/6K1/6Q1/8/8/8/8 w - - 0 1")
    result = shared.search(board, max_nodes=30_000)
    board.push(result.best_move)
    assert board.is_checkmate()
    assert result.score >= MATE_BOUND


def test_independent_returns_legal_move(independent) -> None:
    board = chess.Board()
    result = independent.search(board, max_nodes=3_000)
    assert result.best_move in board.legal_moves
    assert result.per_universe is not None
    assert len(result.per_universe) == len(independent.universes)


def test_independent_respects_total_budget(independent) -> None:
    board = chess.Board()
    budget = 7_000
    result = independent.search(board, max_nodes=budget)
    # Each of N sub-searches can overshoot by ~1 node; total ≤ budget + N.
    n = len(independent.universes)
    assert result.nodes_searched <= budget + n + n  # extra slack for rounding


def test_independent_finds_mate_in_one(independent) -> None:
    board = chess.Board("7k/8/6K1/6Q1/8/8/8/8 w - - 0 1")
    result = independent.search(board, max_nodes=30_000)
    board.push(result.best_move)
    assert board.is_checkmate()


def test_universes_actually_disagree_somewhere(independent) -> None:
    """Across a few midgame positions, at least one must produce universe-disagreement.

    If no position ever shows disagreement, the multiverse architecture degrades
    to single-universe behavior and the entire experiment is moot.
    """
    test_positions = [
        # Open Italian after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.0-0 Nf6 — many plausible plans
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
        # Pawn structure decision: doubled c-pawns from a recapture
        "r1bqkbnr/pp1p1ppp/2p5/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 4",
        # Endgame-ish position with passed pawn potential
        "8/2k5/3p4/p2P1p2/P2P1P2/8/8/4K3 w - - 0 1",
    ]
    saw_disagreement = False
    for fen in test_positions:
        board = chess.Board(fen)
        result = independent.search(board, max_nodes=14_000)
        assert result.per_universe is not None
        moves = {r.best_move for r in result.per_universe if r.best_move is not None}
        if len(moves) >= 2:
            saw_disagreement = True
            break
    assert saw_disagreement, "universes never disagreed across test positions — diversity is dead"
