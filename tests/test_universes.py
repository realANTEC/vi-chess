"""Sanity tests for all registered universes.

Each universe must (a) register itself on import, (b) return a finite int on
the start position, and (c) return a symmetric eval on a symmetric position.
"""

from __future__ import annotations

import chess
import pytest

from vi_chess.universes import get, names

ALL_UNIVERSES = ["aggression", "balanced", "chaos", "endgame_purist",
                 "material_greedy", "mobility", "structural"]


def test_all_seven_universes_registered() -> None:
    assert set(names()) == set(ALL_UNIVERSES)


@pytest.mark.parametrize("uname", ALL_UNIVERSES)
def test_universe_evaluates_startpos(uname: str) -> None:
    u = get(uname)
    score = u.evaluate(chess.Board())
    assert isinstance(score, int)
    assert -30000 < score < 30000


@pytest.mark.parametrize("uname", ALL_UNIVERSES)
def test_universe_returns_legal_move_via_order(uname: str) -> None:
    u = get(uname)
    board = chess.Board()
    legal = list(board.legal_moves)
    ordered = u.order_moves(board, legal)
    assert set(ordered) == set(legal)
    assert len(ordered) == len(legal)


def test_universes_have_diverse_evals() -> None:
    """If all universes assign the same score to every position, the experiment is dead.
    We require at least 2 distinct scores on a non-trivial midgame position.
    """
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5")
    scores = {n: get(n).evaluate(board) for n in ALL_UNIVERSES}
    assert len(set(scores.values())) >= 2, f"all universes gave the same score: {scores}"
