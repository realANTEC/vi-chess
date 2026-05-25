"""Sanity tests for the search: budget enforcement + minimum tactical ability."""

from __future__ import annotations

import chess
import pytest

from vi_chess.core.search import MATE_BOUND, search
from vi_chess.universes import get


@pytest.fixture
def balanced():
    return get("balanced")


def test_search_returns_legal_move_on_startpos(balanced) -> None:
    board = chess.Board()
    result = search(balanced, board, max_nodes=2_000)
    assert result.best_move in board.legal_moves


def test_search_respects_node_budget(balanced) -> None:
    board = chess.Board()
    budget = 5_000
    result = search(balanced, board, max_nodes=budget)
    # We may overshoot by at most one node (the one that triggered the exception).
    assert result.nodes_searched <= budget + 1


def test_search_finds_mate_in_one(balanced) -> None:
    # K+Q vs K. White to play, Qd8# (queen swings to the 8th rank delivering mate).
    board = chess.Board("7k/8/6K1/6Q1/8/8/8/8 w - - 0 1")
    result = search(balanced, board, max_nodes=20_000)
    board.push(result.best_move)
    assert board.is_checkmate(), f"expected mate after {result.best_move}, got {board.fen()}"
    assert result.score >= MATE_BOUND


def test_search_captures_hanging_queen(balanced) -> None:
    # Black blunders queen to g4; White must play Qxg4.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P1q1/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")
    result = search(balanced, board, max_nodes=20_000)
    assert result.best_move == chess.Move.from_uci("d1g4")
    # Net gain should be approximately a queen.
    assert result.score > 500
