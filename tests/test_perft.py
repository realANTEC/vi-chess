"""Perft (performance test) — counts leaf nodes at a given depth from a position.

Movegen comes from python-chess; perft confirms our environment matches the
reference node counts published on https://www.chessprogramming.org/Perft_Results.
If these ever diverge, something is very wrong with the chess library or
Python version, and the rest of the engine is suspect.
"""

from __future__ import annotations

import chess
import pytest


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


# (name, fen, depth, expected_nodes)
FAST_CASES = [
    ("initial",  chess.STARTING_FEN,                                                            4, 197_281),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",        3,  97_862),
    ("pos3",     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                                   4,  43_238),
    ("pos4",     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",            3,   9_467),
    ("pos5",     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",                   3,  62_379),
]

SLOW_CASES = [
    ("initial",  chess.STARTING_FEN,                                                            5, 4_865_609),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",        4, 4_085_603),
    ("pos3",     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                                   5,   674_624),
    ("pos4",     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",            4,   422_333),
    ("pos5",     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",                   4, 2_103_487),
]


@pytest.mark.parametrize("name,fen,depth,expected", FAST_CASES, ids=[c[0] for c in FAST_CASES])
def test_perft_fast(name: str, fen: str, depth: int, expected: int) -> None:
    board = chess.Board(fen)
    assert perft(board, depth) == expected


@pytest.mark.slow
@pytest.mark.parametrize("name,fen,depth,expected", SLOW_CASES, ids=[c[0] for c in SLOW_CASES])
def test_perft_slow(name: str, fen: str, depth: int, expected: int) -> None:
    board = chess.Board(fen)
    assert perft(board, depth) == expected
