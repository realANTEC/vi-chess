"""Curated opening book: ~40 mainstream openings at move 4-6.

Stored as SAN move sequences (readable, self-validating via python-chess
parser) and converted to FENs at import time. If any sequence fails to
parse, ``opening_fens()`` will raise on first call, surfacing the bug.
"""

from __future__ import annotations

import random

import chess

# (name, SAN move sequence) — both colors play from the resulting position.
OPENINGS: list[tuple[str, str]] = [
    # 1.e4 e5
    ("Ruy Lopez Berlin",        "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4"),
    ("Ruy Lopez Closed",        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7"),
    ("Italian Game",            "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6"),
    ("Two Knights Defense",     "e4 e5 Nf3 Nc6 Bc4 Nf6 d4"),
    ("Scotch Game",             "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6"),
    ("Petrov Defense",          "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4"),
    ("Vienna Game",             "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4"),
    ("Kings Gambit Accepted",   "e4 e5 f4 exf4 Nf3 g5 h4"),
    # Sicilian
    ("Najdorf Sicilian",        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    ("Sveshnikov Sicilian",     "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5"),
    ("Dragon Sicilian",         "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6"),
    ("Kan Sicilian",            "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6"),
    ("Alapin Sicilian",         "e4 c5 c3 Nf6 e5 Nd5 d4"),
    ("Closed Sicilian",         "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7"),
    # French
    ("French Tarrasch",         "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7"),
    ("French Winawer",          "e4 e6 d4 d5 Nc3 Bb4 e5 c5"),
    ("French Advance",          "e4 e6 d4 d5 e5 c5 c3 Nc6"),
    # Caro-Kann
    ("Caro-Kann Classical",     "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5"),
    ("Caro-Kann Advance",       "e4 c6 d4 d5 e5 Bf5 Nf3 e6"),
    ("Caro-Kann Panov",         "e4 c6 d4 d5 exd5 cxd5 c4 Nf6"),
    # 1.d4 d5
    ("QGD Orthodox",            "d4 d5 c4 e6 Nc3 Nf6 Bg5"),
    ("Queens Gambit Accepted",  "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6"),
    ("Slav Defense",            "d4 d5 c4 c6 Nc3 Nf6 Nf3 dxc4"),
    ("Catalan Opening",         "d4 Nf6 c4 e6 g3 d5 Bg2"),
    # 1.d4 Nf6
    ("King's Indian Defense",   "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O"),
    ("Nimzo-Indian",            "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O"),
    ("Queens Indian",           "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7"),
    ("Grunfeld Defense",        "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5"),
    ("Modern Benoni",           "d4 Nf6 c4 c5 d5 e6 Nc3 exd5"),
    # Flank
    ("English Reversed Sicilian","c4 e5 Nc3 Nf6 g3 d5"),
    ("English Symmetrical",     "c4 c5 Nc3 Nc6 Nf3 Nf6"),
    ("Reti Opening",            "Nf3 d5 c4 d4 b4"),
    # Less mainstream — adds diversity
    ("Pirc Defense",            "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O"),
    ("Modern Defense",          "e4 g6 d4 Bg7 Nc3 d6 f4 Nf6"),
    ("Alekhine Defense",        "e4 Nf6 e5 Nd5 d4 d6 Nf3"),
    ("Scandinavian Defense",    "e4 d5 exd5 Qxd5 Nc3 Qa5"),
    ("Trompowsky",              "d4 Nf6 Bg5 e6 e4 h6"),
    ("Larsen Opening",          "b3 e5 Bb2 Nc6 e3 Nf6"),
    ("Bird Opening",            "f4 d5 Nf3 Nf6 e3 g6"),
    ("Dutch Defense",           "d4 f5 g3 Nf6 Bg2 e6"),
]


def _moves_to_fen(san_sequence: str) -> str:
    board = chess.Board()
    for san in san_sequence.split():
        board.push_san(san)
    return board.fen()


_FEN_CACHE: list[tuple[str, str]] | None = None


def opening_fens() -> list[tuple[str, str]]:
    """List of (name, FEN). Computed once, cached."""
    global _FEN_CACHE
    if _FEN_CACHE is None:
        _FEN_CACHE = [(name, _moves_to_fen(moves)) for name, moves in OPENINGS]
    return _FEN_CACHE


def sample_opening(rng: random.Random | None = None) -> tuple[str, str]:
    """Pick a random opening. Returns (name, FEN)."""
    rng = rng or random.Random()
    return rng.choice(opening_fens())
