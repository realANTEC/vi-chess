"""Balanced universe: classical material + PST with mg/eg phase interpolation.

Serves as the baseline control. Should play a solid, if generic, brand of chess.
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et
from vi_chess.universes.base import Universe, register


@register("balanced")
class BalancedUniverse(Universe):
    def evaluate(self, board: chess.Board) -> int:
        # Terminal positions are handled by search.py; we are only called on
        # positions with legal moves and no obvious draw.
        ph = et.phase(board)
        white = et.material(board, chess.WHITE) + et.pst_score(board, chess.WHITE, ph)
        black = et.material(board, chess.BLACK) + et.pst_score(board, chess.BLACK, ph)
        score = white - black
        return score if board.turn == chess.WHITE else -score
