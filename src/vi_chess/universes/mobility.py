"""Mobility universe: rewards piece activity heavily.

Plays for piece activity: open files for rooks, long diagonals for bishops,
outposts for knights. Will sacrifice small material to keep pieces working.
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et
from vi_chess.universes.base import Universe, register

MOBILITY_WEIGHT = 4
PST_MULT = 0.5


@register("mobility")
class MobilityUniverse(Universe):
    def evaluate(self, board: chess.Board) -> int:
        ph = et.phase(board)
        white = (
            et.material(board, chess.WHITE)
            + round(PST_MULT * et.pst_score(board, chess.WHITE, ph))
            + MOBILITY_WEIGHT * et.mobility(board, chess.WHITE)
        )
        black = (
            et.material(board, chess.BLACK)
            + round(PST_MULT * et.pst_score(board, chess.BLACK, ph))
            + MOBILITY_WEIGHT * et.mobility(board, chess.BLACK)
        )
        score = white - black
        return score if board.turn == chess.WHITE else -score
