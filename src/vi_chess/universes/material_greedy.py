"""Material-Greedy universe: scales material up, deprioritizes PST.

Plays directly for captures and material gain. Will happily trade positional
considerations for a pawn. Useful as one extreme of the diversity spectrum.
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et
from vi_chess.universes.base import Universe, register

MAT_MULT = 1.5
PST_MULT = 0.2


@register("material_greedy")
class MaterialGreedyUniverse(Universe):
    def evaluate(self, board: chess.Board) -> int:
        ph = et.phase(board)
        white = MAT_MULT * et.material(board, chess.WHITE) + PST_MULT * et.pst_score(board, chess.WHITE, ph)
        black = MAT_MULT * et.material(board, chess.BLACK) + PST_MULT * et.pst_score(board, chess.BLACK, ph)
        score = round(white - black)
        return score if board.turn == chess.WHITE else -score
