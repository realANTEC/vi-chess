"""Structural universe: pawn structure dominates. Plays Karpov-ish chess.

Heavy pawn-structure weighting in mg AND eg. Treats doubled and isolated
pawns as severe; passed pawns are gold. Will refuse pawn moves that wreck
structure even when they look tactically appealing.
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et
from vi_chess.universes.base import Universe, register

STRUCTURE_WEIGHT = 2.5


@register("structural")
class StructuralUniverse(Universe):
    def evaluate(self, board: chess.Board) -> int:
        ph = et.phase(board)
        white = (
            et.material(board, chess.WHITE)
            + et.pst_score(board, chess.WHITE, ph)
            + round(STRUCTURE_WEIGHT * et.pawn_structure(board, chess.WHITE))
        )
        black = (
            et.material(board, chess.BLACK)
            + et.pst_score(board, chess.BLACK, ph)
            + round(STRUCTURE_WEIGHT * et.pawn_structure(board, chess.BLACK))
        )
        score = white - black
        return score if board.turn == chess.WHITE else -score
