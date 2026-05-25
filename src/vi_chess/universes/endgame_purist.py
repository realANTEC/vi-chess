"""Endgame-Purist universe: always uses endgame PST, values pawn structure heavily.

Plays for simplification. Pawn structure dominates. Treats every position as
if the queens are already off the board. Will trade pieces willingly when ahead.
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et
from vi_chess.universes.base import Universe, register

STRUCTURE_WEIGHT = 1.5
SIMPLIFY_BONUS = 5  # cp per non-pawn piece the opponent still has, when we are ahead


@register("endgame_purist")
class EndgamePuristUniverse(Universe):
    def evaluate(self, board: chess.Board) -> int:
        # Force eg-PST regardless of actual phase
        white = (
            et.material(board, chess.WHITE)
            + et.pst_score(board, chess.WHITE, 0.0)
            + round(STRUCTURE_WEIGHT * et.pawn_structure(board, chess.WHITE))
        )
        black = (
            et.material(board, chess.BLACK)
            + et.pst_score(board, chess.BLACK, 0.0)
            + round(STRUCTURE_WEIGHT * et.pawn_structure(board, chess.BLACK))
        )

        # Simplification bonus: when ahead, prefer fewer enemy pieces on the board
        diff = white - black
        if diff > 100:
            diff -= SIMPLIFY_BONUS * et.piece_count(board, chess.BLACK)
        elif diff < -100:
            diff += SIMPLIFY_BONUS * et.piece_count(board, chess.WHITE)

        return diff if board.turn == chess.WHITE else -diff
