"""Chaos universe: rewards positional complexity itself.

Hypothesis: this engine plays better in messy positions than in quiet ones,
so being on move in a messy position is intrinsically valuable. Operationalized
as: small bonus to side-to-move proportional to total mobility (complex piece
interactions) and absolute material imbalance (asymmetric positions are harder
to evaluate correctly).
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et
from vi_chess.universes.base import Universe, register

PST_MULT = 0.5
# Mobility totals run 60-120 in the midgame; 0.3 keeps that contribution near 20-40cp (well under a pawn).
COMPLEXITY_MOBILITY_WEIGHT = 0.3
# With the cap below, this contributes at most 50cp - a stylistic nudge, not enough to justify losing material.
IMBALANCE_WEIGHT = 0.05
IMBALANCE_CAP = 1000  # don't let huge material gaps dominate the chaos term


@register("chaos")
class ChaosUniverse(Universe):
    def evaluate(self, board: chess.Board) -> int:
        ph = et.phase(board)
        white_pos = et.material(board, chess.WHITE) + round(PST_MULT * et.pst_score(board, chess.WHITE, ph))
        black_pos = et.material(board, chess.BLACK) + round(PST_MULT * et.pst_score(board, chess.BLACK, ph))
        positional = white_pos - black_pos

        # Chaos bonus: STM prefers complex positions
        complexity = COMPLEXITY_MOBILITY_WEIGHT * (
            et.mobility(board, chess.WHITE) + et.mobility(board, chess.BLACK)
        )
        imbalance = IMBALANCE_WEIGHT * min(et.material_imbalance(board), IMBALANCE_CAP)
        chaos_bonus = round(complexity + imbalance)

        score = positional + chaos_bonus  # chaos_bonus goes to whoever is STM
        # Flip the positional part for black; the chaos_bonus stays positive for STM
        if board.turn == chess.BLACK:
            score = -positional + chaos_bonus
        return score
