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
    """Chaos prefers complex positions — for *itself*, not for whoever is on move.

    Earlier iterations tied the chaos_bonus to side-to-move. That looks right at
    a leaf but breaks under negamax: the bonus is added at chaos's turns and
    subtracted at the opponent's turns (sign-flip), so chaos's preference
    averages out to roughly zero across a search tree. The fix is to attribute
    the bonus to chaos's own color (``self.playing_as``), which makes it
    survive sign-flips cleanly. The arena sets ``playing_as`` before each game.

    When ``playing_as`` is unset (e.g. unit tests), we fall back to STM-relative
    behavior so the diversity test still observes chaos's signature.
    """

    def evaluate(self, board: chess.Board) -> int:
        ph = et.phase(board)
        white_pos = et.material(board, chess.WHITE) + round(PST_MULT * et.pst_score(board, chess.WHITE, ph))
        black_pos = et.material(board, chess.BLACK) + round(PST_MULT * et.pst_score(board, chess.BLACK, ph))
        positional = white_pos - black_pos

        complexity = COMPLEXITY_MOBILITY_WEIGHT * (
            et.mobility(board, chess.WHITE) + et.mobility(board, chess.BLACK)
        )
        imbalance = IMBALANCE_WEIGHT * min(et.material_imbalance(board), IMBALANCE_CAP)
        chaos_bonus = round(complexity + imbalance)

        # Compute white-pov score with chaos_bonus attributed to chaos's own color.
        if self.playing_as == chess.WHITE:
            white_pov = positional + chaos_bonus
        elif self.playing_as == chess.BLACK:
            white_pov = positional - chaos_bonus
        else:
            # No color set — legacy STM-relative behavior (used by tests).
            white_pov = positional + (chaos_bonus if board.turn == chess.WHITE else -chaos_bonus)

        return white_pov if board.turn == chess.WHITE else -white_pov
