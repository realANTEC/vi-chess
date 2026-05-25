"""Aggression universe: rewards king attacks, biases move ordering toward checks.

Plays for the king. Will sacrifice pawns and minor positional advantages for
attacking chances. Move ordering pushes checks and king-zone attackers first.
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et
from vi_chess.core.ordering import mvv_lva_score
from vi_chess.universes.base import Universe, register

KING_ATTACK_WEIGHT = 8  # centipawns per unit of pressure


@register("aggression")
class AggressionUniverse(Universe):
    def evaluate(self, board: chess.Board) -> int:
        ph = et.phase(board)
        white = (
            et.material(board, chess.WHITE)
            + et.pst_score(board, chess.WHITE, ph)
            + KING_ATTACK_WEIGHT * et.king_attack_pressure(board, chess.WHITE)
        )
        black = (
            et.material(board, chess.BLACK)
            + et.pst_score(board, chess.BLACK, ph)
            + KING_ATTACK_WEIGHT * et.king_attack_pressure(board, chess.BLACK)
        )
        score = white - black
        return score if board.turn == chess.WHITE else -score

    def order_moves(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        enemy_king = board.king(not board.turn)
        king_zone = et._king_zone_mask(enemy_king) if enemy_king is not None else 0

        def key(m: chess.Move) -> int:
            score = mvv_lva_score(board, m)
            if board.gives_check(m):
                score += 600
            # Bonus for moves landing in / attacking the enemy king zone
            if chess.BB_SQUARES[m.to_square] & king_zone:
                score += 200
            return score

        return sorted(moves, key=key, reverse=True)
