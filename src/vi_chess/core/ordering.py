"""Default move ordering: MVV-LVA for captures, quiets after."""

from __future__ import annotations

import chess

from vi_chess.core.pst import PIECE_VALUE


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Higher score = explore earlier. Captures by value-of-victim × 10 − value-of-attacker.

    Quiet moves score 0. Promotions get a small bonus on top.
    """
    score = 0
    if board.is_capture(move):
        if board.is_en_passant(move):
            victim_value = PIECE_VALUE[chess.PAWN]
        else:
            victim = board.piece_at(move.to_square)
            victim_value = PIECE_VALUE[victim.piece_type] if victim else 0
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker.piece_type] if attacker else 0
        score = victim_value * 10 - attacker_value
    if move.promotion is not None:
        score += PIECE_VALUE[move.promotion]
    return score


def order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    return sorted(moves, key=lambda m: mvv_lva_score(board, m), reverse=True)
