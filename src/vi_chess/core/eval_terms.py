"""Reusable evaluation terms. Each universe composes these with its own weights.

All term functions return a centipawn integer from the perspective of `color`
(the side they are evaluating *for*). To get a side-to-move score, subtract
the opponent's term from your own.
"""

from __future__ import annotations

from functools import lru_cache

import chess

from vi_chess.core.pst import PIECE_VALUE, PST_EG, PST_MG, pst_lookup

_PHASE_OPENING = 2 * (
    2 * PIECE_VALUE[chess.KNIGHT]
    + 2 * PIECE_VALUE[chess.BISHOP]
    + 2 * PIECE_VALUE[chess.ROOK]
    + PIECE_VALUE[chess.QUEEN]
)


def non_pawn_material(board: chess.Board, color: chess.Color | None = None) -> int:
    total = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if color is None:
            count = (
                chess.popcount(board.pieces_mask(pt, chess.WHITE))
                + chess.popcount(board.pieces_mask(pt, chess.BLACK))
            )
        else:
            count = chess.popcount(board.pieces_mask(pt, color))
        total += count * PIECE_VALUE[pt]
    return total


def phase(board: chess.Board) -> float:
    """1.0 = opening (full material), 0.0 = endgame (only pawns + kings)."""
    npm = non_pawn_material(board)
    return min(1.0, npm / _PHASE_OPENING) if _PHASE_OPENING else 0.0


def material(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += chess.popcount(board.pieces_mask(pt, color)) * PIECE_VALUE[pt]
    return total


def pst_score(board: chess.Board, color: chess.Color, phase_val: float) -> int:
    """Phase-interpolated PST sum for one color."""
    total = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        table_mg = PST_MG[pt]
        table_eg = PST_EG[pt]
        for sq in board.pieces(pt, color):
            mg = pst_lookup(table_mg, sq, color)
            eg = pst_lookup(table_eg, sq, color)
            total += round(mg * phase_val + eg * (1 - phase_val))
    return total


def mobility(board: chess.Board, color: chess.Color) -> int:
    """Sum of attacked squares across all of color's pieces. Cheap proxy."""
    total = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for sq in board.pieces(pt, color):
            total += chess.popcount(int(board.attacks_mask(sq)))
    return total


@lru_cache(maxsize=64)
def _king_zone_mask(king_sq: int) -> int:
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)
    mask = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = f + df, r + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                mask |= chess.BB_SQUARES[chess.square(nf, nr)]
    return mask


_ATTACKER_WEIGHT = {chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 5}


def king_attack_pressure(board: chess.Board, attacker: chess.Color) -> int:
    """Weighted count of how many of attacker's piece attacks land in the enemy king zone."""
    enemy_king = board.king(not attacker)
    if enemy_king is None:
        return 0
    zone = _king_zone_mask(enemy_king)
    pressure = 0
    for pt, weight in _ATTACKER_WEIGHT.items():
        for sq in board.pieces(pt, attacker):
            attacks = int(board.attacks_mask(sq))
            pressure += chess.popcount(attacks & zone) * weight
    return pressure


def pawn_structure(board: chess.Board, color: chess.Color) -> int:
    """Doubled (-), isolated (-), passed (+) pawn terms."""
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    pawns_by_file: list[list[int]] = [[] for _ in range(8)]
    for sq in pawns:
        pawns_by_file[chess.square_file(sq)].append(sq)

    enemy_by_file: list[list[int]] = [[] for _ in range(8)]
    for sq in enemy_pawns:
        enemy_by_file[chess.square_file(sq)].append(sq)

    score = 0

    for file_pawns in pawns_by_file:
        if len(file_pawns) > 1:
            score -= 20 * (len(file_pawns) - 1)

    for file in range(8):
        if not pawns_by_file[file]:
            continue
        has_neighbor = (file > 0 and pawns_by_file[file - 1]) or (file < 7 and pawns_by_file[file + 1])
        if not has_neighbor:
            score -= 15 * len(pawns_by_file[file])

    for sq in pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        ahead = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
        is_passed = True
        for ef in (file - 1, file, file + 1):
            if 0 <= ef <= 7:
                for ep in enemy_by_file[ef]:
                    if chess.square_rank(ep) in ahead:
                        is_passed = False
                        break
                if not is_passed:
                    break
        if is_passed:
            advance = rank if color == chess.WHITE else (7 - rank)
            score += 20 + advance * 10

    return score


def material_imbalance(board: chess.Board) -> int:
    """Absolute difference in material between sides. Always non-negative."""
    return abs(material(board, chess.WHITE) - material(board, chess.BLACK))


def piece_count(board: chess.Board, color: chess.Color) -> int:
    """All non-king, non-pawn piece count for `color`."""
    return sum(
        chess.popcount(board.pieces_mask(pt, color))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
