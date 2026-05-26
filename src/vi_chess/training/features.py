"""Cheap position features for conditioning the learned aggregator.

These are computed once per leaf eval at search time, so each must be cheap.
We deliberately reuse functions from ``vi_chess.core.eval_terms`` (already
hot in our profile from the universes themselves).

The feature vector is concatenated with the per-universe score vector and
fed to the MLP. ``FEATURE_NAMES`` defines the canonical order — it must be
identical at training and inference time, so the saved model and the
runtime ``LearnedAggregator`` reference the same indices.
"""

from __future__ import annotations

import chess

from vi_chess.core import eval_terms as et

FEATURE_NAMES: list[str] = [
    "phase",
    "npm_white",
    "npm_black",
    "mat_imbalance",
    "mobility_white",
    "mobility_black",
    "king_attack_w",
    "king_attack_b",
    "pawn_struct_w",
    "pawn_struct_b",
    "side_to_move",
]


def extract_features(board: chess.Board) -> list[float]:
    """Return the feature vector for ``board`` in ``FEATURE_NAMES`` order."""
    return [
        et.phase(board),
        float(et.non_pawn_material(board, chess.WHITE)),
        float(et.non_pawn_material(board, chess.BLACK)),
        float(et.material_imbalance(board)),
        float(et.mobility(board, chess.WHITE)),
        float(et.mobility(board, chess.BLACK)),
        float(et.king_attack_pressure(board, chess.WHITE)),
        float(et.king_attack_pressure(board, chess.BLACK)),
        float(et.pawn_structure(board, chess.WHITE)),
        float(et.pawn_structure(board, chess.BLACK)),
        1.0 if board.turn == chess.WHITE else 0.0,
    ]
