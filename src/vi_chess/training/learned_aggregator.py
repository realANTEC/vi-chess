"""Learned ScoreAggregator backed by an MLP.

Reads a pickled sklearn MLPRegressor (trained by ``vi_chess.training.train``)
and uses it at game time to combine N universe scores + position features into
a single STM-relative score.

Input vector layout (must match training):
    [score_1, score_2, ..., score_N,  feature_1, ..., feature_K]
where N = len(universe_names) (training order) and K = len(FEATURE_NAMES).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from vi_chess.training.features import FEATURE_NAMES, extract_features


@dataclass
class LearnedAggregator:
    """sklearn-backed score aggregator. Drop-in replacement for WeightedSum."""

    model: Any                       # sklearn regressor exposing ``predict``
    scaler: Any = None               # optional sklearn StandardScaler
    universe_names: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    name: str = "learned_mlp"

    def combine(self, scores: list[int], board: chess.Board) -> int:
        feats = extract_features(board)
        x = np.array([list(scores) + feats], dtype=np.float64)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        y = self.model.predict(x)[0]
        return int(round(float(y)))

    # ── serialization ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "universe_names": list(self.universe_names),
            "feature_names": list(self.feature_names),
            "name": self.name,
        }
        with p.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedAggregator":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)  # noqa: S301 — trusted local file
        return cls(**payload)
