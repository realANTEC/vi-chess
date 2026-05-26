"""Train the MLP aggregator on the Phase 3 dataset.

Reads ``experiments/data/phase3_dataset.jsonl``, fits an MLP that maps
(universe_scores + position_features) → Stockfish STM-relative score, and
saves the trained ``LearnedAggregator`` as a pickle for inference.

The trained model is small (~5–20 KB) and fast to evaluate (single-sample
prediction ~100 μs on CPU). Reports MSE, MAE, and the per-universe baselines
(MSE of each individual universe against Stockfish, for context — the MLP
should comfortably beat all of them since it can combine).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from vi_chess.training.build_dataset import UNIVERSE_NAMES
from vi_chess.training.features import FEATURE_NAMES
from vi_chess.training.learned_aggregator import LearnedAggregator


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return X (n × (N+K)), y (n,), and the raw rows."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    n_features = len(FEATURE_NAMES)
    n_universes = len(UNIVERSE_NAMES)
    X = np.zeros((n, n_universes + n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, uname in enumerate(UNIVERSE_NAMES):
            X[i, j] = row["scores"][uname]
        for j, fname in enumerate(FEATURE_NAMES):
            X[i, n_universes + j] = row["features"][fname]
        y[i] = row["target_stm_cp"]
    return X, y, rows


def train(
    dataset_path: str | Path = "experiments/data/phase3_dataset.jsonl",
    model_out: str | Path = "experiments/models/phase3_mlp.pkl",
    hidden_layers: tuple[int, ...] = (32, 32),
    max_iter: int = 500,
    random_state: int = 0,
    test_size: float = 0.2,
    target_clip_cp: int = 2000,
) -> dict:
    """Train an MLP aggregator. Return a dict of metrics + the saved model path.

    ``target_clip_cp`` clips the Stockfish targets to ±value cp. Mate scores
    (±30000) otherwise destabilize the regressor — at clipped 2000cp the model
    treats "winning by 2+ pawns" as a single category, which is fine: the
    aggregator drives search ranking, not absolute eval magnitude.
    """
    print(f"Loading dataset from {dataset_path}", flush=True)
    X, y, rows = load_dataset(dataset_path)
    print(f"  {len(rows)} rows, X.shape={X.shape}", flush=True)

    n_clipped = int(np.sum(np.abs(y) > target_clip_cp))
    if n_clipped:
        print(f"  clipping {n_clipped} extreme targets to +/-{target_clip_cp} cp", flush=True)
        y = np.clip(y, -target_clip_cp, target_clip_cp)
    # Also clip universe scores in X so the input range matches what the
    # runtime aggregator will see (universe evals don't return mate scores).
    n_universes_in_X = len(UNIVERSE_NAMES)
    X[:, :n_universes_in_X] = np.clip(
        X[:, :n_universes_in_X], -target_clip_cp, target_clip_cp
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Standardizing features...", flush=True)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Training MLP hidden_layers={hidden_layers}, max_iter={max_iter}", flush=True)
    t0 = time.monotonic()
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train_s, y_train)
    train_time = time.monotonic() - t0
    print(f"  trained in {train_time:.1f}s, n_iter={model.n_iter_}", flush=True)

    y_train_pred = model.predict(X_train_s)
    y_test_pred = model.predict(X_test_s)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"  train MSE={train_mse:.1f}  MAE={train_mae:.1f}", flush=True)
    print(f"   test MSE={test_mse:.1f}  MAE={test_mae:.1f}", flush=True)

    # Baseline: how good is each individual universe at predicting Stockfish?
    print()
    print("Per-universe baselines (MAE between universe's score and Stockfish target):")
    n_universes = len(UNIVERSE_NAMES)
    for j, uname in enumerate(UNIVERSE_NAMES):
        baseline_mae = mean_absolute_error(y_test, X_test[:, j])
        print(f"  {uname:<18} MAE={baseline_mae:>7.1f}", flush=True)
    print(f"  {'(uniform mean)':<18} MAE={mean_absolute_error(y_test, X_test[:, :n_universes].mean(axis=1)):>7.1f}")
    print(f"  {'(MLP)':<18} MAE={test_mae:>7.1f}  <- target", flush=True)

    agg = LearnedAggregator(
        model=model,
        scaler=scaler,
        universe_names=list(UNIVERSE_NAMES),
        feature_names=list(FEATURE_NAMES),
        name="learned_mlp",
    )
    agg.save(model_out)
    print(f"\nSaved -> {model_out}", flush=True)

    return {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "n_iter": int(model.n_iter_),
        "train_time_s": float(train_time),
        "model_path": str(Path(model_out).resolve()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the Phase 3 learned aggregator")
    ap.add_argument("--dataset", default="experiments/data/phase3_dataset.jsonl")
    ap.add_argument("--out", default="experiments/models/phase3_mlp.pkl")
    ap.add_argument("--hidden", default="32,32", help="comma-separated layer sizes")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hidden = tuple(int(s) for s in args.hidden.split(","))
    train(
        dataset_path=args.dataset,
        model_out=args.out,
        hidden_layers=hidden,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
