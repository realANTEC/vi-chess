"""Phase 3 ablation: drop one universe at a time, retrain aggregator, retest vs mobility.

After the main shared-7-learned matchup has landed, this script runs 6 additional
matchups. For each ablation, it:
  1. Loads the full dataset, drops the specified universe's column from X.
  2. Trains a smaller MLP (same hyperparams) on the reduced feature set.
  3. Saves the resulting model as ``phase3_mlp_minus_<name>.pkl``.
  4. Plays shared-6-learned-minus-<name> vs mobility, 40 games at 10k nodes.

The output JSON checkpoints land alongside the main matchup; the analyzer can
read them all together. Resumable -- skips any matchup whose JSON exists.

Ablation skips dropping ``mobility`` itself (it's the best-solo, and dropping
your own benchmark out of the multiverse is methodologically weird) and
``balanced`` (also strong; leaving for a future deeper ablation).
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
import time
from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from vi_chess.harness import MultiversePlayer, SingleUniversePlayer, play_match, summarize
from vi_chess.multiverse import SharedTreeMultiverse
from vi_chess.training.build_dataset import UNIVERSE_NAMES
from vi_chess.training.features import FEATURE_NAMES
from vi_chess.training.learned_aggregator import LearnedAggregator
from vi_chess.training.train import load_dataset
from vi_chess.universes import get

N_GAMES = 40
MAX_NODES = 10_000
TARGET_CLIP_CP = 2000
BEST_SOLO_NAME = "mobility"

# Universes to ablate (one at a time). Skip mobility (it's the baseline) and
# balanced (the other top performer; reserve for a deeper-cut future study).
ABLATION_UNIVERSES: list[str] = [
    "material_greedy",
    "aggression",
    "endgame_purist",
    "structural",
    "chaos",
]

DATASET_PATH = pathlib.Path("experiments/data/phase3_dataset.jsonl")
MODELS_DIR = pathlib.Path("experiments/models")
RESULTS_DIR = pathlib.Path("experiments/results/phase3")


def _matchup_key(a_name: str, b_name: str) -> str:
    return f"{a_name}__vs__{b_name}".replace(":", "_").replace("/", "_")


def train_ablated(dropped: str) -> LearnedAggregator:
    """Train an MLP aggregator with ``dropped`` removed from the universe set."""
    print(f"  Training ablation MLP (dropped: {dropped})...", flush=True)
    X, y, _rows = load_dataset(DATASET_PATH)
    y = np.clip(y, -TARGET_CLIP_CP, TARGET_CLIP_CP)
    X[:, :len(UNIVERSE_NAMES)] = np.clip(
        X[:, :len(UNIVERSE_NAMES)], -TARGET_CLIP_CP, TARGET_CLIP_CP
    )

    drop_idx = UNIVERSE_NAMES.index(dropped)
    keep_universe_cols = [i for i in range(len(UNIVERSE_NAMES)) if i != drop_idx]
    feature_cols = list(range(len(UNIVERSE_NAMES), X.shape[1]))
    X_reduced = X[:, keep_universe_cols + feature_cols]

    kept_names = [n for n in UNIVERSE_NAMES if n != dropped]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)

    t0 = time.monotonic()
    model = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(Xs_train, y_train)

    test_mae = mean_absolute_error(y_test, model.predict(Xs_test))
    print(
        f"    trained in {time.monotonic() - t0:.1f}s, n_iter={model.n_iter_}, "
        f"test MAE={test_mae:.1f}",
        flush=True,
    )

    agg = LearnedAggregator(
        model=model,
        scaler=scaler,
        universe_names=kept_names,
        feature_names=list(FEATURE_NAMES),
        name=f"learned_mlp_minus_{dropped}",
    )
    out = MODELS_DIR / f"phase3_mlp_minus_{dropped}.pkl"
    agg.save(out)
    return agg


def run_matchup(player_a, player_b, seed: int, label: str) -> dict | None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{_matchup_key(player_a.name, player_b.name)}.json"
    if out_file.exists():
        print(f"{label} SKIPPED (already done)", flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))

    print(f"{label} starting at {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.monotonic()
    match = play_match(
        player_a, player_b,
        n_games=N_GAMES, max_nodes=MAX_NODES,
        rng=random.Random(seed),
    )
    elapsed = time.monotonic() - t0
    stats = summarize(match)

    data = {
        "a": player_a.name,
        "b": player_b.name,
        "n_games": match.n_games,
        "wins": match.wins,
        "draws": match.draws,
        "losses": match.losses,
        "score_a": stats.score,
        "elo_diff": stats.elo_diff,
        "elo_ci_95": list(stats.elo_ci_95) if stats.elo_ci_95 else None,
        "los_pct": stats.los_pct,
        "elapsed_seconds": elapsed,
        "max_nodes": MAX_NODES,
    }
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elo_str = f"{stats.elo_diff:+.0f}" if stats.elo_diff is not None else "n/a"
    print(
        f"{label}   done in {elapsed/60:.1f}min: "
        f"W/D/L={match.wins}/{match.draws}/{match.losses} Elo={elo_str} LOS={stats.los_pct:.0f}%",
        flush=True,
    )
    return data


def main() -> int:
    t_start = time.monotonic()
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 2

    best_solo = SingleUniversePlayer(get(BEST_SOLO_NAME))

    print(f"Phase 3 ABLATION starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  vs best-solo: {BEST_SOLO_NAME}", flush=True)
    print(f"  ablating: {ABLATION_UNIVERSES}", flush=True)
    print()

    for idx, dropped in enumerate(ABLATION_UNIVERSES, 1):
        label = f"[ABL {idx}/{len(ABLATION_UNIVERSES)}] minus_{dropped}:"
        print(label)

        model_path = MODELS_DIR / f"phase3_mlp_minus_{dropped}.pkl"
        if model_path.exists():
            print(f"  Reusing existing model {model_path.name}", flush=True)
            agg = LearnedAggregator.load(model_path)
        else:
            agg = train_ablated(dropped)

        kept_universes = [get(n) for n in agg.universe_names]
        learned_mv = MultiversePlayer(
            multiverse=SharedTreeMultiverse(universes=kept_universes, aggregator=agg),
            label=f"shared-{len(kept_universes)}-learned-minus-{dropped}",
        )
        run_matchup(
            learned_mv, best_solo,
            seed=0xAB1A700 + idx,
            label=label,
        )
        print()

    total = time.monotonic() - t_start
    print(f"=== ABLATION COMPLETE in {total/3600:.2f}h ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
