"""Phase 3 experiment: shared-tree multiverse with learned aggregator vs best-solo.

Mirrors run_exp01.py's structure but uses the trained LearnedAggregator instead
of WeightedSum. Single matchup at 10k nodes/move, 40 games, vs mobility (the
final Phase 1 best-solo).

Resumable per-matchup checkpoint. If the model pickle is missing, the script
exits with code 2 — train it first via ``python -m vi_chess.training.train``.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
import time

from vi_chess.harness import (
    MultiversePlayer,
    SingleUniversePlayer,
    play_match,
    summarize,
)
from vi_chess.multiverse import SharedTreeMultiverse
from vi_chess.training.learned_aggregator import LearnedAggregator
from vi_chess.universes import get

N_GAMES = 40
MAX_NODES = 10_000
BEST_SOLO_NAME = "mobility"   # per Phase 1 final standings (mobility +44)

RESULTS_ROOT = pathlib.Path(__file__).parent / "results" / "phase3"
MODEL_PATH = pathlib.Path(__file__).parent / "models" / "phase3_mlp.pkl"


def _matchup_key(a_name: str, b_name: str) -> str:
    return f"{a_name}__vs__{b_name}".replace(":", "_").replace("/", "_")


def run_matchup(player_a, player_b, n_games, max_nodes, out_dir, seed, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{_matchup_key(player_a.name, player_b.name)}.json"
    if out_file.exists():
        print(f"{label} SKIPPED (already done)", flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))

    print(f"{label} starting at {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.monotonic()
    match = play_match(player_a, player_b, n_games=n_games, max_nodes=max_nodes, rng=random.Random(seed))
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
        "max_nodes": max_nodes,
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
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 experiment starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  N_GAMES={N_GAMES}, MAX_NODES={MAX_NODES}", flush=True)
    print(f"  Model: {MODEL_PATH}", flush=True)
    print(f"  Results: {RESULTS_ROOT}", flush=True)
    print(f"  Best-solo: {BEST_SOLO_NAME}", flush=True)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr, flush=True)
        print("Train first via: uv run python -m vi_chess.training.train", file=sys.stderr, flush=True)
        return 2

    print("Loading learned aggregator...", flush=True)
    agg = LearnedAggregator.load(MODEL_PATH)
    print(f"  universes: {agg.universe_names}", flush=True)

    universes = [get(n) for n in agg.universe_names]
    learned_mv = MultiversePlayer(
        multiverse=SharedTreeMultiverse(universes=universes, aggregator=agg),
        label=f"shared-{len(universes)}-learned",
    )
    best_solo = SingleUniversePlayer(get(BEST_SOLO_NAME))

    run_matchup(
        learned_mv,
        best_solo,
        n_games=N_GAMES,
        max_nodes=MAX_NODES,
        out_dir=RESULTS_ROOT,
        seed=0xB44ADE,
        label=f"[MV 1/1] {learned_mv.name} vs {best_solo.name}:",
    )

    total = time.monotonic() - t_start
    print(f"\n=== ALL DONE in {total/3600:.2f}h ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
