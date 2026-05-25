"""Experiment 1: full pairwise singles + multiverse vs best-solo at equal compute.

Outputs (per-matchup, written as soon as the matchup finishes):
  experiments/results/exp01/round_robin/<a>__vs__<b>.json
  experiments/results/exp01/multiverse/<mv>__vs__<best_solo>.json

Resumable: rerun the script and it skips any matchup whose JSON exists.

Phase 1: round-robin among all 7 single universes (21 matchups).
Phase 2: after Phase 1, identify best-solo (highest total round-robin score),
         then play each of 4 multiverse variants against it (4 matchups).
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
import time
from dataclasses import asdict
from typing import Callable

from vi_chess.harness import (
    MultiversePlayer,
    Player,
    SingleUniversePlayer,
    play_match,
    summarize,
)
from vi_chess.multiverse import (
    IndependentMultiverse,
    SharedTreeMultiverse,
    Vote,
    WeightedSum,
    uniform_weights,
)
from vi_chess.universes import get

# === Configuration ===
N_GAMES = 40
MAX_NODES = 10_000
SINGLE_NAMES = [
    "balanced",
    "material_greedy",
    "aggression",
    "endgame_purist",
    "mobility",
    "structural",
    "chaos",
]
FIVE_UNIVERSE_NAMES = [
    "balanced",
    "aggression",
    "endgame_purist",
    "mobility",
    "structural",
]
RESULTS_ROOT = pathlib.Path(__file__).parent / "results" / "exp01"
RR_DIR = RESULTS_ROOT / "round_robin"
MV_DIR = RESULTS_ROOT / "multiverse"

# === Player factories ===


def make_single_players() -> list[SingleUniversePlayer]:
    return [SingleUniversePlayer(get(n)) for n in SINGLE_NAMES]


def make_multiverse_players() -> list[MultiversePlayer]:
    all7 = [get(n) for n in SINGLE_NAMES]
    five = [get(n) for n in FIVE_UNIVERSE_NAMES]
    return [
        MultiversePlayer(
            SharedTreeMultiverse(universes=all7, aggregator=WeightedSum(uniform_weights(7))),
            label="shared-7",
        ),
        MultiversePlayer(
            IndependentMultiverse(universes=all7, aggregator=Vote()),
            label="indep-7",
        ),
        MultiversePlayer(
            SharedTreeMultiverse(universes=five, aggregator=WeightedSum(uniform_weights(5))),
            label="shared-5",
        ),
        MultiversePlayer(
            IndependentMultiverse(universes=five, aggregator=Vote()),
            label="indep-5",
        ),
    ]


# === Matchup runner ===


def _matchup_key(a_name: str, b_name: str) -> str:
    return f"{a_name}__vs__{b_name}".replace(":", "_").replace("/", "_")


def _run_matchup(
    player_a: Player,
    player_b: Player,
    n_games: int,
    max_nodes: int,
    out_dir: pathlib.Path,
    seed: int,
    progress_prefix: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{_matchup_key(player_a.name, player_b.name)}.json"
    if out_file.exists():
        print(f"{progress_prefix} {player_a.name} vs {player_b.name}: SKIPPED (already done)", flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))

    print(f"{progress_prefix} {player_a.name} vs {player_b.name}: starting at {time.strftime('%H:%M:%S')}...", flush=True)
    t0 = time.monotonic()
    match = play_match(
        player_a,
        player_b,
        n_games=n_games,
        max_nodes=max_nodes,
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
        "max_nodes": max_nodes,
        "termination_counts": _termination_counts(match),
    }
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    elo_str = f"{stats.elo_diff:+.0f}" if stats.elo_diff is not None else "n/a"
    print(
        f"{progress_prefix}   done in {elapsed/60:.1f}min: "
        f"W/D/L={match.wins}/{match.draws}/{match.losses} "
        f"Elo={elo_str} "
        f"LOS={stats.los_pct:.0f}%",
        flush=True,
    )
    return data


def _termination_counts(match) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(g.reason.value for g in match.games))


# === Phase 1 ===


def run_round_robin() -> list[dict]:
    players = make_single_players()
    pairs = [(players[i], players[j]) for i in range(len(players)) for j in range(i + 1, len(players))]
    print(f"\n=== PHASE 1: round-robin among {len(players)} single universes ({len(pairs)} matchups) ===\n", flush=True)
    results = []
    for idx, (a, b) in enumerate(pairs):
        results.append(_run_matchup(
            a, b, N_GAMES, MAX_NODES, RR_DIR,
            seed=idx,
            progress_prefix=f"[RR {idx+1}/{len(pairs)}]",
        ))
    return results


def best_solo_from_rr(rr_results: list[dict]) -> str:
    """Return the universe name with the highest total score across round-robin."""
    totals: dict[str, float] = {n: 0.0 for n in SINGLE_NAMES}
    for r in rr_results:
        # r['a'] = "solo:<name>", strip prefix
        a = r["a"].split(":", 1)[1] if ":" in r["a"] else r["a"]
        b = r["b"].split(":", 1)[1] if ":" in r["b"] else r["b"]
        totals[a] += r["score_a"] * r["n_games"]
        totals[b] += (1.0 - r["score_a"]) * r["n_games"]
    return max(totals, key=lambda k: totals[k])


# === Phase 2 ===


def run_multiverse_vs_best(best_solo: str) -> list[dict]:
    mv_players = make_multiverse_players()
    solo_player = SingleUniversePlayer(get(best_solo))
    print(
        f"\n=== PHASE 2: 4 multiverse variants vs best-solo ({best_solo}) ===\n",
        flush=True,
    )
    results = []
    for idx, mv in enumerate(mv_players):
        results.append(_run_matchup(
            mv, solo_player, N_GAMES, MAX_NODES, MV_DIR,
            seed=1000 + idx,
            progress_prefix=f"[MV {idx+1}/{len(mv_players)}]",
        ))
    return results


# === Entry ===


def main() -> int:
    t_start = time.monotonic()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Experiment 1 starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  N_GAMES={N_GAMES}, MAX_NODES={MAX_NODES}", flush=True)
    print(f"  Results: {RESULTS_ROOT}", flush=True)

    rr_results = run_round_robin()
    best = best_solo_from_rr(rr_results)
    print(f"\nBest solo from round-robin: {best}", flush=True)

    (RESULTS_ROOT / "best_solo.txt").write_text(best, encoding="utf-8")

    mv_results = run_multiverse_vs_best(best)

    total = time.monotonic() - t_start
    print(f"\n=== ALL DONE in {total/3600:.1f}h ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
