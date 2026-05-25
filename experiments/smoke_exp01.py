"""Mini smoke run of the experiment infrastructure.

Uses only 2 universes (balanced + material_greedy) and 4 games per matchup, with
results written to results/exp01_smoke/. Should complete in ~5 minutes.

Verifies:
  - per-matchup checkpoints are written to disk
  - re-running the script skips matchups already done
  - phase 2 reads phase 1's best-solo correctly
  - analyzer reads partial / complete results without crashing
"""

from __future__ import annotations

import pathlib
import random
import time
from typing import Any

from vi_chess.harness import (
    MultiversePlayer,
    SingleUniversePlayer,
    play_match,
    summarize,
)
from vi_chess.multiverse import IndependentMultiverse, SharedTreeMultiverse, Vote, WeightedSum, uniform_weights
from vi_chess.universes import get

# Force a smaller config
import experiments.run_exp01 as exp01

exp01.SINGLE_NAMES = ["balanced", "material_greedy"]
exp01.FIVE_UNIVERSE_NAMES = ["balanced", "material_greedy"]
exp01.N_GAMES = 4
exp01.MAX_NODES = 5_000
exp01.RESULTS_ROOT = pathlib.Path(__file__).parent / "results" / "exp01_smoke"
exp01.RR_DIR = exp01.RESULTS_ROOT / "round_robin"
exp01.MV_DIR = exp01.RESULTS_ROOT / "multiverse"

# Multiverse factory needs to be patched because it builds players from SINGLE_NAMES.
_original_make_mv = exp01.make_multiverse_players


def _patched_make_mv() -> list[MultiversePlayer]:
    universes = [get(n) for n in exp01.SINGLE_NAMES]
    return [
        MultiversePlayer(
            SharedTreeMultiverse(universes=universes, aggregator=WeightedSum(uniform_weights(len(universes)))),
            label="shared-mini",
        ),
        MultiversePlayer(
            IndependentMultiverse(universes=universes, aggregator=Vote()),
            label="indep-mini",
        ),
    ]


exp01.make_multiverse_players = _patched_make_mv


if __name__ == "__main__":
    exp01.main()
