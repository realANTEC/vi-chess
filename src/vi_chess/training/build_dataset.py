"""Build the Phase 3 supervised-training dataset.

For each sampled position:
  - Compute STM-relative eval from each of the 7 universes
  - Compute Stockfish's eval (white POV) at fixed depth — the supervised target
  - Compute the position-feature vector

Save one JSONL row per position. The aggregator training script reads this file.

Note on chaos universe: chaos's score depends on ``playing_as``. For training
we leave ``playing_as`` at None so chaos uses its STM-relative legacy behavior;
this means the chaos score in the training set differs slightly from what
chaos contributes at game time (when ``playing_as`` is set by the arena).
The mismatch is bounded by ``±chaos_bonus`` (~90 cp). Acceptable for a
first-cut Phase 3 — if results are promising we'll re-train with proper
per-color handling.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess

from vi_chess.training.extract import collect_positions
from vi_chess.training.features import FEATURE_NAMES, extract_features
from vi_chess.training.stockfish import StockfishEvaluator
from vi_chess.universes import get

UNIVERSE_NAMES: list[str] = [
    "balanced",
    "material_greedy",
    "aggression",
    "endgame_purist",
    "mobility",
    "structural",
    "chaos",
]


def build_dataset(
    n_positions: int = 10_000,
    out_path: str | Path = "experiments/data/phase3_dataset.jsonl",
    stockfish_depth: int = 12,
    seed: int = 42,
    plies_range: tuple[int, int] = (0, 60),
    progress_every: int = 500,
) -> Path:
    """Generate ``n_positions`` training rows. Returns the path written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    universes = {name: get(name) for name in UNIVERSE_NAMES}
    # Make chaos's STM-relative fallback the training-time behavior:
    universes["chaos"].playing_as = None  # type: ignore[attr-defined]

    print(f"Sampling {n_positions} positions...", flush=True)
    t0 = time.monotonic()
    positions = collect_positions(n_positions, seed=seed, plies_range=plies_range)
    print(f"  done in {time.monotonic() - t0:.1f}s  ({len(positions)} unique)", flush=True)

    print(f"Building dataset -> {out}", flush=True)
    print(f"  Stockfish depth: {stockfish_depth}", flush=True)
    rows_written = 0
    t_start = time.monotonic()
    with StockfishEvaluator(depth=stockfish_depth) as sf, out.open("w", encoding="utf-8") as f:
        for i, pos in enumerate(positions):
            board = chess.Board(pos.fen)

            # Universe scores, STM-relative
            scores = {name: int(u.evaluate(board)) for name, u in universes.items()}

            # Position features (in FEATURE_NAMES order)
            feats = extract_features(board)

            # Stockfish ground truth (white POV → STM POV)
            sf_white = sf.evaluate(board)
            sf_stm = sf_white if board.turn == chess.WHITE else -sf_white

            row = {
                "fen": pos.fen,
                "source": pos.source,
                "plies_played": pos.plies_played,
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "scores": scores,                 # dict[name -> int cp, STM-relative]
                "features": dict(zip(FEATURE_NAMES, feats)),
                "stockfish_white_cp": int(sf_white),
                "target_stm_cp": int(sf_stm),     # supervised label
            }
            f.write(json.dumps(row) + "\n")
            rows_written += 1

            if (i + 1) % progress_every == 0:
                elapsed = time.monotonic() - t_start
                rate = (i + 1) / elapsed
                eta = (len(positions) - i - 1) / rate
                print(
                    f"  [{i + 1:>5}/{len(positions)}] "
                    f"rate={rate:.1f} pos/s  ETA {eta/60:.1f} min",
                    flush=True,
                )

    elapsed = time.monotonic() - t_start
    print(f"Done. {rows_written} rows in {elapsed/60:.1f} min ({rows_written/elapsed:.1f} pos/s)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Phase 3 training dataset")
    ap.add_argument("--n", type=int, default=10_000, help="number of positions")
    ap.add_argument("--out", type=str, default="experiments/data/phase3_dataset.jsonl")
    ap.add_argument("--depth", type=int, default=12, help="Stockfish search depth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build_dataset(
        n_positions=args.n,
        out_path=args.out,
        stockfish_depth=args.depth,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
