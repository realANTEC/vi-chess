# VI_CHESS

> Experimental chess engine prototyping a **multi-universe evaluator architecture**: instead of one evaluator, *N* parallel evaluators with different strategic philosophies, aggregated into one decision per move.

This is a Python prototype testing whether ensemble diversity beats the strongest single evaluator **at equal compute** — the necessary precondition for a planned C++ rewrite. If the multiverse architecture doesn't pay for itself at this experimental scale, the C++ project is killed before it starts. Methodology first; performance second.

## The core idea

Most engines compute `Value(position)` with a single evaluator. This one computes `Value_i(position)` for *i ∈ {1..N}* parallel evaluator "universes," each with a different strategic worldview (aggression, endgame purity, mobility, etc.), then aggregates into a final decision.

Two architectures are implemented side by side so they can be compared directly:

| Architecture | Description | Cost accounting |
|---|---|---|
| **SharedTree** | One αβ search tree. At each leaf, all *N* evaluators score the position; the aggregator combines them. | Each eval call ticks the node budget *N* times. Equal `max_nodes` budget = comparable CPU. |
| **Independent** | *N* separate αβ searches, each with `budget / N` nodes. A `MoveAggregator` picks the final move from the *N* per-universe results. | Total nodes across all sub-searches ≈ budget. Each universe is a smaller deeper-or-wider tree. |

The load-bearing invariant is **equal compute**. An ensemble that beats a single eval only by burning more CPU is not interesting; an ensemble that beats single eval at the same total work is the result that justifies the C++ rewrite.

## The seven universes

| Name | One-line philosophy |
|---|---|
| `balanced` | Classical material + PST with mg/eg phase interpolation. The control group. |
| `material_greedy` | Inflated material weights, near-zero PST weight. Plays for captures. |
| `aggression` | Bonus for piece attacks landing in the enemy king-zone. Move ordering biased toward checks. |
| `endgame_purist` | Forces endgame PST even in the middlegame. Heavy pawn-structure weighting. Trades down when ahead. |
| `mobility` | Heavy bonus for attacked-square count. Plays for piece activity. |
| `structural` | Heavy pawn-structure weighting (doubled, isolated, passed). Karpov-ish style. |
| `chaos` | Small bonus to side-to-move proportional to total position complexity + material imbalance. *(First iteration was over-weighted; see commit history.)* |

## Foundations

- **Iterative-deepening alpha-beta** with quiescence search and a hard node-budget primitive (`src/vi_chess/core/search.py`).
- **MVV-LVA move ordering** as the default; some universes override it.
- **Shared eval terms** (`src/vi_chess/core/eval_terms.py`) — material, PST, mobility, king-attack pressure, pawn structure, material imbalance — composed differently by each universe.
- **Arena** (`src/vi_chess/harness/arena.py`) — resign, draw-adjudication, and ply-cap rules to keep games from dragging on indefinitely under weak endgame play.
- **40-position curated opening book** (`src/vi_chess/harness/openings.py`) — Ruy Lopez, Sicilians, French, KID, English, etc.
- **Elo + likelihood-of-superiority** statistics (`src/vi_chess/harness/stats.py`).

## Quick start

Requires Python ≥ 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/realANTEC/vi-chess.git
cd vi-chess
uv sync
uv run pytest -m "not slow"     # 32 fast tests (~7 s)
uv run pytest                    # also runs depth-5 perfts (~minutes)
```

## Reproducing the experiment

The primary experiment is a 21-matchup round-robin among the 7 single universes, followed by 4 multiverse variants playing the strongest solo universe:

```bash
uv run python -m experiments.run_exp01
```

- Per-matchup JSON checkpoints land in `experiments/results/exp01/` — the runner is resumable; re-launch and it skips matchups already done.
- Default config: 40 games per matchup at 10 000 nodes/move. Total runtime ~14 hours on a single CPU core. Edit `experiments/run_exp01.py` to change.

Live analysis at any time:

```bash
uv run python -m experiments.analyze_exp01
```

After the main run, replay the chaos-tainted matchups against the re-tuned chaos universe:

```bash
uv run python experiments/rerun_chaos.py
```

## Project layout

```
src/vi_chess/
├── core/                    αβ search, quiescence, node-budget, eval terms, PSTs, ordering
├── universes/               base + 7 strategic philosophies
├── multiverse/              SharedTree + Independent orchestrators, aggregators
└── harness/                 arena, opening book, Elo/LOS statistics, player abstraction

tests/                       perft, search sanity, universe sanity, multiverse sanity
experiments/                 the actual experimental runner, analyzer, side-scripts
```

## Status & roadmap

- **Phase 1 — Python prototype (now):** validate that the multiverse architecture beats its strongest single universe at equal compute. Open question; experiment in progress.
- **Phase 2 — C++ rewrite (conditional):** bitboards, magic move generation, NNUE eval, one shared αβ tree carrying per-universe score vectors, transposition table holding per-universe scores. Triggered only if Phase 1 produces a clear positive result (e.g. ≥ 50 Elo at LOS > 95 %).
- **Phase 3 — Learned aggregator:** replace hand-set `WeightedSum` weights with a learned gating network conditioned on position features (tactical / positional / endgame, etc.). Out of scope until Phase 2 is in motion.

## License

[MIT](LICENSE) — do whatever you want, but preserve the copyright notice.
