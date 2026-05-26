"""Training pipeline for the learned-aggregator Phase 3 work.

Phase 3 replaces the naive WeightedSum / Vote aggregators in
``vi_chess.multiverse`` with a learned MLP that maps per-universe scores
(plus position features) to a single combined score. The supervised target
is Stockfish's evaluation at fixed depth, treated as ground truth.

Modules here are *training-only* — they don't get imported by the runtime
search/eval path. Models are trained, pickled, and then loaded by a
``LearnedAggregator`` at game time.
"""
