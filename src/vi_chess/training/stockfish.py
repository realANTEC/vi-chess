"""Thin wrapper around python-chess's UCI engine driver for batched Stockfish evals.

The expensive part of evaluating many positions through an external engine is
starting it — a fresh ``popen_uci`` call costs ~100 ms on Windows. So we wrap
the engine in a context manager that opens once, evaluates many positions, and
closes cleanly. Per-position latency at depth 12 is roughly 5–20 ms once the
engine is warm.

Score convention: integer centipawns from White's perspective, with mate scaled
to ±30000 (matching ``vi_chess.core.search.MATE_SCORE``) so it's directly
comparable with our universes' outputs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import chess
import chess.engine

# Default Stockfish path on Vansh's machine. Override with the
# VI_CHESS_STOCKFISH environment variable to point elsewhere.
_DEFAULT_PATH = r"C:\Users\vansh\OneDrive\Desktop\stockfish\stockfish-windows-x86-64-avx2.exe"

STOCKFISH_PATH: str = os.environ.get("VI_CHESS_STOCKFISH", _DEFAULT_PATH)
DEFAULT_DEPTH: int = 12

# Cap mate scores at ±30000 to match our internal MATE_SCORE so learned models
# don't see ±100000 outliers from forced-mate positions.
_MATE_SCORE = 30000


def _resolve_path(path: str | os.PathLike[str] | None) -> str:
    p = str(path or STOCKFISH_PATH)
    if not Path(p).exists():
        raise FileNotFoundError(
            f"Stockfish binary not found at {p!r}. Set VI_CHESS_STOCKFISH env var "
            f"or pass an explicit path."
        )
    return p


class StockfishEvaluator:
    """Context manager wrapping a long-lived Stockfish UCI process.

    Example:
        with StockfishEvaluator(depth=12) as sf:
            for board in many_boards:
                cp = sf.evaluate(board)
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        depth: int = DEFAULT_DEPTH,
        threads: int = 1,
        hash_mb: int = 128,
    ) -> None:
        self._path = _resolve_path(path)
        self._depth = depth
        self._threads = threads
        self._hash_mb = hash_mb
        self._engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> "StockfishEvaluator":
        engine = chess.engine.SimpleEngine.popen_uci(self._path)
        try:
            engine.configure({"Threads": self._threads, "Hash": self._hash_mb})
        except chess.engine.EngineError:
            # Older Stockfish builds may not accept these options; ignore.
            pass
        self._engine = engine
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:  # noqa: BLE001 — best-effort cleanup on shutdown
                pass
            self._engine = None

    def evaluate(self, board: chess.Board, depth: int | None = None) -> int:
        """Stockfish eval of ``board`` in centipawns from White's POV.

        Mate scores are clamped to ±_MATE_SCORE.
        """
        if self._engine is None:
            raise RuntimeError("StockfishEvaluator used outside its context manager")
        info = self._engine.analyse(
            board,
            chess.engine.Limit(depth=depth if depth is not None else self._depth),
        )
        return info["score"].white().score(mate_score=_MATE_SCORE)

    def evaluate_many(
        self,
        boards: Iterable[chess.Board],
        depth: int | None = None,
    ) -> list[int]:
        """Convenience: eval a sequence of boards, returning aligned scores."""
        return [self.evaluate(b, depth=depth) for b in boards]
