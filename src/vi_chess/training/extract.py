"""Position sampler for Phase 3 training data.

Take each curated opening in the book, play a uniform-random number of legal
plies past the book exit, and yield the resulting non-terminal position. This
gives broad coverage across game stages (short walks land in deep theory;
long walks reach simplified middlegames and sometimes endgames) without
needing to replay any actual games.

Why random plies and not engine plies: random play covers a much wider span
of positions than any single engine would visit. For training the aggregator,
*breadth* of positions matters more than realism — we want the aggregator to
learn "trust universe X in position type Y" across many position types, not
just the ones our own engines happen to play.

Terminal positions (checkmate / stalemate / insufficient material / 50-move
/ threefold) are skipped — their evaluation is undefined for our purposes.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from vi_chess.harness.openings import opening_fens


@dataclass(frozen=True)
class Position:
    fen: str
    source: str           # opening name this position descended from
    plies_played: int     # extra plies of random play past the book exit


def book_playout_positions(
    n_target: int,
    rng: random.Random,
    plies_range: tuple[int, int] = (0, 60),
    max_attempts_multiplier: int = 8,
) -> Iterator[Position]:
    """Yield up to ``n_target`` distinct positions reached from book openings.

    Each yielded position is the result of starting at a random book opening,
    playing a uniform-random number of legal plies (in ``plies_range``), and
    accepting the resulting position iff it's non-terminal and we haven't
    yielded it before.

    Gives up after ``n_target * max_attempts_multiplier`` attempts even if
    ``n_target`` wasn't reached (guards against degenerate small books).
    """
    book = opening_fens()
    if not book:
        return

    seen: set[str] = set()
    attempts = 0
    max_attempts = n_target * max_attempts_multiplier

    while len(seen) < n_target and attempts < max_attempts:
        attempts += 1
        opening_name, opening_fen = rng.choice(book)
        target_plies = rng.randint(*plies_range)
        board = chess.Board(opening_fen)
        plies_done = 0

        while plies_done < target_plies:
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            plies_done += 1

        # Discard terminal positions: their eval is degenerate.
        if board.is_game_over(claim_draw=True):
            continue

        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        yield Position(fen=fen, source=opening_name, plies_played=plies_done)


def collect_positions(
    n_target: int,
    seed: int = 42,
    plies_range: tuple[int, int] = (0, 60),
) -> list[Position]:
    """Eager helper around ``book_playout_positions``."""
    rng = random.Random(seed)
    return list(book_playout_positions(n_target, rng, plies_range))


def save_positions_jsonl(positions: list[Position], path: str | Path) -> None:
    """Write positions to a JSON Lines file. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for pos in positions:
            f.write(json.dumps(asdict(pos)) + "\n")


def load_positions_jsonl(path: str | Path) -> list[Position]:
    """Read positions back from a JSON Lines file."""
    with Path(path).open(encoding="utf-8") as f:
        return [Position(**json.loads(line)) for line in f if line.strip()]
