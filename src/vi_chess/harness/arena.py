"""Arena: play games and matches between two players at equal node budget.

Game termination beyond the natural rules of chess:
  - Hard ply cap (default 200) — bad endgame play can otherwise loop indefinitely.
  - Resign: if the mover sees their own position as <= RESIGN_THRESHOLD for
    RESIGN_PLIES consecutive of their own moves, they lose.
  - Draw adjudication: if the last ADJ_DRAW_PLIES half-moves all report
    |score| <= ADJ_DRAW_THRESHOLD AND we're in the endgame (phase < ADJ_PHASE),
    call it a draw.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum

import chess

from vi_chess.core import eval_terms as et
from vi_chess.harness.openings import opening_fens
from vi_chess.harness.player import Player

MAX_PLY_PER_GAME = 200
RESIGN_THRESHOLD = -800
RESIGN_PLIES = 4
ADJ_DRAW_THRESHOLD = 20
ADJ_DRAW_PLIES = 10
ADJ_PHASE = 0.3


class GameOutcome(str, Enum):
    WHITE_WIN = "1-0"
    BLACK_WIN = "0-1"
    DRAW = "1/2-1/2"


class TerminationReason(str, Enum):
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    FIFTY_MOVES = "fifty_moves"
    THREEFOLD_REPETITION = "threefold_repetition"
    MAX_PLY = "max_ply"
    RESIGN = "resign"
    ADJUDICATION_DRAW = "adjudication_draw"


@dataclass
class GameResult:
    outcome: GameOutcome
    reason: TerminationReason
    white_name: str
    black_name: str
    opening_name: str
    ply_count: int
    pgn_moves: list[str]
    elapsed_seconds: float


@dataclass
class MatchResult:
    player_a_name: str
    player_b_name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: list[GameResult] = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score_a(self) -> float:
        return self.wins + 0.5 * self.draws


def _natural_termination(board: chess.Board) -> tuple[GameOutcome, TerminationReason] | None:
    if board.is_checkmate():
        winner = GameOutcome.BLACK_WIN if board.turn == chess.WHITE else GameOutcome.WHITE_WIN
        return winner, TerminationReason.CHECKMATE
    if board.is_stalemate():
        return GameOutcome.DRAW, TerminationReason.STALEMATE
    if board.is_insufficient_material():
        return GameOutcome.DRAW, TerminationReason.INSUFFICIENT_MATERIAL
    if board.can_claim_fifty_moves():
        return GameOutcome.DRAW, TerminationReason.FIFTY_MOVES
    if board.can_claim_threefold_repetition():
        return GameOutcome.DRAW, TerminationReason.THREEFOLD_REPETITION
    return None


def play_game(
    white: Player,
    black: Player,
    opening_fen: str,
    opening_name: str,
    max_nodes: int,
) -> GameResult:
    # Tell each player which color they're playing this game. Most universes
    # ignore this; color-aware ones (e.g. chaos) use it to attribute stylistic
    # preferences to their own side instead of side-to-move.
    white.set_playing_as(chess.WHITE)
    black.set_playing_as(chess.BLACK)

    board = chess.Board(opening_fen)
    moves: list[str] = []
    t_start = time.monotonic()

    white_scores: list[int] = []
    black_scores: list[int] = []
    last_scores: list[int] = []  # most recent N STM scores for adjudication

    outcome: GameOutcome
    reason: TerminationReason

    while True:
        term = _natural_termination(board)
        if term is not None:
            outcome, reason = term
            break

        if board.ply() >= MAX_PLY_PER_GAME:
            outcome, reason = GameOutcome.DRAW, TerminationReason.MAX_PLY
            break

        mover = white if board.turn == chess.WHITE else black
        play = mover.play(board, max_nodes)
        if play.move not in board.legal_moves:
            raise RuntimeError(f"{mover.name} returned illegal move {play.move} on {board.fen()}")

        own_scores = white_scores if board.turn == chess.WHITE else black_scores
        own_scores.append(play.score)
        last_scores.append(play.score)
        if len(last_scores) > ADJ_DRAW_PLIES:
            last_scores = last_scores[-ADJ_DRAW_PLIES:]

        # Resign check: did the side that just moved see themselves losing for N consecutive of own moves?
        if len(own_scores) >= RESIGN_PLIES and all(s <= RESIGN_THRESHOLD for s in own_scores[-RESIGN_PLIES:]):
            # The mover is resigning → they lose
            outcome = GameOutcome.BLACK_WIN if board.turn == chess.WHITE else GameOutcome.WHITE_WIN
            reason = TerminationReason.RESIGN
            moves.append(board.san(play.move))
            board.push(play.move)
            break

        moves.append(board.san(play.move))
        board.push(play.move)

        # Draw adjudication: only after the position has updated
        if (
            len(last_scores) >= ADJ_DRAW_PLIES
            and all(abs(s) <= ADJ_DRAW_THRESHOLD for s in last_scores)
            and et.phase(board) < ADJ_PHASE
        ):
            outcome, reason = GameOutcome.DRAW, TerminationReason.ADJUDICATION_DRAW
            break

    return GameResult(
        outcome=outcome,
        reason=reason,
        white_name=white.name,
        black_name=black.name,
        opening_name=opening_name,
        ply_count=board.ply(),
        pgn_moves=moves,
        elapsed_seconds=time.monotonic() - t_start,
    )


def play_match(
    player_a: Player,
    player_b: Player,
    n_games: int,
    max_nodes: int,
    rng: random.Random | None = None,
    progress: bool = False,
) -> MatchResult:
    rng = rng or random.Random(0xC4E55)
    book = opening_fens()
    if not book:
        raise ValueError("opening book is empty")

    result = MatchResult(player_a_name=player_a.name, player_b_name=player_b.name)

    shuffled = list(book)
    rng.shuffle(shuffled)
    book_pointer = 0

    for game_idx in range(n_games):
        if book_pointer >= len(shuffled):
            rng.shuffle(shuffled)
            book_pointer = 0
        opening_name, opening_fen = shuffled[book_pointer]
        book_pointer += 1

        if game_idx % 2 == 0:
            white, black = player_a, player_b
        else:
            white, black = player_b, player_a

        game = play_game(white, black, opening_fen, opening_name, max_nodes)
        result.games.append(game)

        if game.outcome == GameOutcome.DRAW:
            result.draws += 1
        elif (game.outcome == GameOutcome.WHITE_WIN and white is player_a) or (
            game.outcome == GameOutcome.BLACK_WIN and black is player_a
        ):
            result.wins += 1
        else:
            result.losses += 1

        if progress:
            print(
                f"  [{game_idx + 1:>3}/{n_games}] {opening_name:<28} "
                f"{game.outcome.value:>7}  ({game.reason.value}, {game.ply_count}p, {game.elapsed_seconds:.1f}s)  "
                f"running W/D/L = {result.wins}/{result.draws}/{result.losses}",
                flush=True,
            )

    return result
