"""Elo and likelihood-of-superiority from W/D/L.

LOS is the probability that player A is genuinely stronger than B given the
observed result (Wald, large-sample normal approximation). It's the right
thing to look at when sample sizes are small — Elo confidence intervals
become useless below ~100 games but LOS still tells you "how likely is this
real."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vi_chess.harness.arena import MatchResult


@dataclass
class Stats:
    score: float                # 0..1, draws = 0.5
    n: int
    elo_diff: float | None
    elo_ci_95: tuple[float, float] | None
    los_pct: float              # 0..100


def elo_diff(score: float, n: int) -> float | None:
    """Convert score-rate to Elo difference. Returns None for shutouts (undefined)."""
    if score <= 0 or score >= 1 or n == 0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def _elo_ci_95(wins: int, draws: int, losses: int) -> tuple[float, float] | None:
    """Approximate 95% CI on Elo using a normal approximation around the score-rate."""
    n = wins + draws + losses
    if n < 5:
        return None
    score = (wins + 0.5 * draws) / n
    # Sample variance using BayesElo-style draw handling
    p_w, p_d, p_l = wins / n, draws / n, losses / n
    var = (p_w * (1 - score) ** 2 + p_d * (0.5 - score) ** 2 + p_l * (0.0 - score) ** 2)
    if var <= 0:
        return None
    se = math.sqrt(var / n)
    low_score = max(1e-6, min(1 - 1e-6, score - 1.96 * se))
    high_score = max(1e-6, min(1 - 1e-6, score + 1.96 * se))
    low_elo = elo_diff(low_score, n)
    high_elo = elo_diff(high_score, n)
    if low_elo is None or high_elo is None:
        return None
    return low_elo, high_elo


def los_pct(wins: int, losses: int) -> float:
    """Likelihood of superiority: P(A truly stronger | observed W vs L).

    Standard formula used by cutechess-cli and OpenBench. Draws don't enter:
    they don't tell us anything about which side is better.
    """
    if wins + losses == 0:
        return 50.0
    z = (wins - losses) / math.sqrt(wins + losses)
    return 100.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def summarize(match: MatchResult) -> Stats:
    n = match.n_games
    score = match.score_a / n if n else 0.0
    return Stats(
        score=score,
        n=n,
        elo_diff=elo_diff(score, n) if n else None,
        elo_ci_95=_elo_ci_95(match.wins, match.draws, match.losses),
        los_pct=los_pct(match.wins, match.losses),
    )
