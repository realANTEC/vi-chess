"""Read all checkpoint JSONs from exp01 and print a summary.

Safe to run mid-experiment — uses whatever JSONs exist so far.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

from vi_chess.harness.stats import elo_diff

ROOT = pathlib.Path(__file__).parent / "results" / "exp01"
RR_DIR = ROOT / "round_robin"
MV_DIR = ROOT / "multiverse"

SINGLE_NAMES = ["balanced", "material_greedy", "aggression", "endgame_purist",
                "mobility", "structural", "chaos"]


def _solo_short(name: str) -> str:
    return name.split(":", 1)[1] if ":" in name else name


def load_rr() -> list[dict]:
    if not RR_DIR.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RR_DIR.glob("*.json"))]


def load_mv() -> list[dict]:
    if not MV_DIR.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(MV_DIR.glob("*.json"))]


def print_rr_matrix(rr: list[dict]) -> None:
    print(f"\n=== Round-Robin matrix ({len(rr)} matchups loaded) ===\n")
    if not rr:
        print("  no results yet")
        return

    # cells[a][b] = "W-D-L"
    cells: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rr:
        a, b = _solo_short(r["a"]), _solo_short(r["b"])
        cells[a][b] = f"{r['wins']:>2}-{r['draws']:>2}-{r['losses']:>2}"
        cells[b][a] = f"{r['losses']:>2}-{r['draws']:>2}-{r['wins']:>2}"

    width = 14
    header = " " * width + "".join(f"{n[:8]:>10}" for n in SINGLE_NAMES)
    print(header)
    for a in SINGLE_NAMES:
        row = f"{a[:13]:<{width}}"
        for b in SINGLE_NAMES:
            if a == b:
                row += f"{'   --   ':>10}"
            elif b in cells.get(a, {}):
                row += f"{cells[a][b]:>10}"
            else:
                row += f"{'   ?   ':>10}"
        print(row)


def print_rr_standings(rr: list[dict]) -> None:
    print(f"\n=== Round-Robin standings ===\n")
    if not rr:
        return

    games_played: dict[str, int] = defaultdict(int)
    score_total: dict[str, float] = defaultdict(float)
    for r in rr:
        a, b = _solo_short(r["a"]), _solo_short(r["b"])
        games_played[a] += r["n_games"]
        games_played[b] += r["n_games"]
        score_total[a] += r["score_a"] * r["n_games"]
        score_total[b] += (1.0 - r["score_a"]) * r["n_games"]

    ranked = sorted(SINGLE_NAMES, key=lambda n: score_total[n], reverse=True)
    print(f"  {'universe':<18} {'games':>7} {'score':>8} {'pct':>7} {'~Elo':>8}")
    for n in ranked:
        g = games_played[n]
        if g == 0:
            print(f"  {n:<18} {'-':>7} {'-':>8} {'-':>7} {'-':>8}")
            continue
        pct = score_total[n] / g
        elo = elo_diff(pct, g)
        elo_str = f"{elo:+.0f}" if elo is not None else "  n/a"
        print(f"  {n:<18} {g:>7} {score_total[n]:>8.1f} {pct:>7.3f} {elo_str:>8}")


def print_multiverse_vs_best(mv: list[dict]) -> None:
    best_file = ROOT / "best_solo.txt"
    if not best_file.exists() and not mv:
        print(f"\n=== Multiverse vs best-solo: PHASE 2 not yet started ===")
        return
    best = best_file.read_text(encoding="utf-8").strip() if best_file.exists() else "?"
    print(f"\n=== Multiverse variants vs best-solo ({best}) ===\n")
    print(f"  {'variant':<14} {'W/D/L':>14} {'score':>7} {'Elo':>8} {'LOS%':>7}")
    if not mv:
        print("  no multiverse matchups completed yet")
        return
    for r in sorted(mv, key=lambda x: x["a"]):
        wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
        elo_str = f"{r['elo_diff']:+.0f}" if r["elo_diff"] is not None else "n/a"
        print(f"  {r['a']:<14} {wdl:>14} {r['score_a']:>7.3f} {elo_str:>8} {r['los_pct']:>6.1f}")


def main() -> None:
    rr = load_rr()
    mv = load_mv()
    print(f"=== EXPERIMENT 1 ANALYSIS ===")
    print(f"  Loaded: {len(rr)}/21 round-robin matchups, {len(mv)}/4 multiverse matchups")
    print_rr_matrix(rr)
    print_rr_standings(rr)
    print_multiverse_vs_best(mv)


if __name__ == "__main__":
    main()
