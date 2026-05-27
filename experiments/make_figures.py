"""Generate the two figures used in paper.md.

Figure 1 (paper_fig1.png): horizontal bar chart of Elo vs `mobility` solo for
every multiverse variant from Phase 2 + Phase 3 (main + ablations). Phases are
colour-coded; the y-axis is ordered by Elo ascending (worst at top).

Figure 2 (paper_fig2.png): vertical bar chart of per-predictor MAE against
Stockfish targets on the held-out test split. Highlights that the MLP beats
every single-universe baseline on MAE, foreshadowing the §7 paradox.

Outputs land in the repo root as PNGs at 150 DPI, transparent-free white
background, sans-serif font, paper-grade defaults.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FIG1 = REPO_ROOT / "paper_fig1.png"
OUT_FIG2 = REPO_ROOT / "paper_fig2.png"

# Consistent typography across both figures.
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# --- Figure 1 data: variant -> (elo, phase_label) ---------------------------
# Hardcoded from the recorded JSONs to make the figure script self-contained
# and the paper reproducible without re-running experiments.
PHASE2_RESULTS = [
    ("shared-5",  -53,  "Phase 2 (naive)"),
    ("shared-7", -191,  "Phase 2 (naive)"),
    ("indep-5",  -228,  "Phase 2 (naive)"),
    ("indep-7",  -228,  "Phase 2 (naive)"),
]
PHASE3_MAIN = [
    ("shared-7-learned", -338, "Phase 3 main"),
]
PHASE3_ABLATIONS = [
    ("minus endgame_purist", -359, "Phase 3 ablation"),
    ("minus material_greedy", -407, "Phase 3 ablation"),
    ("minus structural", -407, "Phase 3 ablation"),
    ("minus aggression", -512, "Phase 3 ablation"),
    ("minus chaos", -636, "Phase 3 ablation"),
]

PHASE_COLOURS = {
    "Phase 2 (naive)":     "#3a6ea5",   # muted blue
    "Phase 3 main":        "#d9822b",   # muted orange
    "Phase 3 ablation":    "#a83232",   # muted red
}


def make_fig1() -> None:
    rows = PHASE2_RESULTS + PHASE3_MAIN + PHASE3_ABLATIONS
    # Sort ascending by Elo so the worst loss is at the top of the chart and
    # the (closest-to-zero) `shared-5` is just above the axis.
    rows = sorted(rows, key=lambda r: r[1])

    labels = [r[0] for r in rows]
    elos = [r[1] for r in rows]
    phases = [r[2] for r in rows]
    colors = [PHASE_COLOURS[p] for p in phases]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    bars = ax.barh(labels, elos, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)

    # Annotate each bar with its Elo value, just inside the bar end.
    for bar, elo in zip(bars, elos):
        ax.text(
            elo + 8, bar.get_y() + bar.get_height() / 2,
            f"{elo:+}", va="center", ha="left",
            fontsize=9, color="black",
        )

    ax.set_xlabel("Elo difference vs. `mobility` solo  (negative = multiverse lost)")
    ax.set_title(
        "Multiverse variants vs. best-solo (`mobility`)\n"
        "40 games per matchup, 10 000 nodes/move",
        loc="left",
    )
    ax.set_xlim(-720, 80)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    # Legend in the lower-right where there is empty space.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOURS[p], ec="black", lw=0.4)
        for p in ["Phase 2 (naive)", "Phase 3 main", "Phase 3 ablation"]
    ]
    ax.legend(
        handles,
        ["Phase 2 (naive aggregator)", "Phase 3 (learned MLP, main)", "Phase 3 (learned MLP, ablation)"],
        loc="lower right", frameon=False, fontsize=9,
    )

    fig.savefig(OUT_FIG1)
    plt.close(fig)
    print(f"wrote {OUT_FIG1}")


# --- Figure 2 data: predictor -> MAE in centipawns --------------------------
MAE_RESULTS = [
    ("material_greedy", 502, "universe"),
    ("uniform mean",    394, "uniform"),
    ("mobility",        391, "universe"),
    ("structural",      389, "universe"),
    ("aggression",      387, "universe"),
    ("endgame_purist",  385, "universe"),
    ("balanced",        380, "universe"),
    ("chaos",           364, "universe"),
    ("MLP (ours)",      264, "mlp"),
]

MAE_COLOURS = {
    "universe": "#a8a8a8",   # neutral grey
    "uniform":  "#5a7d9a",   # muted blue
    "mlp":      "#1c4d80",   # darker blue, our model
}


def make_fig2() -> None:
    # Sort by MAE descending so the best (MLP) is at the right and stands out.
    rows = sorted(MAE_RESULTS, key=lambda r: -r[1])
    labels = [r[0] for r in rows]
    maes = [r[1] for r in rows]
    colors = [MAE_COLOURS[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, maes, color=colors, edgecolor="black", linewidth=0.4)

    # Annotate each bar with its MAE value above the bar.
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, mae + 6,
            f"{mae}", ha="center", va="bottom",
            fontsize=9, color="black",
        )

    # Faint reference line at MLP value to emphasise the gap.
    mlp_mae = next(m for n, m, _ in MAE_RESULTS if n == "MLP (ours)")
    ax.axhline(mlp_mae, color="#1c4d80", linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_ylabel("MAE vs. Stockfish target (centipawns)")
    ax.set_title(
        "Per-predictor MAE on Stockfish 17 @ depth 12\n"
        "(lower is better; 20 % held-out split of 10 000 positions)",
        loc="left",
    )
    ax.set_ylim(0, 560)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MAE_COLOURS[k], ec="black", lw=0.4)
        for k in ("universe", "uniform", "mlp")
    ]
    ax.legend(
        handles,
        ["individual universe", "uniform mean of 7", "MLP (ours)"],
        loc="upper right", frameon=False, fontsize=9,
    )

    fig.savefig(OUT_FIG2)
    plt.close(fig)
    print(f"wrote {OUT_FIG2}")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
