"""Generate the four figures referenced in paper.md.

  paper_fig1.png  Architecture diagram comparing Shared-Tree and Independent
                  multiverse layouts. Hand-drawn-feeling vector schematic.
  paper_fig2.png  Horizontal bar chart of Elo vs `mobility` for every
                  multiverse variant across Phases 2 and 3.
  paper_fig3.png  Vertical bar chart of per-predictor Stockfish MAE.
  paper_fig4.png  Pearson correlation heatmap of the seven universes'
                  evaluations across the 10 000-position training set.

All four are 150 DPI PNGs, white background, sans-serif typography.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "experiments" / "data" / "phase3_dataset.jsonl"
FIG_ARCH = REPO_ROOT / "paper_fig1.png"
FIG_ELO = REPO_ROOT / "paper_fig2.png"
FIG_MAE = REPO_ROOT / "paper_fig3.png"
FIG_CORR = REPO_ROOT / "paper_fig4.png"

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
    "mathtext.fontset": "dejavusans",
})

UNIVERSE_NAMES = [
    "balanced", "material_greedy", "aggression", "endgame_purist",
    "mobility", "structural", "chaos",
]


# ============================================================================
# Figure 1 — Architecture diagram
# ============================================================================

def _box(ax, x, y, w, h, text, *, fc, ec="black", fs=9, lw=0.8):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.2",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2, *, color="#444", lw=0.7):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=lw),
    )


def make_fig1_architecture() -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 6.2))

    # --- Panel A: Shared-Tree ---
    for ax, title in [(ax_a, "A. Shared-Tree Multiverse"), (ax_b, "B. Independent Multiverse")]:
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
        ax.text(2, 97, title, fontsize=12, fontweight="bold", ha="left", va="top")

    POS_FC = "#f5ead0"
    UNI_FC = "#cde0f0"
    AGG_FC = "#cce8cc"
    OUT_FC = "#f4d4d4"

    # A: position s -> N universes -> aggregator F -> single score -> alpha-beta tree
    _box(ax_a, 35, 80, 30, 8, "position $s$", fc=POS_FC)
    universe_positions = []
    for i, label in enumerate(["$V_1$", "$V_2$", "$V_3$", "$\\cdots$", "$V_N$"]):
        x = 4 + i * 19
        _box(ax_a, x, 55, 15, 9, label, fc=UNI_FC, fs=11)
        universe_positions.append((x + 7.5, 55, x + 7.5, 64))
        _arrow(ax_a, 50, 80, x + 7.5, 64)
    _box(ax_a, 25, 30, 50, 9, "Aggregator $F(V_1(s),\\ldots,V_N(s),\\phi(s))$", fc=AGG_FC, fs=10)
    for x, y, _, _ in universe_positions:
        _arrow(ax_a, x, y, 50, 39)
    _box(ax_a, 25, 10, 50, 9, "drives one $\\alpha$-$\\beta$ search tree", fc=OUT_FC, fs=10)
    _arrow(ax_a, 50, 30, 50, 19)
    ax_a.text(98, 50, "eval\\_cost = $N$\nbudget visits\n$B/N$ leaves",
              ha="right", va="center", fontsize=9, color="#666", style="italic")

    # B: position s -> N parallel (universe + tree) -> N results -> move aggregator -> final move
    _box(ax_b, 35, 80, 30, 8, "position $s$", fc=POS_FC)
    for i, label in enumerate(["$(V_1,T_1)$", "$(V_2,T_2)$", "$(V_3,T_3)$", "$\\cdots$", "$(V_N,T_N)$"]):
        x = 4 + i * 19
        _box(ax_b, x, 48, 15, 16, label + "\n\n$B/N$ nodes", fc=UNI_FC, fs=9)
        _arrow(ax_b, 50, 80, x + 7.5, 64)
        _arrow(ax_b, x + 7.5, 48, x + 7.5, 41)
        ax_b.text(x + 7.5, 36, "(move$_i$, score$_i$)", ha="center", va="center", fontsize=8, style="italic")
    _box(ax_b, 25, 22, 50, 9, "Move-Aggregator (e.g. plurality vote)", fc=AGG_FC, fs=10)
    for i in range(5):
        x = 4 + i * 19 + 7.5
        _arrow(ax_b, x, 32, 50, 31)
    _box(ax_b, 35, 6, 30, 9, "final move", fc=OUT_FC, fs=10)
    _arrow(ax_b, 50, 22, 50, 15)
    ax_b.text(98, 50, "each tree gets\n$B/N$ nodes\n$\\Rightarrow$ shallow per universe",
              ha="right", va="center", fontsize=9, color="#666", style="italic")

    plt.tight_layout()
    fig.savefig(FIG_ARCH)
    plt.close(fig)
    print(f"wrote {FIG_ARCH}")


# ============================================================================
# Figure 2 — Elo bar chart (Phase 2 + Phase 3)
# ============================================================================

PHASE2_RESULTS = [
    ("shared-5",  -53),
    ("shared-7", -191),
    ("indep-5",  -228),
    ("indep-7",  -228),
]
PHASE3_MAIN = [("shared-7-learned", -338)]
PHASE3_ABLATIONS = [
    ("minus endgame_purist", -359),
    ("minus material_greedy", -407),
    ("minus structural", -407),
    ("minus aggression", -512),
    ("minus chaos", -636),
]

PHASE_COLORS = {
    "phase2": "#3a6ea5",
    "phase3_main": "#d9822b",
    "phase3_ablation": "#a83232",
}


def make_fig2_elo() -> None:
    rows = (
        [(n, e, "phase2") for n, e in PHASE2_RESULTS]
        + [(n, e, "phase3_main") for n, e in PHASE3_MAIN]
        + [(n, e, "phase3_ablation") for n, e in PHASE3_ABLATIONS]
    )
    rows.sort(key=lambda r: r[1])  # most-negative at top
    labels, elos, phases = zip(*rows)
    colors = [PHASE_COLORS[p] for p in phases]

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    bars = ax.barh(labels, elos, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    # Place each label OUTSIDE the bar (to the left of its tip) so the bar
    # interiors stay visually clean. Right-aligned so the text grows leftward
    # away from the bar end.
    for bar, e in zip(bars, elos):
        ax.text(e - 8, bar.get_y() + bar.get_height() / 2, f"{e:+}",
                va="center", ha="right", fontsize=9)
    ax.set_xlabel("Elo difference vs. `mobility` solo  (negative = multiverse lost)")
    ax.set_title("Multiverse variants vs. best-solo (`mobility`)\n"
                 "40 games per matchup, 10 000 nodes/move",
                 loc="left")
    ax.set_xlim(-720, 80)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[k], ec="black", lw=0.4)
        for k in ("phase2", "phase3_main", "phase3_ablation")
    ]
    ax.legend(handles,
              ["Phase 2 (naive aggregator)",
               "Phase 3 (learned MLP, main)",
               "Phase 3 (learned MLP, ablation)"],
              loc="lower right", frameon=False, fontsize=9)
    fig.savefig(FIG_ELO); plt.close(fig); print(f"wrote {FIG_ELO}")


# ============================================================================
# Figure 3 — MAE bar chart
# ============================================================================

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
MAE_COLORS = {"universe": "#a8a8a8", "uniform": "#5a7d9a", "mlp": "#1c4d80"}


def make_fig3_mae() -> None:
    rows = sorted(MAE_RESULTS, key=lambda r: -r[1])
    labels, maes, kinds = zip(*rows)
    colors = [MAE_COLORS[k] for k in kinds]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(labels, maes, color=colors, edgecolor="black", linewidth=0.4)
    for bar, m in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 6, f"{m}",
                ha="center", va="bottom", fontsize=9)
    mlp_mae = next(m for n, m, _ in MAE_RESULTS if n == "MLP (ours)")
    ax.axhline(mlp_mae, color="#1c4d80", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("MAE vs. Stockfish target (cp)")
    ax.set_title("Per-predictor MAE on Stockfish 17 @ depth 12\n"
                 "(lower is better; 20 % held-out split of 10 000 positions)",
                 loc="left")
    ax.set_ylim(0, 560)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MAE_COLORS[k], ec="black", lw=0.4)
               for k in ("universe", "uniform", "mlp")]
    ax.legend(handles, ["individual universe", "uniform mean of 7", "MLP (ours)"],
              loc="upper right", frameon=False, fontsize=9)
    fig.savefig(FIG_MAE); plt.close(fig); print(f"wrote {FIG_MAE}")


# ============================================================================
# Figure 4 — Universe correlation heatmap (computed from real dataset)
# ============================================================================

def make_fig4_correlation() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {DATASET_PATH}")
    scores = {n: [] for n in UNIVERSE_NAMES}
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for n in UNIVERSE_NAMES:
                scores[n].append(row["scores"][n])
    arr = np.array([scores[n] for n in UNIVERSE_NAMES], dtype=np.float64)
    # Pearson correlation matrix
    corr = np.corrcoef(arr)

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(UNIVERSE_NAMES)))
    ax.set_xticklabels(UNIVERSE_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(UNIVERSE_NAMES)))
    ax.set_yticklabels(UNIVERSE_NAMES, fontsize=9)
    for i in range(len(UNIVERSE_NAMES)):
        for j in range(len(UNIVERSE_NAMES)):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.65 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=9)
    ax.set_title("Per-universe evaluation correlation\n"
                 "(10 000 book-playout positions, STM-relative scores)",
                 loc="left")
    fig.savefig(FIG_CORR); plt.close(fig); print(f"wrote {FIG_CORR}")


if __name__ == "__main__":
    make_fig1_architecture()
    make_fig2_elo()
    make_fig3_mae()
    make_fig4_correlation()
