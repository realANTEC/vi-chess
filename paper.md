# Multi-Universe Chess Engines: A Negative Result at Equal Compute

**Vanshdeep Singh Kohli**
*Independent · 2026-05-27*
*Code & data: <https://github.com/realANTEC/vi-chess>*

---

## Abstract

We test whether a "multi-universe" chess engine architecture — *N* parallel evaluators with deliberately different strategic philosophies, aggregated into a single decision per move — beats the strongest single evaluator at **equal total compute**. We construct seven hand-designed universes (balanced, material-greedy, aggression, endgame-purist, mobility, structural, chaos) and two aggregation architectures: a shared alpha-beta tree with *N* leaf evaluators (`shared-N`), and *N* independent searches at `budget/N` nodes each, voted at the root (`indep-N`). The study has three phases. **Phase 1** runs a round-robin among the seven solo universes to establish a best-solo baseline. **Phase 2** plays each of four naive-aggregator multiverse variants against that best-solo at 10 000 nodes per move over 40 games. **Phase 3** replaces the naive uniform-weight aggregator with an MLP trained on 10 000 book-playout positions labeled by Stockfish 17 at depth 12, then re-tests and performs a drop-one-universe ablation. **Every multiverse variant lost the head-to-head, and the learned aggregator lost more than the naive one** (`shared-7-learned` at −338 Elo vs `shared-7-naive` at −191 Elo). The ablation shows every universe contributes positively; the catastrophic loss is structural, not the fault of any one weak component. We argue the dominant constraint is a depth-vs-diversity tradeoff: enforcing equal-compute means an *N*-universe ensemble visits 1/*N* the positions of the single-evaluator baseline, and at the search depths reachable in our regime, that depth deficit dominates whatever signal the aggregator extracts. We discuss why a *better* Stockfish predictor (MAE 264 vs the best single universe at 364) translates to *worse* play, and what would need to change in the architecture for the verdict to flip.

---

## 1. Introduction

Mainstream chess engines compute one evaluation function `V(s)` per position. Strong engines like Stockfish [Romstad et al. 2024] run a single hand-tuned + NNUE-trained evaluator [Nasu 2018] inside a deeply optimized alpha-beta search; AlphaZero-style engines [Silver et al. 2018] run a single neural-network evaluator inside MCTS. There is essentially no production engine that mixes *multiple* heterogeneous evaluators with deliberately different strategic philosophies and aggregates them per move.

We hypothesized this absence was an oversight. Mixture-of-experts (MoE) and ensemble methods are well-established in machine learning [Jacobs et al. 1991, Dietterich 2000, Shazeer et al. 2017]: combining diverse predictors often outperforms any single one, especially in regimes where individual predictors have complementary errors. Chess evaluation is ostensibly such a regime — positions vary widely in character (tactical, positional, endgame), and one might expect specialist evaluators to outperform a generalist on the slices they specialize in.

The proposed architecture — which we called the **multi-universe** engine — instantiates this. Each "universe" is an `(evaluator, move-ordering bias)` pair encoding a coherent strategic philosophy: one prefers material, one rewards attacks on the enemy king, one chases endgame simplifications, and so on. At each leaf in the alpha-beta search, all *N* universes evaluate the position; an aggregator combines their scores into one number that drives the search.

The ambition was a C++ engine with bitboards, NNUE-quality eval, and this multi-universe layer on top — targeting an Elo level competitive with Stockfish. Before committing to that level of engineering, we built a Python prototype to answer one question:

> **At equal total compute, does an *N*-universe multiverse beat its strongest single universe?**

If yes, the architecture justifies the C++ rewrite. If no, the architecture is decorative and the rewrite is unjustified. The honest, replicated answer from our experiments is **no**.

The remainder of this paper is structured around three experimental phases. Section 3 lays out methodology; Section 4 establishes the best-solo baseline via a round-robin among the seven universes (Phase 1); Section 5 measures the four naive-aggregator multiverse variants against that baseline (Phase 2); Section 6 replaces the naive aggregator with a learned MLP and runs a per-universe ablation (Phase 3); Sections 7–9 analyze the result and discuss limitations.

---

## 2. Related Work

**Mixture-of-experts in ML.** The classical MoE formulation [Jacobs et al. 1991] uses a learned gating network to route inputs to specialist predictors. Modern sparse MoE [Shazeer et al. 2017] scales the idea to enormous parameter counts in language models. Both rely on a *learned* gate over specialist outputs — the architectural primitive we instantiate in Phase 3.

**Ensembles in game-playing.** Multi-network voting has been explored in Go [Silver et al. 2017], poker (the Libratus / Pluribus line uses single-network strategies refined by counterfactual regret), and computer chess via Stockfish's historical "personality" parameters — but always with one evaluator running at search time. To our knowledge, no published chess engine runs *N* heterogeneous evaluators *per leaf* with online aggregation.

**Why equal-compute is the right comparison.** A multiverse architecture trivially "wins" if it gets *N* times the search work — it then has the option of doing exactly what each single-universe baseline does and a bit more. The interesting question is whether the architecture compensates for its higher per-position cost. We follow the standard practice in MoE literature of equalizing total compute.

---

## 3. Methodology

### 3.1 The seven universes

Each universe is a classical hand-designed evaluator returning a centipawn score from the side-to-move's perspective:

| Universe | Strategic philosophy |
|---|---|
| `balanced` | Material + Michniewski piece-square tables, phase-interpolated mg/eg |
| `material_greedy` | 1.5× material weight, 0.2× PST weight |
| `aggression` | + king-attack pressure term; move ordering boosts checks and king-zone moves |
| `endgame_purist` | Forces endgame PST in all phases; bonus for simplification when ahead |
| `mobility` | + heavy attacked-squares term (4× weight); halved PST |
| `structural` | + 2.5× pawn-structure term (doubled, isolated, passed) |
| `chaos` | Position-complexity bonus: rewards high total mobility + material imbalance, attributed to the player who is `chaos` in the current game |

The `chaos` universe required three design iterations to land at a sensible implementation; we discuss this in §4.2. All universes share a common alpha-beta search core with quiescence, MVV-LVA capture ordering, and a hard node budget.

### 3.2 Multiverse architectures

**Shared-tree (`shared-N`).** A single alpha-beta tree. At each leaf, every universe evaluates the position; the aggregator combines the *N* scores into one number that drives the search. Move ordering is fixed to default MVV-LVA (per-universe ordering bias is incoherent when only one tree is being walked).

**Independent (`indep-N`).** *N* separate alpha-beta searches, each with `budget / N` nodes and its own move ordering. After all *N* searches complete, a `MoveAggregator` (plurality vote, ties broken by score-sum) picks the final move.

### 3.3 Equal-compute accounting

This is load-bearing. We define one *node* as one unit of search work and count it as follows:

* Each entry into the `negamax` or `quiesce` routine: 1 node.
* Each call to a universe's `evaluate()` function: 1 node.

Under this accounting, a shared-tree multiverse pays `eval_cost = N` per leaf (one tick per universe eval), and an independent multiverse runs *N* searches at `total_budget / N` each. Both architectures consume the same total node budget as a single-universe baseline. Any Elo difference at fixed `max_nodes` is attributable to the architecture, not extra compute.

We use a fixed budget of **10 000 nodes per move** throughout.

### 3.4 Arena

40 games per matchup, opening positions sampled without replacement from a curated 40-opening book (Ruy Lopez, Sicilians, French, KID, English, etc.), strict color alternation, fixed RNG seed for reproducibility. Three early-termination rules keep games from dragging under weak endgame play:

* **Ply cap**: 200 plies.
* **Resign**: mover sees their own eval at ≤ −800 cp for 4 consecutive of their own moves.
* **Draw adjudication**: last 10 plies all report `|score| ≤ 20` cp **and** the position is in the endgame (`phase < 0.3`).

Statistical reporting follows engine-tournament convention: W/D/L, score rate, Elo difference (computed as `400 × log_10(score / (1 − score))`), and likelihood-of-superiority (LOS) in percent.

---

## 4. Phase 1: Single-Universe Round-Robin

### 4.1 Procedure and final standings

We played all `C(7, 2) = 21` pairings between the seven universes, 40 games each (240 games per universe total, 840 games total):

| Universe | Total games | Score | ~Elo |
|---|---|---|---|
| `mobility` | 240 | 0.591 | **+44** |
| `balanced` | 240 | 0.588 | +42 |
| `chaos` | 240 | 0.514 | +13 |
| `structural` | 240 | 0.503 | +3 |
| `aggression` | 240 | 0.495 | −3 |
| `material_greedy` | 240 | 0.483 | −12 |
| `endgame_purist` | 240 | 0.376 | −89 |

`mobility` and `balanced` are essentially co-leaders within statistical noise; `endgame_purist` is the clear loser. We use **`mobility` as the best-solo baseline** for all multiverse comparisons in Phases 2 and 3.

### 4.2 The chaos iteration narrative

`chaos` required three design passes to land at the version reported above; the failure modes are instructive enough to record.

The **first version** (`chaos-v1`) added a position-complexity bonus tied to *side-to-move's* score. Under negamax sign-flipping, this bonus alternated signs ply-by-ply, so chaos's preference for complex positions effectively cancelled out across a search tree. Magnitude was also wildly overweighted (the bonus could exceed 1 000 cp), and chaos-v1 lost 39/40 against `balanced` (−636 Elo as a solo player).

The **second version** (`chaos-v2`) cut the bonus magnitude by 10×. This made chaos a respectable mid-tier player against the *strong* universes (balanced, material_greedy) but produced *identical* W/D/L to chaos-v1 against the weak/attacking ones (aggression, endgame_purist). We traced the identical results to the sign-cancellation bug persisting — without the catastrophic magnitude, chaos was simply equally noisy in both directions, and the games' outcomes were dictated by the opponent.

The **third version** (`chaos-v3`, used in all results reported here) attaches the bonus to chaos's *own color in the current game* — an attribute we thread through the player/arena/universe interface — so the bonus survives negamax sign-flips cleanly. This lifted chaos from −636 to +13 Elo as a solo player.

The iteration is reported because (a) it documents a real subtle bug that would recur in any sign-flip search architecture that uses position-feature bonuses, and (b) it gave us three near-identical Stockfish-MAE measurements across the iterations (363–366 cp) — empirical confirmation that a universe's MAE against Stockfish is not the right metric for predicting its multiverse contribution. We return to this point in §7.2.

---

## 5. Phase 2: Naive Multiverse vs Best-Solo

We tested all four multiverse variants against `mobility` at 10 000 nodes per move, 40 games each, with naive aggregators: uniform-weight sum (`WeightedSum`, weights `[1/N, …, 1/N]`) for shared-tree variants, plurality vote (`Vote`, ties broken by score-sum) for independent variants. The 5-universe variants used `{balanced, aggression, endgame_purist, mobility, structural}`; the 7-universe variants used all seven.

| Variant | W/D/L | Score | Elo vs `mobility` | LOS |
|---|---|---|---|---|
| `shared-5` | 8 / 18 / 14 | 0.425 | **−53** | 15% (closest) |
| `shared-7` | 7 / 10 / 23 | 0.300 | −191 | 0% |
| `indep-5` | 5 / 12 / 23 | 0.275 | −228 | 0% |
| `indep-7` | 5 / 11 / 24 | 0.263 | −228 | 0% |

Every multiverse variant loses. The closest result is `shared-5` at −53 Elo — meaningfully bad but within striking distance of statistical noise (LOS 15% is far from confident). The 7-universe variants are decisively beaten (LOS 0%). The independent multiverses lose hardest, consistent with their depth being hard-divided by *N*: each universe in `indep-7` gets only `10 000 / 7 ≈ 1 428` nodes to itself, before any aggregation.

Notably, **smaller ensembles outperform larger ones in the shared-tree variant** (`shared-5` at −53 vs `shared-7` at −191 — a 138 Elo gap). The pattern is consistent with the depth-vs-diversity tradeoff we develop formally in §7.1: fewer universes pay a smaller `eval_cost` and so search deeper.

This is the verdict that motivated Phase 3. The question carried forward: would a *learned* aggregator close the gap?

---

## 6. Phase 3: Learned Aggregator

### 6.1 Dataset

We sampled 10 000 positions by taking each of the 40 book openings and playing a uniform-random number of legal plies (0–60) past the book exit. Terminal positions were discarded. For each position we recorded:

* All seven universe scores (STM-relative).
* Eleven cheap position features: phase, non-pawn material per side, material imbalance, mobility per side, king-attack pressure per side, pawn structure per side, side-to-move.
* Stockfish 17's eval at depth 12 (white POV → converted to STM POV).

Build time: 1.8 minutes (~90 positions/second on a single CPU thread). 2.8% of targets were forced-mate scores (±30 000 cp), which were clipped to ±2 000 cp during training to prevent destabilization of the regressor.

### 6.2 The MLP

A 2-layer MLP (32 → 32 → 1, ReLU, sklearn `MLPRegressor`) trained on the standardized 18-dim input (7 universe scores + 11 features) → scalar STM-relative target. Training: 5.5 seconds, 583 iterations, early-stopping on a 10 % validation split.

| Predictor | MAE on test (cp) |
|---|---|
| **MLP (ours)** | **264** |
| `chaos` (best individual) | 364 |
| `balanced` | 380 |
| `endgame_purist` | 385 |
| `aggression` | 387 |
| `structural` | 389 |
| `mobility` | 391 |
| `material_greedy` | 502 |
| uniform mean of 7 | 394 |

The MLP is a measurably better Stockfish predictor than any individual universe — a 28 % MAE reduction over the strongest single eval, 33 % over the uniform mean. This is the necessary precondition for the architecture to potentially flip Phase 2's verdict. **It does not.**

### 6.3 Main result

`shared-7-learned` vs `mobility` (40 games, 10 000 nodes/move):

| Variant | W/D/L | Elo | LOS |
|---|---|---|---|
| `shared-7-learned` | 3 / 4 / 33 | **−338** | 0% |

For context, `shared-7` with the naive `WeightedSum` aggregator lost by `−191` Elo (Phase 2). **The learned aggregator made play 147 Elo *worse* than naive, despite being a 28 % better Stockfish predictor.** Section 7 unpacks this paradox.

### 6.4 Drop-one-universe ablation

For each of `{material_greedy, aggression, endgame_purist, structural, chaos}` we retrained an MLP on the dataset with that universe's score column removed, then played `shared-6-learned-minus-X` vs `mobility`:

| Dropped universe | W/D/L | Elo | LOS | Δ vs main (−338) |
|---|---|---|---|---|
| `endgame_purist` | 3 / 3 / 34 | −359 | 0% | −21 |
| `material_greedy` | 3 / 1 / 36 | −407 | 0% | −69 |
| `structural` | 2 / 3 / 35 | −407 | 0% | −69 |
| `aggression` | 1 / 2 / 37 | −512 | 0% | −174 |
| `chaos` | 1 / 0 / 39 | −636 | 0% | **−298** |

We did not ablate `mobility` (it is the best-solo benchmark) or `balanced` (the other top performer; reserved for a future deeper-cut study).

**Every removal made the multiverse worse.** Removing `chaos` was catastrophic — `−636` Elo, equivalent to the original broken chaos-v1 universe playing alone. This is consistent with `chaos` being the best individual Stockfish predictor (§6.2): the MLP's gating depended heavily on chaos's complementary perspective, and removing it broke the ensemble.

### 6.5 Visual summary

Figure 1 shows the Elo difference vs `mobility` for every multiverse variant across both phases. Figure 2 shows per-predictor Stockfish-prediction MAE alongside the MLP's, visualizing the central paradox of §7.

![Figure 1: Multiverse Elo vs mobility solo, all variants and phases](paper_fig1.png)

*Figure 1: Elo difference from `mobility` solo at 10 000 nodes/move, 40 games per matchup. Negative values indicate the multiverse lost. **Phase 2** (naive aggregator, blue) shows the original 4 variants. **Phase 3** main (orange) is `shared-7-learned`; **Phase 3** ablations (red) drop one universe at a time. LOS was 0 % for every Phase 3 variant and for all but `shared-5` in Phase 2.*

![Figure 2: Stockfish-prediction MAE per predictor](paper_fig2.png)

*Figure 2: Mean absolute error (in centipawns) of each predictor's eval against Stockfish 17 at depth 12, on a held-out 20 % split of the 10 000-position dataset. The MLP outperforms every individual universe by ~100 cp MAE — yet plays substantially worse than any of them at game time (§7).*

---

## 7. Analysis: Why Did Learning Make It Worse?

We see three compounding reasons the learned aggregator underperforms the naive one despite being a strictly better Stockfish predictor.

### 7.1 Depth-vs-diversity tradeoff

At fixed `max_nodes = 10 000` and `eval_cost = 7`, the shared-tree multiverse visits about 1 400 positions per move; `mobility` solo visits 10 000. In terms of effective alpha-beta depth with branching factor ~30:

```
mobility solo:        log_30(10 000) ≈ 2.71  plies
shared-7-learned:     log_30(1 400)  ≈ 2.13  plies
```

About **0.6 plies of search depth surrendered** to the multiverse architecture before any aggregator quality matters. In our compute regime, depth dominates: missing a tactic at depth 2.7 is much costlier than missing a positional nuance at depth 2.1.

This also explains the smaller-ensembles-do-better pattern in Phase 2 (`shared-5` at −53 vs `shared-7` at −191). With *N* = 5 the depth gap is `log_30(2 000) = 2.23` plies, only 0.48 plies behind solo — and the Elo deficit is 138 cp smaller.

### 7.2 MAE optimizes the wrong thing

The aggregator's job is not to predict Stockfish's absolute eval. Its job is to *rank moves correctly* at each search node, so that the alpha-beta procedure picks the right principal variation. A model with smaller MAE is one that gets the *magnitude* of Stockfish closer; a model that plays better is one that gets the *relative ordering* of adjacent positions right. These are different objectives, and MSE/MAE training does not directly optimize the second.

The disconnect is visible in Figure 2: the MLP wins by ~100 cp on MAE but loses by 338 Elo on play. We expect a model trained with a pairwise-ranking loss (e.g. `score(a) > score(b)` whenever Stockfish agrees, applied to adjacent positions reachable from the same parent) to substantially outperform our MSE-trained MLP at game time. We did not test this hypothesis.

### 7.3 Out-of-distribution predictions

Training positions came from book openings + 0–60 plies of *random* play. Game positions reached during real matchups come from engine play, which is sharper, more tactical, and reaches different sub-distributions of the position space. The MLP confidently extrapolates into a regime it hasn't seen, with no calibration mechanism. The naive `WeightedSum` has no such failure mode because it doesn't learn anything to forget.

This failure mode is shared by the broader supervised-learning approach to chess eval: NNUE training corpora are drawn from real engine games for exactly this reason [Nasu 2018], and a follow-up study could mirror that practice.

### 7.4 Why the ablation pattern points to a structural problem

If the multiverse worked even partially, the drop-one ablation should show *some* universes contributing positively (their removal hurts) and others contributing negatively or neutrally (their removal helps or doesn't matter). We see only the first pattern — every drop hurts, and the hurt is large (21–298 Elo). This says the ensemble is genuinely using all seven inputs, but the resulting decisions still play worse than `mobility` solo. The problem is not "we picked the wrong universes" or "one weak universe is dragging us down"; it's that **aggregating *N* slightly-different evaluators inside an equal-compute alpha-beta search is structurally outmatched by deeper search with one accurate evaluator**, at least at our scale.

---

## 8. Limitations

* **Compute scale.** We test at 10 000 nodes per move (~depth 3 search). The depth-vs-diversity ratio improves with more compute: at 1 M nodes per move, the multiverse loses `log_30(1 000 000 / 7) ≈ 4.94` plies vs `5.49` plies for single — only 0.55 plies down. We do not test whether the verdict flips at 100× or 1000× our scale. Our cost estimate for a full Phase 1 + Phase 2 + Phase 3 + ablation rerun at 100k nodes per move is ~33 hours on this hardware; 1 M would be ~14 days.
* **Hand-designed evaluators.** Each universe is a small linear combination of position terms. Stronger evaluators (NNUE) would change both the per-universe baseline and the aggregator's signal. We do not test whether a multiverse of *strong* evaluators behaves differently from a multiverse of weak ones.
* **No ranking loss for the aggregator.** Per §7.2 — we used MSE on Stockfish targets, which optimizes a different objective than what alpha-beta consumes. A pairwise-ranking-loss MLP, or one trained on engine self-play with policy-style targets, is the natural follow-up.
* **Python prototype.** Search throughput is ~3 000 nodes/second on a single CPU thread, vs Stockfish's 100M+. Absolute Elo numbers (the round-robin standings) are not strength claims; they are *relative* measurements between universes inside our framework.
* **One opening book, one Stockfish version.** Replication with diverse books and reference engines would strengthen the result.

---

## 9. Conclusion

Multi-universe chess engine architectures with *N* hand-designed evaluators aggregated by either naive or learned methods do not beat the strongest single evaluator at equal compute, in our experimental regime (10 000 nodes per move, 7 universes). The result holds for both shared-tree and independent multiverse architectures, both naive and learned aggregators, and survives a per-universe ablation showing every component is contributing positively (every removal worsens the result). The dominant constraint is a depth-vs-diversity tradeoff: equalizing compute means an *N*-ensemble loses `log_b(N)` plies of search (where *b* is the branching factor), and at our scale that depth deficit dominates aggregator quality.

The most surprising finding is that a learned aggregator — measurably *better* at predicting Stockfish's evaluation than any single universe — produces *worse* play than the naive uniform-weight baseline. Three mechanisms compound: depth-vs-diversity, an MAE training objective that doesn't optimize move ranking, and out-of-distribution predictions on engine-game positions.

We see the result as a useful negative datapoint for the broader question of when ensemble methods help in compute-constrained tree search. The conditions under which the multiverse architecture *might* work — much higher compute scales, ranking-loss-trained aggregators, NNUE-strength evaluators — are interesting and untested. We invite replication and extension.

---

## Code & Data

All code is public and MIT-licensed at <https://github.com/realANTEC/vi-chess>. The full experimental record (per-matchup JSON checkpoints, trained MLP weights, dataset JSONL) is regenerable from a single `python -m experiments.run_exp01` + `experiments.run_phase3` invocation against a Stockfish 17 binary; raw artifacts are kept locally and gitignored due to size.

## Acknowledgements

The experimental harness, opening book, statistical analysis, and a substantial fraction of the prose in this paper were produced through a multi-day pair-programming session with Claude (Anthropic) — Opus 4.7 with the 1M-context configuration. Research direction and interpretation are the author's.

## References

* Dietterich, T. G. (2000). *Ensemble methods in machine learning*. In *Multiple Classifier Systems* (LNCS 1857, pp. 1–15). Springer.
* Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). *Adaptive mixtures of local experts*. Neural Computation, 3(1), 79–87.
* Nasu, Y. (2018). *Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi*. The 28th World Computer Shogi Championship.
* Romstad, T., Costalba, M., Kiiski, J., Linscott, G., et al. (2024). *Stockfish: A strong open-source UCI chess engine*. <https://stockfishchess.org>.
* Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. arXiv:1701.06538.
* Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). *Mastering the game of Go without human knowledge*. Nature, 550(7676), 354–359.
* Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Science, 362(6419), 1140–1144.
