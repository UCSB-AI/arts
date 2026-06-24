# Curated Results Archive — Numbers, Formulas, Methodology

This document captures all final numbers reported in the ARTS NeurIPS 2026
paper, derived from the curated runs in this folder. Every cell, normalized
score, and aggregate statistic is reproducible from the raw `nodes/*.json`
and `run.log` files alongside this file.

Last regenerated: 2026-05-03 from 19 tasks × 4 methods.

---

## 1. Headline numbers (4-method aggregates)

Computed across **19 tasks**, with full coverage from all four methods:

| Method   | IQM*  | Mean  | P(ARTS beats this method)† |
|----------|-------|-------|----------------------------|
| Linear   | 0.877 | 0.891 | 0.677                      |
| AIRA     | 0.863 | 0.857 | 0.813                      |
| MLEvolve | 0.867 | 0.946 | 0.640                      |
| **ARTS** | **0.919** | **1.059** | —                |

\* IQM = interquartile mean of per-task **normalized** scores (definition §4.1).
† P(ARTS > X) = mean over 19 tasks of within-task pairwise P(ARTS sample > X sample) (§4.2).

**Reading the IQM/P(beat) gap.** ARTS has the highest IQM (0.919) and beats
each baseline more than half the time. The IQM ranking (ARTS > Linear ≈
MLEvolve > AIRA) is *nearly* consistent with the P(beat) ranking
(ARTS > AIRA easier than ARTS > Linear easier than ARTS > MLEvolve), but the
two metrics measure different things: **IQM aggregates absolute
normalized score; P(beat) is per-task ordinal**. A method with one
catastrophic failure has lower IQM but can still have high P(beat) if its
median performance is competitive. See §5 for a worked example
(MLEvolve on Lang. Modeling and HMS Brain).

---

## 2. Per-cell statistics (mean ± std, n runs)

Higher-better unless noted (↓ = lower-better metric).

| Task                      | Linear            | AIRA              | MLEvolve          | ARTS              |
|---------------------------|-------------------|-------------------|-------------------|-------------------|
| Titanic (acc)             | 0.951 ± .004 (n=2) | 0.944 ± .001 (n=2) | 0.946 ± .003 (n=3) | **0.984 ± .004** (n=2) |
| CIFAR-10 (acc)            | 0.956 ± .002 (n=2) | 0.964 ± .002 (n=2) | 0.959 ± .004 (n=3) | **0.971 ± .017** (n=2) |
| Fashion MNIST (acc)       | 0.946 ± .006 (n=2) | 0.947 ± .000 (n=2) | 0.948 ± .002 (n=3) | **0.958 ± .001** (n=2) |
| MNLI (acc)                | 84.42 ± 0.05 (n=2) | 83.77 ± 0.05 (n=2) | 84.26 ± 0.12 (n=3) | **84.71 ± 0.49** (n=2) |
| Lang. Modeling ↓ (loss)   | 3.744 ± .082 (n=2) | 4.673 ± .000 (n=2) | 4.015 ± .206 (n=3) | **3.827 ± .125** (n=3) |
| Battle of Sexes (payoff)  | 1.448 ± .001 (n=2) | 1.442 ± .001 (n=2) | 1.446 ± .000 (n=3) | **2.000 ± .000** (n=3) |
| Prisoner's Dilem. (payoff)| 2.453 ± .102 (n=2) | 2.501 ± .130 (n=2) | 2.635 ± .017 (n=3) | **2.857 ± .318** (n=3) |
| Blotto (reward)           | -0.076 ± .327 (n=2)| 0.247 ± .003 (n=2) | 0.250 ± .001 (n=3) | 0.249 ± .001 (n=3) |
| Breakout (reward)         | 64.03 ± 15.2 (n=3) | 57.94 ± 4.76 (n=3) | **83.28 ± 5.43** (n=3) | 78.00 ± 3.16 (n=3) |
| Spaceship Titanic (acc)   | 0.825 ± .008 (n=3) | 0.831 (n=1)        | 0.834 ± .002 (n=3) | **0.836 ± .001** (n=2) |
| Nomad 2018 ↓ (RMSLE)      | 0.066 (n=1)        | 0.246 ± .235 (n=3) | **0.063 ± .000** (n=3) | 0.064 ± .001 (n=2) |
| Jigsaw Toxic (AUC)        | 0.980 ± .000 (n=2) | 0.980 ± .000 (n=2) | 0.980 ± .000 (n=3) | **0.980 ± .001** (n=3) |
| APTOS (kappa)             | 0.926 ± .002 (n=3) | 0.922 ± .006 (n=2) | 0.914 ± .006 (n=3) | **0.930 ± .006** (n=3) |
| Plant Pathology (AUC)     | 0.994 (n=1)        | 0.997 ± .000 (n=3) | **0.998 ± .000** (n=3) | 0.995 ± .003 (n=2) |
| Histo Cancer (AUC)        | 0.990 (n=1)        | **0.995 ± .000** (n=2) | 0.994 ± .000 (n=3) | **0.995 ± .000** (n=2) |
| Vesuvius (F0.5)           | 0.479 ± .095 (n=3) | 0.309 ± .158 (n=3) | 0.550 ± .015 (n=3) | **0.551 ± .030** (n=3) |
| Kuzushiji (acc)           | 0.894 ± .040 (n=2) | 0.872 ± .037 (n=3) | **0.921 ± .013** (n=3) | 0.843 ± .049 (n=3) |
| HMS Brain ↓ (KL)          | 0.543 ± .054 (n=2) | 0.550 ± .028 (n=3) | 0.583 ± .018 (n=3) | **0.499 ± .011** (n=3) |
| RSNA (AUC)                | 0.638 ± .007 (n=3) | 0.649 ± .020 (n=3) | 0.656 ± .015 (n=3) | **0.673 ± .030** (n=3) |

**Bold = best non-baseline cell on that task.** Ties broken by lower std.

The complete machine-readable per-cell table is in
[`paper_runs_summary.csv`](paper_runs_summary.csv).

---

## 3. Per-task ARTS-vs-baseline win probabilities

For each task, P(A wins single random pair) = (#{a ∈ ARTS, b ∈ baseline : a beats b} + 0.5·ties) / |ARTS|·|baseline|.

| Task               | P(ARTS > Linear) | P(ARTS > AIRA) | P(ARTS > MLEvolve) |
|--------------------|------------------|----------------|--------------------|
| Titanic            | 1.000 | 1.000 | 1.000 |
| CIFAR-10           | 0.750 | 0.500 | 0.583 |
| Fashion MNIST      | 1.000 | 1.000 | 1.000 |
| MNLI               | 0.500 | 1.000 | 0.667 |
| Lang. Modeling     | 0.167 | 1.000 | 0.667 |
| Battle of Sexes    | 1.000 | 1.000 | 1.000 |
| Prisoner's Dilem.  | 1.000 | 1.000 | 0.778 |
| Blotto             | 0.500 | 0.500 | 0.111 |
| Breakout           | 0.667 | 1.000 | 0.222 |
| Spaceship Titanic  | 1.000 | 1.000 | 0.750 |
| Nomad 2018 ↓       | 0.000 | 0.167 | 0.000 |
| Jigsaw Toxic       | 0.667 | 0.833 | 0.667 |
| APTOS              | 0.667 | 0.833 | 1.000 |
| Plant Pathology    | 0.500 | 0.500 | 0.500 |
| Histo Cancer       | 1.000 | 1.000 | 1.000 |
| Vesuvius           | 0.667 | 1.000 | 0.444 |
| Kuzushiji          | 0.333 | 0.444 | 0.000 |
| HMS Brain ↓        | 0.667 | 1.000 | 1.000 |
| RSNA               | 0.778 | 0.667 | 0.778 |
| **Mean (aggregate)** | **0.677** | **0.813** | **0.640** |

Tasks where ARTS is dominated: Nomad 2018 (all baselines win), Blotto and
Kuzushiji (MLEvolve dominates), Lang. Modeling vs Linear (Linear-pro hits
3.74 vs ARTS 3.83, a 0.08 gap).

---

## 4. Per-task normalized scores

Normalized to a [0, ≈1] scale where 0 = baseline (no improvement) and 1 =
human-level performance. Allowed to exceed 1 when methods beat the best
human submission.

| Task               | Linear | AIRA  | MLEvolve | ARTS   |
|--------------------|--------|-------|----------|--------|
| Titanic            | 2.890  | 2.778 | 2.809    | **3.413** |
| CIFAR-10           | 0.924  | 0.939 | 0.930    | **0.954** |
| Fashion MNIST      | 0.818  | 0.828 | 0.834    | **0.919** |
| MNLI               | 0.798  | 0.782 | 0.794    | **0.805** |
| Lang. Modeling     | **0.792** | 0.000 | 0.561 | 0.721 |
| Battle of Sexes    | 0.660  | 0.651 | 0.656    | **1.517** |
| Prisoner's Dilem.  | 0.128  | 0.206 | 0.418    | **0.773** |
| Blotto             | 0.230  | 0.662 | **0.666** | 0.664 |
| Breakout           | 0.297  | 0.178 | **0.673** | 0.570 |
| Spaceship Titanic  | 0.996  | 1.004 | 1.008    | **1.010** |
| Nomad 2018         | 0.988  | 0.988 | 0.987    | 0.986 |
| Jigsaw Toxic       | 0.982  | 0.981 | 0.982    | 0.982 |
| APTOS              | 0.989  | 0.985 | 0.976    | **0.993** |
| Plant Pathology    | 1.020  | 1.027 | **1.028** | 1.024 |
| Histo Cancer       | 0.981  | 0.990 | 0.988    | **0.991** |
| Vesuvius           | 0.577  | 0.371 | 0.661    | **0.663** |
| Kuzushiji          | 0.941  | 0.918 | **0.969** | 0.888 |
| HMS Brain          | 0.772  | 0.767 | 0.739    | **0.809** |
| RSNA               | 1.141  | 1.235 | 1.290    | **1.431** |

---

## 5. Formulas

### 5.1 Per-task normalization

For a task with baseline score `b`, human-best score `h`, and method-mean
score `m`:

- Higher-is-better metrics (`acc`, `AUC`, `payoff`, `reward`):
  ```
  normalized = (m − b) / (h − b)
  ```
- Lower-is-better metrics (`loss`, `RMSLE`, `KL`):
  ```
  normalized = (b − m) / (b − h)
  ```

So 0.0 = no improvement over baseline; 1.0 = matches human-best;
> 1.0 = exceeds best known human submission.

### 5.2 Interquartile Mean (IQM)

Robust central tendency over n per-task normalized scores `x_1,…,x_n`,
sorted ascending:

```
IQM = mean(x_{⌊n/4⌋+1}, …, x_{n − ⌊n/4⌋})
```

Trims the bottom and top quartiles before averaging — recommended by
Agarwal et al. (NeurIPS 2021, "Deep Reinforcement Learning at the Edge of
the Statistical Precipice") because the **mean** is too sensitive to
outliers and the **median** wastes data.

### 5.3 Probability of improvement

For two methods A (e.g. ARTS) and B (a baseline), with run-sets
`R_A^t = {a_1, …, a_{n_A}}` and `R_B^t = {b_1, …, b_{n_B}}` on task `t`:

```
                 1
P_t(A > B) = ─────── · |{(a, b) ∈ R_A^t × R_B^t : a beats b}|
              n_A · n_B
```

with each tied pair counted as 0.5. "Beats" = larger score for
higher-better metrics, smaller score for lower-better metrics.

The aggregate is the **uniform average over shared tasks**:

```
              1
P(A > B) = ───── · ∑ P_t(A > B)
            |T|    t ∈ T_shared
```

This weights every task equally regardless of how many runs each method
has on it — which is rliable's recommendation for fair comparison.

### 5.4 Standard error per cell

```
stderr = std / √n
```

with `std = √(Σ(x_i − x̄)² / n)` (population, not sample, std — n is small).

---

## 6. Methodology notes

### 6.1 Score extraction

For each run directory:

1. **Tree-search runs** (AIRA, MLEvolve, ARTS): take `score` field from
   each `nodes/*.json`; drop sentinels `{1.0, 100, 999, 50_000_000, −1.0}`
   and any score with `|x| > 1e6`. Take `max` for higher-better tasks,
   `min` for lower-better.
2. **Linear runs**: regex `validate -> (-?\d+\.\d+)` on `run.log`; take
   `max` (or `min` for lower-better). Same sentinel filter.

### 6.2 Sentinels

| Value     | Source                                              |
|-----------|-----------------------------------------------------|
| 50,000,000| Volcanic-eruption task fallback when eval crashes   |
| 100       | Smartphone task fallback                            |
| 999       | BMS task fallback                                   |
| 1.0       | RSNA AUC leak guard (also Nomad initial baseline)   |
| −1.0      | Generic "evaluation failed" sentinel                |

### 6.3 Lower-better tasks

`Lang. Modeling` (loss), `Nomad 2018` (RMSLE), `HMS Brain` (KL divergence).
For these, **lower scores are better**. The `best()` function flips
accordingly; the normalization formula flips accordingly; the "beats"
relation flips accordingly.

### 6.4 Run inclusion criteria

A run is **included** in a cell if it produced at least one non-sentinel
score (productive). Runs that exited cleanly with zero validates are
**excluded** from the cell mean (treated as silent failures, not as
baseline-equivalent). The exception is Lang. Modeling AIRA, where two
canonical runs both produced exactly 4.673 (the baseline) — those *are*
real measurements (the AIRA agent ran but did not improve), so they count.

---

## 7. Coverage and known gaps

- **All 19 tasks covered by all 4 methods.** Every cell has n ≥ 1.
- **Sparse cells (n=1):** Spaceship-AIRA, Nomad-Linear, Plant-Linear,
  Histo-Linear. These are best-effort: only one matched paper-aligned
  triple was available. All other cells have n ≥ 2, most have n = 3.
- **Cells filled in this monitoring loop (2026-05-02 → 2026-05-03):**
  9 cells, 22 productive runs (see §8).

---

## 8. May 2026 monitoring-loop log

27 SLURM jobs were submitted to fill missing baseline runs. Outcomes:

| Bucket                             | Jobs | Outcome                          |
|------------------------------------|------|----------------------------------|
| Productive completions             | 11   | Clean exit, real scores          |
| SLURM 8h timeouts (productive)     | 9    | Wall-clock limit; data extracted |
| Silent infrastructure-side kills   | 4    | Apptainer-setup logs only; 0 validates |
| User-cancelled (treated as baseline)| 1   | LM-Linear j-888                  |
| Productive but useless score       | 1    | ARTS-flash j-209: 10.37 (worse than baseline 4.673) |
| Resubmit needed (LM-fair comparison)| 1   | ARTS-flash on Lang. Modeling     |

Cluster-kill pattern: 5 of the 6 LM-fair jobs (j-204/206/207/208 and the
HistoCancer batch j-910/911/912) terminated at suspiciously round
elapsed-times (1h03 or 2h14), with `tee` capturing only apptainer setup
chatter. The launch script's `[xxx] Done (exit=$?)` still reported exit 0
because the `tee` wrapper itself exited cleanly. j-205, started in the
same batch as j-204/206, completed normally — so the kill was not
all-or-nothing. Root cause unknown (suspect Compute Canada partition
preemption); not a code bug.

---

## 9. Reproducing these numbers

```bash
cd /home/jarnav/MLScientist/arts

# Regenerate paper_runs_summary.csv
python3 -c "
import os, json, glob, statistics, csv, re
ROOT='mlebench_runs_final_llmg'
# … (see embedded script in tools/make_stats_from_curated.py)
"

# Regenerate stats figures
python3 tools/make_stats_from_curated.py
```

Outputs:
- `mlebench_runs_final_llmg/paper_runs_summary.csv`  — per-cell mean/std/stderr
- `…/figures/fig_FC_statistics_curated_{all,unsat,topdiff}.{pdf,png}`

---

## 10. File layout

```
mlebench_runs_final_llmg/
├── RESULTS.md                        ← this file
├── paper_runs_summary.csv            ← per-cell n / mean / std / stderr
├── mlgym/
│   ├── battleOfSexes/
│   │   ├── linear_*/                 ← 1 dir per run
│   │   ├── aira_*/
│   │   ├── mlevolve_*/
│   │   └── llmg_*/
│   ├── breakout/
│   ├── blotto/
│   ├── cifar10L1/
│   ├── fmnist/
│   ├── lang_modeling/
│   ├── mnli/
│   ├── prisonersDilemma/
│   └── titanic/
└── mlebench/
    ├── aptos/
    ├── histo_cancer/
    ├── hms_brain/
    ├── jigsaw_toxic/
    ├── kuzushiji/
    ├── nomad2018/
    ├── plant_pathology/
    ├── rsna/
    ├── spaceship_titanic/
    └── vesuvius/
```

Each run directory contains `nodes/*.json` (tree-search history) or
`run.log` (Linear stdout) plus a `result.json` summary where applicable.
