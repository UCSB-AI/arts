#!/usr/bin/env python3
"""
Build statistical plots (rliable-style: IQM, performance profile, probability of
improvement) from the *curated* run folder under ARTS_runs/, which
contains exactly the runs that match the paper's reported numbers.

Three figure variants:
  fig_FC_statistics_curated_all      — all tasks with full 4-method coverage
  fig_FC_statistics_curated_unsat    — drops saturated tasks (all methods near 1.0)
  fig_FC_statistics_curated_topdiff  — top-K tasks by ARTS gap over best baseline
"""
from __future__ import annotations
import os, json, glob, re, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

_REPO = Path(__file__).resolve().parents[2]
CURATED = Path(os.environ.get('ARTS_RUNS_DIR', _REPO / 'ARTS_runs'))
OUT_DIR = Path(os.environ.get('ARTS_FIG_DIR', _REPO / 'final_figures'))
SENT = {50_000_000.0, 100.0, 999.0, -1.0}

ORDER = ['Linear', 'AIRA', 'MLEvolve', 'ARTS']
COLORS = {'Linear': '#8D6E63', 'AIRA': '#2E7D32', 'MLEvolve': '#EF6C00', 'ARTS': '#5E35B1'}
LINE_WIDTH = {'Linear': 1.5, 'AIRA': 1.5, 'MLEvolve': 1.5, 'ARTS': 2.3}
PFX = {'Linear': 'linear_', 'AIRA': 'aira_', 'MLEvolve': 'mlevolve_', 'ARTS': 'llmg_'}

# (folder_relpath, label, higher_better, baseline, human_best)
TASKS = [
    ('mlgym/titanic',                'Titanic',           True,  0.766, 0.830),
    ('mlgym/cifar10L1',              'CIFAR-10',          True,  0.497, 0.994),
    ('mlgym/fmnist',                 'Fashion MNIST',     True,  0.848, 0.968),
    ('mlgym/mnli',                   'MNLI',              True,  52.51, 92.50),
    ('mlgym/lang_modeling',          'Lang. Modeling',    False, 4.673, 3.500),
    ('mlgym/battleOfSexes',          'Battle of Sexes',   True,  1.023, 1.667),
    ('mlgym/prisonersDilemma',       "Prisoner's Dilem.", True,  2.372, 3.000),
    ('mlgym/blotto',                 'Blotto',            True, -0.248, 0.500),
    ('mlgym/breakout',               'Breakout',          True,  48.82, 100.0),
    ('mlgym/mountaincar',            'MountainCar',       True,  33.79, 99.00),
    ('mlgym/metamaze',               'Meta Maze',         True,  15.73, 52.50),
    ('mlebench/spaceship_titanic',   'Spaceship Titanic', True,  0.000, 0.828),
    ('mlebench/nomad2018',           'Nomad 2018',        False, 1.000, 0.051),
    ('mlebench/jigsaw_toxic',        'Jigsaw Toxic',      True,  0.500, 0.989),
    ('mlebench/aptos',               'APTOS',             True,  0.000, 0.936),
    ('mlebench/plant_pathology',     'Plant Pathology',   True,  0.500, 0.984),
    ('mlebench/histo_cancer',        'Histo Cancer',      True,  0.500, 1.000),
    ('mlebench/vesuvius',            'Vesuvius',          True,  0.000, 0.831),
    ('mlebench/kuzushiji',           'Kuzushiji',         True,  0.000, 0.950),
    ('mlebench/hms_brain',           'HMS Brain',         False, 1.462, 0.272),
    ('mlebench/rsna',                'RSNA',              True,  0.500, 0.621),
]

# ---------------------------------------------------------------------------
# Score extraction (mirrors the main script)
# ---------------------------------------------------------------------------
def best_tree(rd, higher, leak=False):
    s = []
    for f in glob.glob(os.path.join(rd, 'nodes', '*.json')):
        try:
            v = json.load(open(f)).get('score')
            if not isinstance(v, (int, float)) or v != v: continue
            if v in SENT or abs(v) > 1e6: continue
            if leak and v >= 0.99: continue
            s.append(v)
        except Exception: pass
    if not s: return None
    return max(s) if higher else min(s)

def best_linear(rd, higher, leak=False):
    log = os.path.join(rd, 'run.log')
    if not os.path.exists(log): return None
    s = []
    rx = re.compile(r'validate\s*->\s*(-?\d+(?:\.\d+)?)')
    for line in open(log, errors='ignore'):
        m = rx.search(line)
        if m:
            try:
                v = float(m.group(1))
                if v in SENT or abs(v) > 1e6: continue
                if leak and v >= 0.99: continue
                s.append(v)
            except Exception: pass
    if not s: return None
    return max(s) if higher else min(s)

def normalize(v, baseline, human_best, higher):
    if higher:
        return (v - baseline) / (human_best - baseline)
    return (baseline - v) / (baseline - human_best)

def collect_norm_scores(folder, label, higher, baseline, human_best):
    """Return {method: [normalised final scores per run]} for one task."""
    full = CURATED / folder
    if not full.is_dir(): return {m: [] for m in ORDER}
    leak = (folder == 'mlebench/rsna')
    out = {m: [] for m in ORDER}
    for d in sorted(os.listdir(full)):
        for m in ORDER:
            if d.startswith(PFX[m]):
                ext = best_linear if m == 'Linear' else best_tree
                v = ext(str(full / d), higher, leak)
                if v is not None:
                    out[m].append(normalize(v, baseline, human_best, higher))
                break
    return out

def aggregate(task_pool):
    """Each method's array = scores from EVERY task where that method has runs.
    Rliable convention: per-method aggregation tolerates uneven task coverage.
    Coverage report printed for transparency."""
    per_method = {m: [] for m in ORDER}
    coverage = {m: [] for m in ORDER}
    for folder, label, higher, baseline, human_best in task_pool:
        scores = collect_norm_scores(folder, label, higher, baseline, human_best)
        for m in ORDER:
            if scores[m]:
                per_method[m].extend(scores[m])
                coverage[m].append(label)
    for m in ORDER:
        print(f'  {m:<10}: {len(per_method[m])} runs across {len(coverage[m])} tasks  ({", ".join(coverage[m][:6])}{"..." if len(coverage[m])>6 else ""})')
    return {m: np.array(per_method[m]) for m in ORDER}

def aggregate_paired(task_pool, m_a='ARTS', m_b='Linear'):
    """For probability-of-improvement: only tasks where BOTH methods have runs."""
    a, b = [], []
    shared_tasks = []
    for folder, label, higher, baseline, human_best in task_pool:
        s = collect_norm_scores(folder, label, higher, baseline, human_best)
        if s[m_a] and s[m_b]:
            a.extend(s[m_a]); b.extend(s[m_b]); shared_tasks.append(label)
    return np.array(a), np.array(b), shared_tasks

# ---------------------------------------------------------------------------
# Bootstrap stats
# ---------------------------------------------------------------------------
def bootstrap_iqm(scores, n_boot=2000):
    if len(scores) < 4:
        if len(scores) == 0: return 0.0, 0.0, 0.0
        m = float(np.mean(scores)); return m, m, m
    rng = np.random.default_rng(7); iqms = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        s = np.sort(sample); n = len(s); lo, hi = n // 4, n - n // 4
        iqms.append(float(np.mean(s[lo:hi])) if hi > lo else float(np.mean(s)))
    iqms = np.array(iqms)
    s = np.sort(scores); n = len(s); lo, hi = n // 4, n - n // 4
    iqm = float(np.mean(s[lo:hi])) if hi > lo else float(np.mean(s))
    return iqm, float(np.percentile(iqms, 2.5)), float(np.percentile(iqms, 97.5))

def perf_profile(scores, taus):
    return np.array([(scores > t).mean() for t in taus])

def prob_of_improvement(a, b, n_boot=2000):
    if len(a) == 0 or len(b) == 0: return 0.5, 0.5, 0.5
    rng = np.random.default_rng(11); pis = []
    for _ in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        pis.append((ai[:, None] > bi[None, :]).mean())
    pi = float((a[:, None] > b[None, :]).mean())
    return pi, float(np.percentile(pis, 2.5)), float(np.percentile(pis, 97.5))

def is_saturated(folder, label, higher, baseline, human_best,
                 max_method_spread=0.05, floor_norm=0.85):
    s = collect_norm_scores(folder, label, higher, baseline, human_best)
    if any(len(s[m]) == 0 for m in ORDER): return False
    means = [float(np.mean(s[m])) for m in ORDER]
    return (max(means) - min(means)) < max_method_spread and min(means) >= floor_norm

def task_diff(folder, label, higher, baseline, human_best):
    s = collect_norm_scores(folder, label, higher, baseline, human_best)
    if not s['ARTS']: return float('-inf')
    arts = float(np.mean(s['ARTS']))
    base = [float(np.mean(s[m])) for m in ('Linear','AIRA','MLEvolve') if s[m]]
    if not base: return float('inf')
    return arts - max(base)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def figure_statistics(fname, task_pool, subtitle=None):
    scores = aggregate(task_pool)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    fig.patch.set_facecolor('white')
    if subtitle:
        fig.suptitle(subtitle, fontsize=10, color='#555', y=1.02)

    # IQM
    methods = ORDER
    iqms = [bootstrap_iqm(scores[m]) for m in methods]
    ys = np.arange(len(methods))[::-1]
    means = [v[0] for v in iqms]; ci_lo = [v[1] for v in iqms]; ci_hi = [v[2] for v in iqms]
    bar_h = 0.55
    for y, lo, hi, mu, c in zip(ys, ci_lo, ci_hi, means, [COLORS[m] for m in methods]):
        ax1.barh(y, hi - lo, left=lo, color=c, alpha=0.55, edgecolor='none', height=bar_h, zorder=2)
        ax1.plot([mu, mu], [y - bar_h/2, y + bar_h/2], color='black', lw=1.4, zorder=4)
    ax1.set_yticks(ys); ax1.set_yticklabels([f'{m} (ours)' if m == 'ARTS' else m for m in methods])
    ax1.set_xlabel('IQM normalised score', labelpad=4)
    ax1.set_title('Aggregate performance (IQM)', fontsize=11.5, fontweight='bold', pad=6)
    ax1.set_xlim(0.3, max(1.05, max(ci_hi) + 0.05))
    ax1.axvline(1.0, color='#555', lw=0.9, ls='--')
    ax1.grid(axis='x', ls='--', lw=0.4, alpha=0.45, color='#888'); ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Performance profile with CI
    taus = np.linspace(0, 1.05, 60); rng = np.random.default_rng(13)
    for m in methods:
        if len(scores[m]) == 0: continue
        prof = perf_profile(scores[m], taus)
        boots = []
        for _ in range(500):
            sample = rng.choice(scores[m], size=len(scores[m]), replace=True)
            boots.append(perf_profile(sample, taus))
        boots = np.stack(boots, axis=0)
        lo = np.percentile(boots, 5, axis=0); hi = np.percentile(boots, 95, axis=0)
        label = f'{m} (ours)' if m == 'ARTS' else m
        ax2.plot(taus, prof, color=COLORS[m], lw=LINE_WIDTH[m], label=label, zorder=4)
        ax2.fill_between(taus, lo, hi, color=COLORS[m], alpha=0.18, linewidth=0, zorder=2)
    ax2.set_xlabel(r'Normalised score threshold $\tau$', labelpad=4)
    ax2.set_ylabel(r'Fraction of runs with score $>$ $\tau$', labelpad=4)
    ax2.set_title('Performance profile', fontsize=11.5, fontweight='bold', pad=6)
    ax2.set_xlim(0, 1.05); ax2.set_ylim(-0.02, 1.05)
    ax2.grid(ls='--', lw=0.4, alpha=0.45, color='#888'); ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    leg = ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.80, fancybox=True, handlelength=2.0)
    leg.get_frame().set_facecolor('white')

    # P(improvement) — paired: only tasks where BOTH methods have runs
    others = ['Linear', 'AIRA', 'MLEvolve']
    pis = []
    for m in others:
        a, b, shared = aggregate_paired(task_pool, 'ARTS', m)
        pis.append(prob_of_improvement(a, b))
        print(f'  P(ARTS > {m}): n_shared_tasks={len(shared)}  arts={len(a)} runs vs {m}={len(b)} runs')
    ys = np.arange(len(others))[::-1]
    means = [v[0] for v in pis]; ci_lo = [v[1] for v in pis]; ci_hi = [v[2] for v in pis]
    bar_h = 0.55
    for y, lo, hi, mu, c in zip(ys, ci_lo, ci_hi, means, [COLORS[m] for m in others]):
        ax3.barh(y, hi - lo, left=lo, color=c, alpha=0.55, edgecolor='none', height=bar_h, zorder=2)
        ax3.plot([mu, mu], [y - bar_h/2, y + bar_h/2], color='black', lw=1.4, zorder=4)
    ax3.axvline(0.5, color='#555', lw=0.9, ls='--')
    ax3.set_yticks(ys); ax3.set_yticklabels(['ARTS'] * len(others)); ax3.tick_params(axis='y', length=0)
    ax3.set_xlabel(r'P(ARTS $>$ Y)', labelpad=4)
    ax3.set_title('Probability of improvement', fontsize=11.5, fontweight='bold', pad=6)
    ax3.set_xlim(0, 1.05)
    ax3_r = ax3.twinx(); ax3_r.set_ylim(ax3.get_ylim()); ax3_r.set_yticks(ys); ax3_r.set_yticklabels(others)
    ax3_r.tick_params(axis='y', length=0); ax3_r.spines['top'].set_visible(False); ax3_r.spines['left'].set_visible(False)
    ax3.grid(axis='x', ls='--', lw=0.4, alpha=0.45, color='#888'); ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    for ext in ('pdf', 'png'):
        p = OUT_DIR / f'{fname}.{ext}'
        kw = dict(bbox_inches='tight')
        if ext == 'png': kw['dpi'] = 220
        fig.savefig(p, **kw); print(f'wrote {p}')
    plt.close(fig)

def main():
    # Inclusive pool: every task that has ARTS + at least one baseline.
    # Per-method aggregation handles uneven coverage (rliable convention);
    # P(improvement) uses paired tasks only.
    pool = []
    for t in TASKS:
        s = collect_norm_scores(*t)
        if s['ARTS'] and any(s[m] for m in ('Linear','AIRA','MLEvolve')):
            pool.append(t)
    print(f'[curated] inclusive pool ({len(pool)}/{len(TASKS)}): {[t[1] for t in pool]}')

    # F: full inclusive pool
    figure_statistics('fig_FC_statistics_curated_all', pool,
                      subtitle=f'curated runs, {len(pool)} tasks (uneven method coverage; rliable per-method aggregation)')

    # F2: non-saturated subset
    unsat = [t for t in pool if not is_saturated(*t)]
    sat = [t[1] for t in pool if is_saturated(*t)]
    print(f'[curated] saturated (excluded): {sat}')
    print(f'[curated] non-saturated:         {[t[1] for t in unsat]}')
    if unsat:
        figure_statistics('fig_FC_statistics_curated_unsat', unsat,
                          subtitle=f'non-saturated subset ({len(unsat)} of {len(pool)})')

    # F3: top-K by ARTS gap
    K = 5
    diffs = [(task_diff(*t), t) for t in pool]
    diffs.sort(key=lambda x: x[0], reverse=True)
    top = [t for d, t in diffs[:K] if d != float('-inf')]
    print('[curated] per-task ARTS−best-baseline gap:')
    for d, t in diffs:
        print(f'  {t[1]:<22}  {d:+.3f}')
    if top:
        figure_statistics('fig_FC_statistics_curated_topdiff', top,
                          subtitle=f'top-{K} tasks by ARTS gap over best baseline')

if __name__ == '__main__':
    main()
