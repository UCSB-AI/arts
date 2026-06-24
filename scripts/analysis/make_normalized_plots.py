#!/usr/bin/env python3
"""
Normalized progression plots: (score − baseline) / (human_best − baseline) vs wall-clock time.

Y-axis: 0 = baseline performance, 1 = Kaggle competition winner.
X-axis: wall-clock time (hours, 0–8).

Tasks: Vesuvius (F0.5, higher=better) and HMS Brain (KL divergence, lower=better).
Methods: Linear, AIRA, MLEvolve, LLMG.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE    = Path('/home/jarnav/MLScientist/arts/outputs')
OUT_DIR = Path('/home/jarnav/mlscientist/Formatting_Instructions_For_NeurIPS_2026/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         11,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'axes.linewidth':    0.9,
    'legend.fontsize':   9.5,
    'legend.frameon':    True,
    'legend.framealpha': 0.92,
    'legend.edgecolor':  '#cccccc',
    'xtick.labelsize':   9.5,
    'ytick.labelsize':   9.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
    'xtick.minor.size':  2.0,
    'ytick.minor.size':  2.0,
    'lines.linewidth':   2.0,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

COLORS = {
    'Linear':   '#1565C0',   # deep blue
    'AIRA':     '#00796B',   # deep teal
    'MLEvolve': '#BF360C',   # deep rust/orange
    'LLMG':     '#5E35B1',   # deep purple  (our method)
}
FILL_ALPHA = {
    'Linear':   0.13,
    'AIRA':     0.13,
    'MLEvolve': 0.13,
    'LLMG':     0.18,
}
LINE_WIDTH = {
    'Linear':   1.6,
    'AIRA':     1.6,
    'MLEvolve': 1.6,
    'LLMG':     2.3,   # our method slightly bolder
}
ORDER = ['Linear', 'AIRA', 'MLEvolve', 'LLMG']

BUDGET_H  = 8.0
BUDGET_S  = BUDGET_H * 3600
N_GRID    = 500

# ---------------------------------------------------------------------------
# Task specs: (label, higher, baseline, human_best, linear_globs, aira_globs,
#              mlevolve_globs, llmg_globs)
# ---------------------------------------------------------------------------
# human_best: Kaggle leaderboard #1 score (from leaderboard.csv)
# Vesuvius:  #1 = 0.831399  (F0.5, higher=better)
# HMS Brain: #1 = 0.272332  (KL divergence, lower=better)

VESUVIUS = dict(
    label      = 'Vesuvius (F0.5 score)',
    higher     = True,
    baseline   = 0.0,
    human_best = 0.831399,
    linear_globs  = ['linear_mlebenchVesuvius_gemini3propreview_20260423_085423_j4752137',
                     'linear_mlebenchVesuvius_gemini3propreview_20260423_085424_j4752138'],
    aira_globs    = ['aira_mlebenchVesuvius_gemini3propreview_20260424_030723_j4754503',
                     'aira_mlebenchVesuvius_gemini3propreview_20260424_030723_j4754504'],
    mlevolve_globs= ['mlevolve_v1_mlebenchVesuvius_gemini3propreview_20260425_213726_j4761561',
                     'mlevolve_v2_mlebenchVesuvius_gemini3propreview_20260426_022224_j4763343',
                     'mlevolve_v3_mlebenchVesuvius_gemini3propreview_20260426_022223_j4763347'],
    llmg_globs    = ['llmg_C_vs_extsamp_C_v2_mlebenchVesuvius_o3_gemini3propreview_20260427_183924_j4775756',
                     'llmg_C_vs_extsamp_C_v3_mlebenchVesuvius_o3_gemini3propreview_20260427_183924_j4775757'],
)

HMS = dict(
    label      = 'HMS Brain (KL divergence)',
    higher     = False,
    baseline   = 1.4618,
    human_best = 0.272332,
    linear_globs  = ['linear_mlebenchHMSBrain_gemini3propreview_20260425_034725_j4757792',
                     'linear_mlebenchHMSBrain_gemini3propreview_20260425_034726_j4757793'],
    aira_globs    = ['aira_mlebenchHMSBrain_gemini3propreview_20260424_030823_j4754507',
                     'aira_mlebenchHMSBrain_gemini3propreview_20260424_030823_j4754508'],
    mlevolve_globs= ['mlevolve_v1_mlebenchHMSBrain_gemini3propreview_20260425_213725_j4761562',
                     'mlevolve_v2_mlebenchHMSBrain_gemini3propreview_20260426_022224_j4763344',
                     'mlevolve_v3_mlebenchHMSBrain_gemini3propreview_20260426_022223_j4763348'],
    llmg_globs    = ['llmg_C_vs_C_v1_mlebenchHMSBrain_o3_gemini3propreview_20260427_141721_j4773982',
                     'llmg_C_vs_C_v2_mlebenchHMSBrain_o3_gemini3propreview_20260427_141721_j4773983',
                     'llmg_C_vs_C_v3_mlebenchHMSBrain_o3_gemini3propreview_20260427_141721_j4773984'],
)

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _t0_from_dirname(name: str) -> float | None:
    import datetime as _dt
    m = re.search(r'(\d{8})_(\d{6})', name)
    if not m:
        return None
    try:
        return _dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
    except Exception:
        return None


def extract_tree_run(run_dir: Path) -> list[tuple[float, float]]:
    nodes_dir = run_dir / 'nodes'
    if not nodes_dir.is_dir():
        return []
    t0 = _t0_from_dirname(run_dir.name)
    if t0 is None:
        mtimes = [f.stat().st_mtime for f in nodes_dir.glob('*.json')]
        t0 = min(mtimes) if mtimes else run_dir.stat().st_mtime
    events = []
    for f in sorted(nodes_dir.glob('*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = d.get('score')
        if s is None:
            continue
        try:
            s = float(s)
        except Exception:
            continue
        if not math.isfinite(s):
            continue
        t = max(0.0, f.stat().st_mtime - t0)
        events.append((t, s))
    events.sort()
    return events


_LINEAR_RX = re.compile(r'validate\s*->\s*(-?\d+(?:\.\d+)?)(?:\s*\(best=(-?\d+(?:\.\d+)?)\))?')


def extract_linear_run(run_dir: Path) -> list[tuple[float, float]]:
    log = run_dir / 'run.log'
    if not log.is_file():
        return []
    steps = []
    with open(log, errors='ignore') as fp:
        for i, line in enumerate(fp):
            m = _LINEAR_RX.search(line)
            if m:
                try:
                    sc = float(m.group(1))
                except Exception:
                    continue
                if math.isfinite(sc):
                    steps.append((i, sc))
    if not steps:
        return []
    t0 = _t0_from_dirname(run_dir.name)
    t_end = log.stat().st_mtime
    if t0 is None:
        t0 = log.stat().st_ctime
    dur = max(1.0, t_end - t0)
    max_i = max(s[0] for s in steps)
    return sorted((dur * (i / max(1, max_i)), sc) for i, sc in steps)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def running_best(events: list[tuple[float, float]], higher: bool) -> list[tuple[float, float]]:
    best = None
    out = []
    for t, s in events:
        if best is None or (higher and s > best) or (not higher and s < best):
            best = s
        out.append((t, best))
    return out


def resample(events: list[tuple[float, float]], grid: np.ndarray) -> np.ndarray:
    if not events:
        return np.full_like(grid, np.nan)
    ts = np.array([e[0] for e in events])
    ys = np.array([e[1] for e in events])
    out = np.full_like(grid, np.nan, dtype=float)
    idx = np.searchsorted(ts, grid, side='right') - 1
    mask = idx >= 0
    out[mask] = ys[idx[mask]]
    return out


def aggregate(curves: list[list[tuple[float, float]]], higher: bool):
    grid = np.linspace(0, BUDGET_S, N_GRID)
    mats = np.stack([resample(c, grid) for c in curves], axis=0)
    mean = np.nanmean(mats, axis=0)
    std  = np.nanstd(mats,  axis=0)
    valid = np.sum(~np.isnan(mats), axis=0) >= 1
    mean = np.where(valid, mean, np.nan)
    std  = np.where(valid, std,  np.nan)
    return grid, mean, std


def smooth(x, y, monotone: int = 0):
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return x[mask], y[mask]
    xm, ym = x[mask], y[mask]
    _, uidx = np.unique(xm, return_index=True)
    xm, ym  = xm[uidx], ym[uidx]
    if monotone > 0:
        ym = np.maximum.accumulate(ym)
    elif monotone < 0:
        ym = np.minimum.accumulate(ym)
    xs = np.linspace(xm[0], xm[-1], 500)
    try:
        ys = PchipInterpolator(xm, ym, extrapolate=False)(xs)
    except Exception:
        return xm, ym
    valid = ~np.isnan(ys)
    return xs[valid], ys[valid]


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
def normalize(score: float | np.ndarray, task: dict) -> float | np.ndarray:
    b = task['baseline']
    h = task['human_best']
    if task['higher']:
        return (score - b) / (h - b)
    else:
        return (b - score) / (b - h)


# ---------------------------------------------------------------------------
# Plot one panel
# ---------------------------------------------------------------------------
def plot_panel(ax, task: dict, show_legend: bool = True):
    higher = task['higher']
    b      = task['baseline']

    method_dirs = {
        'Linear':   [BASE / g for g in task['linear_globs']],
        'AIRA':     [BASE / g for g in task['aira_globs']],
        'MLEvolve': [BASE / g for g in task['mlevolve_globs']],
        'LLMG':     [BASE / g for g in task['llmg_globs']],
    }

    # -- collect running-best curves, normalized, with baseline anchor + plateau
    best_curves: dict[str, list[list[tuple[float, float]]]] = {m: [] for m in ORDER}
    for method, dirs in method_dirs.items():
        extractor = extract_linear_run if method == 'Linear' else extract_tree_run
        for d in dirs:
            if not d.is_dir():
                print(f'  [warn] missing: {d}')
                continue
            events = extractor(d)
            if not events:
                continue
            rb = running_best(events, higher)
            if not rb:
                continue
            # anchor at t=0 with baseline score
            rb = [(0.0, b)] + [(t, s) for t, s in rb if t > 0]
            # extend to budget
            rb.append((BUDGET_S, rb[-1][1]))
            # normalize
            rb_norm = [(t, normalize(s, task)) for t, s in rb]
            best_curves[method].append(rb_norm)

    # -- human winner reference line
    ax.axhline(1.0, color='#555555', lw=1.0, ls='--', zorder=1)
    ax.text(BUDGET_H - 0.05, 1.02, 'Kaggle winner', ha='right', va='bottom',
            fontsize=8.5, color='#555555', style='italic')

    # -- baseline reference line
    ax.axhline(0.0, color='#aaaaaa', lw=0.7, ls=':', zorder=1)

    # -- plot each method
    for method in ORDER:
        curves = best_curves[method]
        if not curves:
            continue
        color  = COLORS[method]
        lw     = LINE_WIDTH[method]
        falpha = FILL_ALPHA[method]
        grid, mean, std = aggregate(curves, higher=True)   # normalized → always higher=True
        grid_h = grid / 3600.0
        mdir   = 1   # normalized curves are non-decreasing
        xs, ys_mean = smooth(grid_h, mean, monotone=mdir)
        _,  ys_std  = smooth(grid_h, std,  monotone=0)
        # trim std to same xs
        ys_std = np.interp(xs, grid_h[~np.isnan(std)], std[~np.isnan(std)],
                           left=np.nan, right=np.nan) if (~np.isnan(std)).sum() > 1 else np.zeros_like(xs)
        valid = ~(np.isnan(ys_mean) | np.isnan(ys_std))
        xs_v, ym_v, ys_v = xs[valid], ys_mean[valid], ys_std[valid]
        label = f'LLMG (ours)' if method == 'LLMG' else method
        ax.plot(xs_v, ym_v, color=color, lw=lw, label=label, zorder=4)
        ax.fill_between(xs_v, ym_v - ys_v, ym_v + ys_v,
                        color=color, alpha=falpha, linewidth=0, zorder=3)
        # individual run traces
        for curve in curves:
            if not curve:
                continue
            t_arr = np.array([e[0] for e in curve]) / 3600.0
            y_arr = np.array([e[1] for e in curve])
            ax.plot(t_arr, y_arr, color=color, lw=0.5, alpha=0.28, zorder=2)

    ax.set_xlabel('Wall-clock time (hours)', labelpad=4)
    ax.set_ylabel('Normalized score', labelpad=4)
    ax.set_title(task['label'], fontsize=12, fontweight='bold', pad=8)
    ax.set_xlim(0, BUDGET_H)
    ax.set_ylim(-0.08, 1.18)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax.grid(which='major', ls='--', lw=0.4, alpha=0.45, color='#888888')
    ax.grid(which='minor', ls=':',  lw=0.3, alpha=0.25, color='#aaaaaa')
    if show_legend:
        ax.legend(loc='upper left', fontsize=9.5, handlelength=2.0, handleheight=0.8)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    fig.patch.set_facecolor('white')

    for i, (ax, task) in enumerate(zip(axes, [VESUVIUS, HMS])):
        ax.set_facecolor('#fafafa')
        plot_panel(ax, task, show_legend=(i == 0))
        if i > 0:
            ax.set_ylabel('')

    for ext in ('pdf', 'png'):
        p = OUT_DIR / f'normalized_progression.{ext}'
        kw = dict(bbox_inches='tight')
        if ext == 'png':
            kw['dpi'] = 220
        fig.savefig(p, **kw)
        print(f'wrote {p}')

    plt.close(fig)


if __name__ == '__main__':
    main()
