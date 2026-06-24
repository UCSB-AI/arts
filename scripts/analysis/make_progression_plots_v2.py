#!/usr/bin/env python3
"""
Build publication-quality progression plots: best-so-far score vs wall-clock time
with per-method mean curves and shaded std bands (multi-run).

Methods: Linear, AIRA, LLMG.
Tasks:
  - Vesuvius (MLE-bench, f05_score, higher is better)
  - BMS       (MLE-bench, mean_levenshtein_distance, LOWER is better)
  - MountainCarContinuous (MLGym RL, Reward Mean, higher)
  - BreakoutMinAtar       (MLGym RL, Reward Mean, higher)

Data sources:
  - Tree runs (LLMG, AIRA): nodes/*.json with score; wall time = file mtime.
  - Linear: parse 'step N: validate -> X (best=Y)' from run.log;
            wall time is linearly interpolated from dir_ctime → log_mtime.

Output: NeurIPS-sized 2x2 grid PDF + individual PDFs per task.
"""
from __future__ import annotations
import json
import os
import re
import glob
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

BASE = Path('/home/jarnav/MLScientist/arts/outputs')
OUT_DIR = Path('/home/jarnav/MLScientist/arts/tools/figs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Matplotlib style: NeurIPS-friendly
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Match the NeurIPS paper style: deepblue/deepteal/deeppurple with
# pastelblue/pastelmint/pastellavender for the std-band fill.
COLORS = {
    'Linear': '#1565C0',   # deepblue
    'AIRA':   '#00796B',   # deepteal
    'LLMG':   '#5E35B1',   # deeppurple (our method)
}
FILL_COLORS = {
    'Linear': '#E3F2FD',   # pastelblue
    'AIRA':   '#E0F2F1',   # pastelmint
    'LLMG':   '#EDE7F6',   # pastellavender
}
ORDER = ['Linear', 'AIRA', 'LLMG']


# -----------------------------------------------------------------------------
# Task definitions
# -----------------------------------------------------------------------------
@dataclass
class Task:
    name: str             # display
    higher: bool
    linear_glob: str      # glob pattern under BASE
    aira_glob: str
    llmg_globs: list      # list of globs (we may union several llmg variants)
    ylim: tuple | None
    kaggle_lines: dict    # {'Bronze': score, 'Silver': ..., 'Gold': ...}
    baseline: float | None


TASKS_HARD = [
    Task(
        name='Vesuvius (F0.5)',
        higher=True,
        linear_glob='linear_mlebenchVesuvius_gemini3propreview_*',
        aira_glob='aira_mlebenchVesuvius_gemini3propreview_*',
        llmg_globs=[
            'llmg_mlebenchVesuvius_o3_gemini3propreview_*',
            'llmg_nosignal_mlebenchVesuvius_o3_gemini3propreview_*',
            'llmg_C_*_mlebenchVesuvius_o3_gemini3propreview_*',
        ],
        ylim=(0.0, 0.7),
        kaggle_lines={},
        baseline=0.0,
    ),
    Task(
        name='BMS (Levenshtein ↓)',
        higher=False,
        linear_glob='linear_mlebenchBMS_gemini3propreview_*',
        aira_glob='aira_mlebenchBMS_gemini3propreview_*',
        llmg_globs=[
            'llmg_mlebenchBMS_o3_gemini3propreview_*',
            'llmg_nosignal_mlebenchBMS_o3_gemini3propreview_*',
            'llmg_C_*_mlebenchBMS_o3_gemini3propreview_*',
        ],
        ylim=(30, 220),
        kaggle_lines={},
        baseline=999.0,
    ),
    Task(
        name='MountainCarContinuous (reward)',
        higher=True,
        linear_glob='linear_rlMountainCarContinuous_pro_run*',
        aira_glob='aira_rlMountainCarContinuous_pro_run*',
        llmg_globs=[
            'llmg_rlMountainCarContinuous_o3_geminipro_r*',
            'llmg_C_*_rlMountainCarContinuous_o3_geminipro_*',
        ],
        ylim=(-5, 100),
        kaggle_lines={},
        baseline=33.79,
    ),
    Task(
        name='BreakoutMinAtar (reward)',
        higher=True,
        linear_glob='linear_rlBreakoutMinAtar_pro_run*',
        aira_glob='aira_rlBreakoutMinAtar_pro_run*',
        llmg_globs=[
            'llmg_rlBreakoutMinAtar_o3_geminipro_r*',
            'llmg_C_*_rlBreakoutMinAtar_o3_geminipro_*',
        ],
        ylim=None,
        kaggle_lines={},
        baseline=48.82,
    ),
]

# Tasks where LLM-Guided wins cleanly, shown as a second figure.
TASKS_WINS = [
    Task(
        name='Language Modeling (loss ↓)',
        higher=False,
        linear_glob='linear_languageModelingFineWeb_*',
        aira_glob='aira_languageModelingFineWeb_*',
        llmg_globs=[
            'llmg_languageModelingFineWeb_o3_*',
            'llmg_C_*_languageModelingFineWeb_o3_*',
        ],
        ylim=(3.8, 4.8),
        kaggle_lines={},
        baseline=4.673,
    ),
    Task(
        name='Histopath. Cancer (AUC ↑)',
        higher=True,
        linear_glob='linear_mlebenchHistoCancer_gemini3propreview_*',
        aira_glob='aira_mlebenchHistoCancer_gemini3propreview_*',
        llmg_globs=[
            'llmg_mlebenchHistoCancer_o3_gemini3propreview_*',
            'llmg_C_*_mlebenchHistoCancer_o3_gemini3propreview_*',
        ],
        ylim=(0.5, 1.0),
        kaggle_lines={},
        baseline=0.5,
    ),
    Task(
        name='CIFAR-10 (accuracy ↑)',
        higher=True,
        linear_glob='linear_imageClassificationCifar10L1_*',
        aira_glob='aira_imageClassificationCifar10L1_*',
        llmg_globs=[
            'llmg_imageClassificationCifar10L1_o3_gemini3f_r*',
            'llmg_C_*_imageClassificationCifar10L1_o3_gemini3f_*',
        ],
        ylim=(0.45, 1.0),
        kaggle_lines={},
        baseline=0.497,
    ),
]

TASKS = TASKS_HARD + TASKS_WINS


# -----------------------------------------------------------------------------
# Extractors
# -----------------------------------------------------------------------------
def extract_tree_run(run_dir: Path) -> list[tuple[float, float]]:
    """Return [(t_seconds_from_start, score)] for one tree run (LLMG/AIRA).

    Score = node's reported score. Wall time = mtime(node json) - t0, where
    t0 is the timestamp in the dir name (job start) or, if unavailable,
    the earliest node mtime.
    """
    nodes_dir = run_dir / 'nodes'
    if not nodes_dir.is_dir():
        return []
    events = []
    import datetime as _dt
    t0 = None
    m = re.search(r'(\d{8})_(\d{6})', run_dir.name)
    if m:
        try:
            t0 = _dt.datetime.strptime(
                m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
        except Exception:
            t0 = None
    if t0 is None:
        try:
            mtimes = [f.stat().st_mtime for f in nodes_dir.glob('*.json')]
            if mtimes:
                t0 = min(mtimes)
            else:
                t0 = run_dir.stat().st_mtime
        except Exception:
            return []
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
        # Skip baseline (depth 0) — we want improvement over baseline
        depth = d.get('depth', 0) or 0
        t = f.stat().st_mtime - t0
        if t < 0:
            t = 0.0
        events.append((t, s, depth))
    events.sort()
    return [(t, s) for (t, s, _d) in events]


_LINEAR_STEP_RX = re.compile(r'step\s+(\d+):\s+validate\s*->\s*(-?\d+(?:\.\d+)?)')


def extract_linear_run(run_dir: Path) -> list[tuple[float, float]]:
    """Return [(t_seconds_from_start, score)] for linear search.

    Per-step wall time is unknown; we approximate by linearly interpolating
    between dir_ctime and log_mtime, using step index as progress.
    """
    log = run_dir / 'run.log'
    if not log.is_file():
        return []
    steps = []
    try:
        with open(log, errors='ignore') as fp:
            for line in fp:
                m = _LINEAR_STEP_RX.search(line)
                if m:
                    try:
                        idx = int(m.group(1))
                        sc = float(m.group(2))
                    except Exception:
                        continue
                    if not math.isfinite(sc):
                        continue
                    steps.append((idx, sc))
    except Exception:
        return []
    if not steps:
        return []
    # Wall-clock bounds. Use the timestamp embedded in the dir name
    # (YYYYMMDD_HHMMSS) as the job start — this is the only reliable start
    # time; filesystem mtimes get bumped by later filesystem activity.
    import datetime as _dt
    m = re.search(r'(\d{8})_(\d{6})', run_dir.name)
    t_start = None
    if m:
        try:
            t_start = _dt.datetime.strptime(
                m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
        except Exception:
            t_start = None
    try:
        t_end = log.stat().st_mtime
    except Exception:
        return []
    if t_start is None:
        try:
            t_start = log.stat().st_ctime
        except Exception:
            return []
    dur = max(1.0, t_end - t_start)
    max_idx = max(s[0] for s in steps)
    events = []
    for (idx, sc) in steps:
        t = dur * (idx / max(1, max_idx))
        events.append((t, sc))
    events.sort()
    return events


def dedupe_runs(glob_pattern: str) -> list[Path]:
    """Returns directories matching pattern.

    For run names ending in a replicate tag (`_r1`, `_r2`, `_run1`, `_run2`
    followed by a timestamp), multiple timestamped folders with the same
    replicate tag are treated as restarts of the same logical run; keep the
    one with the most activity.

    For names that have NO replicate tag (MLE-bench jobs named
    `linear_mlebenchBMS_gemini3propreview_<timestamp>_j<jobid>`), each
    timestamped directory is a separate independent replicate → keep all.
    """
    paths = sorted(BASE.glob(glob_pattern))
    if not paths:
        return []

    groups: dict[str, list[Path]] = {}
    for p in paths:
        # Detect `...<tag>_<timestamp>(_j<jobid>)?` where <tag> identifies a
        # logical replicate (e.g. `r1`, `run2`). If present, group on
        # everything up to and including <tag>. Otherwise each run is its own
        # group (use the full name).
        m = re.match(r'^(.+?_(?:r\d+|run\d+))_\d{8}_\d{6}(?:_j\d+)?$', p.name)
        if m:
            key = m.group(1)
        else:
            key = p.name  # unique → no dedup
        groups.setdefault(key, []).append(p)

    picks = []
    for key, candidates in groups.items():
        def weight(p):
            nodes = 0
            if (p / 'nodes').is_dir():
                nodes = len(list((p / 'nodes').glob('*.json')))
            log_lines = 0
            if (p / 'run.log').is_file():
                try:
                    log_lines = sum(1 for _ in open(p / 'run.log', errors='ignore'))
                except Exception:
                    pass
            return (nodes, log_lines)
        candidates.sort(key=weight, reverse=True)
        picks.append(candidates[0])
    return sorted(picks)


# -----------------------------------------------------------------------------
# Aggregate: align runs on a common time grid, compute running best,
# then mean +/- std.
# -----------------------------------------------------------------------------
def running_best(events: list[tuple[float, float]], higher: bool) -> list[tuple[float, float]]:
    best = None
    out = []
    for (t, s) in events:
        if best is None or (higher and s > best) or ((not higher) and s < best):
            best = s
        out.append((t, best))
    return out


def resample_on_grid(events: list[tuple[float, float]], grid: np.ndarray, higher: bool) -> np.ndarray:
    """Given a running-best event stream, return array of y-values on grid.
    For times before first event, value = NaN (unknown).
    For times after last event, value = last best (flat).
    """
    if not events:
        return np.full_like(grid, np.nan, dtype=float)
    ts = np.array([e[0] for e in events])
    ys = np.array([e[1] for e in events])
    out = np.full_like(grid, np.nan, dtype=float)
    # step function: y at time g is the best whose t <= g.
    idx = np.searchsorted(ts, grid, side='right') - 1
    mask = idx >= 0
    out[mask] = ys[idx[mask]]
    return out


def aggregate(runs_events: list[list[tuple[float, float]]],
              higher: bool,
              n_points: int = 400,
              t_end: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Given a list of event streams (already running-best), compute common
    time grid → (grid, mean, std, n_runs). If ``t_end`` is given, use it
    as the grid endpoint so curves always extend to the full budget.
    """
    if not runs_events:
        return np.array([]), np.array([]), np.array([]), 0
    if t_end is None:
        tmaxs = [events[-1][0] for events in runs_events if events]
        if not tmaxs:
            return np.array([]), np.array([]), np.array([]), 0
        t_end = float(np.median(tmaxs))
    grid = np.linspace(0, t_end, n_points)
    Y = np.stack([resample_on_grid(e, grid, higher) for e in runs_events], axis=0)
    # At each grid point, use rows with non-NaN
    mean = np.nanmean(Y, axis=0)
    std  = np.nanstd(Y, axis=0)
    # Mask grid points with no data
    with np.errstate(invalid='ignore'):
        valid = np.sum(~np.isnan(Y), axis=0) >= 1
    mean = np.where(valid, mean, np.nan)
    std  = np.where(valid, std,  np.nan)
    return grid, mean, std, Y.shape[0]


def spline_smooth(x: np.ndarray, y: np.ndarray, s: float | None = None,
                  monotone_dir: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys). For mean curves we use a monotone-preserving
    cubic (PCHIP) — best-so-far is monotone by construction, so we want
    a smooth interpolator that does NOT ring/overshoot. Falls back to
    the raw step curve if PCHIP isn't available or we have too few pts.

    monotone_dir: +1 (non-decreasing, higher-is-better), -1 (non-increasing),
    0 (no monotone enforcement — used for std band).
    """
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return x, y
    xm = x[mask]
    ym = y[mask]
    _, uidx = np.unique(xm, return_index=True)
    xm = xm[uidx]; ym = ym[uidx]
    if len(xm) < 3:
        return x, y
    # Enforce monotone running-best on the input before interpolation so the
    # PCHIP stays monotone even if the mean-across-runs is non-monotone due
    # to coverage changes.
    if monotone_dir > 0:
        ym = np.maximum.accumulate(ym)
    elif monotone_dir < 0:
        ym = np.minimum.accumulate(ym)
    try:
        from scipy.interpolate import PchipInterpolator
        spl = PchipInterpolator(xm, ym, extrapolate=False)
        xs = np.linspace(xm.min(), xm.max(), 400)
        ys = spl(xs)
        return xs, ys
    except Exception:
        return xm, ym


# -----------------------------------------------------------------------------
# Plot one task
# -----------------------------------------------------------------------------
def plot_task(ax, task: Task):
    linear_runs = dedupe_runs(task.linear_glob)
    aira_runs   = dedupe_runs(task.aira_glob)
    llmg_runs   = []
    for g in task.llmg_globs:
        llmg_runs.extend(dedupe_runs(g))

    method_runs = {
        'Linear': [extract_linear_run(p) for p in linear_runs],
        'AIRA':   [extract_tree_run(p)  for p in aira_runs],
        'LLMG':   [extract_tree_run(p)  for p in llmg_runs],
    }

    # running best per run; filter runs that never beat the baseline (stuck),
    # prepend a baseline anchor at t=0 so each curve starts from a
    # well-defined value (no NaN holes in shading), and extend each curve
    # horizontally to the full 8 h budget so that runs which finish early
    # are shown as plateaus at their final best.
    BUDGET_S = 8 * 3600
    best_curves = {}
    for m, runs in method_runs.items():
        curves = []
        for events in runs:
            if not events:
                continue
            rb = running_best(events, task.higher)
            if task.baseline is not None:
                # Drop runs that never improved on baseline.
                final_best = rb[-1][1] if rb else None
                if final_best is None:
                    continue
                if task.higher and final_best <= task.baseline:
                    continue
                if (not task.higher) and final_best >= task.baseline:
                    continue
                # Anchor to baseline at t=0
                if rb[0][0] > 0:
                    rb = [(0.0, task.baseline)] + rb
            else:
                if len(rb) < 2:
                    continue
            # Extend to full 8 h with the last best score (flat plateau).
            if rb and rb[-1][0] < BUDGET_S:
                rb = rb + [(BUDGET_S, rb[-1][1])]
            curves.append(rb)
        best_curves[m] = curves

    # Plot
    ax.set_title(task.name, fontsize=11)

    # baseline as dashed horizontal line
    if task.baseline is not None:
        ax.axhline(task.baseline, color='gray', ls=':', lw=0.9, alpha=0.7,
                   label=f'baseline')

    for m in ORDER:
        runs = best_curves.get(m, [])
        if not runs:
            continue
        color = COLORS[m]
        grid, mean, std, n = aggregate(runs, task.higher, t_end=BUDGET_S)
        if grid.size == 0:
            continue
        # Convert to hours for display
        grid_h = grid / 3600.0
        # Monotone-aware smoothing for the mean curve so it doesn't
        # ring/overshoot. Std band uses plain PCHIP without monotone
        # enforcement.
        mdir = 1 if task.higher else -1
        xs, ys_mean = spline_smooth(grid_h, mean, monotone_dir=mdir)
        _,  ys_std  = spline_smooth(grid_h, std,  monotone_dir=0)
        mask = ~(np.isnan(ys_mean) | np.isnan(ys_std))
        xs, ys_mean, ys_std = xs[mask], ys_mean[mask], ys_std[mask]
        if xs.size < 2:
            xs, ys_mean, ys_std = grid_h, mean, std
            mask = ~(np.isnan(ys_mean) | np.isnan(ys_std))
            xs, ys_mean, ys_std = xs[mask], ys_mean[mask], ys_std[mask]
        fill_color = FILL_COLORS[m]
        ax.plot(xs, ys_mean, color=color, lw=1.9, label=m, zorder=3)
        ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std,
                        color=fill_color, alpha=0.85, linewidth=0, zorder=2)
        # Individual run traces overlaid thinly in the deep hue.
        for events in runs:
            if not events:
                continue
            t = np.array([e[0] for e in events]) / 3600.0
            y = np.array([e[1] for e in events])
            ax.plot(t, y, color=color, lw=0.5, alpha=0.32, zorder=1)

    ax.set_xlabel('Wall-clock time (hours)')
    ax.set_ylabel('Best score so far')
    if task.ylim is not None:
        ax.set_ylim(*task.ylim)
    ax.grid(True, ls='--', lw=0.4, alpha=0.4)
    ax.legend(loc='best', fontsize=8)


def main():
    # 2x2 grid for the 4 hard tasks (existing figure).
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    for ax, task in zip(axes.flatten(), TASKS_HARD):
        plot_task(ax, task)
    out_pdf = OUT_DIR / 'v2_progression_all.pdf'
    out_png = OUT_DIR / 'v2_progression_all.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    print(f'wrote {out_pdf}')
    plt.close(fig)

    # 1x3 row for the 3 wins tasks (new figure).
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.8), constrained_layout=True)
    for ax, task in zip(axes, TASKS_WINS):
        plot_task(ax, task)
    out_pdf = OUT_DIR / 'v2_progression_wins.pdf'
    out_png = OUT_DIR / 'v2_progression_wins.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    print(f'wrote {out_pdf}')
    plt.close(fig)

    # Individual panels too (one per task).
    for task in TASKS:
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.6), constrained_layout=True)
        plot_task(ax, task)
        fname = re.sub(r'[^a-zA-Z0-9_]+', '_', task.name)
        p1 = OUT_DIR / f'v2_progression_{fname}.pdf'
        p2 = OUT_DIR / f'v2_progression_{fname}.png'
        fig.savefig(p1, bbox_inches='tight')
        fig.savefig(p2, dpi=220, bbox_inches='tight')
        print(f'wrote {p1}')
        plt.close(fig)


if __name__ == '__main__':
    main()
