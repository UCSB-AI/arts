#!/usr/bin/env python3
"""Reconstruct Figure 3 of the ARTS paper: selected_tasks_grid_mean_only.pdf.

8 tasks x 4 methods. For each (task, method), pick up to 3 runs, compute
best-so-far at hour boundaries (0..8), normalize via y=(score-B)/(M-B)
for higher-better and (B-score)/(B-M) for lower-better, mean across runs.

Plot lines + open-circle markers at each hour, save 2x4 grid PDF +
companion CSV with the exact per-(task,method,hour) numbers.
"""
from __future__ import annotations
import csv, glob, json, math, os, re, sys
import datetime as _dt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE = Path('/home/jarnav/MLScientist/arts/runs_archive')
OUT_DIR = Path('/home/jarnav/MLScientist/arts/tools/figs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOURS = np.arange(0, 9)  # 0,1,...,8
SENTINELS = {1.0, 50000000.0, 100.0, 999.0, -1.0}
MAX_RUNS_PER_METHOD = 3

# ---------------------------------------------------------------------------
# Per-task config: baseline B, human-best M, direction, run-name token
# (from sections/results/main_results.tex tab:main)
# ---------------------------------------------------------------------------
TASKS = [
    dict(key='LM',       name='Lang. Modeling',     tok='languageModelingFineWeb',   B=4.673, M=3.500, higher=False, ymax=1.05, yticks=[0, 0.3, 0.7, 1.0]),
    dict(key='MCar',     name='MountainCar',        tok='rlMountainCarContinuous',   B=33.79, M=99.00, higher=True,  ymax=1.05, yticks=[0, 0.3, 0.7, 1.0]),
    dict(key='FMnist',   name='Fashion MNIST',      tok='imageClassificationFMnist', B=0.848, M=0.968, higher=True,  ymax=1.05, yticks=[0, 0.3, 0.7, 1.0]),
    dict(key='MetaMaze', name='Meta Maze',          tok='rlMetaMaze',                B=15.73, M=52.50, higher=True,  ymax=1.05, yticks=[0, 0.3, 0.7, 1.0]),
    dict(key='RSNA',     name='RSNA',               tok='mlebenchRSNABrainTumor',    B=0.500, M=0.621, higher=True,  ymax=1.45, yticks=[0, 0.5, 1.0, 1.4]),
    dict(key='Vesuvius', name='Vesuvius',           tok='mlebenchVesuvius',          B=0.000, M=0.831, higher=True,  ymax=1.05, yticks=[0, 0.3, 0.7, 1.0]),
    dict(key='HMS',      name='HMS Brain',          tok='mlebenchHMSBrain',          B=1.462, M=0.272, higher=False, ymax=1.05, yticks=[0, 0.3, 0.7, 1.0]),
    dict(key='PD',       name="Prisoner's Dilem.",  tok='prisoner',                  B=2.372, M=3.000, higher=True,  ymax=1.05, yticks=[0, 0.3, 0.7, 1.0]),
]

METHODS = ['Linear', 'AIRA', 'MLEvo', 'ARTS']
PREFIX = {'Linear': 'linear', 'AIRA': 'aira', 'MLEvo': 'mlevolve', 'ARTS': 'llmg'}

# Color palette to match the reference figure
COLORS = {
    'Linear': '#8b5a2b',   # brown
    'AIRA':   '#1f8b4c',   # green
    'MLEvo':  '#e6a23c',   # orange
    'ARTS':   '#7d3cc5',   # purple
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _valid_score(s):
    try:
        s = float(s)
    except Exception:
        return None
    if not math.isfinite(s):
        return None
    if s in SENTINELS:
        return None
    return s


def load_tree_run(run_dir: Path) -> list[tuple[float, float]]:
    """Tree methods (AIRA, MLEvo, ARTS): list of (t_seconds_from_start, score).
    Score comes from each node json; t comes from file mtime - earliest mtime.
    Sentinels are dropped.
    """
    nodes_dir = run_dir / 'nodes'
    if not nodes_dir.is_dir():
        return []
    files = sorted(nodes_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)
    if not files:
        return []
    t0 = files[0].stat().st_mtime
    events = []
    for f in files:
        try:
            j = json.load(open(f))
        except Exception:
            continue
        s = j.get('score')
        if s is None:
            s = j.get('validation_score')
        s = _valid_score(s)
        if s is None:
            continue
        t = max(0.0, f.stat().st_mtime - t0)
        events.append((t, s))
    return events


_LINEAR_RX = re.compile(r'step\s+(\d+):\s+validate\s*->\s*(-?\d+(?:\.\d+)?)')

def load_linear_run(run_dir: Path) -> list[tuple[float, float]]:
    """Linear: parse 'step N: validate -> X' from run.log; t = linearly
    interpolated from dir-timestamp -> log mtime."""
    log = run_dir / 'run.log'
    if not log.is_file():
        return []
    steps = []
    with open(log, errors='ignore') as fp:
        for line in fp:
            m = _LINEAR_RX.search(line)
            if not m:
                continue
            try:
                idx = int(m.group(1))
                sc = float(m.group(2))
            except Exception:
                continue
            if math.isfinite(sc) and sc not in SENTINELS:
                steps.append((idx, sc))
    if not steps:
        return []
    m = re.search(r'(\d{8})_(\d{6})', run_dir.name)
    t_start = None
    if m:
        try:
            t_start = _dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
        except Exception:
            pass
    if t_start is None:
        t_start = log.stat().st_ctime
    t_end = log.stat().st_mtime
    dur = max(1.0, t_end - t_start)
    max_idx = max(s[0] for s in steps)
    steps.sort(key=lambda x: x[0])
    out = []
    for idx, sc in steps:
        t = dur * (idx / max(1, max_idx))
        out.append((t, sc))
    return out


_TS_RX = re.compile(r'(\d{8})_(\d{6})')

def _ts_key(p: Path) -> float:
    """Sort key: embedded YYYYMMDD_HHMMSS as unix timestamp. Older first when
    not present; we want NEWER first so we negate at call site."""
    m = _TS_RX.search(p.name)
    if not m:
        return 0.0
    try:
        return _dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
    except Exception:
        return 0.0


# Tags we de-prioritise: early experimental variants that aren't the paper config.
_EXCLUDE_TAGS = ('_mcts_', '_smoke_', '_test_', '_debug_', '_ablation_')

# For ARTS, the paper uses the C variant; for tasks where C doesn't exist,
# fall back to the base llmg_<task> pattern. PD only has ABCD/baseline variants.
_ARTS_PREFERENCE = [
    'llmg_C_',        # complexity-cycle (paper main config)
    'llmg_baseline_', # explicit baseline-tag
    'llmg_',          # any llmg
]


def find_runs(task_tok: str, method: str) -> list[Path]:
    """Return up to MAX_RUNS_PER_METHOD run dirs for (task, method).
    Prefers the canonical paper config and most-recent timestamps.
    """
    prefix = PREFIX[method]

    if method == 'ARTS':
        # ARTS = o3 scientist + the C variant by default. Require '_o3_' in name
        # to exclude Qwen-scientist (TTT) runs.
        for tag in _ARTS_PREFERENCE:
            cand = [p for p in BASE.glob(f'{tag}*{task_tok}*') if p.is_dir()]
            cand = [p for p in cand if '_o3_' in p.name]
            cand = [p for p in cand if not any(x in p.name for x in _EXCLUDE_TAGS)]
            cand.sort(key=lambda p: -_ts_key(p))
            productive = _take_productive(cand, method)
            if productive:
                return productive
        return []

    cand = [p for p in BASE.glob(f'{prefix}_*{task_tok}*') if p.is_dir()]
    cand = [p for p in cand if not any(x in p.name for x in _EXCLUDE_TAGS)]
    cand.sort(key=lambda p: -_ts_key(p))
    return _take_productive(cand, method)


def _take_productive(cand: list[Path], method: str) -> list[Path]:
    loader = load_linear_run if method == 'Linear' else load_tree_run
    out = []
    for p in cand:
        ev = loader(p)
        if ev:
            out.append(p)
        if len(out) >= MAX_RUNS_PER_METHOD:
            break
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def best_so_far_at_hours(events: list[tuple[float, float]], higher: bool, B: float) -> np.ndarray:
    """Best-so-far at each hour mark (0..8). Defaults to B before any event."""
    arr = np.full(HOURS.shape, B, dtype=float)
    if not events:
        return arr
    for i, h in enumerate(HOURS):
        cutoff = h * 3600.0
        scores = [s for (t, s) in events if t <= cutoff]
        if not scores:
            arr[i] = B
        else:
            arr[i] = max(scores) if higher else min(scores)
    return arr


def normalize(curve: np.ndarray, B: float, M: float, higher: bool) -> np.ndarray:
    """y in [0, 1] at baseline/human-best. Can exceed 1 or go below 0."""
    if higher:
        return (curve - B) / (M - B)
    else:
        return (B - curve) / (B - M)


def aggregate(task: dict, method: str) -> tuple[np.ndarray, int, list[str]]:
    """Return (mean_normalized_curve, n_runs, run_names)."""
    runs = find_runs(task['tok'], method)
    loader = load_linear_run if method == 'Linear' else load_tree_run
    curves = []
    names = []
    for p in runs:
        ev = loader(p)
        if not ev:
            continue
        raw = best_so_far_at_hours(ev, task['higher'], task['B'])
        norm = normalize(raw, task['B'], task['M'], task['higher'])
        curves.append(norm)
        names.append(p.name)
    if not curves:
        return np.full(HOURS.shape, 0.0), 0, []
    arr = np.stack(curves)
    return arr.mean(axis=0), arr.shape[0], names


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.linewidth': 0.9,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.6), constrained_layout=False)
    axes_flat = axes.flatten()

    rows = []  # for CSV
    for ax, task in zip(axes_flat, TASKS):
        ax.set_title(task['name'])
        ax.set_xlim(0, 8)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_ylim(-0.1, task['ymax'])
        ax.set_yticks(task['yticks'])
        ax.grid(False)
        # baseline horizontal line at y=0
        ax.axhline(0.0, color='lightgray', linewidth=0.7, zorder=0)
        for m in METHODS:
            mean_curve, n, names = aggregate(task, m)
            print(f"  [{task['key']:<9}] {m:<7}  n={n}  hourly={[f'{x:.3f}' for x in mean_curve]}")
            for name in names:
                print(f"      run: {name}")
            color = COLORS[m]
            lw = 2.0 if m == 'ARTS' else 1.5
            ax.plot(
                HOURS, mean_curve,
                color=color, linewidth=lw,
                marker='o', markersize=5, markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=1.2,
                label=m, zorder=3 if m == 'ARTS' else 2,
            )
            for h, v in zip(HOURS, mean_curve):
                rows.append({'task': task['key'], 'method': m, 'hour': int(h),
                             'n_runs': n, 'normalized_mean': float(v)})

        if task is TASKS[0] or task is TASKS[4]:
            ax.set_ylabel('Normalized Score ( ↑ )')
        if task['key'] in {'RSNA', 'Vesuvius', 'HMS', 'PD'}:
            ax.set_xlabel('Hours')

    # legend at bottom
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.10,
                        wspace=0.28, hspace=0.40)

    pdf = OUT_DIR / 'fig3_selected_tasks_grid.pdf'
    png = OUT_DIR / 'fig3_selected_tasks_grid.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=220, bbox_inches='tight')
    print(f'\nwrote {pdf}')
    print(f'wrote {png}')
    plt.close(fig)

    csv_path = OUT_DIR / 'fig3_selected_tasks_grid.csv'
    with open(csv_path, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=['task', 'method', 'hour', 'n_runs', 'normalized_mean'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
