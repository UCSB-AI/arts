#!/usr/bin/env python3
"""HMS-only progression plot — best-so-far KL vs wall-clock time.

Per-method mean + SEM band across n runs. y-axis normalized:
  y = (baseline - score) / (baseline - human_best)   # lower is better
  b = 1.462, m = 0.272

Methods: Baseline, Linear, AIRA, MLEvolve, LLMG (C variant only).
Colors: seaborn colorblind palette.
"""
from __future__ import annotations
import json, os, re, glob, math, datetime as _dt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE = Path('/home/jarnav/MLScientist/arts/outputs')
PAPER_FIG = Path('/home/jarnav/mlscientist/Formatting_Instructions_For_NeurIPS_2026/figures/progression_hms.pdf')

# HMS task constants
B = 1.462   # baseline KL (no model)
M = 0.272   # human best KL
HIGHER_IS_BETTER = False
TIME_GRID = np.arange(0, 8 * 3600 + 1, 60)  # 0..8h, every 60s

# Seaborn colorblind palette (matplotlib RGB tuples)
COLORS = {
    'Baseline': (0.580, 0.580, 0.580),  # #949494 gray
    'Linear':   (0.871, 0.561, 0.020),  # #de8f05 orange
    'AIRA':     (0.004, 0.451, 0.698),  # #0173b2 blue
    'MLEvolve': (0.008, 0.620, 0.451),  # #029e73 green
    'LLMG':     (0.835, 0.369, 0.000),  # #d55e00 red-orange
}


def normalize(score: float) -> float:
    """KL: lower is better. (b-x)/(b-m), 0=baseline, 1=human."""
    if not math.isfinite(score):
        return 0.0
    return (B - score) / (B - M)


# ------------------------------------------------------------------
# Tree-based loader (MLEvolve, AIRA, LLMG): score = best-so-far over
# nodes sorted by mtime; time = mtime - earliest_node_mtime.
# ------------------------------------------------------------------
def load_tree_run(run_dir: Path) -> list[tuple[float, float]]:
    """Returns [(t_seconds, best_so_far_score)]."""
    files = sorted(glob.glob(str(run_dir / 'nodes/*.json')), key=os.path.getmtime)
    events = []
    for f in files:
        try:
            j = json.load(open(f))
            s = j.get('score')
            if s is None:
                s = j.get('validation_score')
            if s is None:
                continue
            mt = os.path.getmtime(f)
            events.append((mt, float(s)))
        except Exception:
            continue
    if not events:
        return []
    t0 = events[0][0]
    best = float('inf') if not HIGHER_IS_BETTER else float('-inf')
    out = []
    for (mt, s) in events:
        if HIGHER_IS_BETTER:
            best = max(best, s)
        else:
            best = min(best, s)
        out.append((mt - t0, best))
    return out


# ------------------------------------------------------------------
# Linear loader: parse "step N: validate -> X" from run.log; time
# linearly interpolated from dir-name-timestamp to log mtime.
# ------------------------------------------------------------------
_LINEAR_RX = re.compile(r'step\s+(\d+):\s+validate\s*->\s*(-?\d+(?:\.\d+)?)')

def load_linear_run(run_dir: Path) -> list[tuple[float, float]]:
    log = run_dir / 'run.log'
    if not log.is_file():
        return []
    steps = []
    with open(log, errors='ignore') as fp:
        for line in fp:
            m = _LINEAR_RX.search(line)
            if m:
                try:
                    idx = int(m.group(1)); sc = float(m.group(2))
                except Exception:
                    continue
                if math.isfinite(sc):
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
    events = []
    best = float('inf') if not HIGHER_IS_BETTER else float('-inf')
    steps.sort(key=lambda x: x[0])
    for (idx, sc) in steps:
        t = dur * (idx / max(1, max_idx))
        if HIGHER_IS_BETTER:
            best = max(best, sc)
        else:
            best = min(best, sc)
        events.append((t, best))
    return events


# ------------------------------------------------------------------
# Per-method run discovery
# ------------------------------------------------------------------
def runs_aira_hms() -> list[Path]:
    return sorted([p for p in BASE.glob('aira_mlebenchHMSBrain_*') if p.is_dir()])

def runs_linear_hms() -> list[Path]:
    seen = set()
    for p in sorted(BASE.glob('linear_mlebenchHMSBrain_*')):
        if p.is_dir() and p.name not in seen:
            seen.add(p.name); yield p

def runs_mlevolve_hms() -> list[Path]:
    out = []
    for jid in [4761562, 4763344, 4763348]:
        for p in BASE.glob(f'mlevolve_*HMSBrain*j{jid}*'):
            if p.is_dir():
                out.append(p)
    return out

def runs_llmg_hms() -> list[Path]:
    out = []
    for jid in [4764330, 4764717, 4764718]:
        for p in BASE.glob(f'llmg_*HMSBrain*j{jid}*'):
            if p.is_dir():
                out.append(p)
    return out


# ------------------------------------------------------------------
# Aggregate to common time grid
# ------------------------------------------------------------------
def best_so_far_on_grid(events: list[tuple[float, float]]) -> np.ndarray:
    """Returns array of shape len(TIME_GRID); best-so-far at each time bucket.
    Uninterpolated grid points before first event are baseline B."""
    arr = np.full_like(TIME_GRID, B, dtype=float)
    if not events:
        return arr
    events = sorted(events)
    j = 0
    cur = B
    for i, t in enumerate(TIME_GRID):
        while j < len(events) and events[j][0] <= t:
            cur = events[j][1]
            j += 1
        arr[i] = cur
    return arr


def aggregate_method(name: str, runs: list[Path], loader) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_normalized, sem_normalized) of shape len(TIME_GRID)."""
    curves = []
    for p in runs:
        events = loader(p)
        if events:
            curves.append(best_so_far_on_grid(events))
    if not curves:
        return None, None
    M_curves = np.stack(curves)  # (n_runs, T)
    norm = (B - M_curves) / (B - M)  # lower-better normalized
    mean = norm.mean(axis=0)
    if M_curves.shape[0] >= 2:
        sem = norm.std(axis=0, ddof=1) / np.sqrt(M_curves.shape[0])
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def best_run_method(name: str, runs: list[Path], loader) -> tuple[np.ndarray, int]:
    """Returns the trajectory (normalized) of the SINGLE best run by final score
    (lowest final KL), plus its index. No SEM band."""
    curves = []
    finals = []
    for p in runs:
        events = loader(p)
        if events:
            arr = best_so_far_on_grid(events)
            curves.append(arr)
            finals.append(arr[-1])  # final raw KL
    if not curves:
        return None, -1
    # For lower-is-better, best = lowest final KL.
    best_idx = int(np.argmin(finals)) if not HIGHER_IS_BETTER else int(np.argmax(finals))
    norm = (B - curves[best_idx]) / (B - M)
    return norm, best_idx


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
def main():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 0.9,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    methods_mean = [
        ('Linear', list(runs_linear_hms()), load_linear_run),
        ('AIRA', runs_aira_hms(), load_tree_run),
        ('MLEvolve', runs_mlevolve_hms(), load_tree_run),
    ]
    for name, runs, loader in methods_mean:
        mean, sem = aggregate_method(name, runs, loader)
        if mean is None:
            print(f"  {name}: no runs found"); continue
        n = len([p for p in runs if loader(p)])
        c = COLORS[name]
        t_h = TIME_GRID / 3600.0
        ax.plot(t_h, mean, color=c, label=name, linewidth=2.0)
        ax.fill_between(t_h, mean - sem, mean + sem, color=c, alpha=0.18, linewidth=0)
        print(f"  {name}: n={n} runs, final mean normalized={mean[-1]:.4f}")

    # LLMG: plot only the single best run's trajectory (no SEM band).
    llmg_runs = runs_llmg_hms()
    norm, best_idx = best_run_method('LLMG', llmg_runs, load_tree_run)
    if norm is not None:
        c = COLORS['LLMG']
        t_h = TIME_GRID / 3600.0
        run_name = llmg_runs[best_idx].name.split('_o3')[0]
        ax.plot(t_h, norm, color=c, label='LLMG', linewidth=2.2)
        print(f"  LLMG: best run = {run_name} | final normalized={norm[-1]:.4f}")

    # Baseline horizontal line at y=0 (it's the reference)
    ax.axhline(0.0, color=COLORS['Baseline'], linestyle='-', linewidth=1.4,
               label='Baseline', alpha=0.8)

    ax.set_xlabel('Wall-clock time (hours)')
    ax.set_ylabel('Normalized score')
    ax.set_title('HMS Brain Activity')
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax.legend(loc='lower right', ncol=1)

    PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PAPER_FIG, bbox_inches='tight')
    print(f"\nWrote {PAPER_FIG}")


if __name__ == '__main__':
    main()
