#!/usr/bin/env python3
"""HMS hourly step-progression plot.

Same data as make_hms_progression.py, but sampled at hour boundaries and
plotted as a step function (constant between hours, jumps at the hour mark).

Linear/AIRA/MLEvolve: mean ± SEM (n=3 each).
LLMG: single best run only (no SEM band).
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/jarnav/MLScientist/arts/tools')
import json, os, glob, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Reuse loaders + run discovery from the per-minute script.
from make_hms_progression import (
    BASE, B, M, COLORS, HIGHER_IS_BETTER,
    load_tree_run, load_linear_run,
    runs_aira_hms, runs_linear_hms, runs_mlevolve_hms, runs_llmg_hms,
)

PAPER_FIG = Path('/home/jarnav/mlscientist/Formatting_Instructions_For_NeurIPS_2026/figures/progression_hms_hourly.pdf')

HOURS = np.arange(0, 9)  # 0,1,2,...,8


def best_so_far_at_hour(events: list[tuple[float, float]]) -> np.ndarray:
    """Return best-so-far value at each hour mark.

    At hour h, value = best across all events with t_seconds <= h*3600.
    If no event yet, value = baseline B.
    """
    arr = np.full(HOURS.shape, B, dtype=float)
    if not events:
        return arr
    for i, h in enumerate(HOURS):
        cutoff = h * 3600.0
        scores = [s for (t, s) in events if t <= cutoff]
        if scores:
            arr[i] = min(scores) if not HIGHER_IS_BETTER else max(scores)
    return arr


def aggregate_method_hourly(name, runs, loader):
    curves = []
    for p in runs:
        events = loader(p)
        if events:
            curves.append(best_so_far_at_hour(events))
    if not curves:
        return None, None
    M_curves = np.stack(curves)
    norm = (B - M_curves) / (B - M)
    mean = norm.mean(axis=0)
    sem = norm.std(axis=0, ddof=1) / np.sqrt(M_curves.shape[0]) if M_curves.shape[0] >= 2 else np.zeros_like(mean)
    return mean, sem


def best_run_hourly(name, runs, loader):
    curves = []; finals = []
    for p in runs:
        events = loader(p)
        if events:
            arr = best_so_far_at_hour(events)
            curves.append(arr); finals.append(arr[-1])
    if not curves:
        return None, -1
    best_idx = int(np.argmin(finals)) if not HIGHER_IS_BETTER else int(np.argmax(finals))
    norm = (B - curves[best_idx]) / (B - M)
    return norm, best_idx


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

    # Mean+SEM methods
    methods_mean = [
        ('Linear', list(runs_linear_hms()), load_linear_run),
        ('AIRA', runs_aira_hms(), load_tree_run),
        ('MLEvolve', runs_mlevolve_hms(), load_tree_run),
    ]
    for name, runs, loader in methods_mean:
        mean, sem = aggregate_method_hourly(name, runs, loader)
        if mean is None:
            print(f"  {name}: no runs"); continue
        n = len([p for p in runs if loader(p)])
        c = COLORS[name]
        # Step plot: 'post' means value V[i] holds from x[i] to x[i+1]
        ax.step(HOURS, mean, where='post', color=c, label=name, linewidth=2.0)
        ax.fill_between(HOURS, mean - sem, mean + sem, step='post', color=c, alpha=0.18, linewidth=0)
        print(f"  {name}: mean per hour = {[f'{x:.3f}' for x in mean]}")

    # LLMG: best run only
    llmg_runs = runs_llmg_hms()
    norm, best_idx = best_run_hourly('LLMG', llmg_runs, load_tree_run)
    if norm is not None:
        c = COLORS['LLMG']
        ax.step(HOURS, norm, where='post', color=c, label='LLMG', linewidth=2.2)
        print(f"  LLMG best-run per hour = {[f'{x:.3f}' for x in norm]}")

    # Baseline reference line at y=0 (no human-best line per user request)
    ax.axhline(0.0, color=COLORS['Baseline'], linestyle='-', linewidth=1.4,
               label='Baseline', alpha=0.8)

    ax.set_xlabel('Wall-clock time (hours)')
    ax.set_ylabel('Normalized score')
    ax.set_title('HMS Brain Activity')
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
    ax.legend(loc='lower right')

    PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PAPER_FIG, bbox_inches='tight')
    print(f"\nWrote {PAPER_FIG}")


if __name__ == '__main__':
    main()
