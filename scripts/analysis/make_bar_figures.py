#!/usr/bin/env python3
"""Paper-styled bar figures for ARTS, matching the progression-curve aesthetic.

Figures (light theme, serif/Times, thin spines, ticks-in, bottom legend):

  fig_results_bars[_h][_iqm].{pdf,png}   normalized score, top-6 tasks by ARTS
                                         margin (ARTS vs MLEvolve vs AIRA)
  fig_ttt_bars[_h][_iqm].{pdf,png}       ARTS* (4B + TTT) vs ARTS-o3, per task

  *_h   = horizontal orientation
  *_iqm = rliable/IQM style: light (alpha) bar fill + a dark cap line at the
          value, in the solid method color.

Data: published normalized scores (ARTS_runs/RESULTS.md) and the
TTT comparison values.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- paper style (mirrors scripts/analysis/make_results_figures.py) ---------
mpl.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        13,
    'axes.titlesize':   15,
    'axes.labelsize':   14,
    'axes.linewidth':   0.9,
    'legend.fontsize':  13,
    'legend.frameon':   False,
    'xtick.labelsize':  12.5,
    'ytick.labelsize':  12,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

# published method colors (Fig. 3 legend)
C_ARTS     = '#5E35B1'   # purple
C_MLEVOLVE = '#EF6C00'   # orange
C_AIRA     = '#2E7D32'   # green
HUMAN      = '#777777'
CAP_LW     = 2.6         # dark cap-line width for the _iqm style

OUT = Path(__file__).resolve().parents[2] / 'final_figures'
OUT.mkdir(exist_ok=True)


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT / f'{name}.{{pdf,png}}')


# ===========================================================================
# Figure 1 — normalized results, top-6 tasks by ARTS margin
# ===========================================================================
# (task, ARTS, MLEvolve, AIRA)
RESULTS = [
    ("Battle of Sexes",    1.52, 0.66, 0.65),
    ("Prisoner's Dilemma", 0.77, 0.42, 0.21),
    ("MountainCar",        0.95, 0.78, 0.72),
    ("Meta Maze",          0.97, 0.81, 0.56),
    ("Language Modeling",  0.72, 0.56, 0.00),
    ("RSNA Brain",         1.43, 1.29, 1.23),
]
BAR_LABELS = ['ARTS', 'MLEvolve', 'AIRA']
BAR_COLORS = [C_ARTS, C_MLEVOLVE, C_AIRA]


def fig_results(alpha=1.0, cap=False, suffix=''):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharey=True)
    ymax = 1.65
    for ax, (task, *vals) in zip(axes.flat, RESULTS):
        x = np.arange(3)
        ax.bar(x, vals, width=0.5, color=BAR_COLORS, alpha=alpha,
               edgecolor='white', linewidth=0.6, zorder=3)
        if cap:  # dark value cap, in the solid method color
            for xi, v, c in zip(x, vals, BAR_COLORS):
                ax.plot([xi - 0.25, xi + 0.25], [v, v], color=c, lw=CAP_LW,
                        solid_capstyle='round', zorder=5)
        ax.axhline(1.0, color=HUMAN, ls=(0, (4, 3)), lw=0.9, zorder=2)
        ax.text(2.45, 1.02, 'human', ha='right', va='bottom', fontsize=10, color=HUMAN)
        for xi, v, c in zip(x, vals, BAR_COLORS):
            ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', va='bottom',
                    fontsize=12, color=c)
        ax.set_title(task, fontsize=14.5, pad=6, loc='left')
        ax.set_xticks(x); ax.set_xticklabels([])
        ax.set_ylim(0, ymax); ax.set_yticks([0, 0.5, 1.0, 1.5])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color='#e8e8e8', lw=0.6, zorder=0); ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel('normalized score')
    handles = [Patch(fc=c, alpha=alpha, label=l) for c, l in zip(BAR_COLORS, BAR_LABELS)]
    fig.legend(handles=handles, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), handlelength=1.4, columnspacing=2.0)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, f'fig_results_bars{suffix}')


def fig_results_h(alpha=1.0, cap=False, suffix=''):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.0), sharex=True)
    xmax = 1.72
    for ax, (task, *vals) in zip(axes.flat, RESULTS):
        y = np.arange(3)[::-1]
        ax.barh(y, vals, height=0.5, color=BAR_COLORS, alpha=alpha,
                edgecolor='white', linewidth=0.6, zorder=3)
        if cap:
            for yi, v, c in zip(y, vals, BAR_COLORS):
                ax.plot([v, v], [yi - 0.25, yi + 0.25], color=c, lw=CAP_LW,
                        solid_capstyle='round', zorder=5)
        ax.axvline(1.0, color=HUMAN, ls=(0, (4, 3)), lw=0.9, zorder=2)
        ax.text(1.0, 2.65, 'human', ha='center', va='bottom', fontsize=10, color=HUMAN)
        for yi, v, c in zip(y, vals, BAR_COLORS):
            ax.text(v + 0.03, yi, f'{v:.2f}', ha='left', va='center', fontsize=12, color=c)
        ax.set_title(task, fontsize=14.5, pad=6, loc='left')
        ax.set_yticks(y); ax.set_yticklabels([])
        ax.set_xlim(0, xmax); ax.set_xticks([0, 0.5, 1.0, 1.5])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e8e8e8', lw=0.6, zorder=0); ax.set_axisbelow(True)
    for ax in axes[-1, :]:
        ax.set_xlabel('normalized score')
    handles = [Patch(fc=c, alpha=alpha, label=l) for c, l in zip(BAR_COLORS, BAR_LABELS)]
    fig.legend(handles=handles, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), handlelength=1.4, columnspacing=2.0)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, f'fig_results_bars_h{suffix}')


# ===========================================================================
# Figure 2 — ARTS* (4B + TTT) vs ARTS-o3, per task (stacked)
# ===========================================================================
# heights are score / human-best (human-best = 1.0)
# (task, o3_h, fourB_h, arts_star_h)
TTT = [
    ("Meta Maze",    0.975, 0.578, 0.996),
    ("MountainCar",  0.967, 0.572, 0.915),
    ("Breakout",     0.780, 0.488, 0.746),
    ("Lang. Model.", 0.915, 0.806, 0.981),
    ("Titanic",      1.186, 1.143, 1.189),
    ("CIFAR-10",     0.977, 0.963, 0.975),
]
C_O3   = '#E0A52E'   # amber benchmark
C_4B   = '#B7AEDB'   # muted lavender (ARTS-4B, no TTT)
C_GAIN = C_ARTS      # purple (+TTT gain)


def fig_ttt(alpha=1.0, cap=False, suffix=''):
    fig, ax = plt.subplots(figsize=(10, 5.0))
    n = len(TTT); group = np.arange(n); bw = 0.30
    for i, (task, o3, fourb, star) in enumerate(TTT):
        xo, xs = group[i] - bw / 2 - 0.03, group[i] + bw / 2 + 0.03
        ax.bar(xo, o3, bw, color=C_O3, alpha=alpha, edgecolor='white', lw=0.6, zorder=3)
        ax.bar(xs, fourb, bw, color=C_4B, alpha=alpha, edgecolor='white', lw=0.6, zorder=3)
        ax.bar(xs, star - fourb, bw, bottom=fourb, color=C_GAIN, alpha=alpha,
               edgecolor='white', lw=0.6, zorder=3)
        if cap:
            ax.plot([xo - bw/2, xo + bw/2], [o3, o3], color=C_O3, lw=CAP_LW,
                    solid_capstyle='round', zorder=5)
            ax.plot([xs - bw/2, xs + bw/2], [star, star], color=C_GAIN, lw=CAP_LW,
                    solid_capstyle='round', zorder=5)
        ax.text(xo, o3 + 0.015, f'{o3:.2f}', ha='center', va='bottom',
                fontsize=11.5, color=C_O3, fontweight='bold')
        ax.text(xs, star + 0.015, f'{star:.2f}', ha='center', va='bottom',
                fontsize=11.5, color=C_GAIN, fontweight='bold')
        ax.text(xo, -0.04, 'o3', ha='center', va='top', fontsize=10.5, color=C_O3)
        ax.text(xs, -0.04, 'ARTS*', ha='center', va='top', fontsize=10.5, color=C_GAIN)
    ax.axhline(1.0, color=HUMAN, ls=(0, (4, 3)), lw=1.0, zorder=2)
    ax.set_xticks(group); ax.set_xticklabels([t[0] for t in TTT], fontsize=13)
    ax.tick_params(axis='x', length=0, pad=22)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel('score ÷ human-best')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e8e8e8', lw=0.6, zorder=0); ax.set_axisbelow(True)
    handles = [
        Patch(fc=C_O3, alpha=alpha, label='ARTS-o3'),
        Patch(fc=C_4B, alpha=alpha, label='ARTS-4B'),
        Patch(fc=C_GAIN, alpha=alpha, label='ARTS* (+TTT) gain'),
        Line2D([0], [0], color=HUMAN, ls=(0, (4, 3)), lw=1.0, label='human-best'),
    ]
    fig.legend(handles=handles, ncol=4, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), handlelength=1.5, columnspacing=2.0)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, f'fig_ttt_bars{suffix}')


def fig_ttt_h(alpha=1.0, cap=False, suffix=''):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    n = len(TTT); group = np.arange(n)[::-1]; bh = 0.30
    for i, (task, o3, fourb, star) in enumerate(TTT):
        yo, ys = group[i] + bh / 2 + 0.03, group[i] - bh / 2 - 0.03
        ax.barh(yo, o3, bh, color=C_O3, alpha=alpha, edgecolor='white', lw=0.6, zorder=3)
        ax.barh(ys, fourb, bh, color=C_4B, alpha=alpha, edgecolor='white', lw=0.6, zorder=3)
        ax.barh(ys, star - fourb, bh, left=fourb, color=C_GAIN, alpha=alpha,
                edgecolor='white', lw=0.6, zorder=3)
        if cap:
            ax.plot([o3, o3], [yo - bh/2, yo + bh/2], color=C_O3, lw=CAP_LW,
                    solid_capstyle='round', zorder=5)
            ax.plot([star, star], [ys - bh/2, ys + bh/2], color=C_GAIN, lw=CAP_LW,
                    solid_capstyle='round', zorder=5)
        ax.text(o3 + 0.012, yo, f'{o3:.2f}', ha='left', va='center',
                fontsize=11.5, color=C_O3, fontweight='bold')
        ax.text(star + 0.012, ys, f'{star:.2f}', ha='left', va='center',
                fontsize=11.5, color=C_GAIN, fontweight='bold')
        ax.text(-0.012, yo, 'o3', ha='right', va='center', fontsize=10.5, color=C_O3)
        ax.text(-0.012, ys, 'ARTS*', ha='right', va='center', fontsize=10.5, color=C_GAIN)
    ax.axvline(1.0, color=HUMAN, ls=(0, (4, 3)), lw=1.0, zorder=2)
    ax.text(1.0, n - 0.35, 'human-best', ha='center', va='bottom', fontsize=10, color=HUMAN)
    ax.set_yticks(group); ax.set_yticklabels([t[0] for t in TTT], fontsize=13)
    ax.tick_params(axis='y', length=0, pad=44)
    ax.set_xlim(0, 1.32); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel('score ÷ human-best')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e8e8e8', lw=0.6, zorder=0); ax.set_axisbelow(True)
    handles = [
        Patch(fc=C_O3, alpha=alpha, label='ARTS-o3'),
        Patch(fc=C_4B, alpha=alpha, label='ARTS-4B'),
        Patch(fc=C_GAIN, alpha=alpha, label='ARTS* (+TTT) gain'),
        Line2D([0], [0], color=HUMAN, ls=(0, (4, 3)), lw=1.0, label='human-best'),
    ]
    fig.legend(handles=handles, ncol=4, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), handlelength=1.5, columnspacing=2.0)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, f'fig_ttt_bars_h{suffix}')


if __name__ == '__main__':
    # solid versions
    fig_results(); fig_ttt(); fig_results_h(); fig_ttt_h()
    # rliable / IQM style: light fill + dark value cap
    fig_results(alpha=0.5, cap=True, suffix='_iqm')
    fig_ttt(alpha=0.5, cap=True, suffix='_iqm')
    fig_results_h(alpha=0.5, cap=True, suffix='_iqm')
    fig_ttt_h(alpha=0.5, cap=True, suffix='_iqm')
