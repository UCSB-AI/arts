#!/usr/bin/env python3
"""TTT (ARTS*) method figure — paper style, clean 4-card flow.

State -> next actions sampled by GRPO (rewards as numbers) -> TTT update ->
context compaction.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

C_ARTS = '#5E35B1'      # purple
C_SOFT = '#EDE7F6'      # very light lavender (card tint)
C_MIDP = '#9A7FD0'      # mid purple
GREY   = '#8a8a8a'
INK    = '#1a1a1a'

OUT = Path(__file__).resolve().parents[2] / 'final_figures'
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(13.5, 3.7))
ax.set_xlim(0, 138); ax.set_ylim(0, 38); ax.axis('off')

CARD_Y, CARD_H = 4, 27
centers = [18, 52, 86, 120]
CW = 30


def card(cx, title):
    x = cx - CW/2
    ax.add_patch(FancyBboxPatch((x, CARD_Y), CW, CARD_H,
                 boxstyle='round,pad=0,rounding_size=1.6',
                 fc='white', ec='#d9d4e6', lw=1.3, zorder=2))
    ax.text(cx, CARD_Y + CARD_H - 3.2, title, ha='center', va='center',
            fontsize=11, color=C_ARTS, fontweight='bold', zorder=4)
    return x


def conn(i):
    x1 = centers[i] + CW/2 + 0.6
    x2 = centers[i+1] - CW/2 - 0.6
    ax.add_patch(FancyArrowPatch((x1, CARD_Y + CARD_H/2), (x2, CARD_Y + CARD_H/2),
                 arrowstyle='-|>', mutation_scale=15, lw=2.0, color=INK, zorder=5,
                 shrinkA=0, shrinkB=0))


def tnode(cx, cy, label, root=False):
    fc = '#efece6' if root else C_SOFT
    ec = '#b9b2a6' if root else C_ARTS
    w, h = 7.0, 4.4
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle='round,pad=0,rounding_size=1.1',
                 fc=fc, ec=ec, lw=1.3, zorder=4))
    ax.text(cx, cy, label, ha='center', va='center', fontsize=8, color=INK, zorder=5)


# ---- Card 1: State ----
c = centers[0]; card(c, 'State')
ax.text(c, CARD_Y + CARD_H - 7, 'current search tree', ha='center', fontsize=8.5,
        color=INK, style='italic')
for (x1,y1),(x2,y2) in [((c,18.5),(c-7,12.5)),((c,18.5),(c,12.5)),((c,18.5),(c+7,12.5))]:
    ax.plot([x1,x2],[y1,y2], color='#cfc7dd', lw=1.5, zorder=1)
tnode(c, 18.5, '0.00', root=True)
tnode(c-7, 12.5, '0.31'); tnode(c, 12.5, '0.10'); tnode(c+7, 12.5, '0.47')

# ---- Card 2: actions sampled by GRPO, rewards as numbers ----
c = centers[1]; card(c, 'Sample next actions')
ax.text(c, CARD_Y + CARD_H - 7, 'GRPO samples a group · reward each', ha='center',
        fontsize=8.2, color=INK, style='italic')
actions = [('deepen 0.10  (richer inputs)', '+1', C_ARTS),
           ('refine 0.47', '+0.2', C_MIDP),
           ('restart from root', '0', GREY)]
ax.text(c - CW/2 + 3, 16.5, 'action', ha='left', fontsize=7.4, color=GREY)
ax.text(c + CW/2 - 3, 16.5, 'reward', ha='right', fontsize=7.4, color=GREY)
for i, (txt, r, col) in enumerate(actions):
    yy = 13.5 - i*3.4
    ax.text(c - CW/2 + 3, yy, f'a{i+1}  {txt}', ha='left', va='center', fontsize=7.8, color=INK)
    ax.text(c + CW/2 - 3, yy, r, ha='right', va='center', fontsize=9.5, color=col, fontweight='bold')

# ---- Card 3: TTT update ----
c = centers[2]; card(c, 'Test-time training')
ax.text(c, CARD_Y + CARD_H - 7, 'GRPO: group-relative advantage', ha='center',
        fontsize=8.2, color=INK, style='italic')
ax.add_patch(FancyBboxPatch((c-12, 9.5), 24, 7.5, boxstyle='round,pad=0,rounding_size=1.4',
             fc=C_SOFT, ec=C_ARTS, lw=1.4, zorder=3))
ax.text(c, 14.4, 'Scientist', ha='center', fontsize=10, color=INK, fontweight='bold', zorder=4)
ax.text(c, 11.1, 'Qwen3-4B + LoRA   (Δθ)', ha='center', fontsize=8, color=C_ARTS, zorder=4)
ax.annotate('', xy=(c, 9.0), xytext=(c, 7.0),
            arrowprops=dict(arrowstyle='-|>', color=C_ARTS, lw=1.6))
ax.text(c, 6.0, 'rewards update the weights', ha='center', fontsize=7.8, color=INK)

# ---- Card 4: context compaction ----
c = centers[3]; card(c, 'Context compaction')
ax.text(c, CARD_Y + CARD_H - 7, 'history lives in the weights', ha='center',
        fontsize=8.2, color=INK, style='italic')
# long history -> compacted into a small chip
for i, yy in enumerate([16.5, 14.8, 13.1, 11.4]):
    ax.plot([c-12, c-3], [yy, yy], color='#cfc7dd', lw=2.0, solid_capstyle='round', zorder=3)
ax.annotate('', xy=(c+2.5, 14), xytext=(c-1.5, 14),
            arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6))
ax.add_patch(FancyBboxPatch((c+3.5, 11.6), 9, 4.8, boxstyle='round,pad=0,rounding_size=1.2',
             fc=C_ARTS, ec=C_ARTS, lw=1.0, zorder=4))
ax.text(c+8, 14, 'θ', ha='center', va='center', fontsize=11, color='white', zorder=5)
ax.text(c, 7.4, 'prompt stays bounded —', ha='center', fontsize=7.8, color=INK)
ax.text(c, 5.0, 'a 4B scientist matches o3', ha='center', fontsize=7.8, color=C_ARTS, fontweight='bold')

for i in range(3):
    conn(i)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'fig_ttt_method.{ext}', dpi=300, bbox_inches='tight')
print('wrote', OUT / 'fig_ttt_method.{pdf,png}')
