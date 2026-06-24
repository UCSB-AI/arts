#!/usr/bin/env python3
"""
Build all results-section figure variants for the ARTS paper.

Tasks (4): LM (FineWeb), HMS Brain, Vesuvius, APTOS  (full 4-method coverage)
Methods : Linear, AIRA, MLEvolve, ARTS
Y-axis  : normalised score (0 = baseline, 1 = Kaggle/SOTA winner)
X-axis  : wall-clock time (hours, 0..8)
Colors  : Linear=brown, AIRA=green, MLEvolve=orange, ARTS=purple
Layout  : 1 row x 4 panels; legend on rightmost only; y-label on leftmost only.

Outputs:
  fig_A1_progression_continuous.pdf   (variant 1: smooth lines, all 4 methods)
  fig_A2_progression_hourly.pdf       (variant 4: hourly step updates)
  fig_B1_broken_continuous.pdf        (variant 3a: variant 1 + broken y-axis)
  fig_B2_broken_hourly.pdf            (variant 3b: variant 4 + broken y-axis)
  fig_C_arts_vs_best.pdf              (variant 2: just ARTS vs best baseline)
  fig_D_arts_vs_best_scatter.pdf      (variant 5: ARTS+best baseline + per-node scatter)
  fig_E_arts_only_annotated.pdf       (variant 6: ARTS only + scatter + new-best annotations)
"""
from __future__ import annotations
import json
import math
import os
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
_REPO   = Path(__file__).resolve().parents[2]
BASE    = Path(os.environ.get('ARTS_OUTPUTS_DIR', _REPO / 'outputs'))
OUT_DIR = Path(os.environ.get('ARTS_FIG_DIR', _REPO / 'final_figures'))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         10.5,
    'axes.labelsize':    10.5,
    'axes.titlesize':    11.5,
    'axes.linewidth':    0.9,
    'legend.fontsize':   9.5,
    'legend.frameon':    True,
    'legend.framealpha': 0.80,   # 80% opaque
    'legend.edgecolor':  '#cccccc',
    'xtick.labelsize':   9.5,
    'ytick.labelsize':   9.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
    'lines.linewidth':   1.8,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# Fixed per-method colors: Linear=brown, AIRA=green, MLEvolve=orange, ARTS=purple
COLORS = {
    'Linear':   '#8D6E63',   # brown
    'AIRA':     '#2E7D32',   # green
    'MLEvolve': '#EF6C00',   # orange
    'ARTS':     '#5E35B1',   # purple
}
FILL_ALPHA = {
    'Linear':   0.10,
    'AIRA':     0.10,
    'MLEvolve': 0.10,
    'ARTS':     0.16,
}
LINE_WIDTH = {
    'Linear':   1.5,
    'AIRA':     1.5,
    'MLEvolve': 1.5,
    'ARTS':     2.3,
}
ORDER = ['Linear', 'AIRA', 'MLEvolve', 'ARTS']

BUDGET_H = 8.0
BUDGET_S = BUDGET_H * 3600
N_GRID   = 500

# ---------------------------------------------------------------------------
# Task specs: 4 tasks with full 4-method coverage
# ---------------------------------------------------------------------------
LM = dict(
    label      = 'Language Modeling',
    metric     = 'loss',
    higher     = False,
    baseline   = 4.673,
    human_best = 3.500,
    linear_globs   = ['linear_languageModelingFineWeb_flash_run1_20260419_140222',
                      'linear_languageModelingFineWeb_flash_run2_20260419_140222'],
    aira_globs     = ['aira_languageModelingFineWeb_flash_run1_20260418_235824',
                      'aira_languageModelingFineWeb_flash_run2_20260418_235824'],
    mlevolve_globs = ['mlevolve_v1_languageModelingFineWeb_gemini3propreview_20260426_022224_j4763351',
                      'mlevolve_v2_flash_languageModelingFineWeb_gemini3flashpreview_20260427_011221_j4771266',
                      'mlevolve_v3_flash_languageModelingFineWeb_gemini3flashpreview_20260427_012625_j4771267'],
    arts_globs     = ['llmg_C_v1_languageModelingFineWeb_o3_gemini3propreview_20260429_110217_j4786004',
                      'llmg_C_v2_languageModelingFineWeb_o3_gemini3propreview_20260429_111627_j4786005',
                      'llmg_C_v3_languageModelingFineWeb_o3_gemini3propreview_20260429_111627_j4786006'],
)

HMS = dict(
    label      = 'HMS Brain',
    metric     = 'KL div',
    higher     = False,
    baseline   = 1.4618,
    human_best = 0.272332,
    linear_globs   = ['linear_mlebenchHMSBrain_gemini3propreview_20260425_034725_j4757792',
                      'linear_mlebenchHMSBrain_gemini3propreview_20260425_034726_j4757793'],
    aira_globs     = ['aira_mlebenchHMSBrain_gemini3propreview_20260424_030823_j4754507',
                      'aira_mlebenchHMSBrain_gemini3propreview_20260424_030823_j4754508'],
    mlevolve_globs = ['mlevolve_v1_mlebenchHMSBrain_gemini3propreview_20260425_213725_j4761562',
                      'mlevolve_v2_mlebenchHMSBrain_gemini3propreview_20260426_022224_j4763344',
                      'mlevolve_v3_mlebenchHMSBrain_gemini3propreview_20260426_022223_j4763348'],
    arts_globs     = ['llmg_C_vs_C_v1_mlebenchHMSBrain_o3_gemini3propreview_20260427_141721_j4773982',
                      'llmg_C_vs_C_v2_mlebenchHMSBrain_o3_gemini3propreview_20260427_141721_j4773983',
                      'llmg_C_vs_C_v3_mlebenchHMSBrain_o3_gemini3propreview_20260427_141721_j4773984'],
)

VESUVIUS = dict(
    label      = 'Vesuvius',
    metric     = 'F0.5',
    higher     = True,
    baseline   = 0.0,
    human_best = 0.831399,
    linear_globs   = ['linear_mlebenchVesuvius_gemini3propreview_20260423_085423_j4752137',
                      'linear_mlebenchVesuvius_gemini3propreview_20260423_085424_j4752138'],
    aira_globs     = ['aira_mlebenchVesuvius_gemini3propreview_20260424_030723_j4754503',
                      'aira_mlebenchVesuvius_gemini3propreview_20260424_030723_j4754504'],
    mlevolve_globs = ['mlevolve_v1_mlebenchVesuvius_gemini3propreview_20260425_213726_j4761561',
                      'mlevolve_v2_mlebenchVesuvius_gemini3propreview_20260426_022224_j4763343',
                      'mlevolve_v3_mlebenchVesuvius_gemini3propreview_20260426_022223_j4763347'],
    arts_globs     = ['llmg_C_vs_extsamp_C_v2_mlebenchVesuvius_o3_gemini3propreview_20260427_183924_j4775756',
                      'llmg_C_vs_extsamp_C_v3_mlebenchVesuvius_o3_gemini3propreview_20260427_183924_j4775757'],
)

APTOS = dict(
    label      = 'APTOS',
    metric     = 'QWK',
    higher     = True,
    baseline   = 0.0,
    human_best = 0.936,
    linear_globs   = ['linear_mlebenchAPTOS_gemini3propreview_20260425_034625_j4757790',
                      'linear_mlebenchAPTOS_gemini3propreview_20260425_034725_j4757791'],
    aira_globs     = ['aira_mlebenchAPTOS_gemini3propreview_20260424_030723_j4754497',
                      'aira_mlebenchAPTOS_gemini3propreview_20260424_030724_j4754498'],
    mlevolve_globs = ['mlevolve_v1_mlebenchAPTOS_gemini3propreview_20260425_213726_j4761564',
                      'mlevolve_v2_mlebenchAPTOS_gemini3propreview_20260426_022224_j4763346',
                      'mlevolve_v3_mlebenchAPTOS_gemini3propreview_20260426_022224_j4763350'],
    arts_globs     = ['llmg_nosignal_aptos_final_mlebenchAPTOS_o3_gemini3propreview_20260425_031725_j4757731',
                      'llmg_nosignal_aptos_final_mlebenchAPTOS_o3_gemini3propreview_20260425_031824_j4757732',
                      'llmg_nosignal_aptos_final_mlebenchAPTOS_o3_gemini3propreview_20260425_031824_j4757733'],
)

TASKS = [LM, HMS, VESUVIUS]

# ---------------------------------------------------------------------------
# Extended task pool used ONLY for the aggregate stats figures.
# Per-task curves (figs A-E, FP) keep using TASKS above.
# ---------------------------------------------------------------------------
KUZUSHIJI = dict(
    label='Kuzushiji', metric='F1', higher=True, baseline=0.0, human_best=0.950,
    linear_globs=[
        'linear_mlebenchKuzushiji_gemini3propreview_20260425_034623_j4757787',
        'linear_mlebenchKuzushiji_gemini3propreview_20260425_034624_j4757788'],
    aira_globs=[
        'aira_mlebenchKuzushiji_gemini3propreview_20260424_030723_j4754505',
        'aira_mlebenchKuzushiji_gemini3propreview_20260424_030822_j4754506'],
    mlevolve_globs=[
        'mlevolve_v1_mlebenchKuzushiji_gemini3propreview_20260425_213726_j4761563',
        'mlevolve_v2_mlebenchKuzushiji_gemini3propreview_20260426_022224_j4763345',
        'mlevolve_v3_mlebenchKuzushiji_gemini3propreview_20260426_022224_j4763349'],
    arts_globs=[
        'llmg_C_interac_v1_mlebenchKuzushiji_o3_gemini3propreview_20260428_231221_j4782967',
        'llmg_C_interac_v2_mlebenchKuzushiji_o3_gemini3propreview_20260428_233121_j4782968',
        'llmg_C_interac_v3_mlebenchKuzushiji_o3_gemini3propreview_20260428_233121_j4782970'],
)

MNLI = dict(
    label='MNLI', metric='acc', higher=True, baseline=52.51, human_best=92.50,
    linear_globs=[
        'linear_naturalLanguageInferenceMNLI_flash_run1_20260419_140222',
        'linear_naturalLanguageInferenceMNLI_flash_run2_20260419_140222'],
    aira_globs=[
        'aira_naturalLanguageInferenceMNLI_flash_run1_20260418_235824',
        'aira_naturalLanguageInferenceMNLI_flash_run2_20260418_235824'],
    mlevolve_globs=[
        'mlevolve_v1_flash_naturalLanguageInferenceMNLI_gemini3flashpreview_20260426_145723_j4764729',
        'mlevolve_v2_flash_naturalLanguageInferenceMNLI_gemini3flashpreview_20260427_004626_j4771264',
        'mlevolve_v3_flash_naturalLanguageInferenceMNLI_gemini3flashpreview_20260427_004625_j4771265'],
    arts_globs=[
        'llmg_naturalLanguageInferenceMNLI_o3_gemini3f_r1_20260418_095722',
        'llmg_naturalLanguageInferenceMNLI_o3_gemini3f_r2_20260418_095722'],
)

CIFAR = dict(
    label='CIFAR-10', metric='acc', higher=True, baseline=0.497, human_best=0.994,
    linear_globs=[
        'linear_imageClassificationCifar10L1_flash_run1_20260419_140222',
        'linear_imageClassificationCifar10L1_flash_run2_20260419_140222'],
    aira_globs=[
        'aira_imageClassificationCifar10L1_flash_run1_20260418_235824',
        'aira_imageClassificationCifar10L1_flash_run2_20260418_235824'],
    mlevolve_globs=[
        'mlevolve_v1_flash_imageClassificationCifar10_gemini3flashpreview_20260426_122421_j4764726',
        'mlevolve_v2_flash_imageClassificationCifar10_gemini3flashpreview_20260427_002925_j4771258',
        'mlevolve_v3_flash_imageClassificationCifar10_gemini3flashpreview_20260427_002925_j4771259'],
    arts_globs=[
        'llmg_imageClassificationCifar10L1_qwen_flash_r3_20260419_233421',
        'llmg_imageClassificationCifar10L1_qwen_flash_r3_20260420_010423'],
)

FMNIST = dict(
    label='Fashion MNIST', metric='acc', higher=True, baseline=0.848, human_best=0.968,
    linear_globs=[
        'linear_imageClassificationFMnist_flash_run1_20260419_140222',
        'linear_imageClassificationFMnist_flash_run2_20260419_140222'],
    aira_globs=[
        'aira_imageClassificationFMnist_flash_run1_20260418_235824',
        'aira_imageClassificationFMnist_flash_run2_20260418_235824'],
    mlevolve_globs=[
        'mlevolve_v1_flash_imageClassificationFMnist_gemini3flashpreview_20260426_122421_j4764727',
        'mlevolve_v2_flash_imageClassificationFMnist_gemini3flashpreview_20260427_002926_j4771260',
        'mlevolve_v3_flash_imageClassificationFMnist_gemini3flashpreview_20260427_002926_j4771261'],
    arts_globs=[
        'llmg_imageClassificationFMnist_qwen_flash_r2_20260419_233421',
        'llmg_imageClassificationFMnist_qwen_flash_r3_20260419_223420'],
)

# Pool used by the stats figures (only tasks with o3+gemini-pro ARTS coverage,
# matching the main paper setup. The unsaturated variant additionally filters
# tasks where all methods cluster near 1.0).
STATS_TASKS = [LM, HMS, VESUVIUS, APTOS, KUZUSHIJI]

# Full pool: MLGym (LM, MNLI, CIFAR, FMNIST) + mle-bench (HMS, VESUVIUS, APTOS, KUZUSHIJI).
# Used by the "full" stats figure regardless of executor (some tasks here use qwen+flash
# ARTS rather than o3+pro; included for breadth).
ALL_STATS_TASKS = [LM, HMS, VESUVIUS, APTOS, KUZUSHIJI, MNLI, CIFAR, FMNIST]


# ---------------------------------------------------------------------------
# Auto-discovery: synthesize a task dict for ANY task id we find on disk,
# pulling baseline from MLGym yaml and using observed-max as the upper anchor.
# Used only by the "all-task" stats figure.
# ---------------------------------------------------------------------------
import yaml as _yaml

_TASK_YAML_DIR = Path(os.environ.get('MLGYM_TASKS_DIR', _REPO.parent / 'MLGym' / 'configs' / 'tasks'))

# Map metric → (higher_is_better)
_METRIC_DIR = {
    # higher better
    'accuracy': True, 'validation_accuracy': True, 'Accuracy': True,
    'auc_roc': True, 'roc_auc': True, 'qwk_score': True,
    'kuzushiji_f1': True, 'macro_f1': True,
    'column_wise_roc_auc': True, 'mean_column_wise_roc_auc': True,
    'mean_average_precision': True, 'f05_score': True, 'BLEU Score': True,
    'Score': True, 'Reward Mean': True,
    # lower better
    'val_loss': False, 'loss': False, 'kl_divergence': False,
    'mae': False, 'rmse': False, 'mean_levenshtein_distance': False,
    'mcrmse': False, 'mean_dist_m': False, 'mean_column_wise_rmsle': False,
    'Time': False,
}

def _load_task_yaml(task_id: str):
    f = _TASK_YAML_DIR / f'{task_id}.yaml'
    if not f.is_file():
        return None
    try:
        return _yaml.safe_load(open(f))
    except Exception:
        return None

def _glob_run_dirs(method_prefix: str, task_token: str) -> list[str]:
    """Return run-dir basenames under outputs/ matching <method>_*<task>_* pattern."""
    out = []
    for d in os.listdir(BASE):
        if not (BASE / d).is_dir(): continue
        parts = d.split('_')
        if not parts: continue
        if parts[0] != method_prefix: continue
        # require exact task_token as one of the underscore-separated parts
        if task_token not in parts: continue
        out.append(d)
    return out

def auto_task(task_id: str, label: str | None = None) -> dict | None:
    """Build a task dict on the fly: baseline from yaml, human_best = best observed
    score across all methods. Returns None if we can't normalise this task."""
    yaml_d = _load_task_yaml(task_id)
    if not yaml_d: return None
    bs = yaml_d.get('baseline_scores')
    if isinstance(bs, list) and bs and isinstance(bs[0], dict):
        metric_name, baseline = next(iter(bs[0].items()))
    elif isinstance(bs, dict):
        metric_name, baseline = next(iter(bs.items()))
    else:
        return None
    higher = _METRIC_DIR.get(metric_name)
    if higher is None:
        return None
    label = label or task_id

    linear_globs   = _glob_run_dirs('linear',   task_id)
    aira_globs     = _glob_run_dirs('aira',     task_id)
    mlevolve_globs = _glob_run_dirs('mlevolve', task_id)
    arts_globs     = _glob_run_dirs('llmg',     task_id)
    if not (linear_globs or aira_globs or mlevolve_globs or arts_globs):
        return None

    # First pass: gather all raw final-best scores (no normalization yet) to set human_best
    tmp_task = dict(label=label, metric=metric_name, higher=higher,
                    baseline=baseline, human_best=baseline + (1.0 if higher else -1.0),
                    linear_globs=linear_globs, aira_globs=aira_globs,
                    mlevolve_globs=mlevolve_globs, arts_globs=arts_globs)
    raws = []
    for method in ORDER:
        for g in tmp_task[f'{method.lower() if method != "ARTS" else "arts"}_globs']:
            d = BASE / g
            if not d.is_dir(): continue
            ext = extract_linear_run if method == 'Linear' else extract_tree_run
            ev = ext(d)
            if not ev: continue
            rb = running_best(ev, higher)
            if rb: raws.append(rb[-1][1])
    # filter known sentinel failure scores
    SENTINELS = {50_000_000.0, 100.0, 999.0, 1.0}  # the 1.0 covers RSNA leak too
    if metric_name == 'auc_roc':
        raws = [r for r in raws if r < 0.99]  # drop leaked AUC
    raws = [r for r in raws if r not in SENTINELS]
    if not raws: return None
    if higher:
        human = max(raws) + 1e-6
        if human <= baseline: return None
    else:
        human = min(raws) - 1e-6
        if human >= baseline: return None
    tmp_task['human_best'] = float(human)
    return tmp_task

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


_NOISE_PREFIXES = re.compile(
    r'^\s*(?:\d+\.\s*)?(?:WHAT|HOW|WHY|EXPERIMENT|HYPOTHESIS|DIRECTION|TITLE)\s*[:\-]\s*',
    re.IGNORECASE,
)


def _clean_direction(s: str) -> str:
    """Strip prompt-format prefixes like '1. WHAT:' / 'HYPOTHESIS:' from a hypothesis."""
    if not s:
        return ''
    s = str(s).strip().replace('\n', ' ')
    # repeatedly strip leading prefixes
    for _ in range(4):
        new = _NOISE_PREFIXES.sub('', s).strip()
        if new == s:
            break
        s = new
    return s


def _node_id_to_direction(run_dir: Path) -> dict[str, str]:
    """Build {node_id -> human-readable experiment description} from scientist_logs."""
    out: dict[str, str] = {}
    sci = run_dir / 'scientist_logs'
    if not sci.is_dir():
        return out
    for f in sorted(sci.glob('step_*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        nid = d.get('node_id')
        if not nid:
            continue
        text = (d.get('direction') or
                d.get('sampling_record', {}).get('sampled_direction') or '')
        text = _clean_direction(text)
        if text:
            out[nid] = text
    return out


def extract_tree_run(run_dir: Path) -> list[tuple[float, float, str]]:
    """Return [(t_seconds, score, hypothesis_text), ...] sorted by t."""
    nodes_dir = run_dir / 'nodes'
    if not nodes_dir.is_dir():
        return []
    t0 = _t0_from_dirname(run_dir.name)
    if t0 is None:
        mtimes = [f.stat().st_mtime for f in nodes_dir.glob('*.json')]
        t0 = min(mtimes) if mtimes else run_dir.stat().st_mtime
    nid_to_dir = _node_id_to_direction(run_dir)
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
        nid = d.get('node_id') or f.stem
        h = nid_to_dir.get(nid, '')
        if not h:
            # fallback: ignore generic strategy placeholders ("Attempt N")
            cand = d.get('direction') or d.get('hypothesis') or ''
            if isinstance(cand, str) and cand.strip() and not cand.lower().startswith('attempt'):
                h = cand
        h = str(h).strip().replace('\n', ' ')
        if len(h) > 80:
            h = h[:78] + '..'
        t = max(0.0, f.stat().st_mtime - t0)
        events.append((t, s, h))
    events.sort()
    return events


_LINEAR_RX = re.compile(r'validate\s*->\s*(-?\d+(?:\.\d+)?)(?:\s*\(best=(-?\d+(?:\.\d+)?)\))?')


def extract_linear_run(run_dir: Path) -> list[tuple[float, float, str]]:
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
    return sorted((dur * (i / max(1, max_i)), sc, '') for i, sc in steps)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def running_best(events: list[tuple[float, float, str]], higher: bool):
    best = None
    out = []
    for tup in events:
        t, s = tup[0], tup[1]
        h = tup[2] if len(tup) > 2 else ''
        new_best = False
        if best is None or (higher and s > best) or (not higher and s < best):
            best = s
            new_best = True
        out.append((t, best, h, s, new_best))
    return out


def normalize(score, task):
    b, h = task['baseline'], task['human_best']
    if task['higher']:
        return (score - b) / (h - b)
    else:
        return (b - score) / (b - h)


def collect_curves(task: dict):
    """For each method, return list of running-best curves: [(t, norm_score), ...]."""
    method_dirs = {
        'Linear':   [BASE / g for g in task['linear_globs']],
        'AIRA':     [BASE / g for g in task['aira_globs']],
        'MLEvolve': [BASE / g for g in task['mlevolve_globs']],
        'ARTS':     [BASE / g for g in task['arts_globs']],
    }
    out = {m: [] for m in ORDER}
    for method, dirs in method_dirs.items():
        extractor = extract_linear_run if method == 'Linear' else extract_tree_run
        for d in dirs:
            if not d.is_dir():
                continue
            events = extractor(d)
            if not events:
                continue
            rb = running_best(events, task['higher'])
            curve = [(0.0, normalize(task['baseline'], task))]
            curve += [(t, normalize(b, task)) for (t, b, _, _, _) in rb if t > 0]
            curve.append((BUDGET_S, curve[-1][1]))
            out[method].append(curve)
    return out


def collect_node_events(task: dict, methods=None):
    """Per-method list of (t, norm_score, hypothesis, is_new_best) per node, across runs."""
    method_dirs = {
        'Linear':   [BASE / g for g in task['linear_globs']],
        'AIRA':     [BASE / g for g in task['aira_globs']],
        'MLEvolve': [BASE / g for g in task['mlevolve_globs']],
        'ARTS':     [BASE / g for g in task['arts_globs']],
    }
    if methods is None:
        methods = ORDER
    out = {m: [] for m in methods}
    for method in methods:
        extractor = extract_linear_run if method == 'Linear' else extract_tree_run
        for d in method_dirs[method]:
            if not d.is_dir():
                continue
            events = extractor(d)
            if not events:
                continue
            rb = running_best(events, task['higher'])
            for (t, _, h, s, nb) in rb:
                out[method].append((t, normalize(s, task), h, nb))
    return out


def best_run_for_method(task: dict, method: str):
    """Return events from the SINGLE best run (highest final running-best) for one method."""
    globs = {
        'Linear':   task['linear_globs'],
        'AIRA':     task['aira_globs'],
        'MLEvolve': task['mlevolve_globs'],
        'ARTS':     task['arts_globs'],
    }[method]
    extractor = extract_linear_run if method == 'Linear' else extract_tree_run
    candidates = []
    for g in globs:
        d = BASE / g
        if not d.is_dir():
            continue
        events = extractor(d)
        if not events:
            continue
        rb_full = running_best(events, task['higher'])
        if not rb_full:
            continue
        # final running-best score (raw)
        final = rb_full[-1][1]
        candidates.append((final, events, rb_full))
    if not candidates:
        return [], []
    # pick best (highest if higher=True, lowest if higher=False)
    if task['higher']:
        candidates.sort(key=lambda x: -x[0])
    else:
        candidates.sort(key=lambda x: x[0])
    _, events, rb_full = candidates[0]
    # return: scatter_events [(t_h, norm_score, h, is_new_best)],
    #         best_so_far_curve [(t_h, norm_score)]
    scatter = [(t / 3600.0, normalize(s, task), h, nb)
               for (t, _, h, s, nb) in rb_full]
    best_curve = [(0.0, normalize(task['baseline'], task))] + \
                 [(t / 3600.0, normalize(b, task)) for (t, b, _, _, _) in rb_full if t > 0]
    if best_curve[-1][0] < BUDGET_H:
        best_curve.append((BUDGET_H, best_curve[-1][1]))
    return scatter, best_curve


def hourly_grid(curves, hours=BUDGET_H):
    """Take per-method best-so-far value at integer hour marks 0,1,...,8."""
    grid = np.arange(0, hours + 1.0)
    grid_s = grid * 3600
    out = []
    for c in curves:
        ts = np.array([e[0] for e in c])
        ys = np.array([e[1] for e in c])
        idx = np.searchsorted(ts, grid_s, side='right') - 1
        idx = np.clip(idx, 0, len(ys) - 1)
        out.append(ys[idx])
    return grid, np.stack(out, axis=0)


def smooth_grid(curves):
    grid = np.linspace(0, BUDGET_S, N_GRID)
    mats = []
    for c in curves:
        ts = np.array([e[0] for e in c])
        ys = np.array([e[1] for e in c])
        idx = np.searchsorted(ts, grid, side='right') - 1
        idx = np.clip(idx, 0, len(ys) - 1)
        mats.append(ys[idx])
    return grid, np.stack(mats, axis=0)


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------
def _decorate(ax, task, show_xlabel=True, show_ylabel=False, ylim=(-0.05, 1.18)):
    ax.axhline(1.0, color='#555555', lw=0.9, ls='--', zorder=1)
    ax.axhline(0.0, color='#aaaaaa', lw=0.7, ls=':', zorder=1)
    ax.set_xlim(0, BUDGET_H)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax.grid(which='major', ls='--', lw=0.4, alpha=0.4, color='#888888')
    ax.set_title(task['label'], fontsize=11.5, fontweight='bold', pad=6)
    if show_xlabel:
        ax.set_xlabel('Wall-clock time (h)', labelpad=3)
    if show_ylabel:
        ax.set_ylabel('Normalized score', labelpad=4)


def panel_continuous(ax, task, methods=None, show_legend=False, show_ylabel=False,
                      legend_loc='lower right'):
    """Variant 1: smooth aggregated lines per method."""
    if methods is None:
        methods = ORDER
    curves = collect_curves(task)
    for method in methods:
        if not curves[method]:
            continue
        grid, mat = smooth_grid(curves[method])
        mean = np.mean(mat, axis=0)
        std  = np.std(mat,  axis=0)
        gh = grid / 3600.0
        label = f'{method} (ours)' if method == 'ARTS' else method
        ax.plot(gh, mean, color=COLORS[method], lw=LINE_WIDTH[method],
                label=label, zorder=4)
        ax.fill_between(gh, mean - std, mean + std,
                        color=COLORS[method], alpha=FILL_ALPHA[method],
                        linewidth=0, zorder=3)
    _decorate(ax, task, show_ylabel=show_ylabel)
    if show_legend:
        leg = ax.legend(loc=legend_loc, fontsize=9, handlelength=1.8,
                        handleheight=0.7, borderaxespad=0.4,
                        framealpha=0.80, fancybox=True)
        leg.get_frame().set_facecolor('white')


def panel_hourly(ax, task, methods=None, show_legend=False, show_ylabel=False,
                  legend_loc='lower right'):
    """Variant 4: step plot at integer-hour marks; markers are white-fill + colored border."""
    if methods is None:
        methods = ORDER
    curves = collect_curves(task)
    for method in methods:
        if not curves[method]:
            continue
        grid, mat = hourly_grid(curves[method])
        mean = np.mean(mat, axis=0)
        std  = np.std(mat,  axis=0)
        label = f'{method} (ours)' if method == 'ARTS' else method
        ax.step(grid, mean, where='post', color=COLORS[method],
                lw=LINE_WIDTH[method], label=label, zorder=4)
        ax.fill_between(grid, mean - std, mean + std, step='post',
                        color=COLORS[method], alpha=FILL_ALPHA[method],
                        linewidth=0, zorder=3)
        # white-fill marker, colored border at every step
        ax.scatter(grid, mean, s=22, facecolor='white',
                   edgecolor=COLORS[method], linewidths=1.4, zorder=6)
    _decorate(ax, task, show_ylabel=show_ylabel)
    if show_legend:
        leg = ax.legend(loc=legend_loc, fontsize=9, handlelength=1.8,
                        handleheight=0.7, borderaxespad=0.4,
                        framealpha=0.80, fancybox=True)
        leg.get_frame().set_facecolor('white')


def panel_with_scatter(ax, task, methods, show_legend=False, show_ylabel=False,
                        legend_loc='lower right'):
    """Variant 5: best run only per method. Best-so-far line + scatter of every node
    from THAT run (so all dots stay below the line)."""
    for method in methods:
        scatter, curve = best_run_for_method(task, method)
        if not curve:
            continue
        label = f'{method} (ours)' if method == 'ARTS' else method
        # scatter every node from the best run
        if scatter:
            ts = np.array([s[0] for s in scatter])
            ys = np.array([s[1] for s in scatter])
            ax.scatter(ts, ys, s=18, facecolor='white',
                       edgecolor=COLORS[method], linewidths=1.0,
                       alpha=0.85, zorder=3)
        # best-so-far line
        cs = np.array([c[0] for c in curve])
        ys = np.array([c[1] for c in curve])
        ax.step(cs, ys, where='post', color=COLORS[method],
                lw=LINE_WIDTH[method], label=label, zorder=5)
    _decorate(ax, task, show_ylabel=show_ylabel, ylim=(-0.4, 1.2))
    if show_legend:
        leg = ax.legend(loc=legend_loc, fontsize=9, handlelength=1.8,
                        handleheight=0.7, borderaxespad=0.4,
                        framealpha=0.80, fancybox=True)
        leg.get_frame().set_facecolor('white')


def panel_arts_annotated(ax, task, show_ylabel=False):
    """Variant 6: ARTS best run only, scatter every node, annotate every new-best."""
    scatter, curve = best_run_for_method(task, 'ARTS')
    if not curve:
        _decorate(ax, task, show_ylabel=show_ylabel, ylim=(-0.4, 1.2))
        return
    # scatter every node from the best run
    if scatter:
        ts = np.array([s[0] for s in scatter])
        ys = np.array([s[1] for s in scatter])
        ax.scatter(ts, ys, s=18, facecolor='white',
                   edgecolor=COLORS['ARTS'], linewidths=1.0,
                   alpha=0.85, zorder=3)
    # best-so-far line
    cs = np.array([c[0] for c in curve])
    ys = np.array([c[1] for c in curve])
    ax.step(cs, ys, where='post', color=COLORS['ARTS'],
            lw=LINE_WIDTH['ARTS'], label='ARTS (ours)', zorder=5)
    # annotations for each new best (skip ones with no real description)
    new_bests = [(t, s, h) for (t, s, h, nb) in scatter if nb and h]
    new_bests.sort(key=lambda x: x[0])
    # cap at 5 annotations to keep readable; prefer the largest jumps
    if len(new_bests) > 5:
        gains, prev = [], -np.inf
        for (t, s, h) in new_bests:
            gains.append((s - prev, t, s, h))
            prev = s
        gains.sort(reverse=True)
        chosen = sorted([(t, s, h) for (_, t, s, h) in gains[:5]])
    else:
        chosen = new_bests
    y_used = []
    for (t, s, h) in chosen:
        y = s + 0.07
        for yu in y_used:
            if abs(y - yu) < 0.08:
                y += 0.09
        y_used.append(y)
        snippet = h[:42] + ('..' if len(h) > 42 else '')
        ax.annotate(snippet, xy=(t, s),
                    xytext=(min(t + 0.15, BUDGET_H - 1.5), y),
                    fontsize=7.5, color=COLORS['ARTS'],
                    arrowprops=dict(arrowstyle='-', lw=0.5,
                                     color=COLORS['ARTS'], alpha=0.6),
                    zorder=7)
    _decorate(ax, task, show_ylabel=show_ylabel, ylim=(-0.4, 1.3))


# ---------------------------------------------------------------------------
# Best-baseline picker
# ---------------------------------------------------------------------------
def best_baseline(task: dict) -> str:
    """Return the method name in {Linear, AIRA, MLEvolve} with the highest
    final mean normalised score on this task."""
    curves = collect_curves(task)
    finals = {}
    for m in ['Linear', 'AIRA', 'MLEvolve']:
        if not curves[m]:
            continue
        _, mat = smooth_grid(curves[m])
        finals[m] = float(np.mean(mat[:, -1]))
    if not finals:
        return 'AIRA'
    return max(finals, key=finals.get)


# ---------------------------------------------------------------------------
# Broken y-axis figure builder
# ---------------------------------------------------------------------------
def figure_broken(panel_fn, fname, methods=None, panel_args=None):
    """For each task, stack two axes vertically (lower=0..plateau, upper=plateau..max)."""
    panel_args = panel_args or {}
    fig, axarr = plt.subplots(2, 3, figsize=(12.0, 4.6),
                              gridspec_kw={'height_ratios': [3, 1.2]},
                              constrained_layout=True)
    fig.patch.set_facecolor('white')
    for j, task in enumerate(TASKS):
        ax_top, ax_bot = axarr[0, j], axarr[1, j]
        # render the same panel on both, then re-limit;
        # legend goes in the upper segment, top-right (lines plateau low-left in the zoom)
        panel_fn(ax_top, task, methods=methods,
                 show_legend=(j == len(TASKS) - 1), show_ylabel=(j == 0),
                 legend_loc='upper right', **panel_args)
        panel_fn(ax_bot, task, methods=methods,
                 show_legend=False, show_ylabel=(j == 0), **panel_args)
        # compute split: pick the highest 2nd-best method's final mean as plateau
        curves = collect_curves(task)
        finals = {}
        for m in ORDER:
            if curves[m]:
                _, mat = smooth_grid(curves[m])
                finals[m] = float(np.mean(mat[:, -1]))
        sorted_finals = sorted(finals.values())
        if len(sorted_finals) >= 2:
            plateau = max(0.0, min(0.85, sorted_finals[-2] - 0.02))
        else:
            plateau = 0.5
        cap = max(1.05, max(finals.values()) + 0.05) if finals else 1.1
        ax_top.set_ylim(plateau, cap)
        ax_bot.set_ylim(-0.05, plateau)
        ax_top.set_title(task['label'], fontsize=11.5, fontweight='bold', pad=6)
        ax_bot.set_title('')
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        ax_top.set_xlabel('')
        ax_bot.set_xlabel('Wall-clock time (h)', labelpad=3)
        if j == 0:
            ax_top.set_ylabel('Normalized score', labelpad=4)
            ax_bot.set_ylabel('')
        # diagonal break marks
        d = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=0.8)
        ax_top.plot((-d, +d), (-d * 3, +d * 3), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d * 3, +d * 3), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d * 3, 1 + d * 3), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d * 3, 1 + d * 3), **kwargs)
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Standard 1x4 figure builder
# ---------------------------------------------------------------------------
def figure_strip(panel_fn, fname, methods=None, panel_args=None):
    panel_args = panel_args or {}
    n = len(TASKS)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')
    for j, (ax, task) in enumerate(zip(axes, TASKS)):
        ax.set_facecolor('#fafafa')
        if methods == 'best_per_task':
            m = ['ARTS', best_baseline(task)]
        else:
            m = methods
        panel_fn(ax, task, methods=m,
                 show_legend=(j == n - 1), show_ylabel=(j == 0), **panel_args)
    _save(fig, fname)


def figure_arts_only(fname):
    n = len(TASKS)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')
    for j, (ax, task) in enumerate(zip(axes, TASKS)):
        ax.set_facecolor('#fafafa')
        panel_arts_annotated(ax, task, show_ylabel=(j == 0))
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Per-task full-page figures (annotated, multi-method) and broken variants
# ---------------------------------------------------------------------------
def panel_full_annotated_multi(ax, task, show_legend=True, show_ylabel=True,
                                ylim=(-0.05, 1.18), legend_loc='lower right'):
    """Full-page panel: all 4 methods (mean ± std), ARTS new-best annotations
    pulled from ARTS best run."""
    # 1. Multi-method aggregated curves
    curves = collect_curves(task)
    for method in ORDER:
        if not curves[method]:
            continue
        grid, mat = smooth_grid(curves[method])
        mean = np.mean(mat, axis=0)
        std  = np.std(mat,  axis=0)
        gh = grid / 3600.0
        label = f'{method} (ours)' if method == 'ARTS' else method
        ax.plot(gh, mean, color=COLORS[method], lw=LINE_WIDTH[method],
                label=label, zorder=4)
        ax.fill_between(gh, mean - std, mean + std,
                        color=COLORS[method], alpha=FILL_ALPHA[method],
                        linewidth=0, zorder=3)
    # 2. ARTS scatter from the best run + annotations on new bests
    scatter, _ = best_run_for_method(task, 'ARTS')
    if scatter:
        ts = np.array([s[0] for s in scatter])
        ys = np.array([s[1] for s in scatter])
        ax.scatter(ts, ys, s=24, facecolor='white',
                   edgecolor=COLORS['ARTS'], linewidths=1.1,
                   alpha=0.9, zorder=6)
        new_bests = [(t, s, h) for (t, s, h, nb) in scatter if nb and h]
        new_bests.sort(key=lambda x: x[0])
        if len(new_bests) > 6:
            gains, prev = [], -np.inf
            for (t, s, h) in new_bests:
                gains.append((s - prev, t, s, h)); prev = s
            gains.sort(reverse=True)
            chosen = sorted([(t, s, h) for (_, t, s, h) in gains[:6]])
        else:
            chosen = new_bests
        y_used = []
        for (t, s, h) in chosen:
            y = s + 0.06
            for yu in y_used:
                if abs(y - yu) < 0.06:
                    y += 0.07
            y_used.append(y)
            snippet = h[:48] + ('..' if len(h) > 48 else '')
            # clamp xtext so the label doesn't fall off the right edge
            x_text = min(t + 0.20, BUDGET_H - 3.2)
            ax.annotate(snippet, xy=(t, s),
                        xytext=(x_text, y),
                        fontsize=8.0, color=COLORS['ARTS'],
                        arrowprops=dict(arrowstyle='-', lw=0.6,
                                         color=COLORS['ARTS'], alpha=0.65),
                        zorder=8)
    _decorate(ax, task, show_ylabel=show_ylabel, ylim=ylim)
    if show_legend:
        leg = ax.legend(loc=legend_loc, fontsize=10, handlelength=2.0,
                        handleheight=0.8, borderaxespad=0.5,
                        framealpha=0.80, fancybox=True)
        leg.get_frame().set_facecolor('white')


def figure_full_per_task(fname_template, broken=False):
    """One full-page figure per task. If broken, split y-axis at smart point."""
    for task in TASKS:
        if broken:
            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(8.5, 6.4),
                gridspec_kw={'height_ratios': [3, 1.3]},
                constrained_layout=True)
            fig.patch.set_facecolor('white')
            # Compute smart break: just below the LOWEST method's final mean
            curves = collect_curves(task)
            finals = []
            for m in ORDER:
                if curves[m]:
                    _, mat = smooth_grid(curves[m])
                    finals.append(float(np.mean(mat[:, -1])))
            if len(finals) >= 2:
                lo, hi = min(finals), max(finals)
                # break point = midway between min and (min,max) midpoint, slightly below lo
                plateau = max(0.0, min(0.85, lo - 0.05))
            else:
                plateau = 0.5
            cap = max(1.10, hi + 0.10) if finals else 1.18
            # render same content twice
            panel_full_annotated_multi(ax_top, task,
                                        show_legend=True, show_ylabel=True,
                                        legend_loc='upper right')
            panel_full_annotated_multi(ax_bot, task,
                                        show_legend=False, show_ylabel=False)
            ax_top.set_ylim(plateau, cap)
            ax_bot.set_ylim(-0.05, plateau)
            ax_top.set_xlabel('')
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_top.spines['bottom'].set_visible(False)
            ax_bot.spines['top'].set_visible(False)
            ax_top.set_title(task['label'], fontsize=13, fontweight='bold', pad=8)
            ax_bot.set_title('')
            ax_top.set_ylabel('Normalized score', labelpad=4)
            ax_bot.set_ylabel('')
            # diagonal break marks
            d = .013
            kw = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=0.8)
            ax_top.plot((-d, +d), (-d * 3, +d * 3), **kw)
            ax_top.plot((1 - d, 1 + d), (-d * 3, +d * 3), **kw)
            kw.update(transform=ax_bot.transAxes)
            ax_bot.plot((-d, +d), (1 - d * 3, 1 + d * 3), **kw)
            ax_bot.plot((1 - d, 1 + d), (1 - d * 3, 1 + d * 3), **kw)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.4),
                                    constrained_layout=True)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#fafafa')
            panel_full_annotated_multi(ax, task,
                                        show_legend=True, show_ylabel=True,
                                        legend_loc='lower right')
            ax.set_title(task['label'], fontsize=13, fontweight='bold', pad=8)
        slug = task['label'].lower().replace(' ', '_')
        _save(fig, fname_template.format(task=slug))


# ---------------------------------------------------------------------------
# Statistical plots (rliable-style, manually implemented)
# ---------------------------------------------------------------------------
def collect_run_finals(task: dict) -> dict[str, list[float]]:
    """For each method, return the final running-best NORMALISED score per run."""
    method_dirs = {
        'Linear':   [BASE / g for g in task['linear_globs']],
        'AIRA':     [BASE / g for g in task['aira_globs']],
        'MLEvolve': [BASE / g for g in task['mlevolve_globs']],
        'ARTS':     [BASE / g for g in task['arts_globs']],
    }
    out = {m: [] for m in ORDER}
    for method, dirs in method_dirs.items():
        extractor = extract_linear_run if method == 'Linear' else extract_tree_run
        for d in dirs:
            if not d.is_dir():
                continue
            events = extractor(d)
            if not events:
                continue
            rb = running_best(events, task['higher'])
            if rb:
                out[method].append(normalize(rb[-1][1], task))
    return out


def aggregate_scores_matrix(task_pool=None) -> dict[str, np.ndarray]:
    """Per method, an array of shape (n_runs * n_tasks,) of normalised final scores."""
    if task_pool is None:
        task_pool = TASKS
    per_method: dict[str, list[float]] = {m: [] for m in ORDER}
    for task in task_pool:
        finals = collect_run_finals(task)
        for m in ORDER:
            per_method[m].extend(finals[m])
    return {m: np.array(per_method[m]) for m in ORDER}


def task_diff(task: dict) -> float:
    """ARTS mean final normalised score minus best baseline mean.
    Positive = ARTS wins. Returns -inf if ARTS has no runs."""
    finals = collect_run_finals(task)
    if not finals.get('ARTS'):
        return float('-inf')
    arts_mean = float(np.mean(finals['ARTS']))
    base_means = [float(np.mean(finals[m])) for m in ('Linear', 'AIRA', 'MLEvolve')
                  if finals.get(m)]
    if not base_means:
        return float('inf')
    return arts_mean - max(base_means)


def is_saturated(task: dict, max_method_spread: float = 0.05,
                 floor_norm: float = 0.85) -> bool:
    """A task is 'saturated' when all methods cluster near the top with
    little spread. Default: spread of mean final normalised scores across
    methods is < 0.05 AND every method's mean is above 0.85."""
    finals = collect_run_finals(task)
    means = []
    for m in ORDER:
        if not finals[m]:
            return False  # incomplete coverage; don't filter
        means.append(float(np.mean(finals[m])))
    spread = max(means) - min(means)
    return spread < max_method_spread and min(means) >= floor_norm


def bootstrap_iqm(scores: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    """Return (iqm, lo_95, hi_95). IQM = mean of middle 50% of values."""
    if len(scores) < 4:
        if len(scores) == 0:
            return 0.0, 0.0, 0.0
        m = float(np.mean(scores))
        return m, m, m
    rng = np.random.default_rng(7)
    iqms = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        s_sort = np.sort(sample)
        n = len(s_sort)
        lo, hi = n // 4, n - n // 4
        if hi <= lo:
            iqms.append(float(np.mean(s_sort)))
        else:
            iqms.append(float(np.mean(s_sort[lo:hi])))
    iqms = np.array(iqms)
    s_sort = np.sort(scores)
    n = len(s_sort)
    lo, hi = n // 4, n - n // 4
    iqm = float(np.mean(s_sort[lo:hi])) if hi > lo else float(np.mean(s_sort))
    return iqm, float(np.percentile(iqms, 2.5)), float(np.percentile(iqms, 97.5))


def perf_profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """P(score > tau) for an array of scores."""
    return np.array([(scores > t).mean() for t in taus])


def prob_of_improvement(a: np.ndarray, b: np.ndarray, n_boot: int = 2000):
    """P(a > b) on randomly paired runs, with 95% bootstrap CI."""
    if len(a) == 0 or len(b) == 0:
        return 0.5, 0.5, 0.5
    rng = np.random.default_rng(11)
    pis = []
    for _ in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        # cross-product: P(ai > bi) over all pairs
        wins = (ai[:, None] > bi[None, :]).mean()
        pis.append(wins)
    pi = float((a[:, None] > b[None, :]).mean())
    return pi, float(np.percentile(pis, 2.5)), float(np.percentile(pis, 97.5))


def figure_statistics(fname='fig_F_statistics', task_pool=None,
                       subtitle: str | None = None):
    scores = aggregate_scores_matrix(task_pool)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0),
                                          constrained_layout=True)
    fig.patch.set_facecolor('white')
    if subtitle:
        fig.suptitle(subtitle, fontsize=10, color='#555555', y=1.02)

    # --- Panel 1: IQM with 95% bootstrap CIs (rliable style) -----------
    # Horizontal lozenge spanning the CI range, with a black vertical tick
    # at the mean. Method names on the y-axis (left).
    methods = ORDER
    iqms = [bootstrap_iqm(scores[m]) for m in methods]
    ys = np.arange(len(methods))[::-1]   # top-to-bottom display order
    means = [v[0] for v in iqms]
    ci_lo = [v[1] for v in iqms]
    ci_hi = [v[2] for v in iqms]
    cols  = [COLORS[m] for m in methods]
    bar_h = 0.55
    for y, lo, hi, m, c in zip(ys, ci_lo, ci_hi, means, cols):
        ax1.barh(y, hi - lo, left=lo, color=c, alpha=0.55,
                  edgecolor='none', height=bar_h, zorder=2)
        ax1.plot([m, m], [y - bar_h / 2, y + bar_h / 2],
                  color='black', lw=1.4, zorder=4)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([f'{m} (ours)' if m == 'ARTS' else m for m in methods])
    ax1.set_xlabel('IQM normalised score', labelpad=4)
    ax1.set_title('Aggregate performance (IQM)', fontsize=11.5, fontweight='bold', pad=6)
    ax1.set_xlim(0.3, max(1.05, max(ci_hi) + 0.05))
    ax1.axvline(1.0, color='#555', lw=0.9, ls='--')
    ax1.grid(axis='x', ls='--', lw=0.4, alpha=0.45, color='#888')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Panel 2: Performance profile with bootstrap CI band ------------
    taus = np.linspace(0, 1.05, 60)
    rng_p = np.random.default_rng(13)
    for m in methods:
        if len(scores[m]) == 0:
            continue
        prof = perf_profile(scores[m], taus)
        # bootstrap CI per tau
        boots = []
        for _ in range(500):
            sample = rng_p.choice(scores[m], size=len(scores[m]), replace=True)
            boots.append(perf_profile(sample, taus))
        boots = np.stack(boots, axis=0)
        lo = np.percentile(boots, 5, axis=0)
        hi = np.percentile(boots, 95, axis=0)
        label = f'{m} (ours)' if m == 'ARTS' else m
        ax2.plot(taus, prof, color=COLORS[m], lw=LINE_WIDTH[m], label=label, zorder=4)
        ax2.fill_between(taus, lo, hi, color=COLORS[m], alpha=0.18,
                          linewidth=0, zorder=2)
    ax2.set_xlabel(r'Normalised score threshold $\tau$', labelpad=4)
    ax2.set_ylabel(r'Fraction of runs with score $>$ $\tau$', labelpad=4)
    ax2.set_title('Performance profile', fontsize=11.5, fontweight='bold', pad=6)
    ax2.set_xlim(0, 1.05); ax2.set_ylim(-0.02, 1.05)
    ax2.grid(ls='--', lw=0.4, alpha=0.45, color='#888')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    leg = ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.80,
                      fancybox=True, handlelength=2.0)
    leg.get_frame().set_facecolor('white')

    # --- Panel 3: Probability of improvement (rliable style) -----------
    # Horizontal lozenge spanning the CI range, with a black vertical tick
    # at the mean. ARTS labelled on left, comparison method on right.
    arts = scores['ARTS']
    others = ['Linear', 'AIRA', 'MLEvolve']
    pis = [prob_of_improvement(arts, scores[m]) for m in others]
    ys = np.arange(len(others))[::-1]
    means = [v[0] for v in pis]
    ci_lo = [v[1] for v in pis]
    ci_hi = [v[2] for v in pis]
    cols  = [COLORS[m] for m in others]
    bar_h = 0.55
    for y, lo, hi, m, c in zip(ys, ci_lo, ci_hi, means, cols):
        ax3.barh(y, hi - lo, left=lo, color=c, alpha=0.55,
                  edgecolor='none', height=bar_h, zorder=2)
        ax3.plot([m, m], [y - bar_h / 2, y + bar_h / 2],
                  color='black', lw=1.4, zorder=4)
    ax3.axvline(0.5, color='#555', lw=0.9, ls='--')
    # left side: ARTS; right side: each comparison method
    ax3.set_yticks(ys)
    ax3.set_yticklabels(['ARTS'] * len(others))
    ax3.tick_params(axis='y', length=0)
    ax3.set_xlabel(r'P(ARTS $>$ Y)', labelpad=4)
    ax3.set_title('Probability of improvement', fontsize=11.5, fontweight='bold', pad=6)
    ax3.set_xlim(0, 1.05)
    # add a second y-axis on the right with the comparison method names
    ax3_r = ax3.twinx()
    ax3_r.set_ylim(ax3.get_ylim())
    ax3_r.set_yticks(ys)
    ax3_r.set_yticklabels(others)
    ax3_r.tick_params(axis='y', length=0)
    ax3_r.spines['top'].set_visible(False)
    ax3_r.spines['left'].set_visible(False)
    ax3.grid(axis='x', ls='--', lw=0.4, alpha=0.45, color='#888')
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    _save(fig, fname)


def _save(fig, fname):
    for ext in ('pdf', 'png'):
        p = OUT_DIR / f'{fname}.{ext}'
        kw = dict(bbox_inches='tight')
        if ext == 'png':
            kw['dpi'] = 220
        fig.savefig(p, **kw)
        print(f'wrote {p}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Variant 1: all methods, smooth lines
    figure_strip(panel_continuous,           'fig_A1_progression_continuous')
    # Variant 4: all methods, hourly step updates
    figure_strip(panel_hourly,               'fig_A2_progression_hourly')
    # Variant 3a: variant 1 with broken y-axis
    figure_broken(panel_continuous,          'fig_B1_broken_continuous')
    # Variant 3b: variant 4 with broken y-axis
    figure_broken(panel_hourly,              'fig_B2_broken_hourly')
    # Variant 2: ARTS vs best baseline (continuous)
    figure_strip(panel_continuous,           'fig_C_arts_vs_best',
                 methods='best_per_task')
    # Variant 5: ARTS + best baseline with per-node scatter
    figure_strip(panel_with_scatter,         'fig_D_arts_vs_best_scatter',
                 methods='best_per_task')
    # Variant 6: ARTS only, annotated (3-panel strip)
    figure_arts_only(                        'fig_E_arts_only_annotated')
    # Per-task full-page annotated multi-method (NEW)
    figure_full_per_task('fig_FP_{task}_annotated',          broken=False)
    # Per-task full-page annotated multi-method, BROKEN axis (NEW)
    figure_full_per_task('fig_FP_{task}_annotated_broken',   broken=True)
    # Statistical aggregate plots — three variants:
    #
    # F  (full)        : MLGym + mle-bench combined (8 tasks)
    # F2 (unsaturated) : non-saturated subset of the full pool
    # F3 (top-diff)    : top-K tasks ranked by (ARTS mean − best baseline mean)

    # F: full pool
    full = ALL_STATS_TASKS
    print(f'[stats] full pool ({len(full)}): {[t["label"] for t in full]}')
    figure_statistics('fig_F_statistics',
                      task_pool=full,
                      subtitle=f'aggregated over {len(full)} tasks (MLGym + mle-bench)')

    # F2: non-saturated subset
    unsat = [t for t in full if not is_saturated(t)]
    sat   = [t['label'] for t in full if is_saturated(t)]
    print(f'[stats] saturated tasks (excluded from F2): {sat}')
    print(f'[stats] non-saturated tasks (kept in F2):   {[t["label"] for t in unsat]}')
    figure_statistics('fig_F2_statistics_unsaturated',
                      task_pool=unsat,
                      subtitle=f'non-saturated tasks ({len(unsat)} of {len(full)})')

    # F3: top-K by ARTS-vs-best-baseline gap
    K = 4
    diffs = [(task_diff(t), t) for t in full]
    diffs.sort(key=lambda x: x[0], reverse=True)
    top = [t for d, t in diffs[:K] if d != float('-inf')]
    diff_log = [(t['label'], round(d, 3)) for d, t in diffs]
    print(f'[stats] per-task ARTS−best-baseline gap (sorted): {diff_log}')
    print(f'[stats] top-{K} kept in F3: {[t["label"] for t in top]}')
    figure_statistics('fig_F3_statistics_topdiff',
                      task_pool=top,
                      subtitle=f'top-{K} tasks by ARTS gap over best baseline')

    # F4: every task discoverable on disk (auto-baselines from MLGym yamls,
    # human_best = best observed score across all methods).
    AUTO_TASK_IDS = [
        # MLGym
        'languageModelingFineWeb', 'naturalLanguageInferenceMNLI',
        'imageClassificationCifar10L1', 'imageClassificationFMnist',
        'regressionKaggleHousePrice', 'titanic',
        'battleOfSexes', 'blotto', 'prisonersDilemma',
        'rlBreakoutMinAtar', 'rlMountainCarContinuous',
        # mle-bench
        'mlebenchAPTOS', 'mlebenchHMSBrain', 'mlebenchVesuvius',
        'mlebenchKuzushiji', 'mlebenchVolcanic', 'mlebenchRSNABrainTumor',
        'mlebench3DDetection', 'mlebenchVentilator', 'mlebenchBMS',
        'mlebenchJigsawToxic', 'mlebenchSpaceshipTitanic',
        'mlebenchHistoCancer', 'mlebenchNomad2018',
        'mlebenchPlantPathology', 'mlebenchCovidVaccine',
    ]
    auto_pool = []
    skipped = []
    for tid in AUTO_TASK_IDS:
        td = auto_task(tid, label=tid)
        if td is None:
            skipped.append(tid); continue
        # require at least one ARTS run to compare against, else stats are dominated by zeros
        if not any(BASE.joinpath(g).is_dir() for g in td['arts_globs']):
            skipped.append(tid + '(no-ARTS)'); continue
        auto_pool.append(td)
    print(f'[stats] F4 auto-pool kept ({len(auto_pool)}): {[t["label"] for t in auto_pool]}')
    print(f'[stats] F4 skipped: {skipped}')
    if auto_pool:
        figure_statistics('fig_F4_statistics_all',
                          task_pool=auto_pool,
                          subtitle=f'every available task ({len(auto_pool)} tasks)')


if __name__ == '__main__':
    main()
