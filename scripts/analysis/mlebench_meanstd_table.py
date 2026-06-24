#!/usr/bin/env python3
"""Mean ± std table per (task, method) plus the best-of-(AIRA,Linear) baseline.

For each task and each method (Linear, AIRA, LLMG) we:
  1. Find all runs that produced ≥1 valid score.
  2. Take the BEST score per run (best across that run's nodes).
  3. Compute mean and std across runs.

For comparison we also report:
  * best-of-(AIRA,Linear) per run — for each run-index across the two
    prior-work methods we take whichever scored better, then aggregate.
  * Δ vs LLMG: difference in mean (signed by direction; positive = LLMG better)

Output: markdown table with mean±std per cell.
"""
from __future__ import annotations
import json
import glob
import os
import re
import math
from pathlib import Path
from statistics import mean, stdev

BASE = Path('/home/jarnav/MLScientist/arts/outputs')

# (short_id, kaggle_slug, higher_is_better, metric_name, baseline)
TASKS = [
    ('mlebenchSpaceshipTitanic', 'spaceship-titanic',                              True,  'accuracy',                 0.0),
    ('mlebenchNomad2018',        'nomad2018-predict-transparent-conductors',       False, 'mean_column_wise_rmsle',   1.0),
    ('mlebenchJigsawToxic',      'jigsaw-toxic-comment-classification-challenge',  True,  'column_wise_roc_auc',      0.5),
    ('mlebenchAPTOS',            'aptos2019-blindness-detection',                  True,  'qwk_score',                0.0),
    ('mlebenchPlantPathology',   'plant-pathology-2020-fgvc7',                     True,  'mean_column_wise_roc_auc', 0.5),
    ('mlebenchHistoCancer',      'histopathologic-cancer-detection',               True,  'roc_auc',                  0.5),
    ('mlebenchVesuvius',         'vesuvius-challenge-ink-detection',               True,  'f05_score',                0.0),
    ('mlebenchBMS',              'bms-molecular-translation',                      False, 'mean_levenshtein_distance', 999.0),
    ('mlebenchKuzushiji',        'kuzushiji-recognition',                          True,  'kuzushiji_f1',             0.0),
    ('mlebenchHMSBrain',         'hms-harmful-brain-activity-classification',      False, 'kl_divergence',            1.462),
]

# Method discovery globs. Each entry yields all matching dirs; we then collapse
# restarts of the same logical run by replicate tag (rN/runN), keeping the dir
# with the most node activity per group.
METHODS = {
    'Linear': ['linear_*{TASK}*'],
    'AIRA':   ['aira_*{TASK}*'],
    'LLMG':   ['llmg_*{TASK}*'],
}


# ─────────────────────── extractors ────────────────────────
def _best_from_tree(d: Path, higher: bool) -> float | None:
    """Best score across nodes/*.json (excluding baseline depth==0)."""
    best = None
    for f in glob.glob(str(d / 'nodes' / '*.json')):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        s = data.get('score')
        if s is None or not isinstance(s, (int, float)) or not math.isfinite(s):
            continue
        if data.get('depth', 0) == 0:
            continue  # skip baseline root
        if best is None or (higher and s > best) or ((not higher) and s < best):
            best = s
    # also check result.json
    r = d / 'result.json'
    if r.exists():
        try:
            bs = json.load(open(r)).get('best_score')
            if bs is not None and isinstance(bs, (int, float)) and math.isfinite(bs):
                if best is None or (higher and bs > best) or ((not higher) and bs < best):
                    best = bs
        except Exception:
            pass
    return best


def _best_from_linear(d: Path, higher: bool) -> float | None:
    """Linear: parse 'step N: validate -> X (best=Y)' from run.log; report
    the final running-best."""
    log = d / 'run.log'
    if not log.exists():
        return None
    best = None
    rx = re.compile(r'validate\s*->\s*(-?\d+(?:\.\d+)?)')
    try:
        with open(log, errors='ignore') as fp:
            for line in fp:
                m = rx.search(line)
                if not m:
                    continue
                try:
                    s = float(m.group(1))
                except Exception:
                    continue
                if not math.isfinite(s):
                    continue
                if best is None or (higher and s > best) or ((not higher) and s < best):
                    best = s
    except Exception:
        return None
    # Prefer result.json if present
    r = d / 'result.json'
    if r.exists():
        try:
            bs = json.load(open(r)).get('best_score') or json.load(open(r)).get('best')
            if bs is not None and isinstance(bs, (int, float)) and math.isfinite(bs):
                if best is None or (higher and bs > best) or ((not higher) and bs < best):
                    best = bs
        except Exception:
            pass
    return best


def _dedupe_restarts(paths: list[Path]) -> list[Path]:
    """Group dirs that look like restarts of the same run (same `_rN`/`_runN`
    tag, only differing in timestamp / job-id). Keep the one with most nodes."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        m = re.match(r'^(.+?_(?:r\d+|run\d+))_\d{8}_\d{6}(?:_j\d+)?$', p.name)
        key = m.group(1) if m else p.name
        groups.setdefault(key, []).append(p)
    picks = []
    for cands in groups.values():
        def w(p):
            n = len(list((p / 'nodes').glob('*.json'))) if (p / 'nodes').is_dir() else 0
            l = sum(1 for _ in open(p / 'run.log', errors='ignore')) if (p / 'run.log').is_file() else 0
            return (n, l)
        cands.sort(key=w, reverse=True)
        picks.append(cands[0])
    return sorted(picks)


def _run_ts(p: Path) -> str:
    """Extract YYYYMMDD_HHMMSS from dir name; falls back to mtime."""
    m = re.search(r'(\d{8})_(\d{6})', p.name)
    if m:
        return m.group(1) + m.group(2)
    try:
        return str(int(p.stat().st_mtime))
    except Exception:
        return '0'


def _oom_job_ids() -> set[str]:
    """Return job IDs that suffered a TRUE mid-run OOM, NOT the cosmetic
    OOM-tag many runs get from the kernel at the 8h SLURM wall (when the
    process happened to be holding lots of GPU memory at kill time).

    Heuristic: a run is a "real" OOM only if its .batch elapsed is < 7h45m
    AND its state is OUT_OF_MEMORY. Anything that ran the full 8h is
    treated as a normal run regardless of the kill-state.
    """
    import subprocess
    try:
        out = subprocess.run(
            ['sacct', '-u', 'jarnav', '--starttime=now-7days',
             '--format=JobID,State,Elapsed', '-P', '-n'],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return set()
    bad = set()
    for line in out.splitlines():
        parts = line.split('|')
        if len(parts) < 3:
            continue
        jid, state, elapsed = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if 'OUT_OF_MEMORY' not in state and 'OOM' not in state:
            continue
        # Parse HH:MM:SS to seconds.
        try:
            h, m, s = map(int, elapsed.split(':'))
            elapsed_s = h * 3600 + m * 60 + s
        except Exception:
            continue
        # 7h45m = 27900s. Wall-hit OOMs are at ~28800s (8h); real mid-run
        # OOMs are anywhere shorter. Pick a generous cutoff.
        if elapsed_s < 27900:
            bad.add(jid.split('.')[0])
    return bad


_OOM_CACHE: set[str] | None = None


def _is_oom_dir(p: Path) -> bool:
    """Check if a run directory's job_id is in the OOM blocklist."""
    global _OOM_CACHE
    if _OOM_CACHE is None:
        _OOM_CACHE = _oom_job_ids()
    m = re.search(r'_j(\d+)$', p.name)
    if not m:
        return False
    return m.group(1) in _OOM_CACHE


def collect_runs(method: str, short: str, higher: bool, last_n: int = 3) -> list[float]:
    """Return per-run best scores for the LATEST `last_n` runs of
    (method, task). Newer dirs (by timestamp in the dir name) come first;
    we keep the last_n and report their best scores. OOM-killed runs are
    filtered out before the last_n cap."""
    paths = []
    for pat in METHODS[method]:
        paths.extend([Path(p) for p in glob.glob(str(BASE / pat.format(TASK=short)))])
    paths = _dedupe_restarts(paths)
    # Drop OOM-killed runs.
    paths = [p for p in paths if not _is_oom_dir(p)]
    # Sort newest-first by timestamp encoded in the dir name.
    paths.sort(key=_run_ts, reverse=True)
    paths = paths[:last_n]
    extractor = _best_from_linear if method == 'Linear' else _best_from_tree
    scores = []
    for p in paths:
        s = extractor(p, higher)
        if s is None:
            continue
        scores.append(s)
    return scores


# ─────────────────────── aggregation ────────────────────────
def fmt(mean_v: float | None, std_v: float | None, p: int = 4) -> str:
    if mean_v is None:
        return '—'
    if std_v is None or math.isnan(std_v):
        return f'{mean_v:.{p}f}'
    return f'{mean_v:.{p}f}$_\\pm$$_{{{std_v:.{p}f}}}$'


def main() -> None:
    print('| Task | Metric | Baseline | Linear | AIRA | LLM-Guided | Best(AIRA,Linear) | Δ vs LLMG |')
    print('|---|---|---|---|---|---|---|---|')
    for short, slug, higher, metric, baseline in TASKS:
        per_method = {m: collect_runs(m, short, higher) for m in METHODS}
        # Filter: drop runs that didn't beat baseline.
        for m in per_method:
            kept = []
            for s in per_method[m]:
                if higher and s > baseline:
                    kept.append(s)
                elif (not higher) and s < baseline:
                    kept.append(s)
            per_method[m] = kept

        # Mean/std per method
        agg = {}
        for m, scores in per_method.items():
            if not scores:
                agg[m] = (None, None, 0)
            else:
                mu = mean(scores)
                sd = stdev(scores) if len(scores) > 1 else 0.0
                agg[m] = (mu, sd, len(scores))

        # Best-prior-method = whichever of {Linear mean, AIRA mean} is better.
        # We report THAT method's mean ± std and number of runs.
        l_mu, l_sd, l_n = agg['Linear']
        a_mu, a_sd, a_n = agg['AIRA']
        if l_mu is None and a_mu is None:
            prior_agg = (None, None, 0)
        elif l_mu is None:
            prior_agg = (a_mu, a_sd, a_n)
        elif a_mu is None:
            prior_agg = (l_mu, l_sd, l_n)
        else:
            l_better = (l_mu > a_mu) if higher else (l_mu < a_mu)
            prior_agg = (l_mu, l_sd, l_n) if l_better else (a_mu, a_sd, a_n)

        # Δ vs LLMG (positive = LLMG better)
        llmg_mu = agg['LLMG'][0]
        delta = None
        if llmg_mu is not None and prior_agg[0] is not None:
            delta = (llmg_mu - prior_agg[0]) if higher else (prior_agg[0] - llmg_mu)

        # Format rows
        short_disp = short.replace('mlebench', '')
        print(
            f"| {short_disp} | {metric}{' ↑' if higher else ' ↓'} | "
            f"{baseline:.4f} | "
            f"{fmt(*agg['Linear'][:2])} (n={agg['Linear'][2]}) | "
            f"{fmt(*agg['AIRA'][:2])} (n={agg['AIRA'][2]}) | "
            f"{fmt(*agg['LLMG'][:2])} (n={agg['LLMG'][2]}) | "
            f"{fmt(*prior_agg[:2])} (n={prior_agg[2]}) | "
            f"{('+' if delta is not None and delta > 0 else '')}{delta:.4f}" + (' ✅' if delta is not None and delta > 0 else (' ✗' if delta is not None and delta < 0 else '')) + ' |'
            if delta is not None else
            f"| {short_disp} | {metric}{' ↑' if higher else ' ↓'} | "
            f"{baseline:.4f} | "
            f"{fmt(*agg['Linear'][:2])} (n={agg['Linear'][2]}) | "
            f"{fmt(*agg['AIRA'][:2])} (n={agg['AIRA'][2]}) | "
            f"{fmt(*agg['LLMG'][:2])} (n={agg['LLMG'][2]}) | "
            f"{fmt(*prior_agg[:2])} (n={prior_agg[2]}) | "
            f"— |"
        )


if __name__ == '__main__':
    main()
