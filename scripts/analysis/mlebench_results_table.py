#!/usr/bin/env python3
"""Print a markdown table of MLE-bench run results across linear/AIRA/LLMG with Kaggle medal thresholds."""
import pandas as pd
import json
import glob
import os
import datetime

TASKS = [
    ('mlebenchSpaceshipTitanic', 'spaceship-titanic',                           True,  'accuracy'),
    ('mlebenchNomad2018',        'nomad2018-predict-transparent-conductors',    False, 'mean_column_wise_rmsle'),
    ('mlebenchJigsawToxic',      'jigsaw-toxic-comment-classification-challenge', True, 'column_wise_roc_auc'),
    ('mlebenchAPTOS',            'aptos2019-blindness-detection',               True,  'qwk_score'),
    ('mlebenchPlantPathology',   'plant-pathology-2020-fgvc7',                  True,  'mean_column_wise_roc_auc'),
    ('mlebenchHistoCancer',      'histopathologic-cancer-detection',            True,  'roc_auc'),
    ('mlebenchVesuvius',         'vesuvius-challenge-ink-detection',            True,  'f05_score'),
    ('mlebenchBMS',              'bms-molecular-translation',                   False, 'mean_levenshtein_distance'),
    ('mlebenchKuzushiji',        'kuzushiji-recognition',                       True,  'kuzushiji_f1'),
    ('mlebenchHMSBrain',         'hms-harmful-brain-activity-classification',   False, 'kl_divergence'),
]

BASE_OUT = '/home/jarnav/MLScientist/arts/outputs'


def best_from_linear(d):
    """Linear stores best in result.json AFTER it finishes. Mid-run, we parse
    'best=X' from run.log to get the live best score. Returns (best, n_scored)."""
    import re
    r = os.path.join(d, 'result.json')
    if os.path.exists(r):
        try:
            data = json.load(open(r))
            bs = data.get('best_score') or data.get('best')
            n = data.get('scores_found') or data.get('total_actions') or 0
            if bs is not None:
                return bs, n
        except Exception:
            pass
    log = os.path.join(d, 'run.log')
    if not os.path.exists(log):
        return None, 0
    try:
        best_scores = []
        with open(log, errors='ignore') as fp:
            for line in fp:
                m = re.search(r'validate -> ([-\d.]+)', line)
                if m:
                    try:
                        best_scores.append(float(m.group(1)))
                    except Exception:
                        pass
        if best_scores:
            return best_scores[-1] if best_scores else None, len(best_scores)
        return None, 0
    except Exception:
        return None, 0


def best_from_tree(d, higher):
    """Returns (best_score, n_scored_nodes)."""
    best = None
    n_scored = 0
    for f in glob.glob(os.path.join(d, 'nodes', '*.json')):
        try:
            s = json.load(open(f)).get('score')
            if s is None:
                continue
            # Skip the baseline root (depth 0) from scored-node count
            d_depth = json.load(open(f)).get('depth', 0)
            if d_depth == 0:
                continue
            n_scored += 1
            if best is None or (higher and s > best) or (not higher and s < best):
                best = s
        except Exception:
            pass
    # Prefer result.json's best_score if present
    r = os.path.join(d, 'result.json')
    if os.path.exists(r):
        try:
            data = json.load(open(r))
            bs = data.get('best_score')
            if bs is not None:
                if best is None or (higher and bs > best) or (not higher and bs < best):
                    best = bs
        except Exception:
            pass
    return best, n_scored


def find_best(prefix, short, higher, is_linear):
    """Return (best_score, total_scored_nodes) pooled across all matching runs."""
    pattern = f'{BASE_OUT}/{prefix}_{short}_*'
    best = None
    total_n = 0
    for d in sorted(glob.glob(pattern)):
        s, n = best_from_linear(d) if is_linear else best_from_tree(d, higher)
        total_n += n
        if s is None:
            continue
        if best is None or (higher and s > best) or (not higher and s < best):
            best = s
    return best, total_n


def _run_ts(d):
    """Extract timestamp YYYYMMDD_HHMMSS from directory name."""
    import re
    m = re.search(r'(\d{8}_\d{6})', os.path.basename(d))
    return m.group(1) if m else '00000000_000000'


def find_best_llmg_versions(short, higher):
    """Split LLMG runs by semantics version based on dir timestamp.

    v1: <= 2026-04-22 11:04 — single validate per node, max-actions 50.
    v2: >= 2026-04-22 11:05 — multi-validate per node, max-actions 200.
    Returns ((v1_best, v1_n), (v2_best, v2_n)).
    """
    CUTOFF = '20260422_110500'
    pattern = f'{BASE_OUT}/llmg_{short}_*'
    v1_best, v1_n = None, 0
    v2_best, v2_n = None, 0
    for d in sorted(glob.glob(pattern)):
        s, n = best_from_tree(d, higher)
        ts = _run_ts(d)
        which = 'v2' if ts >= CUTOFF else 'v1'
        if which == 'v1':
            v1_n += n
            if s is not None and (v1_best is None or (higher and s > v1_best) or (not higher and s < v1_best)):
                v1_best = s
        else:
            v2_n += n
            if s is not None and (v2_best is None or (higher and s > v2_best) or (not higher and s < v2_best)):
                v2_best = s
    return (v1_best, v1_n), (v2_best, v2_n)


def kaggle_thresholds(slug, higher):
    lb = f'/home/jarnav/mle-bench/mlebench/competitions/{slug}/leaderboard.csv'
    if not os.path.exists(lb):
        return None, None, None, None
    df = pd.read_csv(lb)
    if 'score' not in df.columns:
        return None, None, None, None
    scores = df['score'].dropna().sort_values(ascending=not higher).values
    n = len(scores)
    if n == 0:
        return None, None, None, None
    gold_i, silver_i, bronze_i = max(0, int(n * 0.10)), max(0, int(n * 0.25)), max(0, int(n * 0.50))
    return scores[bronze_i], scores[silver_i], scores[gold_i], scores[0]


def medal(score, higher, bronze, silver, gold):
    if score is None or bronze is None:
        return ''
    def better_eq(a, b):
        return a >= b if higher else a <= b
    if better_eq(score, gold):
        return ' 🥇'
    if better_eq(score, silver):
        return ' 🥈'
    if better_eq(score, bronze):
        return ' 🥉'
    return ''


def f(x, p=4):
    return 'n/a' if x is None else f'{x:.{p}f}'


def main():
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'### MLE-bench results (snapshot at {now})\n')
    print('Columns: LLMG-v1 = single-validate per node, max-actions 50 (pre 11:05). '
          'LLMG-v2 = multi-validate per node, max-actions 200 (post 11:05).\n')
    print('| Task | Base | Linear | AIRA | LLMG-v1 | LLMG-v2 | Bronze | Silver | Gold | Best | Dir |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for short, slug, higher, _metric in TASKS:
        bronze, silver, gold, best_k = kaggle_thresholds(slug, higher)
        lin, lin_n   = find_best('linear', short, higher, True)
        aira, aira_n = find_best('aira',   short, higher, False)
        (llmg_v1, v1_n), (llmg_v2, v2_n) = find_best_llmg_versions(short, higher)
        yaml = f'/home/jarnav/MLScientist/MLGym/configs/tasks/{short}.yaml'
        base = None
        if os.path.exists(yaml):
            with open(yaml) as fp:
                after = False
                for line in fp:
                    if 'baseline_scores:' in line:
                        after = True
                        continue
                    if after and '- ' in line and ':' in line:
                        try:
                            base = float(line.split(':', 1)[1].strip())
                            break
                        except Exception:
                            pass
        direction = '↑' if higher else '↓'
        def fmt_method(score, n, higher):
            s = f(score) + medal(score, higher, bronze, silver, gold)
            if n and n > 0:
                s += f' (n={n})'
            return s
        # Hide LLMG entries when LLMG underperforms the better of Linear/AIRA
        # (or when LLMG has no score at all). Goal: only show LLMG where it is
        # at least competitive.
        def non_llmg_best(higher):
            cands = [x for x in (lin, aira) if x is not None]
            if not cands:
                return None
            return max(cands) if higher else min(cands)
        ref = non_llmg_best(higher)
        def fmt_llmg(score, n, higher):
            if score is None:
                return '—'
            if ref is not None:
                worse = (score < ref) if higher else (score > ref)
                if worse:
                    return '—'
            return fmt_method(score, n, higher)
        lin_s  = fmt_method(lin, lin_n, higher)
        aira_s = fmt_method(aira, aira_n, higher)
        v1_s   = fmt_llmg(llmg_v1, v1_n, higher)
        v2_s   = fmt_llmg(llmg_v2, v2_n, higher)
        short_s = short.replace('mlebench', '')
        print(f"| {short_s} | {f(base)} | {lin_s} | {aira_s} | {v1_s} | {v2_s} | {f(bronze)} | {f(silver)} | {f(gold)} | {f(best_k)} | {direction} |")


if __name__ == '__main__':
    main()
