"""Reflexion module: tree-level reflection and error analysis injection.

Provides two capabilities for tree search methods:
1. build_reflection() - LLM call that synthesises what the tree has learned
2. inject_error_analysis_script() - copies analyze_errors.py into the container
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arts.tree_search import ContainerManager, LLMClient, TaskProfile, TreeNode


# ---------------------------------------------------------------------------
# 1. Tree-level reflection
# ---------------------------------------------------------------------------

REFLECTION_PROMPT = """\
You are reflecting on a tree of experiments for: {task_name}.
The primary metric is **{metric}** ({direction} is better).
Baseline score: {baseline:.4f}
Best score so far: {best_score} (node {best_id}: {best_strategy})

EXPERIMENT TREE (score | strategy):
{tree_table}

The next child will be expanded from node **{parent_id}** (score: {parent_score}).
Siblings already tried from this parent:
{sibling_list}

In 3-8 sentences, reflect on:
1. What approaches or techniques have worked best so far and why?
2. What has consistently failed or should be avoided?
3. What specific, concrete direction should the next experiment try?
Be specific — reference actual scores, model types, features, and hyperparameters.\
"""


def _build_tree_table(
    nodes: dict[str, TreeNode],
    higher_is_better: bool,
) -> str:
    """Build a compact table of all nodes for the reflection prompt."""
    rows: list[tuple[float | None, str]] = []
    for nid, node in nodes.items():
        if nid == "root":
            continue
        strategy_short = (node.strategy or "")[:150]
        if node.score is not None:
            rows.append((node.score, f"  {nid}: {node.score:.4f} | {strategy_short}"))
        elif node.error:
            rows.append((None, f"  {nid}: FAIL | {strategy_short} | error: {node.error[:80]}"))
        else:
            rows.append((None, f"  {nid}: FAIL | {strategy_short}"))

    # Sort scored nodes first (best first), then failures
    scored = [r for r in rows if r[0] is not None]
    failed = [r for r in rows if r[0] is None]
    scored.sort(key=lambda x: x[0], reverse=higher_is_better)
    lines = [r[1] for r in scored] + [r[1] for r in failed]
    return "\n".join(lines) if lines else "  (no experiments yet)"


def _build_sibling_list(
    nodes: dict[str, TreeNode],
    parent_id: str,
) -> str:
    """List what siblings from this parent have already tried."""
    parent = nodes.get(parent_id)
    if not parent or not parent.children:
        return "  (none — this is the first child)"
    lines = []
    for cid in parent.children:
        child = nodes.get(cid)
        if not child:
            continue
        score_str = f"{child.score:.4f}" if child.score is not None else "FAIL"
        strategy_short = (child.strategy or "")[:120]
        lines.append(f"  - {cid}: {score_str} | {strategy_short}")
    return "\n".join(lines) if lines else "  (none)"


def build_reflection(
    llm: LLMClient,
    nodes: dict[str, TreeNode],
    parent_id: str,
    task_profile: TaskProfile,
    baseline_score: float = 0.0,
) -> str:
    """Generate a reflection on tree progress before expanding a new node.

    Makes a single LLM call and returns a short reflection string (3-8 sentences)
    to inject into the child's context.
    """
    # Find best node
    higher = task_profile.higher_is_better
    scored_nodes = [(n.score, nid) for nid, n in nodes.items()
                    if n.score is not None and nid != "root"]
    if not scored_nodes:
        return ""
    scored_nodes.sort(key=lambda x: x[0], reverse=higher)
    best_score_val, best_id = scored_nodes[0]
    best_node = nodes[best_id]

    parent = nodes.get(parent_id)
    parent_score = f"{parent.score:.4f}" if parent and parent.score is not None else "N/A"

    prompt = REFLECTION_PROMPT.format(
        task_name=task_profile.name,
        metric=task_profile.primary_metric,
        direction="higher" if higher else "lower",
        baseline=baseline_score,
        best_score=f"{best_score_val:.4f}",
        best_id=best_id,
        best_strategy=(best_node.strategy or "")[:200],
        tree_table=_build_tree_table(nodes, higher),
        parent_id=parent_id,
        parent_score=parent_score,
        sibling_list=_build_sibling_list(nodes, parent_id),
    )

    try:
        reflection = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        # Trim to reasonable length
        reflection = reflection.strip()
        if len(reflection) > 1500:
            reflection = reflection[:1500] + "..."
        print(f"  [reflexion] {len(reflection)} chars:")
        for line in reflection.split("\n"):
            print(f"    | {line}")
        return reflection
    except Exception as e:
        print(f"  [reflexion] failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# 2. Error analysis script injection
# ---------------------------------------------------------------------------

ANALYZE_SCRIPT_CLASSIFICATION = '''\
#!/usr/bin/env python3
"""Analyze misclassifications. Usage: python analyze_errors.py"""
import pandas as pd, sys, os

TARGET = "{target_column}"
ID_COL = "{id_column}"

train_path = "data/train.csv"
pred_path = "submission.csv"
if not os.path.exists(pred_path):
    print("No submission.csv found. Run your model first, then run this script.")
    sys.exit(0)

train = pd.read_csv(train_path)
preds = pd.read_csv(pred_path)

merged = train.merge(preds, on=ID_COL, suffixes=("_true", "_pred"))
tcol = TARGET + "_true" if TARGET + "_true" in merged.columns else TARGET
pcol = TARGET + "_pred" if TARGET + "_pred" in merged.columns else TARGET
if tcol == pcol:
    # Same column name means no suffix was added -- find the right columns
    for c in merged.columns:
        if c.endswith("_true"):
            tcol = c
        elif c.endswith("_pred"):
            pcol = c

errors = merged[merged[tcol] != merged[pcol]]
total = len(merged)
n_err = len(errors)
print(f"Error rate: {{n_err}}/{{total}} = {{n_err/total:.1%}}")

# Breakdown by each feature
print("\\n=== ERROR RATE BY FEATURE ===")
skip = {{ID_COL, tcol, pcol, TARGET + "_true", TARGET + "_pred"}}
for col in merged.columns:
    if col in skip or merged[col].nunique() > 20:
        continue
    print(f"\\n--- {{col}} ---")
    for val in merged[col].value_counts().index[:8]:
        subset = merged[merged[col] == val]
        err_rate = (subset[tcol] != subset[pcol]).mean()
        n = len(subset)
        print(f"  {{col}}={{val}}: {{err_rate:.1%}} error ({{n}} samples)")

print(f"\\n=== TOP 20 MISCLASSIFIED EXAMPLES ===")
print(errors.head(20).to_string(max_colwidth=30))
'''

ANALYZE_SCRIPT_REGRESSION = '''\
#!/usr/bin/env python3
"""Analyze prediction errors. Usage: python analyze_errors.py"""
import pandas as pd, numpy as np, sys, os

TARGET = "{target_column}"
ID_COL = "{id_column}"

train_path = "data/train.csv"
pred_path = "submission.csv"
if not os.path.exists(pred_path):
    print("No submission.csv found. Run your model first, then run this script.")
    sys.exit(0)

train = pd.read_csv(train_path)
preds = pd.read_csv(pred_path)

merged = train.merge(preds, on=ID_COL, suffixes=("_true", "_pred"))
tcol = TARGET + "_true" if TARGET + "_true" in merged.columns else TARGET
pcol = TARGET + "_pred" if TARGET + "_pred" in merged.columns else TARGET
if tcol == pcol:
    for c in merged.columns:
        if c.endswith("_true"):
            tcol = c
        elif c.endswith("_pred"):
            pcol = c

merged["residual"] = merged[tcol] - merged[pcol]
merged["abs_residual"] = merged["residual"].abs()

rmse = np.sqrt((merged["residual"] ** 2).mean())
mae = merged["abs_residual"].mean()
print(f"RMSE: {{rmse:.4f}}, MAE: {{mae:.4f}}")

# Worst predictions
worst = merged.nlargest(20, "abs_residual")
print(f"\\n=== TOP 20 WORST PREDICTIONS ===")
print(worst[[ID_COL, tcol, pcol, "residual"]].to_string())

# Breakdown by numeric feature bins
print("\\n=== ERROR BY FEATURE ===")
skip = {{ID_COL, tcol, pcol, "residual", "abs_residual", TARGET + "_true", TARGET + "_pred"}}
for col in merged.columns:
    if col in skip:
        continue
    if merged[col].dtype in ("object", "category") and merged[col].nunique() <= 15:
        print(f"\\n--- {{col}} ---")
        for val in merged[col].value_counts().index[:8]:
            subset = merged[merged[col] == val]
            mean_err = subset["abs_residual"].mean()
            print(f"  {{col}}={{val}}: MAE={{mean_err:.2f}} ({{len(subset)}} samples)")
    elif pd.api.types.is_numeric_dtype(merged[col]) and merged[col].nunique() > 5:
        try:
            bins = pd.qcut(merged[col], q=4, duplicates="drop")
            print(f"\\n--- {{col}} ---")
            for b in bins.unique().sort_values():
                subset = merged[bins == b]
                mean_err = subset["abs_residual"].mean()
                print(f"  {{col}} in {{b}}: MAE={{mean_err:.2f}} ({{len(subset)}} samples)")
        except Exception:
            pass
'''


ANALYZE_SCRIPT_RL = '''\
#!/usr/bin/env python3
"""Analyze RL training results. Usage: python analyze_errors.py"""
import pickle, glob, os, sys
import numpy as np

ckpt_dir = "checkpoints"
if not os.path.isdir(ckpt_dir):
    print("No checkpoints/ directory found. Run training first.")
    sys.exit(0)

files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pkl")))
if not files:
    print("No checkpoint files found in checkpoints/.")
    sys.exit(0)

print(f"Found {{len(files)}} checkpoint(s)")
print("=" * 60)

all_final_rewards = []
all_curves = []

for f in files:
    seed_name = os.path.basename(f)
    try:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
    except Exception as e:
        print(f"  {{seed_name}}: FAILED to load ({{e}})")
        continue

    log_steps = data.get("log_steps", [])
    log_return = data.get("log_return", [])
    config = data.get("train_config", {{}})

    if not log_return:
        print(f"  {{seed_name}}: No training curve data")
        continue

    rewards = [float(r) for r in log_return]
    all_curves.append(rewards)
    final = rewards[-1]
    all_final_rewards.append(final)
    peak = max(rewards)
    peak_idx = rewards.index(peak)
    total_evals = len(rewards)

    print(f"\\n--- {{seed_name}} ---")
    print(f"  Final reward:  {{final:.2f}}")
    print(f"  Peak reward:   {{peak:.2f}} (at eval {{peak_idx+1}}/{{total_evals}})")
    print(f"  Evaluations:   {{total_evals}}")

    # Detect reward collapse: peak is much higher than final
    if peak > 0 and final < peak * 0.7:
        drop_pct = (1 - final / peak) * 100
        print(f"  WARNING: Reward COLLAPSED ({{drop_pct:.0f}}% drop from peak)")
    elif total_evals >= 3 and rewards[-1] < rewards[-2] < rewards[-3]:
        print(f"  WARNING: Reward is DECLINING (last 3 evals: {{rewards[-3]:.1f}} -> {{rewards[-2]:.1f}} -> {{rewards[-1]:.1f}})")

    # Show training curve (sampled)
    if total_evals > 10:
        indices = np.linspace(0, total_evals - 1, 10, dtype=int)
        curve_str = " -> ".join(f"{{rewards[i]:.1f}}" for i in indices)
        print(f"  Curve (10 pts): {{curve_str}}")
    else:
        curve_str = " -> ".join(f"{{r:.1f}}" for r in rewards)
        print(f"  Curve: {{curve_str}}")

    # Check for NaN/Inf
    nan_count = sum(1 for r in rewards if np.isnan(r) or np.isinf(r))
    if nan_count:
        print(f"  WARNING: {{nan_count}} NaN/Inf values in training curve")

# Summary across seeds
if len(all_final_rewards) >= 2:
    print(f"\\n{{\'=\' * 60}}")
    print("SUMMARY ACROSS SEEDS")
    print(f"  Mean final reward:  {{np.mean(all_final_rewards):.2f}}")
    print(f"  Std final reward:   {{np.std(all_final_rewards):.2f}}")
    print(f"  Min:  {{min(all_final_rewards):.2f}}")
    print(f"  Max:  {{max(all_final_rewards):.2f}}")

    # High variance warning
    if np.std(all_final_rewards) > 0.5 * abs(np.mean(all_final_rewards)):
        print(f"  WARNING: High variance across seeds (std/mean = {{np.std(all_final_rewards)/max(abs(np.mean(all_final_rewards)), 1e-6):.1%}})")
        seed_strs = [f"{{r:.1f}}" for r in all_final_rewards]
        print(f"  Per-seed: {{seed_strs}}")

    # Check if any seed failed badly
    if len(all_final_rewards) >= 3:
        median = np.median(all_final_rewards)
        outliers = [r for r in all_final_rewards if abs(r - median) > 2 * np.std(all_final_rewards)]
        if outliers:
            print(f"  WARNING: {{len(outliers)}} outlier seed(s): {{outliers}}")
elif len(all_final_rewards) == 1:
    print(f"\\nFinal reward: {{all_final_rewards[0]:.2f}} (only 1 seed)")
'''

ANALYZE_SCRIPT_GAME_THEORY = '''\
#!/usr/bin/env python3
"""Analyze game theory strategy. Usage: python analyze_errors.py"""
import sys, os, importlib.util, random

# Load our strategy
spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
strat_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strat_mod)

# Load opponent strategy
spec2 = importlib.util.spec_from_file_location("target", "target.py")
target_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(target_mod)

PAYOFFS = {{
    (0, 0): (2, 1),
    (0, 1): (0, 0),
    (1, 0): (0, 0),
    (1, 1): (1, 2),
}}

NUM_GAMES = 200
ROUNDS = 10

game_scores = []
round_details = []

for game in range(NUM_GAMES):
    history = []
    game_total = 0
    for rnd in range(ROUNDS):
        try:
            my_action = strat_mod.row_strategy(list(history))
        except Exception as e:
            print(f"ERROR in row_strategy at round {{rnd}}: {{e}}")
            sys.exit(1)
        try:
            opp_action = target_mod.column_strategy([(c, r) for r, c in history])
        except Exception as e:
            print(f"ERROR in column_strategy at round {{rnd}}: {{e}}")
            sys.exit(1)

        payoff = PAYOFFS.get((my_action, opp_action), (0, 0))
        game_total += payoff[0]
        history.append((my_action, opp_action))

        if game < 5:  # Track first 5 games round-by-round
            round_details.append((game, rnd, my_action, opp_action, payoff[0]))

    game_scores.append(game_total / ROUNDS)

# Print detailed rounds for first 5 games
print("=== FIRST 5 GAMES (round-by-round) ===")
for game_num in range(min(5, NUM_GAMES)):
    game_rounds = [r for r in round_details if r[0] == game_num]
    actions_str = " ".join(f"{{r[2]}}v{{r[3]}}({{r[4]}})" for r in game_rounds)
    total = sum(r[4] for r in game_rounds)
    print(f"  Game {{game_num+1}}: {{actions_str}} | total={{total}}")

# Summary statistics
import statistics
mean_score = statistics.mean(game_scores)
std_score = statistics.stdev(game_scores) if len(game_scores) > 1 else 0
print(f"\\n=== SUMMARY ({{NUM_GAMES}} games) ===")
print(f"  Mean score per round: {{mean_score:.4f}}")
print(f"  Std:  {{std_score:.4f}}")
print(f"  Min game: {{min(game_scores):.4f}}")
print(f"  Max game: {{max(game_scores):.4f}}")

# Action distribution
all_actions = []
for game in range(min(100, NUM_GAMES)):
    history = []
    for rnd in range(ROUNDS):
        my_action = strat_mod.row_strategy(list(history))
        opp_action = target_mod.column_strategy([(c, r) for r, c in history])
        all_actions.append(my_action)
        history.append((my_action, opp_action))
n_0 = all_actions.count(0)
n_1 = all_actions.count(1)
total_a = len(all_actions)
print(f"\\n=== ACTION DISTRIBUTION ===")
print(f"  Action 0: {{n_0}}/{{total_a}} ({{n_0/total_a:.1%}})")
print(f"  Action 1: {{n_1}}/{{total_a}} ({{n_1/total_a:.1%}})")

# Outcome distribution
outcomes = {{(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}}
for game in range(min(100, NUM_GAMES)):
    history = []
    for rnd in range(ROUNDS):
        my_action = strat_mod.row_strategy(list(history))
        opp_action = target_mod.column_strategy([(c, r) for r, c in history])
        outcomes[(my_action, opp_action)] += 1
        history.append((my_action, opp_action))
total_o = sum(outcomes.values())
print(f"\\n=== OUTCOME DISTRIBUTION ===")
for (a, b), count in sorted(outcomes.items()):
    payoff = PAYOFFS[(a, b)][0]
    print(f"  ({{a}},{{b}}): {{count}}/{{total_o}} ({{count/total_o:.1%}}) -> payoff={{payoff}}")
'''


def inject_error_analysis_script(
    container: ContainerManager,
    task_profile: TaskProfile,
) -> None:
    """Copy analyze_errors.py into the container workspace.

    Supports all task types: classification, regression, RL, game theory.
    """
    if task_profile.task_type == "classification":
        if not task_profile.target_column or not task_profile.id_column:
            return
        script = ANALYZE_SCRIPT_CLASSIFICATION.format(
            target_column=task_profile.target_column,
            id_column=task_profile.id_column,
        )
    elif task_profile.task_type == "regression":
        if not task_profile.target_column or not task_profile.id_column:
            return
        script = ANALYZE_SCRIPT_REGRESSION.format(
            target_column=task_profile.target_column,
            id_column=task_profile.id_column,
        )
    elif task_profile.task_type == "rl":
        script = ANALYZE_SCRIPT_RL
    elif task_profile.task_type == "game_theory":
        script = ANALYZE_SCRIPT_GAME_THEORY
    else:
        return

    # Write to container via heredoc
    container.communicate(
        f"cat << 'ENDOFANALYSIS' > /home/agent/workspace/analyze_errors.py\n"
        f"{script}\n"
        f"ENDOFANALYSIS",
        timeout=10.0,
    )
    print(f"  [reflexion] injected analyze_errors.py ({task_profile.task_type})")


_HINTS = {
    "classification": (
        "After running your model, you can optionally run: python analyze_errors.py\n"
        "This shows which examples your model gets wrong and error patterns by feature."
    ),
    "regression": (
        "After running your model, you can optionally run: python analyze_errors.py\n"
        "This shows the worst predictions and error patterns by feature."
    ),
    "rl": (
        "After training, you can optionally run: python analyze_errors.py\n"
        "This analyzes checkpoints: per-seed rewards, training curves, "
        "reward collapse detection, and variance across seeds."
    ),
    "game_theory": (
        "After writing your strategy, you can optionally run: python analyze_errors.py\n"
        "This simulates games against the opponent and shows round-by-round "
        "decisions, action distribution, and outcome breakdown."
    ),
}


def error_analysis_hint(task_profile: TaskProfile) -> str:
    """Return a short hint about analyze_errors.py for the agent, or empty string."""
    return _HINTS.get(task_profile.task_type, "")
