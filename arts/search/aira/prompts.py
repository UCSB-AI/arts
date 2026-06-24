"""AIRA-dojo prompt templates adapted for MLGym multi-turn execution.

These are adapted from the AIRA-dojo YAML operator configs
(src/dojo/configs/solver/operators/mlebench/aira_operators/*.yaml).

Key adaptations:
- Kaggle Grandmaster persona → ML research agent
- Single-script generation → multi-turn ReAct (one command at a time)
- submission.csv → validate command
- H200 GPU references → generic environment description
- Complexity cycling (simple/normal/complex) preserved from AIRA-dojo
"""


# ---------------------------------------------------------------------------
# DRAFT — generate a solution from scratch
# ---------------------------------------------------------------------------

DRAFT_TEMPLATE = """\
You are an ML research agent working on a task.
Carefully consider the task description, the available data, and the environment.
Your goal is to provide a solution approach that is different from those previously explored.
Be specific about your proposed approach including data processing, modeling method, and evaluation.

# TASK DESCRIPTION
{task_desc}

# PREVIOUSLY EXPLORED IDEAS
{memory}

# DATA OVERVIEW
{data_overview}

{complexity_instruction}

Consider the previously explored ideas, and make sure the idea you propose considers a DIFFERENT ASPECT OF THE SOLUTION.
Brainstorm about possible approaches and WHY THEY ARE LIKELY TO BE EFFECTIVE AND INCREASE THE PERFORMANCE for the given task.

Now implement your approach step by step. Start by writing code, then run it, then validate."""

DRAFT_COMPLEXITY = {
    "simple": (
        "In this iteration focus on PROPOSING A SIMPLE IDEA: one that can serve as a "
        "SIMPLE YET EFFECTIVE BASELINE for the task. Consider battle-tested methods or "
        "models that are known to work well for the task at hand."
    ),
    "normal": (
        "In this iteration focus on PROPOSING A MORE COMPLEX IDEA: one that can beat "
        "the previous baselines at the cost of some complexity. Consider leveraging more "
        "complex models, specialized feature engineering, or basic ensembling and/or "
        "hyper-parameter optimization."
    ),
    "complex": (
        "In this iteration focus on PROPOSING AN ADVANCED IDEA: one that can beat the "
        "previous baselines at the cost of some complexity. Consider using specialized "
        "models, leveraging advanced feature engineering or data augmentation strategies, "
        "advanced ensembling and/or hyper-parameter optimization."
    ),
}


# ---------------------------------------------------------------------------
# IMPROVE — refine an existing solution
# ---------------------------------------------------------------------------

IMPROVE_TEMPLATE = """\
You are an ML research agent working on a task.
You are provided with a previously developed solution and should improve it to further increase performance.
Your goal is to provide EXACTLY ONE improvement idea that is different from those previously explored.
Be specific about the proposed improvement.

# TASK DESCRIPTION
{task_desc}

# PREVIOUS SOLUTION APPROACH
{prev_approach}

# PREVIOUS SCORE
{prev_score}

# PREVIOUSLY EXPLORED IMPROVEMENT IDEAS
{memory}

{complexity_instruction}

Consider the previously explored ideas, and make sure the improvement you propose considers a DIFFERENT IMPROVEMENT, but keep the EVALUATION CONSISTENT.
Brainstorm about possible improvements and WHY THEY ARE LIKELY TO BE EFFECTIVE AND INCREASE THE PERFORMANCE.

Now implement your improvement step by step. Start by examining the current code, then make targeted modifications, run, and validate."""

IMPROVE_COMPLEXITY = {
    "simple": (
        "In this iteration, suggest a minimal, low-risk tweak that keeps the current "
        "solution's core intact. Think: a feature-engineering twist, a lightweight "
        "preprocessing trick, or hyperparameter changes."
    ),
    "normal": (
        "In this iteration, propose a moderate upgrade that builds on the baseline "
        "without deviating dramatically. Options include hyper-parameter tuning, a small "
        "ensemble of similar models, a sturdier preprocessing pipeline, or feature "
        "engineering improvements."
    ),
    "complex": (
        "In this iteration, recommend a substantial extension that pushes the method's "
        "boundaries while preserving its core logic. Consider advanced ensembling/stacking, "
        "fine-tuning specialized models, or exhaustive hyper-parameter searches."
    ),
}


# ---------------------------------------------------------------------------
# DEBUG — fix buggy code
# ---------------------------------------------------------------------------

DEBUG_TEMPLATE = """\
You are an ML research agent fixing bugs in a solution.
Carefully review the previous debugging attempts, the buggy approach and its output.
You must not change the core idea or methodology of the solution, but only fix the bugs.

# TASK DESCRIPTION
{task_desc}

# PREVIOUS DEBUGGING ATTEMPTS
{ancestral_memory}

# BUGGY APPROACH
{buggy_approach}

# ERROR OUTPUT
{error_output}

# Instructions:
- Do NOT alter the core method or underlying idea. Only correct existing bugs.
- If the error is in the code, fix it. If the error is in the approach, make minimal changes.
- Re-run the corrected solution and validate.

Now fix the bug step by step. Start by examining the relevant code, then make targeted fixes, run, and validate."""


# ---------------------------------------------------------------------------
# CROSSOVER — combine two solutions
# ---------------------------------------------------------------------------

CROSSOVER_TEMPLATE = """\
You are an ML research agent working on a task.
Your goal is to combine two previously developed solutions to further increase performance.
Carefully consider the task description, the two provided solutions, and the available environment.

# TASK DESCRIPTION
{task_desc}

# PREVIOUS SOLUTION 1
Approach: {approach_1}
Score: {score_1}

# PREVIOUS SOLUTION 2
Approach: {approach_2}
Score: {score_2}

# Instructions:
Devise a plan to merge or integrate these solutions. Consider:
- Ensembling predictions from both approaches
- Combining the best features/techniques from each
- Using one solution's preprocessing with another's model

Brainstorm how the two solutions can be effectively combined and WHY THIS CROSSOVER IS LIKELY TO BE EFFECTIVE.

Now implement the combined approach step by step. Write code, run it, and validate."""
