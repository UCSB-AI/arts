"""
MLGym tree-search environment for test-time training (TTT) with PRIME-RL / verifiers.

This is the environment used to GRPO-train the ARTS "scientist" so that training
and evaluation share an identical interface. Each turn:

    scientist sees the experiment tree  ->  proposes a direction (+ a PARENT node)
    ->  executor runs it in an MLGym container  ->  real score comes back as reward

One episode = one full tree search of ``node_budget`` nodes. The scientist picks a
PARENT for every proposal (deepen vs. explore), so the episode builds a real
branching tree; reward is computed from the best score reached anywhere in the tree.

Self-contained: defines ``load_environment()`` returning a ``verifiers.MultiTurnEnv``.
It is exposed to PRIME-RL through the thin ``mlgym_tree_env_v3`` package (env id
``mlgym_tree_env_v3`` in the ``configs/prime_rl_*_tree.toml`` files).

Reward schemes (selected via ``reward_scheme``; for "higher is better" tasks, the
env flips signs internally for lower-is-better):

    v6_binary      : {-0.5 fault, 0 if s<=baseline, +1 if s>baseline}
    v7_fixed_tier  : {-0.5 fault, 0 if s<baseline, +0.2 if baseline<=s<=tau, +1 if s>tau}  (tau=0.88)
    v9_percentile  : {-0.5 fault, 0 if s<=baseline, +0.2 if baseline<s<=p, +1 if s>p}
                     p = percentile_q over a rolling window of the last N valid scores
                     (p = baseline until the window reaches ``warmup``). Defaults N=64,
                     q=70, warmup=8. Stateful, persisted to disk, end-of-step snapshot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from arts.tree_search import (
    ContainerManager,
    LLMClient,
    TaskProfile,
    extract_command,
    get_task_profile,
)

logger = logging.getLogger(__name__)

# Where per-rollout / per-reward JSONL logs are written. Defaults to a local
# directory so the env runs off-cluster; override with AIR_ROLLOUT_LOG_DIR.
ROLLOUT_LOG_DIR = Path(os.environ.get("AIR_ROLLOUT_LOG_DIR", "rollout_logs"))


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASKS = {
    "titanic": {
        "task_config": "tasks/titanic.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "battleOfSexes": {
        "task_config": "tasks/battleOfSexes.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "regression": {
        "task_config": "tasks/regressionKaggleHousePrice.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "mountaincar": {
        "task_config": "tasks/rlMountainCarContinuous.yaml",
        "container_image": os.environ.get("MLGYM_RL_IMAGE", "/scratch/jarnav/mlgym_rl.sif"),
    },
    "blotto": {
        "task_config": "tasks/blotto.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "prisonersDilemma": {
        "task_config": "tasks/prisonersDilemma.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "cifar10": {
        "task_config": "tasks/imageClassificationCifar10.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "fashionMnist": {
        "task_config": "tasks/imageClassificationFMnist.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "languageModeling": {
        "task_config": "tasks/languageModelingFineWeb.yaml",
        "container_image": os.environ.get("MLGYM_GPU_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "mnli": {
        "task_config": "tasks/naturalLanguageInferenceMNLI.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
}


# ---------------------------------------------------------------------------
# Single-node execution (synchronous; runs in a worker thread)
# ---------------------------------------------------------------------------

def execute_in_container(
    proposal: str,
    task_profile: TaskProfile,
    task_config: str,
    container_image: str,
    executor_url: str,
    executor_model: str,
    max_actions: int = 15,
    env_gpu: str = "cpu",
) -> tuple[float | None, str, bool]:
    """Execute one proposal in an MLGym container.

    Returns ``(score, feedback_text, executor_fault)``. ``executor_fault=True``
    means the failure was the executor's fault (code errors, ModuleNotFoundError,
    timeouts), NOT the scientist's — the scientist is not penalized for those.
    """
    container = None
    try:
        container = ContainerManager(
            task_config=task_config,
            env_gpu=env_gpu,
            image_name=container_image,
            task_profile=task_profile,
        )
        container.create()

        executor = LLMClient(
            base_url=executor_url,
            model=executor_model,
            temperature=0.9,
        )

        # Build initial messages for the executor
        data_head = ""
        if task_profile.data_head_cmd:
            try:
                data_head = container.env.communicate(task_profile.data_head_cmd, timeout_duration=10)
            except Exception:
                pass

        task_desc = task_profile.root_task_desc.format(
            baseline_score=container.baseline_score,
            data_head=data_head,
        )

        messages = [
            {"role": "system", "content": task_profile.system_prompt},
            {"role": "user", "content": task_desc},
            {"role": "user", "content": f"Strategy to try: {proposal}\n\n{task_profile.branch_write_instruction}"},
        ]

        score = None
        feedback_parts = []
        executor_fault = False

        for _step in range(max_actions):
            try:
                raw = executor.chat(messages)
            except Exception as e:
                feedback_parts.append(f"Executor error: {e}")
                executor_fault = True  # executor LLM failed, not the scientist's fault
                break

            action, _ = extract_command(raw)
            if not action:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "No command detected."})
                continue

            if action.strip().lower() == "submit":
                action = "validate"

            # Wrap python/torchrun commands with a timeout to prevent hanging
            cmd_timeout = int(getattr(task_profile, "step_timeout", 180))
            stripped = action.strip()
            if (stripped.startswith("python ") or stripped.startswith("torchrun ")) and "timeout" not in action:
                action = f"timeout {cmd_timeout} {action}"

            obs, info = container.step(action)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs})

            # Detect executor code errors (not the scientist's fault)
            obs_lower = obs.lower() if obs else ""
            if any(err in obs_lower for err in [
                "modulenotfounderror", "importerror", "syntaxerror",
                "nameerror", "indentationerror", "typeerror: ",
                "filenotfounderror", "permissionerror",
            ]):
                executor_fault = True

            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    score = score_data.get(task_profile.primary_metric, list(score_data.values())[0])
                else:
                    score = score_data
                feedback_parts.append(f"Score: {score}")
                executor_fault = False  # got a score despite errors — counts as valid
                break
        else:
            obs, info = container.step("validate")
            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    score = score_data.get(task_profile.primary_metric, list(score_data.values())[0])
                else:
                    score = score_data
                feedback_parts.append(f"Score (forced validate): {score}")
            else:
                feedback_parts.append("No score produced after max actions.")
                executor_fault = True  # ran out of actions without scoring

        feedback = " | ".join(feedback_parts) if feedback_parts else f"Score: {score}"
        return score, feedback, executor_fault

    except Exception as e:
        return None, f"Execution failed: {e}", True  # infrastructure failure
    finally:
        if container and container.env:
            try:
                container.env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Scientist output parsers (tree-aware)
# ---------------------------------------------------------------------------

def parse_direction(text: str) -> str:
    """Extract the DIRECTION section, stopping at the next section header.

    Accepts both ``DIRECTION:\\n<content>`` and inline ``DIRECTION: <content>``.
    Falls back to the whole text when the model doesn't follow the format.
    """
    m = re.search(
        r"DIRECTION:\s*(.*?)(?=\n(?:MODE|MEMORY|EXECUTOR_GUIDANCE|REASONING|STRATEGIES):|\Z)",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"DIRECTION:\s*(.*)\Z", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:800]
    if len(text.strip()) > 10:
        logger.warning("No DIRECTION section found, using full text as direction")
        return text.strip()[:800]
    return ""


def parse_parent(text: str, valid_ids: set[str]) -> str:
    """Extract the parent node id from the CHOSEN strategy's PARENT annotation.

    Matches e.g. ``1. Strategy ... -> PARENT: node_2 — reason``. Falls back to
    ``root`` when the chosen strategy's parent isn't a known node id.
    """
    chosen_match = re.search(r"CHOSEN:\s*(\d+)", text)
    strat_lines = re.findall(
        r"(\d+)\.\s.*?(?:→|->)\s*PARENT:\s*[\"']?([A-Za-z0-9_]+(?:_\d+)*)[\"']?\s*(?:—|-|$)",
        text,
    )
    if chosen_match and strat_lines:
        chosen_num = int(chosen_match.group(1))
        for num_str, parent in strat_lines:
            if int(num_str) == chosen_num:
                pid = parent.strip().rstrip("—-").strip()
                if pid == "root" or pid in valid_ids:
                    return pid
                return "root"  # invalid → default
    any_parent = re.search(r"PARENT:\s*[\"']?([A-Za-z0-9_]+(?:_\d+)*)[\"']?", text)
    if any_parent:
        pid = any_parent.group(1)
        if pid == "root" or pid in valid_ids:
            return pid
    return "root"


def parse_memory(text: str) -> str:
    """Extract the MEMORY section from scientist output (empty if NONE)."""
    m = re.search(r'MEMORY:\s*\n(.*?)(?:\n[A-Z]+:|\Z)', text, re.DOTALL)
    if m:
        mem = m.group(1).strip()
        if mem.upper() != "NONE":
            return mem
    return ""


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _log_reward(scheme: str, reward: float, best, baseline, any_score, state, extra: dict | None = None):
    try:
        ROLLOUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scheme": scheme,
            "reward": reward,
            "best_score": best,
            "baseline": baseline,
            "any_score": any_score,
            "tree": [(n.get("id"), n.get("score")) for n in state.get("tree", [])],
        }
        if extra:
            entry.update(extra)
        with open(ROLLOUT_LOG_DIR / f"rewards_{scheme}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"reward log write failed: {e}")


def _episode_score(state) -> tuple[float | None, bool]:
    """Return ``(best_score_in_episode, higher_is_better)``.

    The score is the BEST across all non-root tree nodes in the episode (the best
    may appear at any depth), or ``None`` if no node produced a valid score.
    """
    higher = state.get("higher_is_better", True)
    any_score = state.get("any_score_achieved", False)
    if not any_score:
        return None, higher
    best = state.get("best_score")
    if best is not None:
        return float(best), higher
    return None, higher


# -- v6: stateless, binary ---------------------------------------------------

def reward_v6_binary(parser, completion, answer, state, **kwargs) -> float:
    s, higher = _episode_score(state)
    b = state.get("baseline_score", 0)
    if s is None:
        r = -0.5
    elif (higher and s > b) or ((not higher) and s < b):
        r = 1.0
    else:
        r = 0.0
    _log_reward("v6_binary", r, s, b, s is not None, state)
    return r


# -- v7: stateless, fixed-tier (tau=0.88) ------------------------------------

V7_TAU = 0.88


def reward_v7_fixed_tier(parser, completion, answer, state, tau: float = V7_TAU, **kwargs) -> float:
    s, higher = _episode_score(state)
    b = state.get("baseline_score", 0)
    if s is None:
        r = -0.5
    elif higher:
        if s < b:           r = 0.0
        elif s <= tau:      r = 0.2
        else:               r = 1.0
    else:
        # For lower-is-better: tau is the target (lower = better); s must drop below tau.
        if s > b:           r = 0.0
        elif s >= tau:      r = 0.2
        else:               r = 1.0
    _log_reward("v7_fixed_tier", r, s, b, s is not None, state, extra={"tau": tau})
    return r


# -- v9: stateful percentile, end-of-step snapshot, persistent ---------------

class RewardV9Percentile:
    """Reward with a p_q threshold over a rolling window of recent valid scores.

    End-of-step snapshot semantics: all rollouts within a step are scored against
    the window frozen at step-start; after the step completes the window is
    extended with this step's valid scores (FIFO, capped at N).
    """

    # PRIME-RL/verifiers inspects reward_fn.__name__ for logging; class instances
    # lack it by default, which would mark every rollout as failed.
    __name__ = "reward_v9_percentile"
    __qualname__ = "reward_v9_percentile"

    def __init__(self, b: float, N: int, q: int, warmup: int, batch_size: int,
                 task: str, persist_path: Path | None = None):
        self.b = float(b)
        self.N = int(N)
        self.q = int(q)
        self.warmup = int(warmup)
        self.batch_size = int(batch_size)
        self.task = task
        self.window: deque[float] = deque(maxlen=self.N)
        self._pending: list[float] = []
        self._counter = 0
        self._cached_p: float | None = None
        self.persist_path = persist_path or (ROLLOUT_LOG_DIR / f"reward_state_v9_{task}.json")
        self._load()
        self._recompute_threshold()

    def _recompute_threshold(self):
        if len(self.window) < self.warmup:
            self._cached_p = self.b
        else:
            import numpy as np
            self._cached_p = float(np.percentile(list(self.window), self.q))

    def _load(self):
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    d = json.load(f)
                w = d.get("window", [])
                self.window = deque([float(x) for x in w[-self.N:]], maxlen=self.N)
                logger.info(f"[v9] loaded window of {len(self.window)} scores from {self.persist_path}")
        except Exception as e:
            logger.warning(f"[v9] could not load state: {e}")

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump({"window": list(self.window), "baseline": self.b,
                           "N": self.N, "q": self.q, "warmup": self.warmup}, f)
        except Exception as e:
            logger.warning(f"[v9] could not save state: {e}")

    def __call__(self, parser, completion, answer, state, **kwargs) -> float:
        s, higher = _episode_score(state)
        b = state.get("baseline_score", self.b)
        p = self._cached_p if self._cached_p is not None else self.b

        if s is None:
            r = -0.5
        elif higher:
            if s <= b:          r = 0.0
            elif s <= p:        r = 0.2
            else:               r = 1.0
        else:
            if s >= b:          r = 0.0
            elif s >= p:        r = 0.2
            else:               r = 1.0

        if s is not None:
            self._pending.append(s)
        self._counter += 1
        if self._counter >= self.batch_size:
            for ps in self._pending:
                self.window.append(ps)
            self._pending = []
            self._counter = 0
            self._recompute_threshold()
            self._save()

        _log_reward("v9_percentile", r, s, b, s is not None, state,
                    extra={"p_threshold_snapshot": p,
                           "window_size": len(self.window),
                           "pending_in_batch": len(self._pending)})
        return r


def _make_reward(reward_scheme: str, baseline_score: float, batch_size: int,
                 task_name: str, v7_tau: float | None):
    """Resolve a ``reward_scheme`` string to a verifiers reward callable."""
    if reward_scheme == "v6_binary":
        return reward_v6_binary
    if reward_scheme == "v7_fixed_tier":
        tau = v7_tau if v7_tau is not None else V7_TAU
        logger.info(f"v7 tau={tau}")

        def _v7_with_tau(parser, completion, answer, state, **kw):
            return reward_v7_fixed_tier(parser, completion, answer, state, tau=tau, **kw)
        _v7_with_tau.__name__ = "reward_v7_fixed_tier"
        return _v7_with_tau
    if reward_scheme == "v9_percentile":
        return RewardV9Percentile(
            b=baseline_score, N=64, q=70, warmup=8,
            batch_size=batch_size, task=task_name,
        )
    raise ValueError(
        f"Unknown reward_scheme={reward_scheme}. "
        "Options: v6_binary, v7_fixed_tier, v9_percentile"
    )


# ---------------------------------------------------------------------------
# Scientist prompts (tree-aware; match the eval-time format in
# llm_guided_tree_search.py so train and eval are consistent)
# ---------------------------------------------------------------------------

TREE_SYSTEM_PROMPT = """You are a senior ML research scientist. You guide experiment design — you do NOT write code. A separate executor writes and runs the code based on your directions.

Your job: look at the experiment tree, decide what to try next, and give a high-level direction.

IMPORTANT RULES:
- Do NOT write code. Do NOT output python scripts. Only output the structured format below.
- Each direction you give spawns a new node in the experiment tree.
- You must choose which existing node to build on (PARENT field).
- PARENT: root means start a fresh approach. PARENT: node_3 means refine node_3's approach.

## Output Format (follow EXACTLY every turn)

REASONING:
[1-3 sentences: what worked, what failed, what to try next]

STRATEGIES:
1. [idea] → PARENT: root — [why]
2. [idea] → PARENT: node_1 — [why]
3. [idea] → PARENT: root — [why]
CHOSEN: [1/2/3] because [reason]

DIRECTION:
[What the executor should try. Be specific about the approach, hyperparameters, architecture choices. Do NOT write code — the executor handles implementation.]

MODE: explore

MEMORY:
[One-sentence insight from results so far, or NONE if first turn.]

## Example output (first turn, no prior results)

REASONING:
No experiments yet. Start with a simple baseline to establish a reference point.

STRATEGIES:
1. Random Forest on flattened pixels → PARENT: root — simple, fast, establishes baseline
2. Logistic Regression on raw pixels → PARENT: root — even simpler baseline
3. Small CNN with 2 conv layers → PARENT: root — standard image approach
CHOSEN: 1 because Random Forest is reliable and fast for a first experiment

DIRECTION:
Train a Random Forest classifier with 200 trees on the flattened 28x28 pixel features (784 dimensions). Use all 60000 training samples. Predict on the test set and save submission.csv.

MODE: explore

MEMORY: NONE

## Example output (later turn, with prior results)

REASONING:
node_1 (Random Forest) scored 0.87. node_2 (Logistic Regression) scored 0.84. Tree-based methods work better on this data. node_3 (CNN) failed due to timeout. Should try gradient boosting which is stronger than RF.

STRATEGIES:
1. XGBoost with tuned hyperparameters → PARENT: root — stronger tree method, new approach
2. Random Forest with more trees and feature engineering → PARENT: node_1 — refine RF result
3. LightGBM with histogram binning → PARENT: root — fast gradient boosting alternative
CHOSEN: 2 because node_1 already works well and more trees + PCA might push it higher

DIRECTION:
Build on node_1's Random Forest approach: increase to 500 trees, add PCA to reduce to 100 components before training, and use max_depth=20. Keep using all training samples.

MODE: exploit

MEMORY: RF=0.87 beats LR=0.84; tree methods work well on flattened pixels. CNN timed out — avoid for now."""


TREE_INITIAL_PROMPT = """## Task

{task_description}

## Task Details (what the executor sees)

{task_details}

Metric: {metric_name} ({direction} is better)
Baseline score: {baseline_score:.4f}

## Current Experiment Tree

{tree_view}

## Memory

{memory}

You have {budget_left} nodes remaining out of {total_budget} total.

IMPORTANT: You are NOT the executor. Do NOT write code. Do NOT output python scripts.
Instead, output your analysis and direction in this EXACT format:

REASONING:
[1-3 sentences analyzing the tree]

STRATEGIES:
1. [idea] → PARENT: root — [why]
2. [idea] → PARENT: root — [why]
3. [idea] → PARENT: root — [why]
CHOSEN: [number] because [reason]

DIRECTION:
[What approach to try — describe the idea, not the code]

MODE: explore

MEMORY: NONE"""


TREE_TURN_PROMPT = """## Result of Last Expansion

{result}

## Current Experiment Tree

{tree_view}

## Memory

{memory}

You have {budget_left} nodes remaining.

Do NOT write code. Respond with REASONING, STRATEGIES (with PARENT: for each), CHOSEN, DIRECTION, MODE, MEMORY.
TIP: To REFINE a promising node, set PARENT: to that node's ID (e.g. PARENT: root_0). To try something new, use PARENT: root."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MLGymTreeEnv(vf.MultiTurnEnv):
    """Tree-structured MLGym environment for TTT of the ARTS scientist.

    The scientist selects a PARENT for each proposal, so episodes build a real
    branching tree. Reward is computed from the best score reached anywhere in
    the tree (see the reward schemes documented at module level).
    """

    def __init__(
        self,
        task: str = "titanic",
        task_name: str | None = None,
        node_budget: int = 5,
        executor_url: str = "http://localhost:9001/v1",
        executor_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_actions: int = 20,
        num_train_examples: int = 200,
        num_eval_examples: int = 20,
        env_gpu: str = "cpu",
        parser: vf.Parser | None = None,
        rubric: vf.Rubric | None = None,
        reward_scheme: str = "v6_binary",
        reward_batch_size: int = 8,
        v7_tau: float | None = None,
        **kwargs,
    ):
        self.task_name = task_name or task
        logger.info(f"MLGymTreeEnv: task={self.task_name}")
        self.task_cfg = TASKS[self.task_name]
        self.task_profile = get_task_profile(self.task_cfg["task_config"])
        self.node_budget = node_budget
        self.executor_url = executor_url
        self.executor_model = executor_model
        self.max_actions = max_actions
        self.env_gpu = env_gpu

        # Baseline score from the task YAML (sibling MLGym repo).
        tp = self.task_profile
        self._baseline_score = 0.0
        try:
            import yaml
            yaml_path = Path(__file__).resolve().parents[3] / "MLGym" / "configs" / self.task_cfg["task_config"]
            with open(yaml_path) as f:
                task_yaml = yaml.safe_load(f)
            scores = task_yaml.get("baseline_scores", [])
            if scores and isinstance(scores[0], dict):
                self._baseline_score = scores[0].get(tp.primary_metric, 0.0)
            logger.info(f"Loaded baseline score from YAML: {self._baseline_score}")
        except Exception as e:
            logger.warning(f"Could not load baseline from YAML: {e}")

        # What the executor sees for this task.
        self.task_details = tp.root_task_desc.format(
            baseline_score=self._baseline_score,
            data_head="(code files available in workspace)",
        )

        # Parser extracts the proposal from freeform text.
        parser = parser or vf.XMLParser(fields=["proposal"], answer_field="proposal")

        # Reward selection.
        self.reward_scheme = reward_scheme
        self.reward_batch_size = reward_batch_size
        logger.info(f"Using reward scheme: {reward_scheme} (batch_size={reward_batch_size})")
        reward_callable = _make_reward(
            reward_scheme, self._baseline_score, reward_batch_size, self.task_name, v7_tau,
        )
        # Hold a reference to stateful reward objects so their state persists.
        self._reward_obj = reward_callable if isinstance(reward_callable, RewardV9Percentile) else None

        rubric = rubric or vf.Rubric(parser=parser)
        rubric.add_reward_func(reward_callable, weight=1.0)

        dataset = self._build_dataset(num_train_examples)
        eval_dataset = self._build_dataset(num_eval_examples) if num_eval_examples > 0 else None

        # max_turns = node_budget + 1: env_response runs *between* turns, so the
        # final proposal needs one extra turn to actually execute (see env_response).
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=TREE_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            max_turns=node_budget + 1,
            **kwargs,
        )

    def _build_dataset(self, num_examples: int) -> Dataset:
        rows = [{"question": self._format_initial_prompt(), "answer": ""} for _ in range(num_examples)]
        return Dataset.from_list(rows)

    def _format_initial_prompt(self) -> str:
        tp = self.task_profile
        baseline = self._baseline_score
        root_tree = [{"id": "root", "parent_id": None, "depth": 0,
                      "score": baseline, "strategy": "baseline (no experiment)"}]
        return TREE_INITIAL_PROMPT.format(
            task_description=tp.name,
            task_details=self.task_details,
            metric_name=tp.primary_metric,
            direction="higher" if tp.higher_is_better else "lower",
            baseline_score=baseline,
            tree_view=self._format_tree(root_tree),
            memory="No experiments run yet.",
            budget_left=self.node_budget,
            total_budget=self.node_budget,
        )

    def _format_tree(self, tree: list[dict]) -> str:
        """Hierarchical rendering with parent comparison and child counts."""
        if not tree:
            return "  (empty tree)"
        node_by_id = {n["id"]: n for n in tree}
        children_map: dict[str, list[str]] = {}
        for n in tree:
            pid = n.get("parent_id")
            if pid is not None:
                children_map.setdefault(pid, []).append(n["id"])

        higher = tree[0].get("higher_is_better", True) if tree else True
        lines: list[str] = []

        def _render(node_id: str, prefix: str, is_last: bool, is_root: bool):
            n = node_by_id.get(node_id)
            if n is None:
                return
            sc = n.get("score")
            sc_str = f"{sc:.4f}" if sc is not None else "FAILED"
            n_children = len(children_map.get(node_id, []))

            parent_note = ""
            pid = n.get("parent_id")
            if pid and pid in node_by_id and sc is not None:
                parent_sc = node_by_id[pid].get("score")
                if parent_sc is not None:
                    diff = sc - parent_sc
                    better = (diff > 0) if higher else (diff < 0)
                    parent_note = f" ({'better' if better else 'worse'} than parent by {abs(diff):.4f})"

            if is_root:
                lines.append(f"Node {node_id} [Baseline]")
                lines.append(f"  Score: {sc_str} | Children: {n_children}")
                new_prefix = "  "
            else:
                branch = "└─ " if is_last else "├─ "
                strat_full = (n.get("strategy") or "").replace("\n", " ")[:200]
                lines.append(f"{prefix}{branch}Node {node_id} [{strat_full}]")
                lines.append(f"{prefix}{'   ' if is_last else '│  '}"
                             f"  Score: {sc_str} | Children: {n_children}{parent_note}")
                new_prefix = prefix + ("   " if is_last else "│  ")
            kids = children_map.get(node_id, [])
            for i, cid in enumerate(kids):
                _render(cid, new_prefix, i == len(kids) - 1, is_root=False)

        _render("root", "", True, is_root=True)
        best = max((n["score"] for n in tree
                    if n.get("score") is not None and n.get("id") != "root"),
                   default=None)
        if best is None:
            lines.append("\n(no scored children yet)")
        else:
            lines.append(f"\nBest score so far: {best:.4f}")
        return "\n".join(lines)

    async def setup_state(self, state: vf.State) -> vf.State:
        tp = self.task_profile
        baseline = self._baseline_score
        state["tree"] = [{
            "id": "root",
            "parent_id": None,
            "depth": 0,
            "score": baseline,
            "strategy": "Baseline (no experiment)",
        }]
        state["baseline_score"] = baseline
        state["higher_is_better"] = tp.higher_is_better
        state["best_score"] = baseline
        state["node_counter"] = 0
        state["last_score"] = None
        state["memory_lines"] = []
        state["any_score_achieved"] = False
        state["executor_fault_count"] = 0
        # Per-node child counters for hierarchical naming (root_0, root_0_0, ...).
        state["_child_counter"] = {}
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        last_msg = messages[-1] if messages else None
        raw_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        direction = parse_direction(raw_text)
        memory_update = parse_memory(raw_text)
        valid_ids = {n["id"] for n in state.get("tree", [])}
        parent_id = parse_parent(raw_text, valid_ids)

        if not direction or len(direction.strip()) < 5:
            state["last_score"] = None
            state["executor_fault_count"] += 1
            tree_view = self._format_tree(state["tree"])
            memory = "\n".join(state.get("memory_lines", [])) or "No observations yet."
            response = vf.UserMessage(
                content=TREE_TURN_PROMPT.format(
                    result="Invalid direction. Please include DIRECTION: section.",
                    tree_view=tree_view,
                    memory=memory,
                    budget_left=self.node_budget - state["node_counter"],
                )
            )
            return [response]

        if memory_update:
            state.setdefault("memory_lines", []).append(memory_update)

        logger.info(
            f"Executing (parent={parent_id}, depth="
            f"{next((n.get('depth', 0) for n in state['tree'] if n['id'] == parent_id), 0) + 1}): "
            f"{direction[:80]}..."
        )

        # Execute in the container (retry once on fault).
        max_retries = 2
        score = None
        feedback = ""
        executor_fault = False
        for attempt in range(max_retries):
            score, feedback, executor_fault = await asyncio.to_thread(
                execute_in_container,
                proposal=direction,
                task_profile=self.task_profile,
                task_config=self.task_cfg["task_config"],
                container_image=self.task_cfg["container_image"],
                executor_url=self.executor_url,
                executor_model=self.executor_model,
                max_actions=self.max_actions,
                env_gpu=self.env_gpu,
            )
            if not executor_fault or score is not None:
                break
            logger.info(f"Executor fault attempt {attempt + 1}, retrying")

        self._log_rollout(state, direction, score, feedback, executor_fault, raw_text, parent_id)

        # Attach the new node to the chosen parent with hierarchical naming.
        state["node_counter"] += 1
        parent_node = next(
            (n for n in state["tree"] if n["id"] == parent_id),
            state["tree"][0],  # fallback: root
        )
        child_counter = state.get("_child_counter", {})
        child_idx = child_counter.get(parent_id, 0)
        child_counter[parent_id] = child_idx + 1
        state["_child_counter"] = child_counter
        node_id = f"{parent_id}_{child_idx}"
        new_depth = parent_node.get("depth", 0) + 1
        state["tree"].append({
            "id": node_id,
            "parent_id": parent_id,
            "depth": new_depth,
            "score": score,
            "strategy": direction[:200],
        })

        if executor_fault:
            state["executor_fault_count"] += 1

        state["last_score"] = score
        if score is not None:
            state["any_score_achieved"] = True
            higher = state["higher_is_better"]
            if (higher and score > state["best_score"]) or \
               (not higher and score < state["best_score"]):
                state["best_score"] = score
                logger.info(f"NEW BEST: {score:.4f}")

        if score is not None:
            result = f"Score: {score:.4f}. {feedback[:300]}"
        elif executor_fault:
            result = f"FAILED (executor error). {feedback[:300]}"
        else:
            result = f"FAILED. {feedback[:300]}"

        tree_view = self._format_tree(state["tree"])
        memory = "\n".join(state.get("memory_lines", [])) or "No observations yet."
        response = vf.UserMessage(
            content=TREE_TURN_PROMPT.format(
                result=result,
                tree_view=tree_view,
                memory=memory,
                budget_left=self.node_budget - state["node_counter"],
            )
        )

        # Terminal env response after the final execution so the last proposal
        # actually runs (otherwise max_turns would stop one turn early).
        if state["node_counter"] >= self.node_budget:
            state["final_env_response"] = [response]
            logger.info(f"Budget reached ({self.node_budget}), terminating episode")

        return [response]

    def _log_rollout(self, state, direction, score, feedback, executor_fault,
                     scientist_output, parent_id):
        try:
            ROLLOUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_file = ROLLOUT_LOG_DIR / f"{self.task_name}_{self.reward_scheme}_tree_rollouts.jsonl"
            parent_depth = next(
                (n.get("depth", 0) for n in state["tree"] if n["id"] == parent_id), 0,
            )
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task": self.task_name,
                "scheme": self.reward_scheme,
                "tree_mode": True,
                "node_counter": state.get("node_counter", 0),
                "parent_id": parent_id,
                "new_node_depth": parent_depth + 1,
                "baseline_score": state.get("baseline_score", 0),
                "best_so_far": state.get("best_score", 0),
                "score": score,
                "executor_fault": executor_fault,
                "feedback": (feedback or "")[:300],
                "direction": (direction or "")[:300],
                "tree_size": len(state.get("tree", [])),
            }
            reasoning_match = re.search(
                r"REASONING:\s*\n(.*?)(?:\nSTRATEGIES:|\Z)", scientist_output, re.DOTALL,
            )
            if reasoning_match:
                entry["scientist_reasoning"] = reasoning_match.group(1).strip()[:300]
            strategies_match = re.search(
                r"STRATEGIES:\s*\n(.*?)(?:\nCHOSEN:|\nDIRECTION:|\Z)", scientist_output, re.DOTALL,
            )
            if strategies_match:
                entry["scientist_strategies"] = strategies_match.group(1).strip()[:500]
            mode_match = re.search(r"MODE:\s*(explore|exploit)", scientist_output, re.IGNORECASE)
            if mode_match:
                entry["mode"] = mode_match.group(1).lower()
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"rollout log write failed: {e}")


# Backwards-compatible alias (older references / the mlgym_tree_env_v3 package).
MLGymTreeEnvV3 = MLGymTreeEnv


def load_environment(env_args: dict | None = None, **kwargs) -> MLGymTreeEnv:
    env_args = env_args or {}
    return MLGymTreeEnv(**env_args, **kwargs)
