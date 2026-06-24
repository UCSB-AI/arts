"""
Experiment 4: LLM-Guided Tree Search.

Replaces formula-based node selection (UCB, Softmax, Open-Ended) with an
LLM "scientist" that sees the full tree state, reasons about why things
worked or failed, and makes informed decisions about what to expand next.

Two-model setup:
    - Scientist (selector): larger API model (GPT-4o / Claude) that analyzes
      the tree and decides what to expand
    - Executor (worker): Qwen3-4B via vLLM that implements strategies in
      MLGym containers (unchanged from Exp 2/3)

Usage:
    cd /home/ubuntu/MLScientist/MLGym
    uv run --project /home/ubuntu/MLScientist/arts \
        python /home/ubuntu/MLScientist/arts/air/llm_guided_tree_search.py \
        --task-config tasks/titanic.yaml \
        --scientist-model gpt-4o \
        --executor-model Qwen/Qwen3-4B-Instruct-2507 \
        --executor-url http://localhost:8000/v1 \
        --node-budget 15
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env for API keys
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from arts.tree_search import (
    TaskProfile,
    TASK_PROFILES,
    get_task_profile,
    ContainerManager,
    LLMClient,
    TreeNode,
    extract_command,
    classify_execution,
    MLGYM_PATH,
)


# ---------------------------------------------------------------------------
# ScientistDecision — parsed output from the scientist LLM
# ---------------------------------------------------------------------------

@dataclass
class ScientistDecision:
    action: str          # "expand" or "draft_from_root"
    node_id: str         # which node to expand (empty for draft_from_root)
    direction: str       # specific instruction for executor
    mode: str            # "explore" or "exploit"
    memory_update: str   # observation to remember (or empty)
    reasoning: str       # scientist's analysis (logged)
    executor_guidance: str = ""  # warnings/tips passed directly to executor
    axis: str = "hp"     # experiment axis: architecture|loss|data_representation|data_augmentation|regularization|optimizer|hp|combination


# ---------------------------------------------------------------------------
# Scientist Prompt
# ---------------------------------------------------------------------------

SCIENTIST_PROMPT_TURN1 = """You are an ML research ADVISOR. You propose ONE experiment per turn.
A separate coder (the "executor") executes it. You must NEVER write code
(no ``` blocks, no import statements, no def/class).

Think like a scientist: analyze what has been tried, identify gaps, form a
hypothesis, and propose a direction.

Be aware that per-node training cost varies by task — some nodes finish in
minutes, others take 30+ minutes. Coupled changes (LR schedule + longer
training, augs + the regularization that enables them) bundle naturally;
unrelated levers do not.

## Task

{task_description}

## Task Details (this is what the executor sees)

{task_details}

The metric is: {metric_name} ({direction} is better)
Baseline score (no model, just default): {baseline_score}

## How This Works

Each time you propose an experiment, the executor writes code from scratch in a
container, runs it, and validates. It has {max_actions} actions (shell commands) per
attempt. Each attempt creates one "node" in your search tree. The executor can run
any single command for up to 45 minutes.

## Budget

Total run time: {time_budget_min} min. Time elapsed: {time_elapsed_min} min.
Time remaining: {time_remaining_min} min.
Nodes scored so far: {nodes_done} (avg {avg_per_node_min:.1f} min each).
At the current per-node pace, ~{nodes_remaining_est} more nodes fit in the
remaining budget. Factor this into how ambitious each candidate is.

IMPORTANT: The executor already has ALL source files from the workspace pre-loaded in
its context. It can see the full code. Do NOT waste a node asking it to "read" or
"examine" files — it already knows the code. Every experiment you propose should be an
ACTIONABLE change (modify config, swap architecture, tune hyperparameters), never exploration.

INTEGRITY: Do NOT modify evaluation files, opponent strategy files (e.g. target.py),
or any read-only starter code. Do NOT use monkey-patching, sys.modules hacking, or
any technique to manipulate the scoring system. All such modifications are reverted
before evaluation. Only modify YOUR submission files (e.g. strategy.py, baseline.py,
train_and_predict.py). Legitimate improvements only.

## Your Search Tree

{tree_view}

## Your Accumulated Knowledge

{memory_section}

## Your Task Now

Before making a decision, you have two tools:

1. INSPECT nodes — see the actual commands and output the executor ran for any node.
   This lets you understand EXACTLY what was tried and why it succeeded or failed.

2. READ files — see the contents of any workspace file (e.g. target.py, evaluate.py,
   baseline.py, strategy.py). This lets you understand the task's code, opponent
   strategy, evaluation logic, or data format before proposing an experiment.

Respond in EXACTLY this format:

INSPECT: node_id_1, node_id_2
[OR]
INSPECT: NONE

READ: filename1.py, filename2.py
[OR]
READ: NONE

Brief explanation of what you want to understand."""


SCIENTIST_PROMPT_TURN2 = """Good. Now make your decision.

{code_inspection}

You are an ML research ADVISOR. You propose ONE experiment per turn. You must
NEVER write code (no ``` blocks, no import statements, no def/class).

Think like a scientist: analyze what has been tried, identify gaps, form a
hypothesis, then propose a specific experiment.

Tree stats: {explore_stats}
Time remaining: {time_remaining_min} min ({nodes_done} scored, ~{avg_per_node_min:.1f} min/node).

## Rules

- NO CODE. Describe everything in precise English.
- DIVERSITY: Look at the Exploration Summary below. If one axis dominates (e.g.,
  >50% of attempts), deliberately explore an UNEXPLORED axis. Axes are:
  architecture, loss, data_representation, data_augmentation, regularization,
  optimizer, hp, combination. Note: data_representation (what ENTERS the model
  — input channels, slice ranges, resolution, patch size) and data_augmentation
  (transforms applied to each sample — flips, rotations, mixup, etc.) are TWO
  DIFFERENT axes. Changing what the model sees is NOT the same as perturbing it.
  Be specific — name exact technique names, exact parameter values.
- BASELINE AUDIT (mandatory before any loss/optimizer/aug/reg tweak): before
  proposing a downstream tweak, mentally audit the data pipeline based on the
  baseline code you have already seen. Ask: what signal does the raw input
  contain, and what fraction does the current pipeline preserve end-to-end?
  Identify the single largest information-loss step in the pipeline. If a
  downstream tweak (loss/reg/hp/opt) cannot plausibly close a gap that the
  pipeline itself creates, the pipeline is the bottleneck and must be fixed
  FIRST. State the identified bottleneck in your ANALYSIS. (Do NOT request
  additional file reads in this turn — you are in the decision turn.)
- EVOLVE: There are TWO valid ways to build on prior nodes — pick whichever fits.
  (a) LAYER — extend ONE high-scoring parent with a new idea. Set PARENT to that
      parent, COMBINES: NONE. A change from a different axis than the parent's
      counts as a new strategy.
  (b) MERGE — combine two (or more) high-scoring nodes from DIFFERENT branches
      that succeeded on DIFFERENT axes. Example: one branch won by switching
      architecture, another by adding augmentation; the merge runs the new
      architecture WITH the new augmentation. Set PARENT to the stronger of the
      two and list the other(s) in COMBINES. Reserve MERGE for when (i) the tree
      has at least 2 distinct branches with non-trivial scores, AND (ii) you can
      name the specific axis each branch optimized. A merge is especially worth
      trying when 3+ recent single-parent IMPROVE attempts have stagnated — a
      cross-branch combination is more likely to break the plateau than yet
      another tweak on the same parent.
- BUILD ON SIGNAL, NOT ASSUMPTION: Early in a task, prefer focused single-axis
  probes over compound strategies. Before any axis has been individually validated
  in this tree, a minimal change — one loss, one sampler, one architecture swap
  — produces a cleaner signal than a multi-part bundle. Once you've seen a couple
  of axes move the score on their own, you can compound validated changes into
  one direction. Point is to build on signal rather than design on assumption.
- Failed nodes are DATA — analyze WHY they failed and whether the idea could work
  differently. A code failure does NOT mean the approach is wrong.
- REGRESSION INVESTIGATION (STRUCTURAL — enforced by output format below): if
  any recently completed node scored >20% below its parent, you MUST have
  INSPECTed that node (use turn 1 to do so) and emit a CLASSIFICATION: line at
  the bottom of your response. Valid values:
    IDEA-WRONG: <node_id> — <brief why, rooted in inspected evidence>
    IMPLEMENTATION-WRONG: <node_id> — <specific code-level bug found in
      INSPECT, e.g., "conv1 weights mean-replicated across 16 channels
      without rescaling; destroys pretrained features">
    NONE (only if no regression >20% exists among recently completed nodes)
  If you emit IMPLEMENTATION-WRONG, your EXPERIMENT MUST be on the SAME AXIS as
  the regressed node, with explicit instructions that address the bug you
  identified. Do not add "X approach failed" to memory when the real failure
  was implementation.
- Do NOT abandon promising nodes prematurely. A node that scored well but hasn't
  been deepened 3+ times still has unexplored potential.
- If the last 3+ refinements in the same family didn't move the score, switch to
  a DIFFERENT family (classical ML → deep learning, CNN → transformer, etc.).

## Your Output

Respond in EXACTLY this format:

ANALYSIS:
[Thorough analysis of the experiment tree:
 - What has been tried and what scores did they achieve?
 - What axes are MISSING or underexplored? (architecture? loss functions? data_representation? data_augmentation? regularization? training schedule?)
 - BASELINE AUDIT: what is the single largest information-loss step in the current data pipeline? Can any downstream knob plausibly recover that lost signal, or does the pipeline itself need to change?
 - Which high-scoring nodes could be improved further, and how?
 - What do the failures tell us?]

Enumerate {n_candidates} fundamentally different candidate experiments.
Candidates must span DIFFERENT axes OR DIFFERENT families within an axis
(ResNet50 vs ResNet18 = same family; CNN vs Transformer = different;
F-RCNN vs CenterNet = different). {cap_instruction}

The system samples ONE candidate weighted by your <probability> values and
executes that candidate's <plan>. Be honest about your priors — DO NOT
inflate probability on your favorite. The system, not you, picks; your job
is honest enumeration.

<candidates>
<response>
<direction>[1 sentence naming the family/technique]</direction>
<probability>NUMBER</probability>
<plan>
HYPOTHESIS: [1-2 sentences: why this could improve performance]
EXPERIMENT: [3-6 sentences. WHAT (exact models, techniques, values),
             HOW (which components to swap, what to add), WHY (expected
             outcome). As precise as a methods section. No code.]
PARENT: [node_id to build on, or "root" for fresh exploration]
COMBINES: [comma-separated node_ids whose ideas to merge, or NONE]
AXIS: [architecture | loss | data_representation | data_augmentation | regularization | optimizer | hp | combination]
MODE: [explore | exploit]
</plan>
</response>
[…exactly {n_candidates} responses; probabilities sum to 1.0…]
</candidates>

MEMORY:
[One sentence about what you LEARNED. Must include evidence (what was tried,
what score) and an insight. Do NOT repeat anything already in your memory.
GOOD: "CatBoost (0.91) and LightGBM (0.90) both plateau — try feature engineering next."
BAD: "CatBoost works well." (repeats known info, no new insight)
Write NONE if no genuinely new insight.]

CLASSIFICATION:
[Mandatory. Exactly ONE of these three values, nothing else:
  IDEA-WRONG: <node_id> — <brief reason from inspection>
  IMPLEMENTATION-WRONG: <node_id> — <specific code-level bug from inspection>
  NONE
Write NONE if and only if no recently completed node has regressed >20% from
its parent. Otherwise you MUST pick IDEA-WRONG or IMPLEMENTATION-WRONG, and if
IMPLEMENTATION-WRONG, your EXPERIMENT above must be on the SAME AXIS as the
regressed node with corrected instructions for the executor.]"""


SCIENTIST_PROMPT_RESEARCH_TURN2 = """You are in the RESEARCH PHASE — NOT proposing experiments yet.

{code_inspection}

Your job right now is to BUILD A MENTAL MODEL of the task before running experiments.
A good human researcher spends the first day understanding the problem: what the
data looks like, what the metric rewards, what the baseline actually computes,
where the easy wins are, where the hidden traps are. That is what this turn is for.

Do NOT propose an experiment. Do NOT pick a node to expand.
Instead, write structured findings that future-you will use to guide the real search.

## Your Output

Respond in EXACTLY this format:

FINDINGS:
[3-8 specific observations about the task. Each should be concrete and
actionable-in-the-future. Examples:
 - "Input volumes contain 65 z-slices but the baseline only uses slices 28-33
   — the z-axis signal is mostly discarded in the baseline pipeline."
 - "Evaluation uses F0.5 which rewards precision 2× recall — the threshold choice
   will matter a lot."
 - "Fragment 1 has ~20x more positive pixels than fragment 2 — class imbalance
   differs between training fragments."
No proposals. Findings only.]

OPEN QUESTIONS:
[2-4 concrete things you still don't understand and would want to investigate
via READ / INSPECT in future research turns. Be specific — not "how does training
work" but "what exactly does the eval script consider a valid submission format?"]

MEMORY:
[1-3 one-sentence insights to save for the active search phase. Each must be
a novel, evidence-backed observation that would change how you design experiments.
Write NONE if no genuinely new insight.
GOOD: "Baseline averages 5 central z-slices and triples the result — ~90% of z-axis information is discarded at input; any downstream tweak ceiling-limits around this loss."
BAD: "The task is hard." (too vague, no action implied)]"""


# ---------------------------------------------------------------------------
# LLM-Guided Tree Search
# ---------------------------------------------------------------------------

class LLMGuidedTreeSearch:
    """Tree search where an LLM scientist replaces formula-based selection."""

    def __init__(
        self,
        scientist: LLMClient,
        executor: LLMClient,
        container: ContainerManager,
        task_profile: TaskProfile,
        node_budget: int = 12,
        initial_breadth: int = 3,
        max_actions: int = 15,
        output_dir: str = "outputs/llm_guided_search",
        verbose: bool = False,
        time_budget: int = 0,
        resume_from: str = "",
        research_phase_steps: int = 0,
        full_parent_history: bool = False,
        no_build_on_signal: bool = False,
        no_incremental_rules: bool = False,
        complexity_cycle: bool = False,
    ):
        self.scientist = scientist
        self.executor = executor
        self.container = container
        self.task = task_profile
        self.node_budget = node_budget
        self.initial_breadth = initial_breadth
        self.max_actions = max_actions
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.time_budget = time_budget  # seconds, 0 = no limit
        self._resume_path = resume_from
        self.research_phase_steps = research_phase_steps
        self.full_parent_history = full_parent_history
        self.no_build_on_signal = no_build_on_signal
        self.no_incremental_rules = no_incremental_rules
        self.complexity_cycle = complexity_cycle
        self.nodes: dict[str, TreeNode] = {}
        self.memory: list[str] = []
        self._child_counter: dict[str, int] = {}
        self._active_step_counter: int = 0  # incremented per active expansion
        self.start_time: float = time.time()  # set again in run()

    # ------------------------------------------------------------------
    # Prompt-time context helpers
    # ------------------------------------------------------------------

    def _budget_ctx(self) -> dict:
        """Time / nodes-done facts for the scientist prompt."""
        time_budget_min = int(self.time_budget / 60) if self.time_budget > 0 else 0
        time_elapsed_min = int((time.time() - self.start_time) / 60)
        time_remaining_min = (
            max(0, time_budget_min - time_elapsed_min) if time_budget_min > 0 else 0
        )
        nodes_done = sum(1 for n in self.nodes.values() if n.score is not None)
        avg_per_node_min = (time_elapsed_min / nodes_done) if nodes_done > 0 else 0.0
        if avg_per_node_min > 0 and time_remaining_min > 0:
            nodes_remaining_est = int(time_remaining_min / avg_per_node_min)
        else:
            nodes_remaining_est = "?"
        return dict(
            time_budget_min=time_budget_min,
            time_elapsed_min=time_elapsed_min,
            time_remaining_min=time_remaining_min,
            nodes_done=nodes_done,
            avg_per_node_min=avg_per_node_min,
            nodes_remaining_est=nodes_remaining_est,
        )

    def _vs_ctx(self, n_candidates: int = 3) -> dict:
        """Verbalized-sampling knobs (cap regime keyed on tree-state)."""
        max_scored_depth = max(
            (n.depth for n in self.nodes.values() if n.score is not None),
            default=0,
        )
        if max_scored_depth <= 2:
            cap_instruction = "No probability cap; floor 0.05 per candidate."
        elif max_scored_depth <= 4:
            cap_instruction = "Soft cap: each <probability> ≤ 0.50."
        else:
            cap_instruction = (
                "Hard cap: each <probability> ≤ 0.20 "
                "(deep-tree mode-collapse guard)."
            )
        return dict(n_candidates=n_candidates, cap_instruction=cap_instruction)

    def _external_sample(self, turn2_resp: str) -> tuple[str, dict]:
        """Parse <candidates>, weighted-sample externally, synthesize a flat
        response that the legacy parser can read.

        Returns (synthesized_text, sampling_record). If parsing fails, returns
        (turn2_resp, {}) — falls through to the legacy single-EXPERIMENT path.
        """
        max_scored_depth = max(
            (n.depth for n in self.nodes.values() if n.score is not None),
            default=0,
        )
        if max_scored_depth <= 2:
            cap = None
        elif max_scored_depth <= 4:
            cap = 0.50
        else:
            cap = 0.20

        pat = (
            r"<response>\s*"
            r"<direction>\s*(.*?)\s*</direction>\s*"
            r"<probability>\s*([0-9.]+)\s*</probability>\s*"
            r"<plan>\s*(.*?)\s*</plan>\s*"
            r"</response>"
        )
        matches = re.findall(pat, turn2_resp, re.DOTALL)
        if not matches:
            return turn2_resp, {}

        cands = []
        for direction, p_str, plan in matches:
            try:
                p = float(p_str)
            except ValueError:
                p = 0.0
            cands.append({
                "direction": direction.strip(),
                "probability": p,
                "plan": plan.strip(),
            })

        # Apply cap, renormalize
        if cap is not None:
            for c in cands:
                c["probability"] = min(c["probability"], cap)
        Z = sum(c["probability"] for c in cands) or 1.0
        weights = [c["probability"] / Z for c in cands]

        idx = random.choices(range(len(cands)), weights=weights, k=1)[0]
        chosen = cands[idx]

        # Pull shared blocks (ANALYSIS, MEMORY, CLASSIFICATION) from outside
        # the <candidates> region so the legacy parser sees the full picture.
        before_cands = re.split(r"<candidates>", turn2_resp, maxsplit=1)[0]
        after_cands = ""
        m_after = re.search(r"</candidates>(.*)", turn2_resp, re.DOTALL)
        if m_after:
            after_cands = m_after.group(1)

        synthesized = (
            f"{before_cands}\n\n"
            f"{chosen['plan']}\n\n"
            f"{after_cands}"
        )

        sampling_record = {
            "n_candidates": len(cands),
            "cap": cap,
            "weights": weights,
            "raw_probabilities": [m[1] for m in matches],
            "directions": [c["direction"] for c in cands],
            "sampled_index": idx,
            "sampled_direction": chosen["direction"],
        }
        return synthesized, sampling_record

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _load_resumed_tree(self, resume_path: str):
        """Load a previous result.json and rebuild the tree for resuming."""
        prev = json.loads(Path(resume_path).read_text())
        prev_nodes = prev.get("nodes", {})
        prev_memory = prev.get("memory", [])

        print(f"  Resuming from {resume_path}")
        print(f"  Previous run: {prev.get('total_nodes', 0)} nodes, "
              f"best={prev.get('best_score')}")

        # Rebuild TreeNode objects from saved data (without snapshots/conversations)
        for nid, ndata in prev_nodes.items():
            if nid == "root":
                continue  # root is recreated fresh with a live container
            node = TreeNode(
                node_id=ndata["node_id"],
                parent_id=ndata.get("parent_id"),
                depth=ndata.get("depth", 1),
                strategy=ndata.get("strategy", ""),
                score=ndata.get("score"),
                actions=[],  # no action replay
                children=list(ndata.get("children", [])),
                conversation_history=[],  # no conversation replay
                snapshot_path="",  # no snapshot — will expand from root
                error=ndata.get("error"),
            )
            self.nodes[nid] = node
            # Track child counters so new children get correct indices
            parent = ndata.get("parent_id")
            if parent:
                self._child_counter[parent] = self._child_counter.get(parent, 0) + 1

        # Restore memory
        if isinstance(prev_memory, list):
            self.memory = list(prev_memory)

        # Wire up root's children from the loaded tree
        root_node = self.nodes.get("root")
        if root_node:
            root_children = prev_nodes.get("root", {}).get("children", [])
            for cid in root_children:
                if cid not in root_node.children:
                    root_node.children.append(cid)

        resumed_count = len(self.nodes) - 1  # minus root which was already there
        print(f"  Loaded {resumed_count} previous nodes, {len(self.memory)} memory entries")

    def run(self) -> dict:
        start = time.time()
        self.start_time = start
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "nodes").mkdir(exist_ok=True)

        # ---- Phase 1: Create baseline root ----
        print("\n" + "=" * 60)
        print("LLM-GUIDED TREE SEARCH - Phase 1: Root (baseline)")
        print("=" * 60)
        root = self._execute_root()
        self.nodes[root.node_id] = root
        self._save_node(root)

        # ---- Phase 1b: Resume from previous run if requested ----
        if self._resume_path:
            self._load_resumed_tree(self._resume_path)

        # ---- Phase 1c: Research phase (no node expansion) ----
        if self.research_phase_steps > 0:
            print(f"\n{'=' * 60}")
            print(f"Phase 1c: Research phase ({self.research_phase_steps} steps, no expansion)")
            print("=" * 60)
            for rstep in range(self.research_phase_steps):
                if self.time_budget > 0 and time.time() - start >= self.time_budget:
                    break
                print(f"\n--- Research step {rstep + 1}/{self.research_phase_steps} ---")
                self._scientist_research_step(rstep)

        # ---- Phase 2: Scientist-guided loop (all budget) ----
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Scientist-guided search ({self.node_budget} expansions)")
        print("=" * 60)

        stopped_by = "node_budget"
        for step in range(self.node_budget):
            # Check time budget before each expansion
            if self.time_budget > 0:
                elapsed = time.time() - start
                if elapsed >= self.time_budget:
                    print(f"\n--- Time budget reached ({elapsed:.0f}s >= {self.time_budget}s). Stopping after {step} expansions. ---")
                    stopped_by = "time_budget"
                    break

            print(f"\n--- Scientist step {step + 1}/{self.node_budget} ---")
            budget_left = self.node_budget - step
            self._active_step_counter = step

            # Ask scientist what to do
            decision = self._scientist_decide(budget_left)

            # Update memory (unbounded; scientist keeps full history of insights)
            if decision.memory_update and decision.memory_update.upper() != "NONE":
                self.memory.append(decision.memory_update)
                print(f"  Memory updated: {decision.memory_update[:80]}")

            # Execute the decision
            if decision.action == "draft_from_root":
                parent_id = "root"
            else:
                # Validate node_id exists
                parent_id = decision.node_id
                if parent_id not in self.nodes:
                    print(f"  WARNING: Scientist chose non-existent node '{parent_id}', falling back to root")
                    parent_id = "root"

            # Enforce initial breadth: force the first `initial_breadth`
            # expansions to be root children. This guarantees a diverse
            # starting fan-out before the scientist goes depth-first.
            root_children = len(self.nodes.get("root").children) if "root" in self.nodes else 0
            if root_children < self.initial_breadth and parent_id != "root":
                print(f"  [initial-breadth enforce] scientist chose {parent_id}, "
                      f"but only {root_children}/{self.initial_breadth} root "
                      f"children exist — rerouting to root")
                parent_id = "root"
                decision.mode = "explore"

            print(f"  -> Expanding {parent_id} (mode={decision.mode})")
            print(f"  -> Direction: {decision.direction[:100]}")

            child = self._expand_one(
                parent_id,
                mode=decision.mode,
                direction=decision.direction,
                executor_guidance=decision.executor_guidance,
                axis=decision.axis,
            )

            # Periodic snapshot cleanup (every 10 nodes) to prevent tmpfs overflow
            if len(self.nodes) > 15 and len(self.nodes) % 10 == 0:
                self._cleanup_snapshots(keep_top_k=15)

        # ---- Results ----
        return self._compile_results(start, stopped_by=stopped_by)

    # ------------------------------------------------------------------
    # Root node
    # ------------------------------------------------------------------

    def _read_workspace_files(self) -> str:
        """Read all source files from the workspace and return as a string."""
        file_list = self.container.communicate(
            "find /home/agent/workspace -type f "
            "\\( -name '*.py' -o -name '*.yaml' -o -name '*.yml' "
            "-o -name '*.json' -o -name '*.cfg' -o -name '*.txt' "
            "-o -name '*.sh' \\) "
            "! -path '*/checkpoints/*' ! -path '*/__pycache__/*' "
            "| sort"
        ).strip()
        if not file_list:
            return ""

        parts = []
        total_chars = 0
        max_total = 50000  # cap total workspace context to ~12K tokens
        for fpath in file_list.split("\n"):
            fpath = fpath.strip()
            if not fpath:
                continue
            content = self.container.communicate(f"cat {fpath}")
            # Skip very large files (>5000 chars)
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            rel = fpath.replace("/home/agent/workspace/", "")
            parts.append(f"=== {rel} ===\n{content}")
            total_chars += len(content)
            if total_chars > max_total:
                parts.append(f"... ({len(file_list.split(chr(10))) - len(parts)} more files omitted)")
                break

        return "\n\n".join(parts)

    def _execute_root(self) -> TreeNode:
        data_head = ""
        if self.task.data_head_cmd:
            data_head = self.container.communicate(self.task.data_head_cmd)

        from arts.tree_search import generate_code_outline
        task_desc = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head=data_head,
            code_outline=generate_code_outline(getattr(self.task, "starter_code_host_path", "")),
        )

        # Read all workspace source files and inject into context
        workspace_files = self._read_workspace_files()
        if workspace_files:
            file_context = (
                "Here are the current source files in your workspace. "
                "You do NOT need to read them again — they are already provided:\n\n"
                f"{workspace_files}"
            )
            task_desc = f"{task_desc}\n\n{file_context}"

        messages = [
            {"role": "system", "content": self.task.system_prompt},
            {"role": "user", "content": task_desc},
        ]

        snap = self.container.save_snapshot("root")
        baseline = self.container.baseline_score
        print(f"  [root] Baseline (score={baseline:.4f})")

        return TreeNode(
            node_id="root", parent_id=None, depth=0,
            strategy="Baseline (no model execution)",
            score=baseline, actions=[],
            conversation_history=messages,
            snapshot_path=snap,
        )

    # ------------------------------------------------------------------
    # Snapshot cleanup
    # ------------------------------------------------------------------

    def _cleanup_snapshots(self, keep_top_k: int = 15):
        """Delete snapshots for nodes unlikely to be expanded, freeing tmpfs.

        Keeps snapshots for:
        - root (always needed as fallback parent)
        - top-K scoring nodes (likely expansion candidates)
        - ancestors of top-K (needed to understand lineage, though not for restore)
        """
        scored = [
            (nid, n.score) for nid, n in self.nodes.items()
            if n.score is not None
        ]
        if len(scored) <= keep_top_k:
            return  # not enough nodes to bother cleaning

        # Find top-K nodes by score
        if self.task.higher_is_better:
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored.sort(key=lambda x: x[1])
        top_k_ids = {nid for nid, _ in scored[:keep_top_k]}

        # Always keep root
        keep = {"root"} | top_k_ids

        # Also keep ancestors of top-K
        for nid in list(top_k_ids):
            cur = nid
            while cur and cur in self.nodes:
                keep.add(cur)
                cur = self.nodes[cur].parent_id

        # Delete snapshots for nodes not in keep set
        deleted = 0
        for nid, node in self.nodes.items():
            if nid not in keep and node.snapshot_path:
                try:
                    self.container.communicate(
                        f"rm -f {node.snapshot_path}", timeout=10,
                    )
                    node.snapshot_path = ""
                    deleted += 1
                except Exception:
                    pass
        if deleted:
            print(f"  [cleanup] Deleted {deleted} old snapshots, keeping {len(keep)}")

    # ------------------------------------------------------------------
    # Tree view for scientist
    # ------------------------------------------------------------------

    def _build_explore_stats(self) -> str:
        """Build exploration summary with axis distribution for the scientist."""
        if len(self.nodes) <= 1:
            return "Tree is empty — start by exploring."

        root_children = len(self.nodes.get("root", TreeNode(
            node_id="root", parent_id=None, depth=0, strategy="",
            score=None, actions=[], conversation_history=[], snapshot_path="",
        )).children) if "root" in self.nodes else 0

        max_depth = max((n.depth for n in self.nodes.values()), default=0)

        # Best branch info
        scored = [(nid, n.score) for nid, n in self.nodes.items() if n.score is not None]
        if scored:
            if self.task.higher_is_better:
                best_id, best_score = max(scored, key=lambda x: x[1])
            else:
                best_id, best_score = min(scored, key=lambda x: x[1])
            best_depth = self.nodes[best_id].depth
            best_children = len(self.nodes[best_id].children)
        else:
            best_id, best_score, best_depth, best_children = "none", 0, 0, 0

        parts = [
            f"- {root_children} approaches tried from root (breadth), deepest branch is depth {max_depth}",
            f"- Best node: {best_id} (score={best_score:.4f}, depth={best_depth}, {best_children} children)",
        ]

        # Axis distribution
        all_axes = ["architecture", "loss", "data_representation", "data_augmentation", "regularization", "optimizer", "hp", "combination"]
        axis_stats: dict[str, dict] = {a: {"count": 0, "best": None, "fails": 0} for a in all_axes}
        total_experiments = 0
        for nid, node in self.nodes.items():
            if nid == "root":
                continue
            total_experiments += 1
            ax = getattr(node, "axis", None) or self._categorize_strategy(node.strategy)
            if ax not in axis_stats:
                ax = "hp"
            axis_stats[ax]["count"] += 1
            if node.score is None:
                axis_stats[ax]["fails"] += 1
            elif axis_stats[ax]["best"] is None:
                axis_stats[ax]["best"] = node.score
            elif self.task.higher_is_better and node.score > axis_stats[ax]["best"]:
                axis_stats[ax]["best"] = node.score
            elif not self.task.higher_is_better and node.score < axis_stats[ax]["best"]:
                axis_stats[ax]["best"] = node.score

        if total_experiments > 0:
            parts.append(f"\n## Exploration Summary\nTotal experiments: {total_experiments}")
            for ax in all_axes:
                s = axis_stats[ax]
                pct = 100 * s["count"] / total_experiments if total_experiments else 0
                best_str = f"best={s['best']:.4f}" if s["best"] is not None else "best=N/A"
                parts.append(f"  - {ax}: {s['count']} attempts ({pct:.0f}%), {best_str}, fails={s['fails']}")

        # Score improvement along best path
        if best_id != "root" and best_id in self.nodes:
            path_scores = []
            cur = best_id
            while cur and cur in self.nodes:
                n = self.nodes[cur]
                if n.score is not None:
                    path_scores.append((cur, n.score))
                cur = n.parent_id
            path_scores.reverse()
            if len(path_scores) >= 2:
                parts.append(
                    f"- Best path scores: {' → '.join(f'{s:.4f}' for _, s in path_scores[-5:])}"
                )

        return "\n   ".join(parts)

    _TREE_VIEW_MAX_NODES = 25  # show at most this many nodes to the scientist

    def _build_tree_view(self) -> str:
        """Build a compact structured view of nodes for the scientist.

        When the tree exceeds _TREE_VIEW_MAX_NODES, only the top-K scoring
        nodes (plus root and their ancestors) are shown. This prevents the
        scientist's context window from being overwhelmed in long runs.
        """
        # Decide which nodes to show
        show_all = len(self.nodes) <= self._TREE_VIEW_MAX_NODES
        if not show_all:
            visible = self._pick_visible_nodes(self._TREE_VIEW_MAX_NODES)
        else:
            visible = set(self.nodes.keys())

        lines = []

        def _node_line(nid: str, indent: int = 0):
            if nid not in visible:
                # Count hidden subtree
                hidden = self._count_subtree(nid, visible)
                if hidden > 0:
                    prefix = "  " * indent
                    lines.append(f"{prefix}... ({hidden} lower-scoring nodes omitted)")
                return

            n = self.nodes[nid]
            prefix = "  " * indent

            # Score
            if n.score is not None:
                score_str = f"{n.score:.4f}"
            else:
                score_str = "FAILED"

            # Strategy summary
            strategy = n.strategy[:120] if n.strategy else "N/A"

            # Environment feedback — give the scientist a clear status label
            error_str = ""
            if n.node_id != "root":
                status = n.execution_status or ""
                err_t = n.error_type or ""

                if status == "success":
                    error_str = f"\n{prefix}  STATUS: success"
                elif status == "training_failed":
                    fallback_note = ""
                    if self.container.baseline_score is not None and n.score is not None:
                        if abs(n.score - self.container.baseline_score) < 0.02:
                            fallback_note = " → score is likely baseline fallback, NOT real training result"
                    error_str = (
                        f"\n{prefix}  STATUS: training_failed"
                        + (f" ({err_t})" if err_t else "")
                        + fallback_note
                    )
                elif status in ("no_validate_called", "no_submission_produced", "env_error"):
                    error_str = (
                        f"\n{prefix}  STATUS: {status}"
                        + (f" ({err_t})" if err_t else "")
                    )
                elif n.score is None and n.actions:
                    # Fallback for old nodes without execution_status
                    error_actions = [
                        a for a in n.actions
                        if "Traceback" in a.get("observation", "")
                    ]
                    if error_actions:
                        last_err = error_actions[-1]["observation"]
                        err_lines = last_err.strip().split("\n")
                        err_msg = err_lines[-1][:120] if err_lines else last_err[:120]
                        error_str = (
                            f"\n{prefix}  STATUS: training_failed"
                            f" — Last error: {err_msg}"
                        )
                    elif len(n.actions) >= self.max_actions:
                        error_str = f"\n{prefix}  STATUS: no_submission_produced (ran out of {len(n.actions)} actions)"
                    else:
                        error_str = f"\n{prefix}  STATUS: failed — {n.error[:120]}" if n.error else ""
                elif n.error:
                    error_str = f"\n{prefix}  STATUS: env_error — {n.error[:120]}"

            # Parent comparison
            parent_note = ""
            if n.parent_id and n.parent_id in self.nodes and n.score is not None:
                parent = self.nodes[n.parent_id]
                if parent.score is not None:
                    diff = n.score - parent.score
                    if self.task.higher_is_better:
                        parent_note = f" ({'better' if diff > 0 else 'worse'} than parent by {abs(diff):.4f})"
                    else:
                        parent_note = f" ({'better' if diff < 0 else 'worse'} than parent by {abs(diff):.4f})"

            lines.append(
                f"{prefix}Node {nid} [{strategy}]\n"
                f"{prefix}  Score: {score_str} | Actions: {len(n.actions)} | "
                f"Children: {len(n.children)}{parent_note}{error_str}"
            )
            # Include LLM-generated execution summary so the scientist can see
            # what actually happened without explicitly inspecting the node.
            summary = getattr(n, "log_summary", "") or ""
            if summary and nid != "root":
                indented = "\n".join(f"{prefix}  | {ln}" for ln in summary.splitlines()[:10])
                lines.append(f"{prefix}  Summary:\n{indented}")

            for cid in n.children:
                if cid in self.nodes:
                    _node_line(cid, indent + 1)

        if not show_all:
            n_hidden = len(self.nodes) - len(visible)
            lines.append(
                f"[Showing {len(visible)}/{len(self.nodes)} nodes — "
                f"{n_hidden} lower-scoring nodes omitted for brevity]"
            )
            lines.append("")

        if "root" in self.nodes:
            _node_line("root")
        else:
            lines.append("(empty tree)")

        return "\n".join(lines)

    def _pick_visible_nodes(self, max_nodes: int) -> set[str]:
        """Pick the most relevant nodes to show the scientist."""
        # Always show root
        visible = {"root"}

        # Rank all scored nodes
        scored = [
            (nid, n.score) for nid, n in self.nodes.items()
            if n.score is not None and nid != "root"
        ]
        if self.task.higher_is_better:
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored.sort(key=lambda x: x[1])

        # Take top-K by score
        for nid, _ in scored[:max_nodes - 1]:
            visible.add(nid)

        # Add ancestors of visible nodes so the tree structure makes sense
        for nid in list(visible):
            cur = self.nodes[nid].parent_id
            while cur and cur in self.nodes:
                visible.add(cur)
                cur = self.nodes[cur].parent_id

        # Also include recently added nodes (last 5) even if low-scoring
        # so the scientist sees what just happened
        all_ids = list(self.nodes.keys())
        for nid in all_ids[-5:]:
            visible.add(nid)
            # Add their ancestors too
            cur = self.nodes[nid].parent_id
            while cur and cur in self.nodes:
                visible.add(cur)
                cur = self.nodes[cur].parent_id

        return visible

    def _count_subtree(self, nid: str, visible: set[str]) -> int:
        """Count hidden nodes in subtree rooted at nid."""
        count = 0 if nid in visible else 1
        if nid in self.nodes:
            for cid in self.nodes[nid].children:
                if cid in self.nodes:
                    count += self._count_subtree(cid, visible)
        return count

    # ------------------------------------------------------------------
    # Scientist decision
    # ------------------------------------------------------------------

    def _format_node_code(self, node_id: str) -> str:
        """Format a node's executor actions for the scientist to inspect."""
        if node_id not in self.nodes:
            return f"Node {node_id} not found."

        node = self.nodes[node_id]
        if not node.actions:
            return f"Node {node_id}: No actions (baseline node)."

        status_label = node.execution_status or "unknown"
        if node.error_type:
            status_label += f":{node.error_type}"
        lines = [f"=== Node {node_id} (score: {node.score}, status: {status_label}) ==="]
        for i, action in enumerate(node.actions):
            cmd = action.get("action", "")
            obs = action.get("observation", "")
            # Truncate very long observations (e.g. training logs)
            if len(obs) > 500:
                obs = obs[:500] + "\n... (truncated)"
            lines.append(f"--- Action {i} ---")
            lines.append(f"$ {cmd}")
            lines.append(obs)

        return "\n".join(lines)

    def _parse_inspect_response(self, text: str) -> list[str]:
        """Parse the INSPECT: line from turn 1 to get node IDs."""
        match = re.search(r"INSPECT:\s*(.+)", text)
        if not match:
            return []
        raw = match.group(1).strip()
        if raw.upper() == "NONE":
            return []
        # Split by comma and clean up
        node_ids = [nid.strip() for nid in raw.split(",") if nid.strip()]
        # Validate they exist and cap at 3
        valid = [nid for nid in node_ids if nid in self.nodes]
        return valid[:3]

    def _parse_read_response(self, text: str) -> list[str]:
        """Parse the READ: line from turn 1 to get filenames."""
        match = re.search(r"READ:\s*(.+)", text)
        if not match:
            return []
        raw = match.group(1).strip()
        if raw.upper() == "NONE":
            return []
        filenames = [f.strip() for f in raw.split(",") if f.strip()]
        return filenames[:3]  # cap at 3 files

    def _read_workspace_files_for_scientist(self, filenames: list[str]) -> list[str]:
        """Read file contents from the workspace for the scientist's READ request."""
        parts = []
        try:
            workspace = Path(self.container.env.container_obj.workspace_host_dir)
        except Exception:
            return ["(Could not access workspace)"]
        for fname in filenames:
            fpath = workspace / fname
            if fpath.exists() and fpath.is_file():
                content = fpath.read_text()
                if len(content) > 5000:
                    content = content[:2500] + "\n... (truncated) ...\n" + content[-2500:]
                parts.append(f"--- {fname} ---\n{content}")
            else:
                parts.append(f"--- {fname} --- (file not found)")
        return parts

    def _detect_regression_preamble(self) -> str:
        """Look at the most recent non-root child nodes; if any scored >20% below
        its parent's score, emit a preamble enforcing CLASSIFICATION in turn 2."""
        # Gather recently-scored non-root nodes ordered by insertion
        recent = [n for n in self.nodes.values()
                  if n.parent_id and n.score is not None and n.parent_id in self.nodes]
        if not recent:
            return ""
        # Consider the last 3 nodes added (latest last)
        recent = recent[-3:]
        offenders = []
        higher_is_better = self.task.higher_is_better
        for node in recent:
            parent = self.nodes.get(node.parent_id)
            if not parent or parent.score is None:
                continue
            if higher_is_better and parent.score > 0:
                drop_pct = 100.0 * (parent.score - node.score) / parent.score
                if drop_pct > 20.0:
                    offenders.append((node.node_id, parent.node_id,
                                      parent.score, node.score, drop_pct))
            elif not higher_is_better and parent.score > 0:
                rise_pct = 100.0 * (node.score - parent.score) / parent.score
                if rise_pct > 20.0:
                    offenders.append((node.node_id, parent.node_id,
                                      parent.score, node.score, rise_pct))
        if not offenders:
            return ""
        lines = ["=== REGRESSION DETECTED — CLASSIFICATION MANDATORY ==="]
        for (nid, pid, ps, ns, pct) in offenders:
            lines.append(
                f"Node {nid} (score {ns:.4f}) regressed {pct:.0f}% below its "
                f"parent {pid} (score {ps:.4f}). You MUST (a) have INSPECTed "
                f"{nid} in turn 1 so you can see the executor's actual code, "
                f"and (b) emit a CLASSIFICATION line that is NOT 'NONE'. If "
                f"the bug is visible in the inspected code (broken init, "
                f"missing validation, mismatched shapes, dropped step, etc.), "
                f"use IMPLEMENTATION-WRONG and propose a retry on the SAME "
                f"axis with corrected executor instructions. Only use "
                f"IDEA-WRONG if you can affirmatively argue from the code that "
                f"the approach itself cannot work here."
            )
        return "\n".join(lines)

    def _detect_stagnation_preamble(self, window: int = 3) -> str:
        """If the global best score has not improved over the last `window`
        scored expansions, emit a preamble nudging the scientist to try a
        cross-branch MERGE rather than another single-parent extension."""
        scored = [n for n in self.nodes.values()
                  if n.parent_id and n.score is not None]
        if len(scored) < window + 1:
            return ""
        higher = self.task.higher_is_better
        # Reconstruct best-so-far trajectory in expansion order.
        best_seq = []
        cur_best = None
        for n in scored:
            s = n.score
            cur_best = s if cur_best is None else (max(cur_best, s) if higher else min(cur_best, s))
            best_seq.append(cur_best)
        # Stagnant if best hasn't moved across the most recent (window+1) entries.
        recent = best_seq[-(window + 1):]
        if len(recent) < window + 1:
            return ""
        if higher:
            improved = max(recent) > recent[0] + 1e-9
        else:
            improved = min(recent) < recent[0] - 1e-9
        if improved:
            return ""
        # Identify branch leaders for cross-branch context (top scored node per
        # immediate root child).
        branch_top = {}
        for n in scored:
            cur = n
            while cur.parent_id and cur.parent_id in self.nodes:
                p = self.nodes[cur.parent_id]
                if p.parent_id is None:
                    break
                cur = p
            br = cur.node_id  # first-level branch root
            prev = branch_top.get(br)
            if prev is None or (higher and n.score > prev.score) or (not higher and n.score < prev.score):
                branch_top[br] = n
        if len(branch_top) < 2:
            return (
                "=== STAGNATION DETECTED — best score has not improved in "
                f"{window} consecutive scored expansions. Consider opening a NEW "
                "branch (PARENT: root, MODE: explore) on an unexplored axis "
                "rather than another tweak on the same parent. Keep this in "
                "mind when picking PARENT and AXIS below."
            )
        # 2+ branches — encourage MERGE.
        leaders_str = ", ".join(
            f"{nid}={n.score:.4f}" for nid, n in
            sorted(branch_top.items(), key=lambda kv: kv[1].score, reverse=higher)[:3]
        )
        return (
            f"=== STAGNATION DETECTED — best score has not improved in {window} "
            f"consecutive scored expansions. Branch leaders: {leaders_str}. "
            "Strongly consider a cross-branch MERGE this turn: pick PARENT "
            "= the strongest branch leader and list other strong leaders in "
            "COMBINES, with an EXPERIMENT that explicitly fuses what each "
            "branch did well. A merge is more likely to break the plateau "
            "than another single-parent IMPROVE."
        )

    def _scientist_research_step(self, rstep: int) -> None:
        """Research-phase step: scientist reads / inspects and saves findings to
        memory, but does NOT propose an experiment. Used to build a mental model
        of the task before the active search loop begins."""
        tree_view = self._build_tree_view()
        if self.memory:
            memory_section = "\n".join(f"- {m}" for m in self.memory)
        else:
            memory_section = "(No accumulated knowledge yet — you are in research step 1.)"

        task_desc = f"{self.task.name}"
        if self.task.task_type:
            task_desc += f" ({self.task.task_type})"

        from arts.tree_search import generate_code_outline
        task_details = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head="(data preview omitted)",
            code_outline=generate_code_outline(getattr(self.task, "starter_code_host_path", "")),
        )
        task_details = (
            f"EXECUTOR SYSTEM PROMPT:\n{self.task.system_prompt}\n\n"
            f"TASK DESCRIPTION:\n{task_details}"
        )

        turn1_prompt = SCIENTIST_PROMPT_TURN1.format(
            task_description=task_desc,
            task_details=task_details,
            metric_name=self.task.primary_metric,
            direction="higher" if self.task.higher_is_better else "lower",
            baseline_score=f"{self.container.baseline_score:.4f}",
            max_actions=self.max_actions,
            budget_left=self.node_budget,
            total_budget=self.node_budget,
            tree_view=tree_view,
            memory_section=memory_section,
            **self._budget_ctx(),
        )
        messages = [{"role": "user", "content": turn1_prompt}]

        try:
            turn1_resp = self.scientist.chat(messages, temperature=0.3)
        except Exception as e:
            print(f"  WARNING: Research turn 1 failed: {e}")
            turn1_resp = "INSPECT: NONE\n\nREAD: baseline.py, evaluate.py"

        inspect_ids = self._parse_inspect_response(turn1_resp)
        read_files = self._parse_read_response(turn1_resp)
        # In research step 1, if the scientist didn't ask to read anything,
        # default to reading the baseline and evaluate code so it sees the task.
        if rstep == 0 and not read_files:
            read_files = ["baseline.py", "evaluate.py"]

        parts = []
        if inspect_ids:
            print(f"  Research inspecting: {inspect_ids}")
            code_parts = [self._format_node_code(nid) for nid in inspect_ids]
            parts.append(
                "Here is the code and output from the nodes you requested:\n\n"
                + "\n\n".join(code_parts)
            )
        if read_files:
            print(f"  Research reading files: {read_files}")
            file_parts = self._read_workspace_files_for_scientist(read_files)
            if file_parts:
                parts.append(
                    "Here are the file contents you requested:\n\n"
                    + "\n\n".join(file_parts)
                )
        code_inspection = "\n\n".join(parts) if parts else "(No nodes inspected, no files read.)"

        turn2_prompt = SCIENTIST_PROMPT_RESEARCH_TURN2.format(code_inspection=code_inspection)
        messages.append({"role": "assistant", "content": turn1_resp})
        messages.append({"role": "user", "content": turn2_prompt})

        try:
            turn2_resp = self.scientist.chat(messages, temperature=0.3)
        except Exception as e:
            print(f"  WARNING: Research turn 2 failed: {e}")
            return

        # Print a snippet of findings so the user can see what was learned
        snippet = turn2_resp[:500].replace("\n", "\n  ")
        print(f"  RESEARCH FINDINGS:\n  {snippet}...")

        # Extract and save memory entries. Format: MEMORY: followed by one or
        # more lines (may be dash-bulleted), continuing to end of response.
        mem_match = re.search(
            r"MEMORY:\s*\n?(.*)\Z",
            turn2_resp, re.DOTALL,
        )
        if mem_match:
            mem_block = mem_match.group(1).strip()
            if mem_block and mem_block.upper() != "NONE":
                new_entries = []
                for line in mem_block.splitlines():
                    line = line.strip().lstrip("-*•").strip()
                    if line and len(line) > 15:
                        new_entries.append(line)
                # De-duplicate against existing memory (substring match)
                for entry in new_entries:
                    if not any(entry[:60] in m for m in self.memory):
                        self.memory.append(entry)
                        print(f"  Research memory +: {entry[:100]}")

        # Persist the research turn to disk for post-hoc inspection
        self.output_dir.mkdir(parents=True, exist_ok=True)
        research_dir = self.output_dir / "research_logs"
        research_dir.mkdir(exist_ok=True)
        try:
            with open(research_dir / f"research_{rstep:03d}.json", "w") as f:
                json.dump({
                    "step": rstep,
                    "inspected": inspect_ids,
                    "read_files": read_files,
                    "turn1_response": turn1_resp,
                    "turn2_response": turn2_resp,
                    "memory_after": list(self.memory),
                }, f, indent=2)
        except Exception:
            pass

    def _scientist_decide(self, budget_left: int) -> ScientistDecision:
        """Call the scientist LLM in two turns: inspect, then decide."""
        tree_view = self._build_tree_view()

        # Memory section
        if self.memory:
            memory_section = "\n".join(f"- {m}" for m in self.memory)
        else:
            memory_section = "(No accumulated knowledge yet — this is the first scientist step.)"

        # Task description
        task_desc = f"{self.task.name}"
        if self.task.task_type:
            task_desc += f" ({self.task.task_type})"

        # Build task details from what the executor sees
        from arts.tree_search import generate_code_outline
        task_details = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head="(data preview omitted)",
            code_outline=generate_code_outline(getattr(self.task, "starter_code_host_path", "")),
        )
        task_details = (
            f"EXECUTOR SYSTEM PROMPT:\n{self.task.system_prompt}\n\n"
            f"TASK DESCRIPTION:\n{task_details}"
        )

        # --- Turn 1: Show tree, ask what to inspect ---
        turn1_prompt = SCIENTIST_PROMPT_TURN1.format(
            task_description=task_desc,
            task_details=task_details,
            metric_name=self.task.primary_metric,
            direction="higher" if self.task.higher_is_better else "lower",
            baseline_score=f"{self.container.baseline_score:.4f}",
            max_actions=self.max_actions,
            budget_left=budget_left,
            total_budget=self.node_budget,
            tree_view=tree_view,
            memory_section=memory_section,
            **self._budget_ctx(),
        )
        if self.no_incremental_rules:
            # Strip the cost-aware bundling block in TURN1 (the "DEFAULT to one
            # focused edit / LOW cost / HIGH cost / COUPLED / kitchen sink"
            # paragraph) so the scientist is not pushed toward incremental
            # default. Match from "The bundling decision" through the kitchen-
            # sink paragraph end.
            import re as _re_t1
            turn1_prompt = _re_t1.sub(
                r"The bundling decision \(one focused edit vs.*?destroys signal without reason\.",
                "Propose whichever experiment you think is best for the task — bundle aggressively if that is what wins, or keep edits focused if that gives cleaner signal.",
                turn1_prompt,
                flags=_re_t1.DOTALL,
            )

        messages = [{"role": "user", "content": turn1_prompt}]

        try:
            turn1_resp = self.scientist.chat(messages, temperature=0.3)
        except Exception as e:
            print(f"  WARNING: Scientist turn 1 failed: {e}")
            turn1_resp = "INSPECT: NONE"

        # Parse which nodes to inspect
        inspect_ids = self._parse_inspect_response(turn1_resp)

        # Parse which files to read
        read_files = self._parse_read_response(turn1_resp)

        # --- Build code inspection content ---
        parts = []
        if inspect_ids:
            print(f"  Scientist inspecting: {inspect_ids}")
            code_parts = [self._format_node_code(nid) for nid in inspect_ids]
            parts.append(
                "Here is the code and output from the nodes you requested:\n\n"
                + "\n\n".join(code_parts)
            )

        if read_files:
            print(f"  Scientist reading files: {read_files}")
            file_parts = self._read_workspace_files_for_scientist(read_files)
            if file_parts:
                parts.append(
                    "Here are the file contents you requested:\n\n"
                    + "\n\n".join(file_parts)
                )

        code_inspection = "\n\n".join(parts) if parts else "(No nodes inspected, no files read.)"

        # --- Turn 2: Make decision with code context ---
        explore_stats = self._build_explore_stats()
        turn2_prompt = SCIENTIST_PROMPT_TURN2.format(
            code_inspection=code_inspection,
            budget_left=budget_left,
            max_actions=self.max_actions,
            explore_stats=explore_stats,
            **self._budget_ctx(),
            **self._vs_ctx(n_candidates=3),
        )
        import re as _re
        if self.no_build_on_signal or self.no_incremental_rules:
            # Strip the BUILD ON SIGNAL bullet (ablation: free scientist to
            # propose compound strategies from step 0).
            turn2_prompt = _re.sub(
                r"\n- BUILD ON SIGNAL.*?design on assumption\.",
                "",
                turn2_prompt,
                flags=_re.DOTALL,
            )
        if self.no_incremental_rules:
            # Stronger ablation: also strip the EVOLVE bullet so the scientist
            # is not pushed toward "layer ONE new idea on top" defaults.
            turn2_prompt = _re.sub(
                r"\n- EVOLVE:.*?counts as a new strategy\.",
                "",
                turn2_prompt,
                flags=_re.DOTALL,
            )
        if self.complexity_cycle and self._active_step_counter < 3:
            # AIRA-dojo-style complexity hint, cycled by active expansion step.
            tier = ["simple", "normal", "complex"][self._active_step_counter]
            tier_text = {
                "simple": "FOR THIS EXPANSION — propose a SIMPLE IDEA: a battle-tested baseline that establishes a working solution. Consider widely-used, well-understood methods that are known to work for tasks of this kind.",
                "normal": "FOR THIS EXPANSION — propose a MORE COMPLEX IDEA: one that beats the simple baseline at the cost of some complexity. Consider stronger pretrained models, specialized feature engineering, or basic ensembling and/or hyperparameter optimization.",
                "complex": "FOR THIS EXPANSION — propose an ADVANCED IDEA: consider specialized SOTA-class models, advanced feature engineering or data augmentation, advanced ensembling/stacking, fine-tuning on specialized backbones. This is your chance to escape weak baselines.",
            }[tier]
            turn2_prompt = f"COMPLEXITY TIER (cycled simple->normal->complex over first 3 expansions): {tier}\n{tier_text}\n\n" + turn2_prompt

        # Detect a recent >20% regression vs parent and, if present, prepend a
        # mandatory preamble forcing the scientist to classify it and (if
        # implementation-wrong) retry on the same axis with corrected advice.
        regression_preamble = self._detect_regression_preamble()
        if regression_preamble:
            turn2_prompt = regression_preamble + "\n\n" + turn2_prompt

        # Stagnation nudge — pushes scientist toward MERGE when best hasn't
        # improved in 3 consecutive scored expansions.
        stagnation_preamble = self._detect_stagnation_preamble(window=3)
        if stagnation_preamble:
            turn2_prompt = stagnation_preamble + "\n\n" + turn2_prompt

        messages.append({"role": "assistant", "content": turn1_resp})
        messages.append({"role": "user", "content": turn2_prompt})

        sampling_record: dict = {}
        try:
            turn2_resp_raw = self.scientist.chat(messages, temperature=0.3)
            turn2_resp, sampling_record = self._external_sample(turn2_resp_raw)
            if sampling_record:
                print(
                    f"  [VS] {sampling_record['n_candidates']} candidates, "
                    f"sampled #{sampling_record['sampled_index'] + 1} "
                    f"(P_raw={sampling_record['raw_probabilities']}, "
                    f"weights={[f'{w:.2f}' for w in sampling_record['weights']]}, "
                    f"cap={sampling_record['cap']}): "
                    f"{sampling_record['sampled_direction'][:100]}"
                )
            decision = self._parse_scientist_response(turn2_resp)
        except Exception as e:
            print(f"  WARNING: Scientist turn 2 failed: {e}")
            decision = ScientistDecision(
                action="draft_from_root", node_id="root",
                direction="Try a robust sklearn pipeline with cross-validation",
                mode="explore", memory_update="",
                reasoning=f"Fallback due to scientist error: {e}",
            )

        # Log scientist reasoning
        print(f"\n  SCIENTIST REASONING:\n  {decision.reasoning[:300]}")
        if inspect_ids:
            print(f"  Inspected nodes: {inspect_ids}")
        self._save_scientist_log(
            decision, budget_left, tree_view,
            inspected_nodes=inspect_ids, turn1_response=turn1_resp,
            sampling_record=sampling_record,
        )

        return decision

    def _summarize_execution(self, actions: list[dict], score: float | None, strategy: str) -> str:
        """Use a cheap LLM call to extract key signals from execution logs.

        Returns a compact multi-line summary the scientist can read without
        asking for full log inspection.
        """
        if not actions:
            return "(no actions)"

        # Build a compact dump of actions and observations
        log_lines = []
        for i, a in enumerate(actions):
            cmd = a.get("action", "")[:300]
            obs = a.get("observation", "")
            # Keep error-heavy observations fuller, truncate normal ones
            if "error" in obs.lower()[:300] or "traceback" in obs.lower()[:300]:
                obs_trim = obs[:800]
            else:
                obs_trim = obs[:400]
            log_lines.append(f"[{i}] $ {cmd}\n{obs_trim}")
        full_log = "\n---\n".join(log_lines)
        # Cap total length
        if len(full_log) > 20000:
            full_log = full_log[:10000] + "\n...[truncated middle]...\n" + full_log[-10000:]

        score_str = f"{score:.4f}" if score is not None else "FAILED"
        prompt = f"""Summarize this ML experiment run for the next attempt.

Strategy tried: {strategy[:300]}
Final validation score: {score_str}

OUTPUT FORMAT:
1. A 4-7 bullet point summary covering:
   - Did training converge? (final loss, loss trajectory)
   - Model architecture / size actually used
   - Hyperparameters (epochs, batch size, lr)
   - Train vs validation gap if visible
   - Red flags (overfitting, underfitting, NaN, OOM, nothing learned)

2. If ANY error, crash, timeout, OOM, or fatal warning occurred during the run,
   append a dedicated ERROR section in this exact format (one per distinct error):

   ERROR:
     type: <exception class or "Timeout"/"OOM"/"Hang">
     message: "<exact quoted error message, not paraphrased>"
     location: <file:line from the traceback if visible, else the action index>
     cause: <one-sentence root-cause analysis based on surrounding code/context>
     fix: <specific, code-level instruction the NEXT node should apply — e.g., "replace torch.cuda.amp.GradScaler(...) with torch.amp.GradScaler('cuda', ...)"; "add print('...', flush=True) before DataLoader init"; "load slices on-the-fly in __getitem__ instead of preloading full volume in __init__">

   If multiple errors, list each separately. Be specific — a vague "fix imports" is
   useless; "add `from torch.cuda.amp import autocast` at line 7" is actionable.

3. If no error: just the bullet list.

Execution log:
{full_log}

Output ONLY the bullet list (+ optional ERROR section). No preamble."""

        try:
            summary = self.executor.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return summary.strip()[:2500]  # hard cap (raised from 1500 to fit ERROR section)
        except Exception as e:
            return f"(summary failed: {type(e).__name__})"

    def _parse_scientist_response(self, text: str) -> ScientistDecision:
        """Parse the scientist's structured output.

        Supports two formats:
        - New: ANALYSIS / HYPOTHESIS / EXPERIMENT / PARENT / COMBINES / MODE / MEMORY
        - Legacy: REASONING / STRATEGIES / CHOSEN / DIRECTION / EXECUTOR_GUIDANCE / MODE / MEMORY
        """
        # --- Extract reasoning (ANALYSIS or REASONING) ---
        reasoning = ""
        for header in ["ANALYSIS:", "REASONING:"]:
            m = re.search(
                rf"{header}\s*\n(.*?)(?=\nHYPOTHESIS:|\nSTRATEGIES:|\nEXPERIMENT:|\nACTION:|$)",
                text, re.DOTALL,
            )
            if m:
                reasoning = m.group(1).strip()
                break
        if not reasoning:
            reasoning = text[:200]

        # --- Extract hypothesis (new format only) ---
        hyp_match = re.search(
            r"HYPOTHESIS:\s*\n(.*?)(?=\nEXPERIMENT:|$)", text, re.DOTALL
        )
        hypothesis = hyp_match.group(1).strip() if hyp_match else ""

        # --- Extract direction (EXPERIMENT or DIRECTION) ---
        direction = ""
        for header, stop in [
            ("EXPERIMENT:", r"\nPARENT:|\nCOMBINES:|\nMODE:"),
            ("DIRECTION:", r"\nEXECUTOR_GUIDANCE:|\nMODE:"),
        ]:
            m = re.search(rf"{header}\s*\n(.*?)(?={stop})", text, re.DOTALL)
            if m:
                direction = m.group(1).strip()
                break

        # --- Extract parent node ---
        action = "draft_from_root"
        node_id = "root"

        # New format: standalone PARENT: line
        parent_match = re.search(r"PARENT:\s*[\"']?(\S+?)[\"']?\s*$", text, re.MULTILINE)
        if parent_match:
            node_id = parent_match.group(1).strip().rstrip("—-").strip()
            action = "expand" if node_id != "root" else "draft_from_root"
        else:
            # Legacy: STRATEGIES + CHOSEN
            strat_lines = re.findall(
                r"(\d+)\.\s.*?(?:→|->)\s*PARENT:\s*[\"']?(\S+?)[\"']?\s*(?:—|-|$)",
                text,
            )
            chosen_match = re.search(r"CHOSEN:\s*(\d+)", text)
            if strat_lines and chosen_match:
                chosen_num = int(chosen_match.group(1))
                for num_str, parent in strat_lines:
                    if int(num_str) == chosen_num:
                        node_id = parent.strip().rstrip("—-").strip()
                        break
                action = "expand" if node_id != "root" else "draft_from_root"
            else:
                # Fallback: old ACTION format
                action_match = re.search(r"ACTION:\s*(expand\s+(\S+)|draft_from_root)", text)
                if action_match:
                    full = action_match.group(1).strip()
                    if full.startswith("expand"):
                        action = "expand"
                        node_id = action_match.group(2) or "root"

        # --- Extract COMBINES (new format, informational) ---
        combines_match = re.search(r"COMBINES:\s*(.*?)$", text, re.MULTILINE)
        combines = ""
        if combines_match:
            c = combines_match.group(1).strip()
            if c.upper() != "NONE":
                combines = c

        # --- Extract EXECUTOR_GUIDANCE (legacy format) ---
        guidance_match = re.search(
            r"EXECUTOR_GUIDANCE:\s*\n(.*?)(?=\nMODE:)", text, re.DOTALL
        )
        executor_guidance = ""
        if guidance_match:
            g = guidance_match.group(1).strip()
            if g.upper() != "NONE":
                executor_guidance = g

        # --- Extract AXIS ---
        axis = ""
        axis_match = re.search(r"AXIS:\s*(architecture|loss|data|regularization|optimizer|hp|combination)", text, re.IGNORECASE)
        if axis_match:
            axis = axis_match.group(1).lower()
        else:
            axis = self._categorize_strategy(direction)

        # --- Extract MODE ---
        mode = "explore"
        mode_match = re.search(r"MODE:\s*(explore|exploit)", text, re.IGNORECASE)
        if mode_match:
            mode = mode_match.group(1).lower()

        # --- Extract MEMORY (stop at CLASSIFICATION to avoid swallowing it) ---
        memory_update = ""
        memory_match = re.search(
            r"MEMORY:\s*\n?(.*?)(?=\nCLASSIFICATION:|$)", text, re.DOTALL
        )
        if memory_match:
            memory_update = memory_match.group(1).strip()

        # --- Extract CLASSIFICATION (regression investigation — logged) ---
        classification = ""
        cls_match = re.search(r"CLASSIFICATION:\s*\n?(.*?)$", text, re.DOTALL)
        if cls_match:
            classification = cls_match.group(1).strip()
            if classification and classification.upper() != "NONE":
                print(f"  CLASSIFICATION: {classification[:200]}")

        # Append hypothesis to direction if present (gives executor the "why")
        if hypothesis and direction:
            direction = f"{direction}\n\nHypothesis: {hypothesis}"
        if combines:
            direction = f"{direction}\n\nCombines ideas from: {combines}"

        return ScientistDecision(
            action=action, node_id=node_id,
            direction=direction, mode=mode,
            memory_update=memory_update,
            reasoning=reasoning,
            executor_guidance=executor_guidance,
            axis=axis,
        )

    @staticmethod
    def _categorize_strategy(strategy: str) -> str:
        """Keyword-based fallback to categorize a strategy into an axis."""
        s = strategy.lower()
        arch_kw = ["resnet", "vgg", "transformer", "cnn", "lstm", "gru", "unet",
                    "u-net", "efficientnet", "convnext", "densenet", "mobilenet",
                    "backbone", "encoder", "decoder", "attention", "head", "layer",
                    "architecture", "model", "network", "deeplabv3"]
        loss_kw = ["loss", "focal", "label smooth", "contrastive", "cross entropy",
                    "bce", "dice", "criterion"]
        data_aug_kw = ["augment", "flip", "rotate", "mixup", "cutout", "cutmix",
                    "randaugment", "autoaugment", "jitter", "elastic", "randaffine",
                    "colorjitter", "distortion", "noise", "blur"]
        data_rep_kw = ["channel", "slice", "z-slice", "z_slice", "zrange", "z range",
                    "input representation", "patch size", "resolution", "crop size",
                    "normalize", "preprocessing", "fragment", "sampling",
                    "positive patch", "data loader", "dataset"]
        reg_kw = ["dropout", "weight decay", "batch norm", "regulariz", "ema",
                   "swa", "stochastic weight", "early stop"]
        opt_kw = ["optimizer", "adam", "sgd", "cosine anneal", "onecycle",
                   "learning rate schedule", "warmup", "lr schedule"]
        hp_kw = ["epoch", "batch size", "learning rate", "lr=", "threshold",
                  "cutoff", "grid search", "hyperparameter"]

        scores = {
            "architecture": sum(1 for k in arch_kw if k in s),
            "loss": sum(1 for k in loss_kw if k in s),
            "data_augmentation": sum(1 for k in data_aug_kw if k in s),
            "data_representation": sum(1 for k in data_rep_kw if k in s),
            "regularization": sum(1 for k in reg_kw if k in s),
            "optimizer": sum(1 for k in opt_kw if k in s),
            "hp": sum(1 for k in hp_kw if k in s),
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "hp"
        if sum(1 for v in scores.values() if v > 0) >= 3:
            return "combination"
        return best

    def _save_scientist_log(self, decision: ScientistDecision, budget_left: int,
                            tree_view: str, inspected_nodes: list[str] | None = None,
                            turn1_response: str = "",
                            sampling_record: dict | None = None):
        """Save scientist decision to a log file for post-hoc analysis."""
        log_dir = self.output_dir / "scientist_logs"
        log_dir.mkdir(exist_ok=True)
        step_num = self.node_budget - budget_left
        log_file = log_dir / f"step_{step_num:03d}.json"
        data = {
            "step": step_num,
            "budget_left": budget_left,
            "action": decision.action,
            "node_id": decision.node_id,
            "direction": decision.direction,
            "executor_guidance": decision.executor_guidance,
            "mode": decision.mode,
            "memory_update": decision.memory_update,
            "reasoning": decision.reasoning,
            "tree_view": tree_view,
            "memory_state": list(self.memory),
            "inspected_nodes": inspected_nodes or [],
            "turn1_response": turn1_response,
            "sampling_record": sampling_record or {},
        }
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Expand a single node
    # ------------------------------------------------------------------

    def _expand_one(self, parent_id: str, mode: str,
                    direction: str = "",
                    executor_guidance: str = "",
                    axis: str = "hp") -> TreeNode | None:
        """Create and execute a single child from the given parent."""
        parent = self.nodes[parent_id]

        # Assign child index
        if parent_id not in self._child_counter:
            self._child_counter[parent_id] = 0
        child_idx = self._child_counter[parent_id]
        self._child_counter[parent_id] += 1
        child_id = f"{parent_id}_{child_idx}"

        # Strategy comes directly from scientist — no separate VS
        strategy_text = direction or f"Attempt {child_idx}"

        print(f"  [{child_id}] mode={mode}, strategy: {strategy_text[:80]}")

        # Restore parent workspace (fall back to root if no snapshot — e.g. resumed nodes)
        snap = parent.snapshot_path
        if not snap and "root" in self.nodes and self.nodes["root"].snapshot_path:
            snap = self.nodes["root"].snapshot_path
        if snap:
            self.container.restore_snapshot(snap)
        if self.task.submission_file:
            self.container.communicate(
                f"rm -f /home/agent/workspace/{self.task.submission_file}"
            )

        # Build child conversation
        child_msgs = self._build_child_messages(parent, strategy_text, mode, executor_guidance)

        # Execute until validate
        try:
            score, actions, final_msgs = self._execute_until_validate(
                child_msgs, child_id
            )
            snap = self.container.save_snapshot(child_id)
            error = None
        except Exception as e:
            print(f"  ERROR: {e}")
            score, actions, final_msgs = None, [], child_msgs
            snap = ""
            error = str(e)

        exec_status, err_type = classify_execution(actions, score)

        # Summarize execution logs with a cheap LLM call for the scientist's tree view
        log_summary = ""
        try:
            log_summary = self._summarize_execution(actions, score, strategy_text)
        except Exception as e:
            log_summary = f"(summary failed: {type(e).__name__})"

        child = TreeNode(
            node_id=child_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            strategy=strategy_text,
            score=score,
            actions=actions,
            conversation_history=final_msgs,
            snapshot_path=snap,
            error=error,
            execution_status=exec_status,
            error_type=err_type,
            axis=axis,
            log_summary=log_summary,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)
        self._save_node(child)

        if score is not None:
            try:
                print(f"  [{child_id}] score={float(score):.4f}")
            except (ValueError, TypeError):
                print(f"  [{child_id}] score={score}")
        else:
            print(f"  [{child_id}] FAILED")

        # Periodic save — write intermediate result.json after every node
        # so crashes don't lose hours of work
        self._save_intermediate_result()

        return child

    def _save_intermediate_result(self):
        """Write result.json with current best, so progress survives crashes."""
        scored_nodes = []
        for nid, n in self.nodes.items():
            if n.score is None:
                continue
            try:
                scored_nodes.append((nid, float(n.score)))
            except (ValueError, TypeError):
                continue  # skip nodes with non-numeric scores (e.g. error strings)
        if not scored_nodes:
            return
        if self.task.higher_is_better:
            best_id, best_score = max(scored_nodes, key=lambda x: x[1])
        else:
            best_id, best_score = min(scored_nodes, key=lambda x: x[1])

        result = {
            "task": self.task.name,
            "primary_metric": self.task.primary_metric,
            "higher_is_better": self.task.higher_is_better,
            "selection_strategy": "llm_guided",
            "best_node_id": best_id,
            "best_score": best_score,
            "baseline_score": self.container.baseline_score,
            "improvement": best_score - self.container.baseline_score,
            "total_nodes": len(self.nodes),
            "node_budget": self.node_budget,
            "time_budget_seconds": self.time_budget,
            "stopped_by": "in_progress",
            "memory": list(self.memory),
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "strategy": n.strategy[:100],
                    "score": n.score,
                    "actions_count": len(n.actions),
                    "children": n.children,
                    "error": n.error,
                }
                for nid, n in self.nodes.items()
            },
        }
        try:
            with open(self.output_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass  # don't crash the run over a save failure

    # ------------------------------------------------------------------
    # Build child conversation messages
    # ------------------------------------------------------------------

    def _extract_parent_code(self, parent: TreeNode) -> str:
        """Return the last-written version of the task's main script from the
        parent's action history. Falls back to empty string if not found."""
        script = self.task.script_name or "train_and_predict.py"
        last_content = ""
        for a in parent.actions:
            cmd = a.get("action", "") if isinstance(a, dict) else ""
            if cmd.startswith(f"WRITE_FILE: {script}"):
                # cmd format: "WRITE_FILE: <path>\n<body>\n(END_WRITE_FILE)?"
                lines = cmd.split("\n", 1)
                body = lines[1] if len(lines) > 1 else ""
                # Strip trailing END_WRITE_FILE if present
                if body.rstrip().endswith("END_WRITE_FILE"):
                    body = body.rstrip()[: -len("END_WRITE_FILE")].rstrip()
                last_content = body
            elif cmd.lstrip().startswith(("cat <<", "cat <<- ")) and script in cmd:
                # heredoc: capture body up to the terminator
                # best-effort; if parse fails, skip
                try:
                    m = re.match(r"cat\s*<<-?\s*'?(\w+)'?[^\n]*\n(.*?)\n\1\s*$",
                                 cmd, re.DOTALL)
                    if m:
                        last_content = m.group(2)
                except Exception:
                    pass
        # Cap at 20K chars to protect against pathological writes
        if len(last_content) > 20000:
            last_content = last_content[:20000] + "\n# ... (truncated at 20K chars)"
        return last_content

    def _build_child_messages(
        self, parent: TreeNode, strategy_text: str, mode: str,
        executor_guidance: str = "",
    ) -> list[dict]:
        """Build the conversation messages for a child node.

        Default: FRESH conversation with compact summary of parent's work +
        final validated code. Avoids 100K+ char chat transcripts that degrade
        long-context executor reliability.

        Alternative (--full-parent-history): deep-copy parent's full message
        transcript and append the scientist's new direction. Heavier context
        but preserves complete conversational continuity.
        """
        if self.full_parent_history and getattr(parent, "messages", None):
            import copy as _copy
            child_msgs = _copy.deepcopy(parent.messages)
            direction_msg = (
                f"Based on the previous attempt (score: "
                f"{parent.score if parent.score is not None else 'FAILED'}), "
                f"your supervisor proposes the following next direction "
                f"({mode} mode):\n\n{strategy_text}"
            )
            if executor_guidance:
                direction_msg += f"\n\nSupervisor guidance: {executor_guidance}"
            child_msgs.append({"role": "user", "content": direction_msg})
            return child_msgs
        try:
            from arts.tree_search import generate_code_outline
            task_desc = self.task.root_task_desc.format(
                baseline_score=self.container.baseline_score,
                data_head="(use 'cat' to inspect files)",
                code_outline=generate_code_outline(getattr(self.task, "starter_code_host_path", "")),
            )
        except Exception:
            task_desc = f"Task: {self.task.name}, baseline={self.container.baseline_score}"
        child_msgs = [
            {"role": "system", "content": self.task.system_prompt},
            {"role": "user", "content": task_desc},
        ]
        write_instr = self.task.branch_write_instruction
        is_from_baseline = len(parent.actions) == 0

        parts = []

        # Inject executor guidance from the scientist (warnings, tips)
        if executor_guidance:
            parts.append(
                f"IMPORTANT WARNINGS FROM YOUR SUPERVISOR:\n{executor_guidance}"
            )

        # If the scientist said to combine multiple nodes, expand the referenced
        # nodes into concrete context (strategy + score + log summary) so the
        # executor can actually merge them rather than guessing from IDs.
        import re as _re
        combines_match = _re.search(r"Combines ideas from:\s*(.+)", strategy_text)
        if combines_match:
            ref_ids = [s.strip() for s in combines_match.group(1).split(",") if s.strip()]
            combine_ctx_lines = ["=== IDEAS TO COMBINE (from previous nodes) ==="]
            for rid in ref_ids:
                if rid in self.nodes:
                    rn = self.nodes[rid]
                    score_s = f"{rn.score:.4f}" if rn.score is not None else "FAILED"
                    combine_ctx_lines.append(f"\n-- Node {rid} (score: {score_s}) --")
                    combine_ctx_lines.append(f"Strategy: {rn.strategy[:500]}")
                    if getattr(rn, "log_summary", ""):
                        combine_ctx_lines.append(f"What happened:\n{rn.log_summary[:800]}")
            combine_ctx_lines.append(
                "\nYour task: merge the key successful elements of the above nodes."
            )
            parts.append("\n".join(combine_ctx_lines))

        # Provide a compact history of recent ancestor nodes so the
        # executor can avoid repeating mistakes and build on successes.
        history_lines = []
        seen = set()
        # Walk up ancestors
        cur = parent
        depth_back = 0
        while cur is not None and depth_back < 4:
            if cur.node_id != "root" and cur.node_id not in seen:
                seen.add(cur.node_id)
                score_s = f"{cur.score:.4f}" if cur.score is not None else "FAILED"
                line = f"- {cur.node_id} (score {score_s}): {cur.strategy[:200]}"
                summ = getattr(cur, "log_summary", "")
                if summ:
                    # Show the whole summary (bullets + ERROR section if any),
                    # not just the first line, so failures and fix-hints are
                    # visible to the child. Cap to keep the prompt bounded.
                    summ_trim = summ[:800]
                    # Indent for readability
                    summ_indent = "\n    ".join(summ_trim.splitlines())
                    line += f"\n    {summ_indent}"
                history_lines.append(line)
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
            depth_back += 1
        if history_lines:
            parts.append(
                "=== ANCESTOR CONTEXT (what already happened on this branch) ===\n"
                + "\n".join(history_lines)
            )

        # Parent's final working code (from its snapshot, which the orchestrator
        # has already restored into the workspace). Extracting the last WRITE_FILE
        # for the task's script lets the executor see what to modify without
        # having to spend a turn on `cat`.
        if not is_from_baseline and parent.score is not None:
            parent_code = self._extract_parent_code(parent)
            if parent_code:
                parts.append(
                    f"=== CURRENT CODE ({self.task.script_name}) — from parent "
                    f"{parent.node_id} (score {parent.score:.4f}) ===\n"
                    "This code is already in your workspace. Modify it per the "
                    "strategy below — do not rewrite from scratch unless the "
                    "strategy explicitly asks for a full rebuild.\n"
                    f"```python\n{parent_code}\n```"
                )

        score_str = f"{parent.score:.4f}" if parent.score is not None else "N/A (previous attempt failed)"

        write_file_rule = (
            "\n=== FILE WRITING — USE WRITE_FILE FOR LONG FILES ===\n"
            "For files longer than ~30 lines (e.g. a full training pipeline),\n"
            "use the out-of-band WRITE_FILE command INSTEAD of `cat << ENDOFFILE`.\n"
            "The `cat` heredoc breaks silently on long code. WRITE_FILE syntax:\n"
            "\n"
            "  WRITE_FILE: train_and_predict.py\n"
            "  import torch\n"
            "  ...full file contents on subsequent lines...\n"
            "  END_WRITE_FILE\n"
            "\n"
            "Write the ENTIRE file in one response — do not try to chunk or\n"
            "'continue' in a later response. If your file is 400 lines, emit\n"
            "all 400 lines between WRITE_FILE: and END_WRITE_FILE.\n"
            "Short one-off edits can still use cat/sed/echo as usual.\n"
        )

        logging_rule = (
            "\n=== PRINT DIAGNOSTICS — AND STAY ALIVE ===\n"
            "CRITICAL: MLGym captures stdout via pipe (not tty). If your script\n"
            "emits NO output for 15 minutes, MLGym kills it and you lose the run.\n"
            "Silent data loading / DataLoader init / image preprocessing on a few\n"
            "thousand samples easily crosses this threshold.\n"
            "\n"
            "TWO MANDATORY HABITS:\n"
            "(a) Run with `python -u <script>.py` (unbuffered), OR put\n"
            "    `flush=True` on EVERY print. Default buffering hides your output\n"
            "    until a 4KB flush, which looks identical to a hang.\n"
            "(b) Print a status line BEFORE every potentially-slow step, not after.\n"
            "    'After' means MLGym never sees it if the step hangs.\n"
            "\n"
            "REQUIRED PRINT CHECKPOINTS (in this order, every one is mandatory):\n"
            "  print('Starting script...', flush=True)                       # line 1 of __main__\n"
            "  print('Loading dataset...', flush=True)                       # BEFORE reading files\n"
            "  print(f'DATA: train={N_TRAIN} val={N_VAL}', flush=True)       # after dataset built\n"
            "  print('Building model...', flush=True)                        # before model init\n"
            "  print(f'CONFIG: epochs={E} batch={B} lr={LR} model={M}', flush=True)\n"
            "  print('Starting epoch 1...', flush=True)                      # before training loop\n"
            "  # then every epoch:\n"
            "  print(f'Epoch {e+1}/{E}: train_loss={tl:.4f} val_metric={vm:.4f}', flush=True)\n"
            "  print('Writing submission.csv...', flush=True)                # before save\n"
            "\n"
            "If your dataset has heavy __init__ (opening thousands of files,\n"
            "pre-decoding images, scanning labels for a weighted sampler),\n"
            "print INSIDE the init too (e.g. every 500 files). Otherwise a 15-min\n"
            "silent init will be killed even though it's making real progress.\n"
            "\n"
            "Catch handled errors plainly: print(f'WARN: {e}', flush=True).\n"
            "A summary like 'trained 10 epochs, loss 0.5->0.04, val 0.91' is far\n"
            "more useful to the supervisor than 'model trained successfully'.\n"
        )

        fidelity_rule = (
            "\n=== CRITICAL RULE — STRATEGY FIDELITY ===\n"
            "You MUST implement exactly the strategy described above. Your job is\n"
            "to make the supervisor's experiment work well — not to substitute\n"
            "your own idea. Specifically:\n"
            "- If the strategy says 'threshold tuning', do threshold tuning. Do\n"
            "  NOT swap in a CNN/LR/transformer instead.\n"
            "- If the strategy says 'add post-processing X', add X on top of the\n"
            "  existing pipeline. Do NOT rewrite the whole pipeline.\n"
            "- If the strategy says 'train for 12 epochs', train for 12 epochs.\n"
            "  Do NOT silently reduce to 2 to save time.\n"
            "- You MAY choose implementation details (library calls, exact\n"
            "  hyperparameter names, helper functions, error handling). But the\n"
            "  fundamental algorithm/architecture/recipe MUST match the strategy.\n"
            "- If you think the strategy is a bad idea, implement it anyway. The\n"
            "  supervisor will see the result and pivot next turn.\n"
            "- A failing implementation of the requested strategy is more useful\n"
            "  than a successful implementation of a different one.\n"
        )

        iterate_rule = (
            "\n=== VALIDATE AND SUBMIT ===\n"
            "You may call `validate` multiple times within this node if useful;\n"
            "it is a progress check and does not close the node. The node's score\n"
            "is the best of all your validate calls. Call `submit` when you are\n"
            "ready to finalize the node.\n"
        )

        edit_rule = (
            "\n=== EDIT IN PLACE — PARENT CODE IS ALREADY IN YOUR WORKSPACE ===\n"
            "Your parent's validated code is loaded. The default expectation is\n"
            "that you make SMALL, TARGETED edits to it — not rewrite it.\n"
            "\n"
            "HARD RULE: if the strategy describes any of the following, you MUST\n"
            "use sed / python-regex-patching and MUST NOT use WRITE_FILE:\n"
            "  • Change a hyperparameter value (epochs, batch, lr, weight_decay,\n"
            "    threshold, etc.)\n"
            "  • Swap one function call for another (e.g., Adam -> AdamW)\n"
            "  • Add a single line (a print, a scheduler step, a flush=True)\n"
            "  • Tune a numeric constant (224 -> 384, 0.5 -> 0.7, etc.)\n"
            "  • Toggle a boolean flag\n"
            "Examples:\n"
            "  sed -i 's/epochs=8/epochs=20/' train_and_predict.py\n"
            "  sed -i 's/Adam(/AdamW(/' train_and_predict.py\n"
            "  sed -i '47a\\    print(\"debug\", flush=True)' train_and_predict.py\n"
            "  python -c \"import re; s=open('train_and_predict.py').read(); s=re.sub(r'lr=1e-4','lr=3e-4',s); open('train_and_predict.py','w').write(s)\"\n"
            "\n"
            "IMPORTANT — DO NOT FAKE A REWRITE AS AN EDIT:\n"
            "Shell heredocs that overwrite the whole file count as WRITE_FILE and\n"
            "will trigger the write-throttle the same way. Do NOT do:\n"
            "  cat << 'EOF' > train_and_predict.py ...   # banned as small edit\n"
            "  tee train_and_predict.py << EOF ...       # banned as small edit\n"
            "  printf '...' > train_and_predict.py       # banned as small edit\n"
            "If you need to write the whole file, use the WRITE_FILE: command\n"
            "explicitly so the orchestrator can count it.\n"
            "\n"
            "Use WRITE_FILE (full file replacement) ONLY when:\n"
            "  - The strategy explicitly says 'reimplement', 'replace', 'rewrite'\n"
            "  - The change is structural (new model class, new training loop)\n"
            "    affecting many scattered lines\n"
            "  - You've tried sed edits and they didn't land correctly\n"
            "\n"
            "A sed + run cycle is ~30 seconds; a full WRITE_FILE regenerates\n"
            "hundreds of lines (~2-5 min LLM time). A tree search that spends\n"
            "its budget on regenerations will exhaust actions without testing.\n"
            "\n"
            "After ANY edit, `cat` / `sed -n 'start,endp'` the changed region to\n"
            "confirm it landed before running python:\n"
            "  sed -n '45,55p' train_and_predict.py\n"
            "If the pattern didn't match, re-read the file to find the actual\n"
            "line, then retry. Do NOT blindly re-sed the same pattern.\n"
        )

        if is_from_baseline:
            parts.append(f"Strategy to try: {strategy_text}")
            parts.append(fidelity_rule)
            parts.append(iterate_rule)
            parts.append(logging_rule)
            parts.append(write_file_rule)
            parts.append(write_instr)
        elif mode == "exploit":
            parts.append(
                f"Your current score is {score_str}. "
                f"Refine your current approach to improve it."
            )
            parts.append(f"Variation to try: {strategy_text}")
            parts.append(
                "IMPORTANT: Stay within the same approach as before. "
                "Do NOT switch to a completely different algorithm. "
                "Just tune or tweak the existing approach."
            )
            parts.append(fidelity_rule)
            parts.append(iterate_rule)
            parts.append(logging_rule)
            parts.append(write_file_rule)
            parts.append(edit_rule)  # overrides write_file_rule for exploit tweaks
            parts.append(write_instr)
        else:  # explore
            parts.append(
                f"Your current score is {score_str}. "
                f"Try a FUNDAMENTALLY DIFFERENT approach to improve it."
            )
            parts.append(f"Strategy: {strategy_text}")
            parts.append(fidelity_rule)
            parts.append(iterate_rule)
            parts.append(logging_rule)
            parts.append(write_file_rule)
            parts.append(edit_rule)  # still encourage edits even on explore
            parts.append(write_instr)

        child_msgs.append({"role": "user", "content": "\n\n".join(parts)})
        return child_msgs

    # ------------------------------------------------------------------
    # Execute until validate (adapted from adaptive_tree_search)
    # ------------------------------------------------------------------

    def _execute_until_validate(
        self, messages: list[dict], node_id: str
    ) -> tuple[float | None, list[dict], list[dict]]:
        """Execute actions until validate is called.

        Returns (score, action_log, final_messages).
        """
        action_log = []
        score = None
        consecutive_writes = 0  # throttle: block 3rd+ consecutive WRITE_FILE without a run in between

        for step in range(self.max_actions):
            # Debug: log prompt size before LLM call
            total_chars = sum(len(m.get('content', '')) for m in messages)
            if step == 0:
                print(f"  [{node_id}] Initial prompt: {len(messages)} msgs, {total_chars} chars (~{total_chars//4} tokens)")
            try:
                raw = self.executor.chat(messages)
            except Exception as e:
                print(f"  [{node_id}] LLM error at step {step}, prompt={total_chars} chars: {str(e)[:100]}")
                time.sleep(2)
                try:
                    raw = self.executor.chat(messages)
                except Exception:
                    raise RuntimeError(f"LLM failed: {e}")

            action, _ = extract_command(raw)
            if not action:
                # Truncate raw if too long (can happen with long thinking blocks)
                raw_capped = raw[:5000] if len(raw) > 5000 else raw
                messages.append({"role": "assistant", "content": raw_capped})
                messages.append({
                    "role": "user",
                    "content": "No command detected. Output a valid command.",
                })
                action_log.append({
                    "action": raw[:100], "observation": "No command", "step": step,
                })
                # Bail early if we've had too many parse failures
                if sum(1 for a in action_log if a.get("observation") == "No command") >= 3:
                    print(f"  [{node_id}] Bailing after 3 parse failures")
                    break
                continue

            # New node semantics (2026-04-22):
            # - `validate` is a CHEAP progress check. Executor may call it many
            #   times during a node to iterate on its score. The node does NOT
            #   close on validate. The returned score is surfaced to the
            #   executor so it can decide whether to keep iterating.
            # - `submit` FINALIZES the node with whatever the current
            #   submission.csv scores. This is the score the scientist sees.
            is_submit = action.strip().lower() == "submit"

            # Write-throttle: block 3rd+ consecutive WRITE_FILE without a run.
            # Rewriting the same file repeatedly without ever running python on
            # it is a known pathological executor loop (observed 175 writes / 0
            # pythons on Vesuvius). Force a test cycle by rejecting the write
            # and injecting a strong hint telling the executor to run first.
            # A "write" is anything that overwrites a .py file, not just WRITE_FILE.
            # Shell heredocs (cat<<EOF>file.py, tee, printf>file.py) are functionally
            # identical rewrites and previously bypassed the throttle.
            import re as _re_w
            _act_s = action.lstrip()
            is_write = (
                _act_s.upper().startswith("WRITE_FILE:")
                or bool(_re_w.search(r"(cat|tee|printf)\s*<<\s*[-'\"]*\w+[-'\"]*.*?>\s*\S+\.py", _act_s, _re_w.DOTALL))
                or bool(_re_w.search(r"^\s*(cat|tee|printf|echo)\s+[^|]*>\s*\S+\.py\s*$", _act_s, _re_w.MULTILINE))
            )
            if is_write:
                consecutive_writes += 1
            else:
                consecutive_writes = 0
            if consecutive_writes >= 3:
                hint = (
                    "[orchestrator] You've written this file 3 times without "
                    "running it. BLOCKED — rewriting an untested file is not "
                    "allowed. Your next command MUST run the script (e.g. "
                    "`python -u train_and_predict.py`) so you can see what "
                    "actually breaks. After that you may rewrite based on the "
                    "real error."
                )
                messages.append({"role": "assistant", "content": action[:2000]})
                messages.append({"role": "user", "content": hint})
                action_log.append({
                    "action": action[:200],
                    "observation": "BLOCKED: 3rd consecutive WRITE_FILE without run",
                    "step": step,
                })
                consecutive_writes = 2  # stay at 2 so a *different* next write would also be blocked, but a python run resets
                continue

            exec_action = "validate" if is_submit else action

            if self.verbose:
                print(f"    [{node_id}] step {step}: {exec_action[:80]}")

            obs, info = self.container.step(exec_action)

            if self.verbose:
                if "validate" in exec_action.strip().lower():
                    print(f"    [{node_id}] validate score={info.get('score')}")
                elif exec_action.strip().startswith("python"):
                    has_error = "Traceback" in (obs or "") or "Error" in (obs or "")
                    if has_error:
                        print(f"    [{node_id}] python ERROR: {(obs or '')[-150:]}")

            action_log.append({
                "action": (action[:2000] if not is_submit else "submit"),
                "observation": obs[:2000] if obs else "",
                "step": step,
            })
            raw_capped = raw[:5000] if len(raw) > 5000 else raw
            if obs and len(obs) > 8000:
                obs_capped = obs[:2000] + "\n... [truncated middle, showing head+tail] ...\n" + obs[-6000:]
            else:
                obs_capped = obs
            messages.append({"role": "assistant", "content": raw_capped})
            messages.append({"role": "user", "content": obs_capped})

            score_found = self._extract_score(info, obs)
            if score_found is not None:
                # Track BEST score within the node (symmetric with linear).
                if score is None or (
                    self.task.higher_is_better and score_found > score
                ) or (
                    not self.task.higher_is_better and score_found < score
                ):
                    score = score_found
                if is_submit:
                    print(f"  [{node_id}] submit score={score_found:.4f} "
                          f"(node best={score:.4f})")
                    break
                # Otherwise: validate, keep iterating. Hint the executor it can
                # iterate further or submit to finalize.
                hint = (
                    f"\n[node-loop] validate returned score={score_found:.4f} "
                    f"(node best so far={score:.4f}). "
                    "You may iterate (write more code, run again, validate again), "
                    "or issue `submit` to finalize this node."
                )
                messages[-1] = {"role": "user", "content": (obs_capped or "") + hint}
        else:
            # Max actions hit with no explicit submit. Force a final validate
            # so the node has a score. Treat it as the finalized submission.
            print(f"  [{node_id}] Max actions reached, forcing final submit")
            obs, info = self.container.step("validate")
            messages.append({"role": "assistant", "content": "submit (forced)"})
            messages.append({"role": "user", "content": obs})
            action_log.append({
                "action": "submit (forced at max actions)",
                "observation": obs[:500],
                "step": self.max_actions,
            })
            score = self._extract_score(info, obs)

        return score, action_log, messages

    def _extract_score(self, info: dict, obs: str) -> float | None:
        """Extract the primary metric score from validate output. Always returns float|None."""
        raw = None
        if info.get("score"):
            score_data = info["score"][-1]
            if isinstance(score_data, dict):
                raw = score_data.get(
                    self.task.primary_metric, list(score_data.values())[0]
                )
            else:
                raw = score_data

        if raw is None and obs and "Evaluation Score" in obs:
            import ast
            m = re.search(r"Evaluation Score:\s*(\{[^}]+\})", obs)
            if m:
                try:
                    score_dict = ast.literal_eval(m.group(1))
                    raw = list(score_dict.values())[0]
                except Exception:
                    pass

        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            # raw was a string error message (e.g. "Grading failed: ..."); return None
            return None

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _global_best_score(self) -> float:
        scored = [n.score for n in self.nodes.values() if n.score is not None]
        if not scored:
            return 0.0
        return max(scored) if self.task.higher_is_better else min(scored)

    def _compile_results(self, start_time: float, stopped_by: str = "node_budget") -> dict:
        scored_nodes = [
            (nid, n.score) for nid, n in self.nodes.items()
            if n.score is not None
        ]
        if not scored_nodes:
            best_id, best_score = "root", 0.0
        elif self.task.higher_is_better:
            best_id, best_score = max(scored_nodes, key=lambda x: x[1])
        else:
            best_id, best_score = min(scored_nodes, key=lambda x: x[1])

        elapsed = time.time() - start_time

        result = {
            "task": self.task.name,
            "primary_metric": self.task.primary_metric,
            "higher_is_better": self.task.higher_is_better,
            "selection_strategy": "llm_guided",
            "best_node_id": best_id,
            "best_score": best_score,
            "baseline_score": self.container.baseline_score,
            "improvement": best_score - self.container.baseline_score,
            "total_nodes": len(self.nodes),
            "elapsed_seconds": round(elapsed, 1),
            "node_budget": self.node_budget,
            "time_budget_seconds": self.time_budget,
            "stopped_by": stopped_by,
            "memory": list(self.memory),
            "tree_shape": {
                nid: {
                    "depth": n.depth,
                    "num_children": len(n.children),
                    "score": n.score,
                    "strategy": n.strategy[:100],
                }
                for nid, n in self.nodes.items()
            },
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "strategy": n.strategy[:100],
                    "score": n.score,
                    "actions_count": len(n.actions),
                    "children": n.children,
                    "error": n.error,
                }
                for nid, n in self.nodes.items()
            },
        }

        with open(self.output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        self._print_tree(best_id)
        return result

    def _save_node(self, node: TreeNode):
        path = self.output_dir / "nodes" / f"{node.node_id}.json"
        data = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "strategy": node.strategy,
            "score": node.score,
            "error": node.error,
            "execution_status": node.execution_status,
            "error_type": node.error_type,
            "axis": getattr(node, "axis", "hp"),
            "log_summary": getattr(node, "log_summary", ""),
            "actions": node.actions,
            "conversation_length": len(node.conversation_history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _print_tree(self, best_id: str):
        print(f"\n{'=' * 70}")
        print("LLM-GUIDED TREE SEARCH RESULTS")
        print(f"{'=' * 70}")

        best = self.nodes.get(best_id)
        if best and best.score is not None:
            print(
                f"Baseline: {self.container.baseline_score:.4f} | "
                f"Best: {best.score:.4f} (node: {best_id}) | "
                f"Improvement: {best.score - self.container.baseline_score:+.4f}"
            )
        print(f"Nodes explored: {len(self.nodes)}")
        if self.memory:
            print(f"Accumulated memory ({len(self.memory)} entries):")
            for m in self.memory:
                print(f"  - {m[:100]}")
        print(f"{'=' * 70}\n")

        def _print_node(nid: str, prefix: str = "", is_last: bool = True):
            n = self.nodes[nid]
            connector = "└── " if is_last else "├── "
            marker = " *** BEST ***" if nid == best_id else ""
            score_str = f"{n.score:.4f}" if n.score is not None else "FAIL"
            strategy_short = n.strategy[:50]
            print(f"{prefix}{connector}{nid} [{score_str}] {strategy_short}{marker}")

            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, cid in enumerate(n.children):
                if cid in self.nodes:
                    _print_node(cid, child_prefix, i == len(n.children) - 1)

        _print_node("root")

        # Print best path
        path = []
        nid = best_id
        while nid:
            path.append(nid)
            nid = self.nodes[nid].parent_id
        path.reverse()
        print(f"\nBest path: {' -> '.join(path)}")
        for p in path:
            n = self.nodes[p]
            score_str = f"{n.score:.4f}" if n.score is not None else "N/A"
            print(f"  {p}: [{score_str}] {n.strategy[:80]}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: LLM-guided tree search with scientist + executor",
    )
    parser.add_argument("--task-config", default="tasks/titanic.yaml")
    parser.add_argument("--output-dir", default="outputs/llm_guided_search")
    parser.add_argument("--node-budget", type=int, default=12)
    parser.add_argument("--initial-breadth", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--time-budget", type=int, default=0,
                        help="Max seconds for search (0 = no limit). Stops when either node or time budget is exhausted.")
    parser.add_argument("--env-gpu", default="7")
    parser.add_argument("--image-name", default="aigym/mlgym-agent:latest")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume-from", default="",
                        help="Path to a previous result.json to resume from")
    parser.add_argument("--research-phase-steps", type=int, default=0,
                        help="Number of no-expansion 'research' steps at the start where the scientist only reads files / inspects nodes and writes memory entries (builds task understanding before the active search)")
    parser.add_argument("--full-parent-history", action="store_true",
                        help="Inherit parent's full chat transcript instead of the compact summary. Ablation: test whether long context helps or hurts executor reliability.")
    parser.add_argument("--no-build-on-signal", action="store_true",
                        help="Disable the BUILD ON SIGNAL bullet in scientist prompt. Ablation: let scientist freely propose compound multi-axis strategies from step 0.")
    parser.add_argument("--no-incremental-rules", action="store_true",
                        help="Stronger ablation: strip EVOLVE bullet, BUILD ON SIGNAL bullet, and the cost-aware bundling block in TURN1. Lets the scientist propose freely without any incrementalism scaffolding.")
    parser.add_argument("--complexity-cycle", action="store_true",
                        help="Add an AIRA-dojo style simple->normal->complex hint to the first 3 active expansion steps. Ablation: test whether explicit complexity scaffolding helps the scientist escape weak first recipes.")

    # Scientist model
    parser.add_argument("--scientist-model", default="gpt-4o",
                        help="Model for scientist (e.g., gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--scientist-url", default="",
                        help="API base URL for scientist (empty = OpenAI default)")
    parser.add_argument("--scientist-temperature", type=float, default=0.3)
    parser.add_argument("--scientist-thinking-budget", type=int, default=0,
                        help="Thinking budget tokens for scientist (0 = disabled)")

    # Executor model
    parser.add_argument("--executor-model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model for executor (e.g., local vLLM model)")
    parser.add_argument("--executor-url", default="http://localhost:8000/v1",
                        help="vLLM URL for executor")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Executor temperature")
    parser.add_argument("--executor-thinking-budget", type=int, default=0,
                        help="Thinking budget tokens for executor (0 = disabled)")

    # Convenience aliases
    parser.add_argument("--model", default="",
                        help="Override executor model (backward compat)")
    parser.add_argument("--vllm-url", default="",
                        help="Override executor URL (backward compat)")

    args = parser.parse_args()

    # Handle backward-compat overrides
    executor_model = args.model or args.executor_model
    executor_url = args.vllm_url or args.executor_url

    task_profile = get_task_profile(args.task_config)
    print("=" * 60)
    print(f"Task: {task_profile.name}")
    print(f"LLM-Guided Tree Search (Experiment 4)")
    print("=" * 60)
    print(f"Scientist: {args.scientist_model}")
    print(f"Executor:  {executor_model}")
    print(f"Node budget: {args.node_budget}, Initial breadth: {args.initial_breadth}")
    time_str = f"{args.time_budget}s" if args.time_budget > 0 else "unlimited"
    print(f"Max actions/node: {args.max_actions}, Time budget: {time_str}")
    print(f"Primary metric: {task_profile.primary_metric} "
          f"({'higher' if task_profile.higher_is_better else 'lower'} is better)")
    print()

    # Create LLM clients
    scientist = LLMClient(
        base_url=args.scientist_url or "",
        model=args.scientist_model,
        temperature=args.scientist_temperature,
        thinking_budget=args.scientist_thinking_budget,
    )
    executor = LLMClient(
        base_url=executor_url,
        model=executor_model,
        temperature=args.temperature,
        thinking_budget=args.executor_thinking_budget,
    )

    container = ContainerManager(
        args.task_config, args.env_gpu, args.image_name,
        task_profile=task_profile,
    )
    print("Creating MLGym container...")
    container.create()

    search = LLMGuidedTreeSearch(
        scientist=scientist,
        executor=executor,
        container=container,
        task_profile=task_profile,
        node_budget=args.node_budget,
        initial_breadth=args.initial_breadth,
        max_actions=args.max_actions,
        output_dir=args.output_dir,
        verbose=args.verbose,
        time_budget=args.time_budget,
        resume_from=args.resume_from,
        research_phase_steps=args.research_phase_steps,
        full_parent_history=args.full_parent_history,
        no_build_on_signal=args.no_build_on_signal,
        no_incremental_rules=args.no_incremental_rules,
        complexity_cycle=args.complexity_cycle,
    )

    try:
        result = search.run()
        print(f"Results saved to {args.output_dir}/result.json")
    finally:
        container.close()


if __name__ == "__main__":
    main()
