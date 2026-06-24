"""MLEvolve search policy adapted for MLGym.

Re-implementation of the MLEvolve search algorithm
(https://github.com/InternScience/MLEvolve) adapted for our multi-turn
ReAct executor.

Differs from AIRA-MCTS (arts.search.aira.search.MCTSSearch) in four ways:
  1. Piecewise exploration decay — uct_c starts high (broad search)
     and is multiplied by `decay_factor` at planned milestones.
  2. Time-aware explore->exploit — when wall-clock elapsed exceeds
     `exploit_after_frac` of `time_budget`, exploration is forced to 0.
  3. Stagnation detection — if the global best score has not improved
     in `stagnation_window` consecutive expansions, the next step is
     forced to be a cross-branch FUSION.
  4. Cross-branch fusion — pick the top-scoring node from each of two
     distinct root-children subtrees and merge them via the AIRA-dojo
     CROSSOVER operator. Falls back to fitness-proportional pairing
     when fewer than two branches exist.

Implementation reuses BaseSearch + AiraOperators from arts.search.aira;
only the search policy (selection, decay, stagnation, fusion) is new.

Usage (CLI):
    python -m arts.search.mlevolve.search \\
        --task-config tasks/mlebenchKuzushiji.yaml \\
        --node-budget 100 --max-actions 50 \\
        --time-budget 27600 \\
        --model gemini-3-pro-preview \\
        --output-dir outputs/mlevolve_kuz_run1 \\
        --env-gpu 0
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from arts.search.aira.operators import AiraOperators, OperatorType
from arts.search.aira.search import BaseSearch
from arts.tree_search import (
    ContainerManager,
    LLMClient,
    get_task_profile,
)


class MLEvolveSearch(BaseSearch):
    """MLEvolve: MCGS with decay, stagnation, and cross-branch fusion."""

    def __init__(
        self,
        uct_c_init: float = 0.5,
        decay_milestones: tuple[float, ...] = (0.33, 0.66),
        decay_factor: float = 0.5,
        exploit_after_frac: float = 0.8,
        stagnation_window: int = 4,
        initial_drafts: int = 3,
        max_debug_depth: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.uct_c_init = uct_c_init
        self.decay_milestones = tuple(sorted(decay_milestones))
        self.decay_factor = decay_factor
        self.exploit_after_frac = exploit_after_frac
        self.stagnation_window = stagnation_window
        self.initial_drafts = min(initial_drafts, self.node_budget)
        self.max_debug_depth = max_debug_depth

        self._visit_count: dict[str, int] = {}
        self._cumulative_value: dict[str, float] = {}
        self._best_history: list[float] = []
        self._fusion_count = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        start = time.time()
        self._start_time = start
        print("\n" + "=" * 60)
        print("MLEVOLVE SEARCH")
        print(
            f"  uct_c_init={self.uct_c_init}, "
            f"decay@{self.decay_milestones}*{self.decay_factor}, "
            f"exploit_after={self.exploit_after_frac}, "
            f"stagnation_window={self.stagnation_window}, "
            f"node_budget={self.node_budget}"
        )
        if self.time_budget > 0:
            print(f"  time_budget={self.time_budget}s")
        print("=" * 60)

        # Phase 1: root
        root = self._execute_root()
        self._visit_count["root"] = 1
        self._cumulative_value["root"] = root.score or 0.0
        self._record_best()
        budget_used = 0

        # Phase 2: initial drafts (root children = branches)
        print(f"\n--- Phase 2: Initial drafts ({self.initial_drafts}) ---")
        for _ in range(self.initial_drafts):
            if budget_used >= self.node_budget or self._time_budget_exceeded():
                break
            memory = self._build_simple_memory()
            msg = self.ops.draft(memory)
            child = self._expand_node("root", OperatorType.DRAFT, msg)
            budget_used += 1
            self._post_expand_bookkeeping(child)

        # Phase 3: MCGS loop with decay/stagnation/fusion
        remaining = self.node_budget - budget_used
        print(f"\n--- Phase 3: MCGS loop ({remaining} iterations) ---")

        for step in range(remaining):
            if self._time_budget_exceeded():
                break

            stagnant = self._is_stagnant()
            current_c = self._current_uct_c(budget_used)
            print(
                f"\n  MLEvolve step {step + 1}/{remaining} "
                f"| uct_c={current_c:.3f} | stagnant={stagnant} "
                f"| best={self._global_best_score():.4f}"
            )

            if stagnant and self._has_two_branches():
                child = self._do_fusion_step()
                self._fusion_count += 1
                self._best_history.clear()  # reset stagnation window after fusion
            else:
                child = self._do_uct_step(current_c)

            budget_used += 1
            self._post_expand_bookkeeping(child)

        result = self._compile_results(start, "mlevolve")
        result["mlevolve"] = {
            "uct_c_init": self.uct_c_init,
            "decay_milestones": list(self.decay_milestones),
            "decay_factor": self.decay_factor,
            "exploit_after_frac": self.exploit_after_frac,
            "stagnation_window": self.stagnation_window,
            "fusion_count": self._fusion_count,
        }
        return result

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _do_uct_step(self, uct_c: float):
        """Standard MCGS step: UCT-select a leaf, decide op, expand."""
        selected_id = self._uct_select(uct_c)
        selected = self.nodes[selected_id]

        if selected.score is None and selected_id != "root":
            ancestral_mem = self._build_ancestral_memory(selected_id)
            msg = self.ops.debug(
                buggy_approach=selected.strategy[:300],
                error_output=selected.error or "Unknown error",
                ancestral_memory=ancestral_mem,
            )
            op_type = OperatorType.DEBUG
        elif not selected.children:
            memory = self._build_simple_memory()
            msg = self.ops.draft(memory)
            op_type = OperatorType.DRAFT
        else:
            memory = self._build_simple_memory()
            msg = self.ops.improve(
                prev_approach=selected.strategy[:300],
                prev_score=selected.score or 0.0,
                memory=memory,
            )
            op_type = OperatorType.IMPROVE

        return self._expand_node(selected_id, op_type, msg)

    def _do_fusion_step(self):
        """Cross-branch fusion: top-of-branch-A + top-of-branch-B -> child of A."""
        a_id, b_id = self._pick_cross_branch_pair()
        a, b = self.nodes[a_id], self.nodes[b_id]
        print(
            f"  [fusion] merging {a_id} (score={a.score}) "
            f"x {b_id} (score={b.score})"
        )
        msg = self.ops.crossover(
            approach_1=a.strategy[:300],
            score_1=a.score or 0.0,
            approach_2=b.strategy[:300],
            score_2=b.score or 0.0,
        )
        return self._expand_node(
            a_id, OperatorType.CROSSOVER, msg, second_parent_id=b_id,
        )

    # ------------------------------------------------------------------
    # MLEvolve mechanisms
    # ------------------------------------------------------------------

    def _current_uct_c(self, budget_used: int) -> float:
        """Piecewise decay + time-aware exploit zone."""
        if self.time_budget > 0:
            elapsed_frac = (time.time() - self._start_time) / self.time_budget
            if elapsed_frac >= self.exploit_after_frac:
                return 0.0

        if self.node_budget <= 0:
            return self.uct_c_init
        progress = budget_used / self.node_budget
        c = self.uct_c_init
        for m in self.decay_milestones:
            if progress >= m:
                c *= self.decay_factor
        return c

    def _is_stagnant(self) -> bool:
        """No improvement in best score over the last stagnation_window steps."""
        if len(self._best_history) < self.stagnation_window + 1:
            return False
        recent = self._best_history[-(self.stagnation_window + 1):]
        if self.task.higher_is_better:
            return max(recent) <= recent[0] + 1e-9
        return min(recent) >= recent[0] - 1e-9

    def _has_two_branches(self) -> bool:
        """At least two distinct root-children subtrees contain a scored node."""
        scored_branches = set()
        for nid, node in self.nodes.items():
            if node.score is None or nid == "root":
                continue
            branch_root = self._branch_root(nid)
            if branch_root is not None:
                scored_branches.add(branch_root)
            if len(scored_branches) >= 2:
                return True
        return False

    def _branch_root(self, node_id: str) -> str | None:
        """Walk up to the immediate child of root for the given node."""
        cur = node_id
        while cur and cur in self.nodes:
            parent = self.nodes[cur].parent_id
            if parent == "root":
                return cur
            cur = parent
        return None

    def _pick_cross_branch_pair(self) -> tuple[str, str]:
        """Return (a, b): top-scoring scored node from each of two branches.

        Picks the two branches whose best scored node is highest (or lowest
        for minimization), then returns those two nodes.
        """
        per_branch: dict[str, str] = {}  # branch_root -> best node_id in branch
        for nid, node in self.nodes.items():
            if node.score is None or nid == "root":
                continue
            br = self._branch_root(nid)
            if br is None:
                continue
            cur_best = per_branch.get(br)
            if cur_best is None:
                per_branch[br] = nid
                continue
            if self._is_better(node.score, self.nodes[cur_best].score):
                per_branch[br] = nid

        ranked = sorted(
            per_branch.values(),
            key=lambda n: self.nodes[n].score,
            reverse=self.task.higher_is_better,
        )
        if len(ranked) < 2:
            # Shouldn't happen — caller checks _has_two_branches first.
            best = self._global_best_id() or "root"
            return best, best
        return ranked[0], ranked[1]

    def _is_better(self, a: float, b: float) -> bool:
        return (a > b) if self.task.higher_is_better else (a < b)

    # ------------------------------------------------------------------
    # UCT (parameterised by current uct_c)
    # ------------------------------------------------------------------

    def _uct_select(self, uct_c: float) -> str:
        current = "root"
        while True:
            node = self.nodes[current]
            children = [c for c in node.children if c in self.nodes]
            if not children:
                return current
            current = max(children, key=lambda c: self._uct_value(c, uct_c))

    def _uct_value(self, node_id: str, uct_c: float) -> float:
        node = self.nodes[node_id]
        n_child = self._visit_count.get(node_id, 1)
        parent_id = node.parent_id or "root"
        n_parent = self._visit_count.get(parent_id, 1)

        q_raw = self._cumulative_value.get(node_id, 0.0) / max(n_child, 1)

        all_scores = [n.score for n in self.nodes.values() if n.score is not None]
        if all_scores:
            s_min, s_max = min(all_scores), max(all_scores)
            if s_max > s_min:
                q_norm = (q_raw - s_min) / (s_max - s_min)
                if not self.task.higher_is_better:
                    q_norm = 1.0 - q_norm
            else:
                q_norm = 0.5
        else:
            q_norm = 0.5

        explore = uct_c * math.sqrt(math.log(max(n_parent, 1)) / max(n_child, 1))
        return q_norm + explore

    def _backpropagate(self, leaf_id: str, score: float):
        current = leaf_id
        while current is not None:
            self._visit_count[current] = self._visit_count.get(current, 0) + 1
            self._cumulative_value[current] = (
                self._cumulative_value.get(current, 0.0) + score
            )
            parent = self.nodes.get(current)
            current = parent.parent_id if parent else None

    # ------------------------------------------------------------------
    # Bookkeeping after each expansion
    # ------------------------------------------------------------------

    def _post_expand_bookkeeping(self, child):
        if child is None:
            self._record_best()
            return
        score = child.score if child.score is not None else 0.0
        self._visit_count[child.node_id] = 1
        self._cumulative_value[child.node_id] = score
        self._backpropagate(child.node_id, score)
        self._record_best()

    def _record_best(self):
        scored = [n.score for n in self.nodes.values() if n.score is not None]
        if not scored:
            return
        best = max(scored) if self.task.higher_is_better else min(scored)
        self._best_history.append(best)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(description="MLEvolve search over MLGym tasks")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-budget", type=int, default=100)
    parser.add_argument("--time-budget", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--model", required=True,
                        help="Executor model id (e.g. gemini-3-pro-preview)")
    parser.add_argument("--vllm-url", default="")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--env-gpu", default="0")
    parser.add_argument("--image-name", default="aigym/mlgym-agent:latest")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reflexion", action="store_true", default=True)
    parser.add_argument("--no-reflexion", dest="reflexion", action="store_false")
    parser.add_argument("--thinking-budget", type=int, default=0)

    parser.add_argument("--uct-c-init", type=float, default=0.5)
    parser.add_argument("--decay-milestones", type=str, default="0.33,0.66",
                        help="Comma-separated fractions of node budget")
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--exploit-after-frac", type=float, default=0.8)
    parser.add_argument("--stagnation-window", type=int, default=4)
    parser.add_argument("--initial-drafts", type=int, default=3)

    args = parser.parse_args()

    task_profile = get_task_profile(args.task_config)
    decay_milestones = tuple(
        float(x) for x in args.decay_milestones.split(",") if x.strip()
    )

    print("=" * 60)
    print(f"MLEVOLVE Search | Task: {task_profile.name} | Model: {args.model}")
    time_str = f"{args.time_budget}s" if args.time_budget > 0 else "unlimited"
    print(
        f"Node budget: {args.node_budget} | Max actions: {args.max_actions} "
        f"| Time budget: {time_str}"
    )
    print("=" * 60)

    llm = LLMClient(
        args.vllm_url, args.model, args.temperature,
        thinking_budget=args.thinking_budget,
    )
    container = ContainerManager(
        args.task_config, args.env_gpu, args.image_name,
        task_profile=task_profile,
    )

    print("Creating MLGym container...")
    container.create()

    from arts.tree_search import generate_code_outline
    task_desc = task_profile.root_task_desc.format(
        baseline_score=container.baseline_score,
        data_head="",
        code_outline=generate_code_outline(
            getattr(task_profile, "starter_code_host_path", "")
        ),
    )
    data_overview = ""
    if task_profile.data_head_cmd:
        data_overview = container.communicate(task_profile.data_head_cmd)
    operators = AiraOperators(task_desc=task_desc, data_overview=data_overview)

    search = MLEvolveSearch(
        llm=llm,
        container=container,
        task_profile=task_profile,
        operators=operators,
        node_budget=args.node_budget,
        max_actions=args.max_actions,
        output_dir=args.output_dir,
        verbose=args.verbose,
        reflexion=args.reflexion,
        time_budget=args.time_budget,
        uct_c_init=args.uct_c_init,
        decay_milestones=decay_milestones,
        decay_factor=args.decay_factor,
        exploit_after_frac=args.exploit_after_frac,
        stagnation_window=args.stagnation_window,
        initial_drafts=args.initial_drafts,
    )

    try:
        result = search.run()
        print(f"\nResults saved to {args.output_dir}/result.json")
        print(
            f"Best score: {result['best_score']:.4f} "
            f"(improvement: {result['improvement']:+.4f}, "
            f"fusions: {result['mlevolve']['fusion_count']})"
        )
    finally:
        container.close()


if __name__ == "__main__":
    main()
