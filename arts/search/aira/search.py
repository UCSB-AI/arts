"""AIRA-dojo search policies adapted for MLGym.

Implements three search strategies from AIRA-dojo (Meta, arXiv:2507.02554):
  - GreedySearch: draft N → pick best → improve/debug best repeatedly
  - MCTSSearch: UCT selection → expand → backpropagate
  - EvolutionarySearch: population-based with crossover + mutation

All strategies use the same execution infrastructure as our adaptive tree
search: multi-turn ReAct trajectories over MLGym containers, with the same
model (Qwen3-4B via vLLM).

Usage:
    cd /home/ubuntu/MLScientist/MLGym
    uv run --project /home/ubuntu/MLScientist/arts \
        python -m arts.search.aira.search \
        --task-config tasks/titanic.yaml \
        --search-policy greedy \
        --node-budget 12 --max-actions 15 \
        --output-dir outputs/adaptive_search_v3/titanic/aira_greedy.g/run1 \
        --env-gpu 1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

from arts.search.aira.operators import AiraOperators, OperatorType
from arts.reflexion import build_reflection, error_analysis_hint, inject_error_analysis_script
from arts.tree_search import (
    ContainerManager,
    LLMClient,
    TaskProfile,
    TreeNode,
    extract_command,
    classify_execution,
    get_task_profile,
)


# ---------------------------------------------------------------------------
# Base Search — shared infrastructure
# ---------------------------------------------------------------------------

class BaseSearch:
    """Shared infrastructure for all AIRA-dojo search policies."""

    def __init__(
        self,
        llm: LLMClient,
        container: ContainerManager,
        task_profile: TaskProfile,
        operators: AiraOperators,
        node_budget: int,
        max_actions: int,
        output_dir: str,
        verbose: bool = False,
        reflexion: bool = True,
        time_budget: int = 0,
    ):
        self.llm = llm
        self.container = container
        self.task = task_profile
        self.ops = operators
        self.node_budget = node_budget
        self.max_actions = max_actions
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.reflexion = reflexion
        self.time_budget = time_budget  # seconds, 0 = no limit
        self._start_time: float = 0.0  # set in run()
        self.nodes: dict[str, TreeNode] = {}
        self._child_counter: dict[str, int] = {}

    def _time_budget_exceeded(self) -> bool:
        """Check if time budget has been exceeded."""
        if self.time_budget <= 0:
            return False
        elapsed = time.time() - self._start_time
        if elapsed >= self.time_budget:
            print(f"\n--- Time budget reached ({elapsed:.0f}s >= {self.time_budget}s). Stopping. ---")
            return True
        return False

    # --- Root node (baseline) ---

    def _execute_root(self) -> TreeNode:
        """Create baseline root node (no model execution, just snapshot)."""
        data_head = ""
        if self.task.data_head_cmd:
            data_head = self.container.communicate(self.task.data_head_cmd)

        from arts.tree_search import generate_code_outline
        task_desc = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head=data_head,
            code_outline=generate_code_outline(getattr(self.task, "starter_code_host_path", "")),
        )
        messages = [
            {"role": "system", "content": self.task.system_prompt},
            {"role": "user", "content": task_desc},
        ]

        # Inject error analysis script before snapshot so it's baked in
        if self.reflexion:
            inject_error_analysis_script(self.container, self.task)

        snap = self.container.save_snapshot("root")
        baseline = self.container.baseline_score
        try:
            print(f"  [root] Baseline (score={float(baseline):.4f})")
        except (TypeError, ValueError):
            print(f"  [root] Baseline (score={baseline})")

        root = TreeNode(
            node_id="root", parent_id=None, depth=0,
            strategy="Baseline (no model execution)",
            score=baseline, actions=[],
            conversation_history=messages,
            snapshot_path=snap,
        )
        self.nodes[root.node_id] = root
        self._save_node(root)

        return root

    # --- Node expansion ---

    def _next_child_id(self, parent_id: str) -> str:
        if parent_id not in self._child_counter:
            self._child_counter[parent_id] = 0
        idx = self._child_counter[parent_id]
        self._child_counter[parent_id] += 1
        return f"{parent_id}_{idx}"

    def _expand_node(
        self,
        parent_id: str,
        op_type: OperatorType,
        user_message: str,
        second_parent_id: str | None = None,
    ) -> TreeNode | None:
        """Create and execute a single child node.

        Args:
            parent_id: node to branch from (workspace restored from this node)
            op_type: which AIRA-dojo operator is being applied
            user_message: the operator-generated prompt to inject
            second_parent_id: for crossover, the second parent (not used for workspace)
        """
        parent = self.nodes[parent_id]
        child_id = self._next_child_id(parent_id)

        print(f"\n  [{child_id}] op={op_type.value}, parent={parent_id}")

        # Restore parent workspace
        self.container.restore_snapshot(parent.snapshot_path)
        if self.task.submission_file:
            self.container.communicate(
                f"rm -f /home/agent/workspace/{self.task.submission_file}"
            )

        # Tree-level reflection (if enabled and we have scored nodes)
        reflection = ""
        if self.reflexion and len(self.nodes) > 1:
            reflection = build_reflection(
                self.llm, self.nodes, parent_id, self.task,
                baseline_score=self.container.baseline_score,
            )

        # Build conversation: always start from ROOT conversation (system
        # prompt + task description) + the operator message. This keeps context
        # short regardless of tree depth, matching AIRA-dojo's approach where
        # each node gets a fresh context with only the operator-provided info.
        root = self.nodes["root"]
        child_msgs = copy.deepcopy(root.conversation_history)
        write_instr = self.task.branch_write_instruction

        # Prepend reflection to operator message if available
        parts = []
        if reflection:
            parts.append(f"REFLECTION ON PREVIOUS EXPERIMENTS:\n{reflection}")
        parts.append(user_message)
        parts.append(write_instr)
        if self.reflexion:
            hint = error_analysis_hint(self.task)
            if hint:
                parts.append(hint)

        child_msgs.append({
            "role": "user",
            "content": "\n\n".join(parts),
        })

        # Execute multi-turn trajectory
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

        # Extract a short description from the operator message (first line or key phrase)
        first_line = user_message.split('\n')[0][:100]
        strategy_desc = f"{op_type.value}: {first_line}"
        exec_status, err_type = classify_execution(actions, score)
        child = TreeNode(
            node_id=child_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            strategy=strategy_desc,
            score=score,
            actions=actions,
            conversation_history=final_msgs,
            snapshot_path=snap,
            error=error,
            reflection=reflection,
            execution_status=exec_status,
            error_type=err_type,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)
        self._save_node(child)

        if score is not None:
            try:
                print(f"  [{child_id}] score={float(score):.4f}")
            except (TypeError, ValueError):
                print(f"  [{child_id}] score={score}")
        else:
            print(f"  [{child_id}] FAILED (buggy)")

        # Periodic save so crashes don't lose progress
        self._save_intermediate_result()

        # Snapshot cleanup every 10 nodes to prevent tmpfs overflow
        if len(self.nodes) > 15 and len(self.nodes) % 10 == 0:
            self._cleanup_snapshots(keep_top_k=15)

        return child

    # --- Multi-turn execution ---

    def _execute_until_validate(
        self, messages: list[dict], node_id: str
    ) -> tuple[float | None, list[dict], list[dict]]:
        """Execute actions until validate or max_actions."""
        action_log = []
        score = None

        for step in range(self.max_actions):
            try:
                raw = self.llm.chat(messages)
            except Exception as e:
                time.sleep(2)
                try:
                    raw = self.llm.chat(messages)
                except Exception:
                    raise RuntimeError(f"LLM failed: {e}")

            action, _ = extract_command(raw)
            if not action:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "No command detected. Output a valid command.",
                })
                action_log.append({
                    "action": raw[:100], "observation": "No command",
                    "step": step,
                })
                continue

            if action.strip().lower() == "submit":
                action = "validate"

            if self.verbose:
                print(f"    [{node_id}] step {step}: {action[:80]}")

            obs, info = self.container.step(action)

            if self.verbose and action.strip().startswith("python"):
                if "Traceback" in (obs or "") or "Error" in (obs or ""):
                    print(f"    [{node_id}] python ERROR: {(obs or '')[-150:]}")

            action_log.append({
                "action": action[:2000],
                "observation": obs[:2000] if obs else "",
                "step": step,
            })
            # Truncate obs to prevent context overflow (32K model limit)
            obs_for_msg = obs[:8000] if obs else ""
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs_for_msg})

            score_found = self._extract_score(info, obs)
            if score_found is not None:
                score = score_found
                break
        else:
            # Force validate at max actions
            print(f"  [{node_id}] Max actions reached, forcing validate")
            obs, info = self.container.step("validate")
            obs_for_msg = obs[:8000] if obs else ""
            messages.append({"role": "assistant", "content": "validate"})
            messages.append({"role": "user", "content": obs_for_msg})
            action_log.append({
                "action": "validate (forced)",
                "observation": obs[:500],
                "step": self.max_actions,
            })
            score = self._extract_score(info, obs)

        return score, action_log, messages

    def _extract_score(self, info: dict, obs: str) -> float | None:
        """Extract the primary metric score from validate output."""
        import ast
        import re

        def _to_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        if info.get("score"):
            score_data = info["score"][-1]
            if isinstance(score_data, dict):
                val = score_data.get(
                    self.task.primary_metric, list(score_data.values())[0]
                )
                return _to_float(val)
            return _to_float(score_data)

        if obs and "Evaluation Score" in obs:
            m = re.search(r"Evaluation Score:\s*(\{[^}]+\})", obs)
            if m:
                try:
                    score_dict = ast.literal_eval(m.group(1))
                    return _to_float(list(score_dict.values())[0])
                except Exception:
                    pass
        return None

    # --- Memory systems (matching AIRA-dojo) ---

    def _build_simple_memory(self) -> str:
        """Simple memory: summaries of all non-buggy nodes.

        Used by DRAFT and IMPROVE operators. Shows what has been tried,
        scores achieved, and execution status (so executor knows if score
        is genuine vs baseline fallback).
        """
        lines = []
        for nid, node in self.nodes.items():
            if nid == "root":
                continue
            status = node.execution_status or ""
            err_t = node.error_type or ""
            if node.score is not None:
                status_tag = ""
                if status == "training_failed":
                    fallback = ""
                    if (self.container.baseline_score is not None
                            and abs(node.score - self.container.baseline_score) < 0.02):
                        fallback = " [BASELINE FALLBACK — training crashed]"
                    status_tag = f" [training_failed:{err_t or 'unknown'}]{fallback}"
                elif status == "success":
                    status_tag = " [success]"
                lines.append(
                    f"- {node.strategy[:120]} -> Score: {node.score:.4f}{status_tag}"
                )
            elif node.error:
                lines.append(
                    f"- {node.strategy[:120]} -> FAILED ({err_t or status or 'unknown'}): {node.error[:80]}"
                )
        return "\n".join(lines) if lines else "(no previous attempts)"

    def _build_ancestral_memory(self, node_id: str) -> str:
        """Ancestral memory: chain of buggy ancestors for DEBUG.

        Walks up the parent chain collecting buggy nodes so the debugger
        knows what fixes were already attempted.
        """
        lines = []
        current = self.nodes.get(node_id)
        attempt = 1
        while current is not None:
            if current.node_id == "root":
                break
            if current.score is None and current.error:
                lines.append(
                    f"Attempt {attempt} ({current.node_id}): "
                    f"{current.strategy[:100]} -> Error: {current.error[:200]}"
                )
                attempt += 1
            current = (
                self.nodes.get(current.parent_id)
                if current.parent_id else None
            )
        lines.reverse()
        return "\n".join(lines) if lines else "(first debugging attempt)"

    # --- Utility ---

    def _is_buggy(self, node_id: str) -> bool:
        """A node is buggy if it has no valid score."""
        node = self.nodes.get(node_id)
        return node is not None and node.score is None

    def _global_best_score(self) -> float:
        scored = [n.score for n in self.nodes.values() if n.score is not None]
        if not scored:
            return 0.0
        return max(scored) if self.task.higher_is_better else min(scored)

    def _global_best_id(self) -> str | None:
        scored = [
            (nid, n.score) for nid, n in self.nodes.items()
            if n.score is not None
        ]
        if not scored:
            return None
        if self.task.higher_is_better:
            return max(scored, key=lambda x: x[1])[0]
        return min(scored, key=lambda x: x[1])[0]

    # --- Results ---

    def _compile_results(self, start_time: float, policy_name: str) -> dict:
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
            "search_policy": policy_name,
            "best_node_id": best_id,
            "best_score": best_score,
            "baseline_score": self.container.baseline_score,
            "improvement": best_score - self.container.baseline_score,
            "total_nodes": len(self.nodes),
            "elapsed_seconds": round(elapsed, 1),
            "node_budget": self.node_budget,
            "time_budget_seconds": self.time_budget,
            "stopped_by": "time_budget" if self._time_budget_exceeded() else "node_budget",
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        self._print_tree(best_id)
        return result

    def _save_node(self, node: TreeNode):
        nodes_dir = self.output_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        path = nodes_dir / f"{node.node_id}.json"
        data = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "strategy": node.strategy,
            "score": node.score,
            "error": node.error,
            "execution_status": node.execution_status,
            "error_type": node.error_type,
            "actions": node.actions,
            "conversation_length": len(node.conversation_history),
            "reflection": node.reflection or None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_intermediate_result(self):
        """Write result.json with current best, so progress survives crashes."""
        scored = [
            (nid, n.score) for nid, n in self.nodes.items()
            if n.score is not None
        ]
        if not scored:
            return
        if self.task.higher_is_better:
            best_id, best_score = max(scored, key=lambda x: x[1])
        else:
            best_id, best_score = min(scored, key=lambda x: x[1])

        result = {
            "task": self.task.name,
            "best_node_id": best_id,
            "best_score": best_score,
            "baseline_score": self.container.baseline_score,
            "total_nodes": len(self.nodes),
            "node_budget": self.node_budget,
            "time_budget_seconds": self.time_budget,
            "stopped_by": "in_progress",
            "nodes": {
                nid: {
                    "node_id": n.node_id, "score": n.score,
                    "strategy": n.strategy[:100], "children": n.children,
                }
                for nid, n in self.nodes.items()
            },
        }
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

    def _cleanup_snapshots(self, keep_top_k: int = 15):
        """Delete snapshots for nodes unlikely to be expanded, freeing tmpfs."""
        scored = [
            (nid, n.score) for nid, n in self.nodes.items()
            if n.score is not None
        ]
        if len(scored) <= keep_top_k:
            return
        if self.task.higher_is_better:
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored.sort(key=lambda x: x[1])
        top_k_ids = {nid for nid, _ in scored[:keep_top_k]}
        keep = {"root"} | top_k_ids
        for nid in list(top_k_ids):
            cur = nid
            while cur and cur in self.nodes:
                keep.add(cur)
                cur = self.nodes[cur].parent_id
        deleted = 0
        for nid, node in self.nodes.items():
            if nid not in keep and node.snapshot_path:
                try:
                    self.container.communicate(f"rm -f {node.snapshot_path}", timeout=10)
                    node.snapshot_path = ""
                    deleted += 1
                except Exception:
                    pass
        if deleted:
            print(f"  [cleanup] Deleted {deleted} old snapshots, keeping {len(keep)}")

    def _print_tree(self, best_id: str):
        print(f"\n{'=' * 70}")
        print("AIRA-DOJO SEARCH RESULTS")
        print(f"{'=' * 70}")

        best = self.nodes.get(best_id)
        if best and best.score is not None:
            print(
                f"Baseline: {self.container.baseline_score:.4f} | "
                f"Best: {best.score:.4f} (node: {best_id}) | "
                f"Improvement: {best.score - self.container.baseline_score:+.4f}"
            )
        print(f"Nodes explored: {len(self.nodes)}")
        print(f"{'=' * 70}\n")

        def _print_node(nid: str, prefix: str = "", is_last: bool = True):
            n = self.nodes[nid]
            connector = "\\-- " if is_last else "|-- "
            marker = " *** BEST ***" if nid == best_id else ""
            score_str = f"{n.score:.4f}" if n.score is not None else "FAIL"
            strategy_short = n.strategy[:50]
            print(f"{prefix}{connector}{nid} [{score_str}] {strategy_short}{marker}")
            child_prefix = prefix + ("    " if is_last else "|   ")
            for i, cid in enumerate(n.children):
                if cid in self.nodes:
                    _print_node(cid, child_prefix, i == len(n.children) - 1)

        _print_node("root")

        # Best path
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
# Greedy Search
# ---------------------------------------------------------------------------

class GreedySearch(BaseSearch):
    """AIRA-dojo Greedy search: draft N -> pick best -> improve/debug repeatedly.

    Hyperparameters (from paper):
        num_drafts: 5 (initial drafts, capped by node_budget)
        debug_prob: 1.0 (always debug buggy nodes)
        improvement_steps: remaining budget after drafts
    """

    def __init__(self, num_drafts: int = 5, debug_prob: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_drafts = min(num_drafts, self.node_budget)
        self.debug_prob = debug_prob

    def run(self) -> dict:
        start = time.time()
        self._start_time = start
        print("\n" + "=" * 60)
        print("AIRA-DOJO GREEDY SEARCH")
        print(f"  num_drafts={self.num_drafts}, node_budget={self.node_budget}")
        if self.time_budget > 0:
            print(f"  time_budget={self.time_budget}s")
        print("=" * 60)

        # Phase 1: Root
        root = self._execute_root()
        budget_used = 0

        # Phase 2: Generate initial drafts
        print(f"\n--- Phase 2: Generate {self.num_drafts} initial drafts ---")
        for i in range(self.num_drafts):
            if budget_used >= self.node_budget or self._time_budget_exceeded():
                break
            memory = self._build_simple_memory()
            msg = self.ops.draft(memory)
            self._expand_node("root", OperatorType.DRAFT, msg)
            budget_used += 1

        # Phase 3: Iteratively improve/debug the best
        remaining = self.node_budget - budget_used
        print(f"\n--- Phase 3: Improve/debug best ({remaining} steps remaining) ---")

        for step in range(remaining):
            if self._time_budget_exceeded():
                break
            best_id = self._global_best_id()
            if best_id is None or best_id == "root":
                # All drafts failed, try another draft
                memory = self._build_simple_memory()
                msg = self.ops.draft(memory)
                self._expand_node("root", OperatorType.DRAFT, msg)
                continue

            best_node = self.nodes[best_id]

            # Check if any recent child is buggy and needs debugging
            buggy_child = None
            for cid in reversed(best_node.children):
                if self._is_buggy(cid):
                    buggy_child = cid
                    break

            if buggy_child and random.random() < self.debug_prob:
                # DEBUG the buggy child
                buggy = self.nodes[buggy_child]
                ancestral_mem = self._build_ancestral_memory(buggy_child)
                msg = self.ops.debug(
                    buggy_approach=buggy.strategy[:300],
                    error_output=buggy.error or "Unknown error",
                    ancestral_memory=ancestral_mem,
                )
                self._expand_node(best_id, OperatorType.DEBUG, msg)
            else:
                # IMPROVE the best
                memory = self._build_simple_memory()
                msg = self.ops.improve(
                    prev_approach=best_node.strategy[:300],
                    prev_score=best_node.score or 0.0,
                    memory=memory,
                )
                self._expand_node(best_id, OperatorType.IMPROVE, msg)

        return self._compile_results(start, "greedy")


# ---------------------------------------------------------------------------
# MCTS Search
# ---------------------------------------------------------------------------

class MCTSSearch(BaseSearch):
    """AIRA-dojo MCTS with UCT selection and backpropagation.

    Hyperparameters (from paper):
        uct_c: 0.25
        num_children_per_expansion: 5 (capped by node_budget)
        max_debug_depth: 20

    Backpropagation: after expanding a leaf, walk from new node to root,
    incrementing visit_count and cumulative_value at each ancestor.
    UCT uses Q = cumulative_value / visit_count, normalized to [0,1]
    via global min/max scores.
    """

    def __init__(
        self,
        uct_c: float = 0.25,
        num_children_per_expansion: int = 5,
        max_debug_depth: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.uct_c = uct_c
        self.num_children_per_expansion = num_children_per_expansion
        self.max_debug_depth = max_debug_depth
        # MCTS stats
        self._visit_count: dict[str, int] = {}
        self._cumulative_value: dict[str, float] = {}

    def run(self) -> dict:
        start = time.time()
        self._start_time = start
        print("\n" + "=" * 60)
        print("AIRA-DOJO MCTS SEARCH")
        print(f"  uct_c={self.uct_c}, node_budget={self.node_budget}")
        if self.time_budget > 0:
            print(f"  time_budget={self.time_budget}s")
        print("=" * 60)

        # Phase 1: Root
        root = self._execute_root()
        self._visit_count["root"] = 1
        self._cumulative_value["root"] = root.score or 0.0
        budget_used = 0

        # Phase 2: Initial draft(s)
        initial_drafts = min(3, self.node_budget)
        print(f"\n--- Phase 2: Initial drafts ({initial_drafts}) ---")
        for i in range(initial_drafts):
            if budget_used >= self.node_budget or self._time_budget_exceeded():
                break
            memory = self._build_simple_memory()
            msg = self.ops.draft(memory)
            child = self._expand_node("root", OperatorType.DRAFT, msg)
            budget_used += 1
            if child:
                child_score = child.score if child.score is not None else 0.0
                self._visit_count[child.node_id] = 1
                self._cumulative_value[child.node_id] = child_score
                self._backpropagate(child.node_id, child_score)

        # Phase 3: MCTS loop
        remaining = self.node_budget - budget_used
        print(f"\n--- Phase 3: MCTS loop ({remaining} iterations) ---")

        for step in range(remaining):
            if self._time_budget_exceeded():
                break
            print(f"\n  MCTS step {step + 1}/{remaining}")

            # SELECT: traverse from root to leaf via UCT
            selected_id = self._uct_select()
            selected = self.nodes[selected_id]

            # Decide operation
            if selected.score is None and selected.node_id != "root":
                # Buggy node -> DEBUG
                ancestral_mem = self._build_ancestral_memory(selected_id)
                msg = self.ops.debug(
                    buggy_approach=selected.strategy[:300],
                    error_output=selected.error or "Unknown error",
                    ancestral_memory=ancestral_mem,
                )
                op_type = OperatorType.DEBUG
            elif not selected.children:
                # Leaf with valid score -> DRAFT new approach
                memory = self._build_simple_memory()
                msg = self.ops.draft(memory)
                op_type = OperatorType.DRAFT
            else:
                # Internal node -> IMPROVE
                memory = self._build_simple_memory()
                msg = self.ops.improve(
                    prev_approach=selected.strategy[:300],
                    prev_score=selected.score or 0.0,
                    memory=memory,
                )
                op_type = OperatorType.IMPROVE

            child = self._expand_node(selected_id, op_type, msg)
            if child:
                child_score = child.score if child.score is not None else 0.0
                self._visit_count[child.node_id] = 1
                self._cumulative_value[child.node_id] = child_score
                self._backpropagate(child.node_id, child_score)

        return self._compile_results(start, "mcts")

    def _uct_select(self) -> str:
        """Traverse tree from root to leaf using UCT at each level."""
        current = "root"
        while True:
            node = self.nodes[current]
            children_in_tree = [
                c for c in node.children if c in self.nodes
            ]
            if not children_in_tree:
                return current

            # Pick child with highest UCT value
            best_child = max(children_in_tree, key=lambda c: self._uct_value(c))
            current = best_child

    def _uct_value(self, node_id: str) -> float:
        """Compute UCT value for a node: Q_normalized + c * sqrt(ln(N_parent) / N_child)."""
        node = self.nodes[node_id]
        n_child = self._visit_count.get(node_id, 1)
        parent_id = node.parent_id or "root"
        n_parent = self._visit_count.get(parent_id, 1)

        # Q = average value (cumulative / visits), normalized to [0,1]
        q_raw = self._cumulative_value.get(node_id, 0.0) / max(n_child, 1)

        # Normalize Q to [0,1] using global min/max
        all_scores = [
            n.score for n in self.nodes.values()
            if n.score is not None
        ]
        if all_scores:
            s_min, s_max = min(all_scores), max(all_scores)
            if s_max > s_min:
                q_norm = (q_raw - s_min) / (s_max - s_min)
            else:
                q_norm = 0.5
        else:
            q_norm = 0.5

        # Exploration bonus
        explore = self.uct_c * math.sqrt(
            math.log(max(n_parent, 1)) / max(n_child, 1)
        )

        return q_norm + explore

    def _backpropagate(self, leaf_id: str, score: float):
        """Walk from leaf to root, updating visit_count and cumulative_value."""
        current = leaf_id
        while current is not None:
            self._visit_count[current] = self._visit_count.get(current, 0) + 1
            self._cumulative_value[current] = (
                self._cumulative_value.get(current, 0.0) + score
            )
            parent = self.nodes.get(current)
            current = parent.parent_id if parent else None


# ---------------------------------------------------------------------------
# Evolutionary Search
# ---------------------------------------------------------------------------

class EvolutionarySearch(BaseSearch):
    """AIRA-dojo Evolutionary search with crossover + fitness-proportional selection.

    Hyperparameters (from paper):
        num_islands: 1 (single population)
        max_island_size: 500 (effectively unlimited)
        crossover_prob: 0.5
        migration_prob: 0.0 (disabled)
        initial_temp: 1.0, final_temp: 1.0 (no annealing)
        num_generations_till_crossover: 2
        individuals_per_generation: 5 (capped by node_budget)
    """

    def __init__(
        self,
        crossover_prob: float = 0.5,
        num_generations_till_crossover: int = 2,
        individuals_per_generation: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.crossover_prob = crossover_prob
        self.crossover_gen = num_generations_till_crossover
        self.indiv_per_gen = individuals_per_generation
        self._population: list[str] = []  # node_ids in population

    def run(self) -> dict:
        start = time.time()
        self._start_time = start
        print("\n" + "=" * 60)
        print("AIRA-DOJO EVOLUTIONARY SEARCH")
        print(f"  crossover_prob={self.crossover_prob}, "
              f"crossover_gen={self.crossover_gen}, "
              f"node_budget={self.node_budget}")
        if self.time_budget > 0:
            print(f"  time_budget={self.time_budget}s")
        print("=" * 60)

        # Phase 1: Root
        root = self._execute_root()
        budget_used = 0
        generation = 0

        # Phase 2: Initial population (drafts)
        initial_pop = min(self.indiv_per_gen, self.node_budget)
        print(f"\n--- Phase 2: Initial population ({initial_pop} drafts) ---")
        for i in range(initial_pop):
            if budget_used >= self.node_budget or self._time_budget_exceeded():
                break
            memory = self._build_simple_memory()
            msg = self.ops.draft(memory)
            child = self._expand_node("root", OperatorType.DRAFT, msg)
            budget_used += 1
            if child and child.score is not None:
                self._population.append(child.node_id)
        generation += 1

        # Phase 3: Evolutionary loop
        remaining = self.node_budget - budget_used
        print(f"\n--- Phase 3: Evolution ({remaining} evaluations remaining) ---")

        for step in range(remaining):
            if self._time_budget_exceeded():
                break
            print(f"\n  Evo step {step + 1}/{remaining} (gen={generation})")

            # Decide: crossover or improve
            do_crossover = (
                generation >= self.crossover_gen
                and len(self._population) >= 2
                and random.random() < self.crossover_prob
            )

            if do_crossover:
                # CROSSOVER: select two parents
                p1_id, p2_id = self._select_two_parents()
                p1, p2 = self.nodes[p1_id], self.nodes[p2_id]
                msg = self.ops.crossover(
                    approach_1=p1.strategy[:300],
                    score_1=p1.score or 0.0,
                    approach_2=p2.strategy[:300],
                    score_2=p2.score or 0.0,
                )
                child = self._expand_node(
                    p1_id, OperatorType.CROSSOVER, msg,
                    second_parent_id=p2_id,
                )
            else:
                # IMPROVE: select one parent
                parent_id = self._select_parent()
                parent = self.nodes[parent_id]
                memory = self._build_simple_memory()
                msg = self.ops.improve(
                    prev_approach=parent.strategy[:300],
                    prev_score=parent.score or 0.0,
                    memory=memory,
                )
                child = self._expand_node(parent_id, OperatorType.IMPROVE, msg)

            if child and child.score is not None:
                self._population.append(child.node_id)

            # Track generations
            if (step + 1) % self.indiv_per_gen == 0:
                generation += 1

        return self._compile_results(start, "evolutionary")

    def _select_parent(self) -> str:
        """Fitness-proportional selection from population."""
        if not self._population:
            return "root"

        valid = [
            pid for pid in self._population
            if pid in self.nodes and self.nodes[pid].score is not None
        ]
        if not valid:
            return "root"

        scores = [self.nodes[pid].score for pid in valid]
        # Shift scores to be positive for proportional selection
        min_score = min(scores)
        shifted = [s - min_score + 1e-6 for s in scores]

        if self.task.higher_is_better:
            weights = shifted
        else:
            # For minimization, invert
            max_shifted = max(shifted)
            weights = [max_shifted - s + 1e-6 for s in shifted]

        total = sum(weights)
        probs = [w / total for w in weights]

        return random.choices(valid, weights=probs, k=1)[0]

    def _select_two_parents(self) -> tuple[str, str]:
        """Select two different parents for crossover."""
        p1 = self._select_parent()
        # Try to get a different parent
        for _ in range(10):
            p2 = self._select_parent()
            if p2 != p1:
                return p1, p2
        # Fallback: just use same parent twice
        valid = [
            pid for pid in self._population
            if pid in self.nodes and self.nodes[pid].score is not None
            and pid != p1
        ]
        return p1, (valid[0] if valid else p1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SEARCH_POLICIES = {
    "greedy": GreedySearch,
    "mcts": MCTSSearch,
    "evolutionary": EvolutionarySearch,
}


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(
        description="AIRA-dojo search policies over MLGym tasks",
    )
    parser.add_argument(
        "--search-policy", required=True,
        choices=list(SEARCH_POLICIES.keys()),
        help="Search policy to use",
    )
    parser.add_argument("--node-budget", type=int, default=12)
    parser.add_argument("--time-budget", type=int, default=0,
                        help="Max seconds for search (0 = no limit)")
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--task-config", default="tasks/titanic.yaml")
    parser.add_argument("--output-dir", default="outputs/aira_dojo")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--env-gpu", default="7")
    parser.add_argument("--image-name", default="aigym/mlgym-agent:latest")
    parser.add_argument("--verbose", action="store_true")

    # Reflexion
    parser.add_argument("--reflexion", action="store_true", default=True,
                        help="Enable tree-level reflection (default: on)")
    parser.add_argument("--no-reflexion", dest="reflexion", action="store_false",
                        help="Disable tree-level reflection")
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="Thinking budget tokens for Claude models (0 = disabled)")

    # MCTS-specific
    parser.add_argument("--uct-c", type=float, default=0.25)
    parser.add_argument("--num-children", type=int, default=5)
    parser.add_argument("--max-debug-depth", type=int, default=20)

    # Greedy-specific
    parser.add_argument("--num-drafts", type=int, default=5)
    parser.add_argument("--debug-prob", type=float, default=1.0)

    # Evolutionary-specific
    parser.add_argument("--crossover-prob", type=float, default=0.5)
    parser.add_argument("--crossover-gen", type=int, default=2)
    parser.add_argument("--indiv-per-gen", type=int, default=5)

    args = parser.parse_args()

    task_profile = get_task_profile(args.task_config)

    print("=" * 60)
    print(f"AIRA-DOJO Search: {args.search_policy}")
    print(f"Task: {task_profile.name}")
    print(f"Model: {args.model}")
    time_str = f"{args.time_budget}s" if args.time_budget > 0 else "unlimited"
    print(f"Node budget: {args.node_budget}, Max actions: {args.max_actions}, Time budget: {time_str}")
    print(f"Temperature: {args.temperature}")
    print(f"Reflexion: {'enabled' if args.reflexion else 'disabled'}")
    print("=" * 60)

    llm = LLMClient(args.vllm_url, args.model, args.temperature,
                    thinking_budget=args.thinking_budget)
    container = ContainerManager(
        args.task_config, args.env_gpu, args.image_name,
        task_profile=task_profile,
    )

    print("Creating MLGym container...")
    container.create()

    # Build task description and data overview for operators
    from arts.tree_search import generate_code_outline
    task_desc = task_profile.root_task_desc.format(
        baseline_score=container.baseline_score,
        data_head="",
        code_outline=generate_code_outline(getattr(task_profile, "starter_code_host_path", "")),
    )
    data_overview = ""
    if task_profile.data_head_cmd:
        data_overview = container.communicate(task_profile.data_head_cmd)

    operators = AiraOperators(task_desc=task_desc, data_overview=data_overview)

    # Common kwargs
    base_kwargs = dict(
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
    )

    if args.search_policy == "greedy":
        search = GreedySearch(
            num_drafts=args.num_drafts,
            debug_prob=args.debug_prob,
            **base_kwargs,
        )
    elif args.search_policy == "mcts":
        search = MCTSSearch(
            uct_c=args.uct_c,
            num_children_per_expansion=args.num_children,
            max_debug_depth=args.max_debug_depth,
            **base_kwargs,
        )
    elif args.search_policy == "evolutionary":
        search = EvolutionarySearch(
            crossover_prob=args.crossover_prob,
            num_generations_till_crossover=args.crossover_gen,
            individuals_per_generation=args.indiv_per_gen,
            **base_kwargs,
        )
    else:
        raise ValueError(f"Unknown policy: {args.search_policy}")

    try:
        result = search.run()
        print(f"\nResults saved to {args.output_dir}/result.json")
        print(f"Best score: {result['best_score']:.4f} "
              f"(improvement: {result['improvement']:+.4f})")
    finally:
        container.close()


if __name__ == "__main__":
    main()
