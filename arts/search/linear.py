"""Linear search: single continuous ReAct trajectory (no branching).

Baseline for comparison against tree search and AIRA-dojo. Uses the same
LLMClient, ContainerManager, and TaskProfile as tree_search.py but runs
a single long trajectory with total_budget = node_budget * max_actions.

Usage:
    cd /home/ubuntu/MLScientist/MLGym
    uv run --project /home/ubuntu/MLScientist/arts \
        python -m arts.search.linear \
        --task-config tasks/titanic.yaml \
        --max-actions 180 \
        --output-dir outputs/adaptive_search_v3/titanic/gpt4o_linear/run1 \
        --model gpt-4o --vllm-url "" --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from arts.tree_search import (
    ContainerManager,
    LLMClient,
    extract_command,
    get_task_profile,
)


def run_linear(
    llm: LLMClient,
    container: ContainerManager,
    task_config: str,
    max_actions: int,
    output_dir: str,
    verbose: bool = False,
) -> dict:
    task = get_task_profile(task_config)
    container.create()

    data_head = ""
    if task.data_head_cmd:
        data_head = container.communicate(task.data_head_cmd)

    from arts.tree_search import generate_code_outline
    task_desc = task.root_task_desc.format(
        baseline_score=container.baseline_score,
        data_head=data_head,
        code_outline=generate_code_outline(getattr(task, "starter_code_host_path", "")),
    )
    messages = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": task_desc + "\n\n" + task.branch_write_instruction},
    ]

    baseline = container.baseline_score
    best_score = None
    all_scores: list[float] = []
    action_log: list[dict] = []
    start = time.time()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _persist():
        """Save actions + running result so slurm-killed runs still have logs."""
        try:
            (out / "actions.json").write_text(json.dumps(action_log, indent=2))
            partial = {
                "task": task.name,
                "primary_metric": task.primary_metric,
                "higher_is_better": task.higher_is_better,
                "search_policy": "linear",
                "best_score": best_score if best_score is not None else baseline,
                "baseline_score": baseline,
                "total_actions": len(action_log),
                "all_scores": all_scores,
                "partial": True,
            }
            (out / "result.json").write_text(json.dumps(partial, indent=2))
        except Exception:
            pass

    for step in range(max_actions):
        try:
            raw = llm.chat(messages)
        except Exception as e:
            print(f"  LLM error at step {step}: {e}")
            break

        action, _ = extract_command(raw)
        if not action:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "No command detected. Output a valid command."})
            action_log.append({"action": raw[:100], "observation": "No command", "step": step})
            continue

        if action.strip().lower() == "submit":
            action = "validate"

        if verbose:
            print(f"  step {step}: {action[:80]}")

        obs, info = container.step(action)

        action_log.append({
            "action": action[:2000],
            "observation": (obs or "")[:2000],
            "step": step,
        })

        obs_for_msg = (obs or "")[:8000]
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": obs_for_msg})

        # Check for score
        score = _extract_score(info, obs, task.primary_metric)
        if score is not None:
            all_scores.append(score)
            if best_score is None or (
                task.higher_is_better and score > best_score
            ) or (
                not task.higher_is_better and score < best_score
            ):
                best_score = score
            print(f"  step {step}: validate -> {score:.4f} (best={best_score:.4f})")

        # Persist every 5 actions so slurm-killed runs retain full action log
        if step % 5 == 4:
            _persist()

    elapsed = time.time() - start

    result = {
        "task": task.name,
        "primary_metric": task.primary_metric,
        "higher_is_better": task.higher_is_better,
        "search_policy": "linear",
        "best_score": best_score if best_score is not None else baseline,
        "baseline_score": baseline,
        "improvement": (best_score - baseline) if best_score is not None else 0.0,
        "total_actions": len(action_log),
        "max_actions": max_actions,
        "all_scores": all_scores,
        "elapsed_seconds": round(elapsed, 1),
    }

    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "actions.json").write_text(json.dumps(action_log, indent=2))

    print(f"\n{'=' * 60}")
    print(f"LINEAR SEARCH RESULTS")
    print(f"{'=' * 60}")
    print(f"Baseline: {baseline:.4f}")
    print(f"Best:     {best_score:.4f}" if best_score is not None else "Best:     N/A")
    if best_score is not None:
        print(f"Improvement: {best_score - baseline:+.4f}")
    print(f"Actions:  {len(action_log)}/{max_actions}")
    print(f"Scores found: {len(all_scores)}")
    print(f"Elapsed:  {elapsed:.0f}s")
    print(f"{'=' * 60}\n")

    return result


def _extract_score(info: dict, obs: str, primary_metric: str) -> float | None:
    import ast
    import re

    if info.get("score"):
        score_data = info["score"][-1]
        if isinstance(score_data, dict):
            return score_data.get(primary_metric, list(score_data.values())[0])
        return score_data

    if obs and "Evaluation Score" in obs:
        m = re.search(r"Evaluation Score:\s*(\{[^}]+\})", obs)
        if m:
            try:
                score_dict = ast.literal_eval(m.group(1))
                return list(score_dict.values())[0]
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Linear search (single trajectory)")
    parser.add_argument("--max-actions", type=int, default=180)
    parser.add_argument("--task-config", default="tasks/titanic.yaml")
    parser.add_argument("--output-dir", default="outputs/linear_search")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--env-gpu", default="7")
    parser.add_argument("--image-name", default="aigym/mlgym-agent:latest")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    task = get_task_profile(args.task_config)
    print("=" * 60)
    print(f"LINEAR SEARCH")
    print(f"Task: {task.name}")
    print(f"Model: {args.model}")
    print(f"Max actions: {args.max_actions}")
    print(f"Temperature: {args.temperature}")
    print("=" * 60)

    llm = LLMClient(args.vllm_url, args.model, args.temperature)
    container = ContainerManager(
        args.task_config, args.env_gpu, args.image_name, task_profile=task,
    )

    try:
        run_linear(
            llm=llm,
            container=container,
            task_config=args.task_config,
            max_actions=args.max_actions,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    finally:
        container.close()


if __name__ == "__main__":
    main()
