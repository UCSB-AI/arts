#!/usr/bin/env python3
"""
Simple terminal trajectory viewer.

Usage:
    # List all trajectories
    uv run python air/view_trajectory.py --list

    # View a specific trajectory
    uv run python air/view_trajectory.py --file titanic_0_20260205_174124.json

    # View latest trajectory
    uv run python air/view_trajectory.py --latest

    # Compare two trajectories
    uv run python air/view_trajectory.py --compare file1.json file2.json
"""

import argparse
import json
from pathlib import Path

DEFAULT_DIR = "/home/ubuntu/MLScientist/MLGym/outputs/trajectories_exp1.2"


def truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s


def extract_action(completion: list) -> str:
    if not completion:
        return "(no action)"
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "<think>" in content:
                content = content.split("</think>")[-1]
            return content.strip()
    return "(no action)"


def extract_feedback_from_next_prompt(next_turn: dict | None) -> str:
    """Extract env feedback from the next turn's prompt (last user message)."""
    if not next_turn:
        return "(end of episode)"

    prompt = next_turn.get("prompt", [])
    if not prompt:
        return "(no prompt)"

    # Find the last user message - that's the env response
    user_msgs = [m for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
    if user_msgs:
        return user_msgs[-1].get("content", "(empty)")

    return "(no user message)"


def view_trajectory(traj: dict, filename: str = "", verbose: bool = False):
    print("=" * 70)
    print(f"File: {filename}")
    print(f"Task: {traj.get('task', '?')} | Example: {traj.get('example_id', '?')}")

    m = traj.get("metrics", {})
    print(f"Baseline: {m.get('baseline_score')} | Final: {m.get('final_score')} | Improvement: {m.get('improvement')}")
    print(f"Validation history: {m.get('validation_history', [])}")
    print("=" * 70)

    trajectory = traj.get("trajectory", [])
    print(f"\nTotal steps: {len(trajectory)}\n")

    for i, turn in enumerate(trajectory):
        action = extract_action(turn.get("completion", []))
        # Get feedback from NEXT turn's prompt
        next_turn = trajectory[i + 1] if i + 1 < len(trajectory) else None
        feedback = extract_feedback_from_next_prompt(next_turn)
        reward = turn.get("reward", None)

        if verbose:
            print(f"--- Step {i+1} ---")
            print(f"ACTION:\n{action}")
            print(f"\nFEEDBACK:\n{feedback}")
            print(f"\nREWARD: {reward}")
            print()
        else:
            print(f"[{i+1:2d}] Action: {truncate(action, 60)}")
            print(f"     Feedback: {truncate(feedback, 60)}")
            if reward is not None:
                print(f"     Reward: {reward}")
            print()


def list_trajectories(traj_dir: Path):
    files = sorted(traj_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"Trajectories in {traj_dir} ({len(files)} total):\n")
    print(f"{'Filename':<45} {'Turns':>6} {'Baseline':>10} {'Final':>10} {'Impr':>8}")
    print("-" * 85)

    for f in files:
        with open(f) as fp:
            t = json.load(fp)
        m = t.get("metrics", {})
        turns = len(t.get("trajectory", []))
        baseline = m.get("baseline_score", "N/A")
        final = m.get("final_score", "N/A")
        impr = m.get("improvement", "N/A")

        if isinstance(baseline, float):
            baseline = f"{baseline:.4f}"
        if isinstance(final, float):
            final = f"{final:.4f}"
        if isinstance(impr, float):
            impr = f"{impr:.4f}"

        print(f"{f.name:<45} {turns:>6} {baseline:>10} {final:>10} {impr:>8}")


def compare_trajectories(traj1: dict, traj2: dict, name1: str, name2: str):
    print("=" * 140)
    print(f"COMPARING: {name1} vs {name2}")
    print("=" * 140)

    m1 = traj1.get("metrics", {})
    m2 = traj2.get("metrics", {})
    t1 = traj1.get("trajectory", [])
    t2 = traj2.get("trajectory", [])

    print(f"\n{'Metric':<20} {'Trajectory 1':>20} {'Trajectory 2':>20}")
    print("-" * 62)
    print(f"{'Task':<20} {traj1.get('task', '?'):>20} {traj2.get('task', '?'):>20}")
    print(f"{'Example ID':<20} {str(traj1.get('example_id', '?')):>20} {str(traj2.get('example_id', '?')):>20}")
    print(f"{'Turns':<20} {len(t1):>20} {len(t2):>20}")
    print(f"{'Baseline':<20} {str(m1.get('baseline_score', 'N/A')):>20} {str(m2.get('baseline_score', 'N/A')):>20}")
    print(f"{'Final':<20} {str(m1.get('final_score', 'N/A')):>20} {str(m2.get('final_score', 'N/A')):>20}")
    print(f"{'Improvement':<20} {str(m1.get('improvement', 'N/A')):>20} {str(m2.get('improvement', 'N/A')):>20}")

    print("\n" + "=" * 140)
    print("STEP-BY-STEP COMPARISON")
    print("=" * 140)

    max_steps = max(len(t1), len(t2))

    for i in range(max_steps):
        print(f"\n--- Step {i+1} ---")

        # Trajectory 1
        if i < len(t1):
            a1 = truncate(extract_action(t1[i].get("completion", [])), 60)
            next1 = t1[i + 1] if i + 1 < len(t1) else None
            f1 = truncate(extract_feedback_from_next_prompt(next1), 60)
        else:
            a1 = "(ended)"
            f1 = ""

        # Trajectory 2
        if i < len(t2):
            a2 = truncate(extract_action(t2[i].get("completion", [])), 60)
            next2 = t2[i + 1] if i + 1 < len(t2) else None
            f2 = truncate(extract_feedback_from_next_prompt(next2), 60)
        else:
            a2 = "(ended)"
            f2 = ""

        print(f"T1 Action:   {a1}")
        print(f"T1 Feedback: {f1}")
        print(f"T2 Action:   {a2}")
        print(f"T2 Feedback: {f2}")


def main():
    parser = argparse.ArgumentParser(description="View MLGym trajectories")
    parser.add_argument("--dir", type=str, default=DEFAULT_DIR, help="Trajectory directory")
    parser.add_argument("--list", action="store_true", help="List all trajectories")
    parser.add_argument("--file", type=str, help="View specific trajectory file")
    parser.add_argument("--latest", action="store_true", help="View latest trajectory")
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"), help="Compare two trajectories")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full action/feedback text")

    args = parser.parse_args()
    traj_dir = Path(args.dir)

    if not traj_dir.exists():
        print(f"Directory not found: {traj_dir}")
        return

    if args.list:
        list_trajectories(traj_dir)
        return

    if args.latest:
        files = sorted(traj_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if files:
            with open(files[0]) as f:
                view_trajectory(json.load(f), files[0].name, args.verbose)
        else:
            print("No trajectories found")
        return

    if args.file:
        filepath = traj_dir / args.file
        if filepath.exists():
            with open(filepath) as f:
                view_trajectory(json.load(f), args.file, args.verbose)
        else:
            print(f"File not found: {filepath}")
        return

    if args.compare:
        f1, f2 = args.compare
        p1 = traj_dir / f1
        p2 = traj_dir / f2

        if not p1.exists():
            print(f"File not found: {p1}")
            return
        if not p2.exists():
            print(f"File not found: {p2}")
            return

        with open(p1) as f:
            t1 = json.load(f)
        with open(p2) as f:
            t2 = json.load(f)

        compare_trajectories(t1, t2, f1, f2)
        return

    # Default: list
    list_trajectories(traj_dir)


if __name__ == "__main__":
    main()
