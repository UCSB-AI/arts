"""
Tree Search Viewer - Streamlit app for visualizing tree search results.

Usage:
    cd /home/ubuntu/MLScientist/arts
    uv run streamlit run air/tree_viewer.py -- --dir /home/ubuntu/MLScientist/MLGym/outputs/tree_search

Then SSH port forward: ssh -L 8501:localhost:8501 user@server
Open http://localhost:8501 in your browser
"""

import json
import sys
from pathlib import Path

import streamlit as st

DEFAULT_DIR = "/home/ubuntu/MLScientist/MLGym/outputs/tree_search"


def load_tree(tree_dir: str) -> tuple[dict | None, dict[str, dict]]:
    """Load result.json and all node JSONs."""
    tree_dir = Path(tree_dir)
    result = None
    nodes = {}

    result_path = tree_dir / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            result = json.load(f)

    nodes_dir = tree_dir / "nodes"
    if nodes_dir.exists():
        for file in sorted(nodes_dir.glob("*.json")):
            try:
                with open(file) as f:
                    data = json.load(f)
                    nodes[data["node_id"]] = data
            except (json.JSONDecodeError, KeyError):
                continue

    return result, nodes


def score_color(score: float | None, baseline: float) -> str:
    """Return color based on score relative to baseline."""
    if score is None:
        return "#888888"
    improvement = score - baseline
    if improvement > 0.15:
        return "#00cc00"  # bright green
    elif improvement > 0.10:
        return "#33aa33"  # green
    elif improvement > 0.05:
        return "#66aa00"  # yellow-green
    elif improvement > 0.0:
        return "#aaaa00"  # yellow
    else:
        return "#cc3333"  # red


def score_bar(score: float | None, baseline: float, max_score: float) -> str:
    """Return an HTML progress bar for the score."""
    if score is None:
        return '<span style="color: #888">FAIL</span>'
    pct = max(0, min(100, (score - baseline) / max(0.001, max_score - baseline) * 100))
    color = score_color(score, baseline)
    return (
        f'<div style="background: #333; border-radius: 4px; height: 20px; width: 100%;">'
        f'<div style="background: {color}; width: {pct:.0f}%; height: 100%; border-radius: 4px; '
        f'display: flex; align-items: center; justify-content: center; min-width: 60px;">'
        f'<span style="color: white; font-size: 12px; font-weight: bold;">{score:.4f}</span>'
        f'</div></div>'
    )


def render_tree_node(node_id: str, nodes: dict, result: dict, depth: int = 0,
                     is_last: bool = True, prefix: str = ""):
    """Render a single tree node with its children recursively."""
    if node_id not in nodes:
        return

    node = nodes[node_id]
    baseline = result.get("baseline_score", 0.76555)
    best_id = result.get("best_node_id", "")
    score = node.get("score")
    strategy = node.get("strategy", "")[:80]
    is_best = node_id == best_id

    # Build the display
    connector = "└── " if is_last else "├── "
    tree_prefix = prefix + connector if depth > 0 else ""

    score_str = f"{score:.4f}" if score is not None else "FAIL"
    color = score_color(score, baseline)
    best_marker = " ⭐" if is_best else ""

    st.markdown(
        f'<code style="font-size: 14px;">{tree_prefix}</code>'
        f'<span style="color: {color}; font-weight: {"bold" if is_best else "normal"}; font-size: 14px;">'
        f'{node_id} [{score_str}]{best_marker}</span> '
        f'<span style="color: #aaa; font-size: 13px;">{strategy}</span>',
        unsafe_allow_html=True,
    )

    # Recurse children
    children = node.get("children", [])
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child_id in enumerate(children):
        render_tree_node(child_id, nodes, result, depth + 1,
                         i == len(children) - 1, child_prefix)


def display_node_detail(node: dict, baseline: float):
    """Display detailed view of a single node."""
    score = node.get("score")
    strategy = node.get("strategy", "N/A")
    error = node.get("error")
    actions = node.get("actions", [])

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score", f"{score:.4f}" if score else "FAIL")
    col2.metric("Improvement", f"+{(score - baseline)*100:.1f}%" if score else "N/A")
    col3.metric("Actions", len(actions))
    col4.metric("Depth", node.get("depth", 0))

    # Strategy
    st.markdown(f"**Strategy:** {strategy}")
    if error:
        st.error(f"Error: {error}")

    st.markdown("---")

    # Actions timeline
    st.markdown("### Actions")

    for action_data in actions:
        step = action_data.get("step", "?")
        action = action_data.get("action", "")
        obs = action_data.get("observation", "")

        # Determine action type for icon/color
        if action.startswith("cat <<"):
            action_type = "write"
            icon = "📝"
            short = "Write train_and_predict.py"
        elif action.startswith("python"):
            action_type = "run"
            icon = "▶️"
            short = action[:60]
        elif "validate" in action:
            action_type = "validate"
            icon = "✅" if "Evaluation Score" in obs else "❌"
            short = action
        elif action.startswith("ls") or action.startswith("cat ") or action.startswith("head"):
            action_type = "explore"
            icon = "🔍"
            short = action[:60]
        elif action.startswith("pip"):
            action_type = "install"
            icon = "📦"
            short = action[:60]
        else:
            action_type = "other"
            icon = "⚙️"
            short = action[:60]

        # Determine if error
        has_error = "Traceback" in obs or "Error" in obs or "not found" in obs.lower()
        title_color = "#cc3333" if has_error else "#cccccc"

        with st.expander(f"{icon} Step {step}: {short}", expanded=False):
            st.markdown("**Command:**")
            # For heredoc, show the full script
            if action.startswith("cat <<"):
                st.code(action, language="bash")
            else:
                st.code(action, language="bash")

            st.markdown("**Output:**")
            if has_error:
                st.error(obs[:1500] if obs else "(empty)")
            elif "Evaluation Score" in obs:
                st.success(obs[:500])
            else:
                st.code(obs[:1500] if obs else "(empty)", language=None)


def main():
    st.set_page_config(page_title="Tree Search Viewer", layout="wide")
    st.title("Tree Search Viewer")

    # Parse CLI args for default dir
    default_dir = DEFAULT_DIR
    for i, arg in enumerate(sys.argv):
        if arg == "--dir" and i + 1 < len(sys.argv):
            default_dir = sys.argv[i + 1]

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        tree_dir = st.text_input("Tree Search Directory", default_dir)

        if st.button("Refresh"):
            st.rerun()

        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("- [experiments2.md](file:///home/ubuntu/MLScientist/arts/experiments2.md)")

    # Load data
    result, nodes = load_tree(tree_dir)

    if not result or not nodes:
        st.warning(f"No tree search results found in `{tree_dir}`")
        st.info("Run a tree search first, then refresh.")
        return

    baseline = result.get("baseline_score", 0.76555)
    best_id = result.get("best_node_id", "root")
    best_score = result.get("best_score", 0)

    # Summary bar
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Baseline", f"{baseline:.4f}")
    col2.metric("Best Score", f"{best_score:.4f}" if best_score else "N/A")
    col3.metric("Improvement", f"+{(best_score - baseline)*100:.1f}%" if best_score else "N/A")
    col4.metric("Nodes", result.get("total_nodes", len(nodes)))
    col5.metric("Runtime", f"{result.get('elapsed_seconds', 0):.0f}s")

    st.markdown("---")

    # Tabs: Tree view + Node detail
    tab_tree, tab_table, tab_detail = st.tabs(["Tree View", "Score Table", "Node Detail"])

    # --- Tab 1: Tree View ---
    with tab_tree:
        st.markdown("### Search Tree")
        render_tree_node("root", nodes, result)

        # Best path
        if best_id and best_id in nodes:
            st.markdown("---")
            st.markdown("### Best Path")
            path = []
            nid = best_id
            while nid and nid in nodes:
                path.append(nid)
                nid = nodes[nid].get("parent_id")
            path.reverse()
            st.markdown(" → ".join(
                f"**{p}** [{nodes[p].get('score', 0):.4f}]" for p in path
            ))

    # --- Tab 2: Score Table ---
    with tab_table:
        st.markdown("### All Nodes by Score")

        # Sort nodes by score (descending), None scores last
        sorted_nodes = sorted(
            nodes.values(),
            key=lambda n: n.get("score") or 0,
            reverse=True,
        )

        # Table header
        header_cols = st.columns([1, 2, 1, 1, 1, 1])
        header_cols[0].markdown("**Node**")
        header_cols[1].markdown("**Strategy**")
        header_cols[2].markdown("**Score**")
        header_cols[3].markdown("**Improvement**")
        header_cols[4].markdown("**Actions**")
        header_cols[5].markdown("**Depth**")

        for node in sorted_nodes:
            nid = node["node_id"]
            score = node.get("score")
            is_best = nid == best_id
            cols = st.columns([1, 2, 1, 1, 1, 1])

            name = f"⭐ {nid}" if is_best else nid
            cols[0].markdown(f"**{name}**" if is_best else name)
            cols[1].markdown(node.get("strategy", "")[:60])

            if score is not None:
                color = score_color(score, baseline)
                cols[2].markdown(
                    f'<span style="color: {color}; font-weight: bold;">{score:.4f}</span>',
                    unsafe_allow_html=True,
                )
                imp = (score - baseline) * 100
                cols[3].markdown(f"+{imp:.1f}%")
            else:
                cols[2].markdown("FAIL")
                cols[3].markdown("N/A")

            cols[4].markdown(str(len(node.get("actions", []))))
            cols[5].markdown(str(node.get("depth", 0)))

        # Score distribution chart
        st.markdown("---")
        st.markdown("### Score Distribution")
        scores = [n.get("score") for n in sorted_nodes if n.get("score") is not None]
        if scores:
            import pandas as pd
            chart_data = pd.DataFrame({
                "node": [n["node_id"] for n in sorted_nodes if n.get("score") is not None],
                "score": scores,
            })
            chart_data = chart_data.set_index("node")
            st.bar_chart(chart_data, horizontal=True)

    # --- Tab 3: Node Detail ---
    with tab_detail:
        st.markdown("### Node Details")
        selected_node = st.selectbox(
            "Select node",
            list(nodes.keys()),
            format_func=lambda nid: (
                f"{'⭐ ' if nid == best_id else ''}{nid} "
                f"[{nodes[nid].get('score', 0):.4f if nodes[nid].get('score') else 'FAIL'}] "
                f"- {nodes[nid].get('strategy', '')[:50]}"
            ),
        )

        if selected_node:
            display_node_detail(nodes[selected_node], baseline)


if __name__ == "__main__":
    main()
