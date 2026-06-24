# MLEvolve Search Policy for MLGym

Re-implementation of [MLEvolve](https://github.com/InternScience/MLEvolve)
(InternScience, ranked #1 on the MLE-bench leaderboard) adapted for our
multi-turn ReAct executor over MLGym containers.

## What This Is

MLEvolve performs **Monte Carlo Graph Search (MCGS)** with four mechanisms
that distinguish it from vanilla AIRA-MCTS (`air.aira_dojo.search.MCTSSearch`):

1. **Piecewise exploration decay** — `uct_c` starts high (broad search) and
   is multiplied by `decay_factor` at planned milestones (default: at 33%
   and 66% of node budget, halve each time).
2. **Time-aware explore→exploit** — when wall-clock elapsed exceeds
   `exploit_after_frac` of `time_budget`, exploration is forced to 0
   (pure greedy exploitation in the final stretch).
3. **Stagnation detection** — if global best has not improved in
   `stagnation_window` consecutive expansions, the next step is forced
   to be a cross-branch fusion.
4. **Cross-branch fusion** — pick the top-scoring node from each of two
   distinct root-children subtrees and merge them via the AIRA-dojo
   CROSSOVER operator. Falls back to top-2 fitness pairs when only one
   branch has scored nodes.

## What's Reused vs New

| Component | Source |
|-----------|--------|
| `BaseSearch` infrastructure | `air.aira_dojo.search.BaseSearch` |
| DRAFT/IMPROVE/DEBUG/CROSSOVER prompts | `air.aira_dojo.prompts` |
| `AiraOperators` wrapper | `air.aira_dojo.operators` |
| Multi-turn executor + container | `air.tree_search` |
| **MLEvolve search policy** | `air.mlevolve.search.MLEvolveSearch` (new) |

No prompt or operator code is duplicated. The only new code is the
search-policy class that subclasses `BaseSearch`.

## Adaptations from Original

The upstream MLEvolve repository pairs MCGS with three further systems
that we **do not** reimplement, because they're orthogonal to the
search-policy comparison the paper makes:

| Upstream MLEvolve | This re-implementation |
|-------------------|------------------------|
| Multi-agent (planner / coder / feedback) over OpenAI-compatible APIs | Single-executor ReAct loop matching our LLMG/AIRA setup |
| BM25 + FAISS retrieval over global memory | In-process simple memory (string summary of all scored nodes) |
| Three code-generation strategies (single-pass, multi-agent, SEARCH/REPLACE diff) | Multi-turn ReAct (executor chooses commands one at a time) |
| MLE-Bench data-loader directly | MLGym task profile (same as AIRA / LLMG) |

The **search algorithm** (UCT + decay + stagnation + cross-branch fusion)
is preserved. This keeps the comparison apples-to-apples: only the search
policy differs between AIRA, MLEvolve, and our LLM-guided method —
executor, task interface, validator, and node accounting are held constant.

## Usage

```bash
python -m air.mlevolve.search \
    --task-config tasks/mlebenchKuzushiji.yaml \
    --node-budget 100 \
    --max-actions 50 \
    --time-budget 27600 \
    --model gemini-3-pro-preview \
    --output-dir outputs/mlevolve_kuz_run1 \
    --env-gpu 0
```

For SLURM submission see `run_mlebench_mlevolve.sh`.

## Hyperparameters

| Flag | Default | Notes |
|------|---------|-------|
| `--uct-c-init` | 0.5 | Initial exploration coefficient (2× AIRA's 0.25) |
| `--decay-milestones` | `0.33,0.66` | Fractions of node budget where `uct_c` is multiplied by `decay_factor` |
| `--decay-factor` | 0.5 | Multiplier applied at each milestone |
| `--exploit-after-frac` | 0.8 | Force exploration to 0 after this fraction of `time_budget` elapses |
| `--stagnation-window` | 4 | Consecutive non-improving expansions before forcing FUSION |
| `--initial-drafts` | 3 | Number of root children before MCGS loop begins |

## Reference

InternScience MLEvolve (#1 on MLE-bench): <https://github.com/InternScience/MLEvolve>
