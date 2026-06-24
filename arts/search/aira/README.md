# AIRA-dojo Search Policies for MLGym

Re-implementation of [AIRA-dojo](https://github.com/facebookresearch/aira-dojo) (Meta, arXiv:2507.02554) search policies adapted for MLGym multi-turn execution.

## What This Is

AIRA-dojo implements three search strategies for AI research agents:
- **Greedy**: Draft N solutions, pick the best, iteratively improve/debug
- **MCTS**: UCT tree selection with backpropagation
- **Evolutionary**: Population-based with crossover and fitness-proportional selection

We use these as comparison baselines against our UCB adaptive tree search. Same model (Qwen3-4B via vLLM), same tasks (Titanic, MountainCar, MetaMaze via MLGym), same node budgets.

## Adaptations from Original

| Aspect | Original AIRA-dojo | Our Adaptation |
|--------|-------------------|----------------|
| Execution | Single LLM call → complete Python script | Multi-turn ReAct trajectories (10-20 steps per node) |
| Container | Apptainer (Singularity) | Docker via MLGym |
| Prompts | Kaggle Grandmaster persona | ML research agent persona |
| Analysis | Separate ANALYZE LLM call for bug detection | Score extraction from `validate` command output |
| Environment | H200 GPU, mlebench tasks | MLGym tasks with task-specific containers |

The core search logic (selection policies, operators, memory system, backpropagation) is preserved from the paper.

## Search Policies

### Greedy
1. Generate `num_drafts` (5) initial solutions from root
2. Pick the best non-buggy solution
3. Iteratively: if buggy → DEBUG, else → IMPROVE the best

### MCTS
1. Generate initial drafts as root children
2. SELECT: traverse root→leaf using UCT (Q_normalized + c*sqrt(ln(N_parent)/N_child))
3. EXPAND: DRAFT (leaf) or IMPROVE (internal node) or DEBUG (buggy)
4. BACKPROPAGATE: walk new node→root, updating visit_count and cumulative_value

### Evolutionary
1. Generate initial population via DRAFT
2. Each step: with probability `crossover_prob`, CROSSOVER two parents; else IMPROVE one parent
3. Parents selected via fitness-proportional selection
4. Crossover enabled after `num_generations_till_crossover` generations

## Hyperparameters (from paper)

### MCTS
- `uct_c = 0.25`
- `num_children_per_expansion = 5`
- `max_debug_depth = 20`

### Greedy
- `num_drafts = 5`
- `debug_prob = 1.0`

### Evolutionary
- `crossover_prob = 0.5`
- `num_generations_till_crossover = 2`
- `individuals_per_generation = 5`

### Shared
- LLM temperature: 0.6 (draft/improve/debug)
- Node budgets match adaptive search: 12 (titanic), 8 (RL tasks)
- Max actions per node: 15 (titanic), 20 (RL tasks)

## Operators

Four operator types generate the initial context for each node's trajectory:

- **DRAFT**: Generate a novel solution from scratch. Uses complexity cycling (simple → normal → complex)
- **IMPROVE**: Refine an existing solution. Also uses complexity cycling
- **DEBUG**: Fix buggy code using ancestral memory (chain of failed attempts)
- **CROSSOVER**: Combine two solutions into one

## Memory System

- **Simple memory** (DRAFT/IMPROVE): Summaries of all non-buggy nodes with scores
- **Ancestral memory** (DEBUG): Chain of buggy ancestors showing what fixes were tried

## Directory Structure

```
air/aira_dojo/
├── __init__.py       # Package marker
├── README.md         # This file
├── prompts.py        # Adapted AIRA-dojo prompt templates
├── operators.py      # Operator implementations (build messages for each op type)
└── search.py         # BaseSearch + GreedySearch + MCTSSearch + EvolutionarySearch + CLI
```

## Result Storage

Results are stored alongside adaptive search results:
```
outputs/adaptive_search_v3/
  titanic/
    ucb_c1.g/run1/          # our adaptive search
    aira_greedy.g/run1/     # AIRA-dojo greedy
    aira_mcts.g/run1/       # AIRA-dojo MCTS
    aira_evo.g/run1/        # AIRA-dojo evolutionary
  mountaincar/
    ...same pattern...
```

Each run directory contains:
- `result.json` — tree summary (best score, baseline, improvement, tree shape)
- `nodes/*.json` — per-node data (score, strategy, actions, conversation length)

## How to Run

### Standalone
```bash
cd /home/ubuntu/MLScientist/MLGym
uv run --project /home/ubuntu/MLScientist/arts \
    python -m air.aira_dojo.search \
    --task-config tasks/titanic.yaml \
    --search-policy greedy \
    --node-budget 12 --max-actions 15 \
    --output-dir outputs/adaptive_search_v3/titanic/aira_greedy.g/run1 \
    --env-gpu 1
```

### Via run_parallel.py
```bash
uv run --project /home/ubuntu/MLScientist/arts \
    python /home/ubuntu/MLScientist/arts/air/run_parallel.py \
    --suite aira-titanic --gpus 1,2,6,7
```

## Reference

```
@article{aira-dojo,
  title={AIRA-dojo: An Agentic Framework for AI Research Agents},
  author={Meta AI},
  journal={arXiv preprint arXiv:2507.02554},
  year={2025}
}
```
