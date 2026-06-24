"""MLEvolve search policy adapted for MLGym multi-turn execution.

Re-implementation of the MLEvolve search algorithm (InternScience,
ranked #1 on MLE-bench leaderboard) adapted for our multi-turn
ReAct executor over MLGym containers. Reuses the BaseSearch
infrastructure and operator prompts from arts.search.aira.

Reference: https://github.com/InternScience/MLEvolve
"""
