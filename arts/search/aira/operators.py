"""AIRA-dojo operator implementations.

Operators build conversation messages using AIRA-dojo-style prompts,
then the search policy executes them as multi-turn trajectories.

Each operator produces (user_message, operator_type) to inject into the
child node's conversation before the execution loop begins.
"""

from __future__ import annotations

from enum import Enum

from arts.search.aira.prompts import (
    CROSSOVER_TEMPLATE,
    DEBUG_TEMPLATE,
    DRAFT_COMPLEXITY,
    DRAFT_TEMPLATE,
    IMPROVE_COMPLEXITY,
    IMPROVE_TEMPLATE,
)


class OperatorType(Enum):
    DRAFT = "draft"
    IMPROVE = "improve"
    DEBUG = "debug"
    CROSSOVER = "crossover"


_COMPLEXITY_CYCLE = ["simple", "normal", "complex"]


class AiraOperators:
    """Wraps AIRA-dojo operator prompts for multi-turn MLGym execution."""

    def __init__(self, task_desc: str, data_overview: str):
        self.task_desc = task_desc
        self.data_overview = data_overview
        self._draft_idx = 0
        self._improve_idx = 0

    def draft(self, memory: str) -> str:
        """Generate a DRAFT operator user message."""
        complexity = _COMPLEXITY_CYCLE[self._draft_idx % len(_COMPLEXITY_CYCLE)]
        self._draft_idx += 1
        return DRAFT_TEMPLATE.format(
            task_desc=self.task_desc,
            memory=memory or "(none yet)",
            data_overview=self.data_overview or "(no data overview)",
            complexity_instruction=DRAFT_COMPLEXITY[complexity],
        )

    def improve(self, prev_approach: str, prev_score: float, memory: str) -> str:
        """Generate an IMPROVE operator user message."""
        complexity = _COMPLEXITY_CYCLE[self._improve_idx % len(_COMPLEXITY_CYCLE)]
        self._improve_idx += 1
        return IMPROVE_TEMPLATE.format(
            task_desc=self.task_desc,
            prev_approach=prev_approach,
            prev_score=f"{prev_score:.4f}" if prev_score is not None else "N/A",
            memory=memory or "(none yet)",
            complexity_instruction=IMPROVE_COMPLEXITY[complexity],
        )

    def debug(self, buggy_approach: str, error_output: str,
              ancestral_memory: str) -> str:
        """Generate a DEBUG operator user message."""
        return DEBUG_TEMPLATE.format(
            task_desc=self.task_desc,
            buggy_approach=buggy_approach,
            error_output=error_output[:2000],
            ancestral_memory=ancestral_memory or "(first attempt)",
        )

    def crossover(self, approach_1: str, score_1: float,
                  approach_2: str, score_2: float) -> str:
        """Generate a CROSSOVER operator user message."""
        return CROSSOVER_TEMPLATE.format(
            task_desc=self.task_desc,
            approach_1=approach_1,
            score_1=f"{score_1:.4f}" if score_1 is not None else "N/A",
            approach_2=approach_2,
            score_2=f"{score_2:.4f}" if score_2 is not None else "N/A",
        )
