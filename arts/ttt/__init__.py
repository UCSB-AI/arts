"""Test-Time Training (TTT) for the scientist model.

Step-level GRPO training: at each search step, generate K candidate
scientist decisions, execute in parallel, and update LoRA weights
to reinforce the best decisions.
"""
