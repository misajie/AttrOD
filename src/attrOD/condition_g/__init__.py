"""Condition G: generators fit on R only; emit with target outflows."""
from .pipeline import score_condition_g, run_generators
from .constraints import project_to_O_i_T, audit_constraints
from .masks import build_few_shot_masks
from .generators import generate_production_constrained, run_registered_generators

__all__ = [
    "score_condition_g",
    "run_generators",
    "project_to_O_i_T",
    "audit_constraints",
    "build_few_shot_masks",
    "generate_production_constrained",
    "run_registered_generators",
]
