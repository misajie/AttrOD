"""Condition S: observed overlap of T vs mass-scaled R / Rᵀ / independence."""
from .pipeline import (
    bootstrap_days,
    compute_delta,
    compute_lambda_r,
    compute_lambda_rt,
    day_bootstrap,
    evaluate_independence_null,
    run_condition_s_full,
    score_condition_s,
    split_half_cpc,
)

__all__ = [
    "score_condition_s",
    "bootstrap_days",
    "split_half_cpc",
    "compute_lambda_r",
    "compute_lambda_rt",
    "compute_delta",
    "evaluate_independence_null",
    "day_bootstrap",
    "run_condition_s_full",
]
