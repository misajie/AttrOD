"""Few-shot masks for neuroGravity (draft2 defaults; Ticket 3).

Default masks (Yang et al. / draft2):
  - random 1% of internal OD pairs
  - random 10% of internal pairs
  - internal-only edges among a random 10% of zones
Temperature τ=2 for softmax weighting of observed edges by meta-Gravity prior.

No-leakage notes:
  - Masks act only on HBW ``R`` pretrain + declared target edges for few-shot.
  - Held-out Condition G spatial-block *test origins* must never appear as
    observed few-shot edges (draft2).
  - Target cells used for scoring are excluded from the revealed set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

FewshotMask = Literal["1pct_edges", "10pct_edges", "10pct_zones"]
DEFAULT_MASKS: Tuple[FewshotMask, ...] = ("1pct_edges", "10pct_edges", "10pct_zones")
DEFAULT_TAU = 2.0


def build_few_shot_masks(
    n_zones: int,
    *,
    masks: Sequence[FewshotMask] = DEFAULT_MASKS,
    seed: int = 0,
    holdout_origins: Optional[Sequence[int]] = None,
    exclude_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    tau: float = DEFAULT_TAU,
) -> Dict[str, Any]:
    """Build few-shot mask manifests without using target-side leakage beyond declared edges.

    Returns a dict with per-mask edge lists and an audit block proving holdout
    origins were excluded from candidates.
    """
    rng = np.random.default_rng(seed)
    holdout: Set[int] = set(int(i) for i in (holdout_origins or []))
    exclude: Set[Tuple[int, int]] = set((int(i), int(j)) for i, j in (exclude_pairs or []))

    # Candidates: internal pairs whose origin is NOT a spatial-block test origin
    candidates: List[Tuple[int, int]] = [
        (i, j)
        for i in range(n_zones)
        for j in range(n_zones)
        if i not in holdout and (i, j) not in exclude
    ]

    out_masks: Dict[str, Any] = {}
    for mask in masks:
        if mask == "1pct_edges":
            k = max(1, int(0.01 * n_zones * n_zones))
            chosen_idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
            chosen = [candidates[int(i)] for i in chosen_idx]
        elif mask == "10pct_edges":
            k = max(1, int(0.10 * n_zones * n_zones))
            chosen_idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
            chosen = [candidates[int(i)] for i in chosen_idx]
        elif mask == "10pct_zones":
            allowed_zones = [i for i in range(n_zones) if i not in holdout]
            z = rng.choice(
                allowed_zones,
                size=max(1, int(0.10 * n_zones)),
                replace=False,
            )
            zset = set(int(x) for x in z)
            chosen = [(i, j) for i in zset for j in zset if (i, j) not in exclude]
        else:
            raise ValueError(f"unknown few-shot mask: {mask}")

        leaked = [p for p in chosen if p[0] in holdout]
        out_masks[mask] = {
            "edges": chosen,
            "n_edges": len(chosen),
            "leaked_holdout_origins": leaked,
            "ok_no_holdout_leak": len(leaked) == 0,
        }

    return {
        "n_zones": n_zones,
        "seed": seed,
        "tau": tau,
        "holdout_origins": sorted(holdout),
        "n_candidates": len(candidates),
        "masks": out_masks,
        "no_leakage_notes": [
            "Masks built from zone counts + holdout origin exclusion only.",
            "No target-partition cell values were read to choose edges.",
            "Few-shot must not use Condition G spatial-block test origins as observed edges.",
            "Scoring cells must remain disjoint from revealed edges at evaluation time.",
        ],
        "source": "draft2 / Yang et al. defaults (1%, 10% edges, 10% zones; τ=2)",
        "stub": True,  # builder only; model training may still be stub backend
    }
