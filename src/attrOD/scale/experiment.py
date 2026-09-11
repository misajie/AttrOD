"""From native tessellation: official dissolve + random contiguous merge (3 realisations).

Table 8: Spearman correlation of partition-wise CPC(T,λR) with native CPC;
fraction of partitions that keep the sign of Δ.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

from attrOD.condition_s.pipeline import score_condition_s


def scale_spearman_and_delta_sign(
    native_scores: Sequence[Dict[str, float]],
    coarse_scores: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """native_scores / coarse_scores: one dict per partition with CPC_R and Delta."""
    c_nat = np.asarray([s["CPC_R"] for s in native_scores], dtype=float)
    c_coa = np.asarray([s["CPC_R"] for s in coarse_scores], dtype=float)
    if len(c_nat) != len(c_coa) or len(c_nat) < 2:
        rho = float("nan")
    else:
        rho = float(spearmanr(c_nat, c_coa).correlation)
    d_nat = np.asarray([s["Delta"] for s in native_scores], dtype=float)
    d_coa = np.asarray([s["Delta"] for s in coarse_scores], dtype=float)
    mask = np.isfinite(d_nat) & np.isfinite(d_coa)
    if mask.sum() == 0:
        frac = float("nan")
    else:
        same = np.sign(d_nat[mask]) == np.sign(d_coa[mask])
        # treat 0 sign carefully: both zero counts as keep
        frac = float(same.mean())
    return {"spearman_CPC": rho, "frac_delta_sign_kept": frac}
