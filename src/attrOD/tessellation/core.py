"""Tessellation helpers (draft2 Object and tessellation / Scale experiment)."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from attrOD.data.flow_table import h_intra


def intra_zonal_distance(area: float) -> float:
    """d_ii = sqrt(A / π) — radius of equal-area circle."""
    a = float(area)
    if a < 0:
        raise ValueError("area must be non-negative")
    return float(np.sqrt(a / np.pi))


def pairwise_centroid_distances(
    xy: np.ndarray,
    areas: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Euclidean distance matrix; diagonal = sqrt(A_i/π) if areas given else 0."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2)")
    n = xy.shape[0]
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    if areas is not None:
        areas = np.asarray(areas, dtype=float).ravel()
        if areas.shape[0] != n:
            raise ValueError("areas length mismatch")
        np.fill_diagonal(d, [intra_zonal_distance(a) for a in areas])
    else:
        np.fill_diagonal(d, 0.0)
    return d


def inclusion_ok(
    mat_weekday_total: np.ndarray,
    min_N: int = 25,
    max_h_intra: float = 0.70,
) -> Tuple[bool, Dict[str, float]]:
    """Inclusion on unsliced weekday total: N≥25 and h_intra≤0.70."""
    N = int(mat_weekday_total.shape[0])
    h = h_intra(mat_weekday_total)
    ok = (N >= min_N) and (np.isfinite(h) and h <= max_h_intra)
    return ok, {"N": float(N), "h_intra": float(h)}


def official_dissolve(
    mat: np.ndarray,
    zone_to_group: Mapping[str, str],
    zones: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Aggregate OD matrix by mapping each zone to a coarser group id."""
    zones = [str(z) for z in zones]
    groups = sorted({str(zone_to_group[z]) for z in zones if z in zone_to_group})
    # zones without mapping stay as own group
    for z in zones:
        if z not in zone_to_group:
            g = z
            if g not in groups:
                groups.append(g)
    g_index = {g: i for i, g in enumerate(groups)}
    n = len(groups)
    out = np.zeros((n, n), dtype=float)
    for i, zi in enumerate(zones):
        gi = g_index[str(zone_to_group.get(zi, zi))]
        for j, zj in enumerate(zones):
            gj = g_index[str(zone_to_group.get(zj, zj))]
            out[gi, gj] += mat[i, j]
    return out, groups


def random_contiguous_merge(
    adjacency: Mapping[str, Sequence[str]],
    zones: Sequence[str],
    mat: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    target_n: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Repeatedly join a zone to a random neighbour until N is halved (default).

    Returns (merged_mat, group_ids_ordered, zone_to_group).
    """
    rng = rng or np.random.default_rng(0)
    zones = [str(z) for z in zones]
    parent = {z: z for z in zones}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def n_components() -> int:
        return len({find(z) for z in zones})

    target = target_n if target_n is not None else max(1, len(zones) // 2)
    # build undirected adj
    adj = {z: set(map(str, adjacency.get(z, []))) for z in zones}
    for z, nbrs in list(adj.items()):
        for v in nbrs:
            if v in adj:
                adj[v].add(z)

    safety = 0
    while n_components() > target and safety < len(zones) * 20:
        safety += 1
        # pick a random zone whose component has a neighbour in another component
        order = rng.permutation(len(zones))
        merged = False
        for ix in order:
            z = zones[int(ix)]
            rz = find(z)
            candidates = []
            for v in adj.get(z, []):
                if find(v) != rz:
                    candidates.append(v)
            if not candidates:
                continue
            v = str(rng.choice(candidates))
            union(z, v)
            merged = True
            break
        if not merged:
            break

    zone_to_group = {z: find(z) for z in zones}
    return official_dissolve(mat, zone_to_group, zones)[0:2] + (zone_to_group,)
