"""Spatial split (mechanism subset).

Blocks on zone adjacency graph (queen contiguity). Partition into five connected
blocks of origins by greedy graph growing from five farthest-point seeds.
Block sizes differ by at most one zone. Each fold: train on four, generate fifth.
Destinations remain full tessellation. Target N≥80 for mechanism subset.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def farthest_point_seeds(
    adjacency: Mapping[int, Sequence[int]],
    n_seeds: int = 5,
    n_nodes: Optional[int] = None,
) -> List[int]:
    """Farthest-point seeding on the unweighted graph (graph distance)."""
    nodes = list(range(n_nodes if n_nodes is not None else (max(adjacency) + 1 if adjacency else 0)))
    if not nodes:
        return []
    # BFS distances
    def bfs(src: int) -> np.ndarray:
        dist = np.full(len(nodes), np.inf)
        dist[src] = 0
        q = [src]
        while q:
            u = q.pop(0)
            for v in adjacency.get(u, []):
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    seeds = [int(nodes[0])]
    while len(seeds) < min(n_seeds, len(nodes)):
        # distance to nearest seed
        mind = np.full(len(nodes), np.inf)
        for s in seeds:
            mind = np.minimum(mind, bfs(s))
        # unreachable → treat as large
        mind[~np.isfinite(mind)] = -1
        nxt = int(np.argmax(mind))
        if nxt in seeds:
            # pick any unused
            unused = [i for i in nodes if i not in seeds]
            if not unused:
                break
            nxt = unused[0]
        seeds.append(nxt)
    return seeds


def make_queen_blocks(
    adjacency: Mapping[int, Sequence[int]],
    n_blocks: int = 5,
    n_nodes: Optional[int] = None,
) -> List[List[int]]:
    """Greedy graph growing from farthest-point seeds; sizes differ by ≤1."""
    n = n_nodes if n_nodes is not None else (max(adjacency.keys()) + 1 if adjacency else 0)
    if n == 0:
        return [[] for _ in range(n_blocks)]
    seeds = farthest_point_seeds(adjacency, n_seeds=n_blocks, n_nodes=n)
    # target sizes
    base, rem = divmod(n, n_blocks)
    targets = [base + (1 if b < rem else 0) for b in range(n_blocks)]
    blocks: List[List[int]] = [[s] for s in seeds]
    assigned = set(seeds)
    # frontier per block
    frontiers = [set(adjacency.get(s, [])) - assigned for s in seeds]

    # grow until targets met or stuck
    progress = True
    while len(assigned) < n and progress:
        progress = False
        order = np.argsort([len(b) - targets[i] for i, b in enumerate(blocks)])  # most underfilled first
        for bi in order:
            if len(blocks[bi]) >= targets[bi]:
                continue
            # pick a frontier node (stable: min id)
            frontier = {v for v in frontiers[bi] if v not in assigned}
            if not frontier:
                # try expand frontier from block members
                for u in blocks[bi]:
                    frontier |= set(adjacency.get(u, []))
                frontier -= assigned
            if not frontier:
                continue
            v = min(frontier)
            blocks[bi].append(v)
            assigned.add(v)
            frontiers[bi] |= set(adjacency.get(v, []))
            frontiers[bi].discard(v)
            progress = True
            break
        if not progress:
            # assign leftovers to nearest underfilled block via adjacency / any
            leftover = [i for i in range(n) if i not in assigned]
            for v in leftover:
                # choose block with room and adjacency if possible
                candidates = []
                for bi, b in enumerate(blocks):
                    if len(b) >= targets[bi] + 1:
                        continue
                    if any(v in adjacency.get(u, []) or u in adjacency.get(v, []) for u in b):
                        candidates.append(bi)
                if not candidates:
                    candidates = [int(np.argmin([len(b) for b in blocks]))]
                bi = candidates[0]
                blocks[bi].append(v)
                assigned.add(v)
            break
    return blocks


def random_origin_holdout(
    n: int,
    holdout_size: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], List[int]]:
    """Supplement: random origin hold-out of the same size as one spatial block."""
    rng = rng or np.random.default_rng(0)
    hold = sorted(rng.choice(n, size=min(holdout_size, n), replace=False).tolist())
    train = [i for i in range(n) if i not in set(hold)]
    return train, hold


def folds_from_blocks(blocks: Sequence[Sequence[int]]) -> List[Dict[str, List[int]]]:
    """Each fold trains on all but one block."""
    folds = []
    for k, held in enumerate(blocks):
        train = [i for j, b in enumerate(blocks) if j != k for i in b]
        folds.append({"fold": k, "train_origins": train, "test_origins": list(held)})
    return folds
