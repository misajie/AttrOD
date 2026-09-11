"""Tiny connected-graph block test."""
from __future__ import annotations

from attrOD.spatial.blocks import folds_from_blocks, make_queen_blocks


def test_five_blocks_cover_all():
    # path graph 0-1-2-3-4-5-6-7-8-9
    n = 10
    adj = {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}
    blocks = make_queen_blocks(adj, n_blocks=5, n_nodes=n)
    assert len(blocks) == 5
    flat = sorted(i for b in blocks for i in b)
    assert flat == list(range(n))
    sizes = sorted(len(b) for b in blocks)
    assert max(sizes) - min(sizes) <= 1
    folds = folds_from_blocks(blocks)
    assert len(folds) == 5
