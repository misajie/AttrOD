"""Distances, inclusion criteria, dissolve / merge."""
from .core import (
    inclusion_ok,
    intra_zonal_distance,
    pairwise_centroid_distances,
    official_dissolve,
    random_contiguous_merge,
)

__all__ = [
    "inclusion_ok",
    "intra_zonal_distance",
    "pairwise_centroid_distances",
    "official_dissolve",
    "random_contiguous_merge",
]
