"""Queen-contiguity 5-block connected origin splits."""
from .blocks import farthest_point_seeds, make_queen_blocks, random_origin_holdout

__all__ = ["farthest_point_seeds", "make_queen_blocks", "random_origin_holdout", "METRIC_DISTANCE", "build_metric_distance_matrix", "build_from_lonlat", "intra_zonal_distance", "resolve_metric_crs", "write_distance_artifacts"]
from .metric_distance import (
    METRIC_DISTANCE,
    build_from_lonlat,
    build_metric_distance_matrix,
    intra_zonal_distance,
    resolve_metric_crs,
    write_distance_artifacts,
)
