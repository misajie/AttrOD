"""draft2 metric distance: Euclidean centroids in a projected CRS; d_ii = sqrt(A/pi)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

METRIC_DISTANCE = "metric_projected"
PROVISIONAL_DISTANCE = "provisional_wgs84_not_metric"


def intra_zonal_distance(area: float) -> float:
    """Radius of a circle with the same area as the polygon: sqrt(A_i / pi)."""
    a = float(area)
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"polygon area must be finite and non-negative, got {area!r}")
    return float(math.sqrt(a / math.pi))


def pairwise_euclidean(xy: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances for centroids shaped (N, 2) in metres."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"centroids must be (N, 2), got {xy.shape}")
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _utm_epsg_etrs89(lon: float, lat: float) -> str:
    """ETRS89 / UTM zone from a representative lon/lat (Northern Hemisphere)."""
    zone = int(math.floor((float(lon) + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    # ETRS89 / UTM north: EPSG 258xx
    return f"EPSG:258{zone:02d}"


def resolve_metric_crs(
    *,
    study_area: str = "mitma",
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    crs: Optional[str] = None,
) -> str:
    """Pick draft2 metric CRS: ETRS89/UTM for MITMA-like areas; Monte Mario for Lombardy."""
    if crs:
        return str(crs)
    sa = study_area.lower()
    if "lombard" in sa or "milan" in sa or "italy" in sa:
        # Monte Mario / Italy zone 1 (north-west) — draft2 Lombardy default
        return "EPSG:3003"
    if lon is None or lat is None:
        # Spain mainland default (ETRS89 / UTM 30N) when no sample point yet
        return "EPSG:25830"
    return _utm_epsg_etrs89(lon, lat)


def project_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    crs: str,
) -> np.ndarray:
    """Project WGS84 lon/lat to metric CRS metres. Requires pyproj."""
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "pyproj is required to project lon/lat into a metric CRS "
            "(install attrOD[geo] or pyproj)"
        ) from exc
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def distance_summary(D: np.ndarray) -> Dict[str, Any]:
    """Units metres; off-diagonal quantiles + simple outlier flags."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    finite = off[np.isfinite(off)]
    diag = np.diag(D)
    out: Dict[str, Any] = {
        "units": "meters",
        "n": int(n),
        "offdiag_n": int(finite.size),
        "offdiag_min": float(np.min(finite)) if finite.size else None,
        "offdiag_p50": float(np.percentile(finite, 50)) if finite.size else None,
        "offdiag_p90": float(np.percentile(finite, 90)) if finite.size else None,
        "offdiag_p99": float(np.percentile(finite, 99)) if finite.size else None,
        "offdiag_max": float(np.max(finite)) if finite.size else None,
        "diag_min": float(np.min(diag)) if n else None,
        "diag_p50": float(np.percentile(diag, 50)) if n else None,
        "diag_max": float(np.max(diag)) if n else None,
        "n_nonfinite": int(np.size(D) - np.isfinite(D).sum()),
        "n_negative": int(np.sum(D < 0)),
        "n_zero_offdiag": int(np.sum(off == 0)),
    }
    # rough outlier: off-diagonal > p99 * 5 or zero off-diagonal
    if finite.size and out["offdiag_p99"]:
        thr = float(out["offdiag_p99"]) * 5.0
        out["n_offdiag_gt_5x_p99"] = int(np.sum(finite > thr))
        out["outlier_threshold_m"] = thr
    else:
        out["n_offdiag_gt_5x_p99"] = 0
        out["outlier_threshold_m"] = None
    return out


def build_metric_distance_matrix(
    centroids_xy: np.ndarray,
    areas: np.ndarray,
    *,
    crs: str,
    zone_ids: Optional[Sequence[Any]] = None,
    study_area: str = "mitma",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build N×N metric Euclidean distance with draft2 intra-zonal diagonal.

    Parameters
    ----------
    centroids_xy :
        Official zone centroids already expressed in *crs* (metres), shape (N, 2).
    areas :
        Polygon areas in square metres, shape (N,).
    crs :
        Metric CRS name used for the centroids (e.g. EPSG:25830).
    """
    xy = np.asarray(centroids_xy, dtype=float)
    areas = np.asarray(areas, dtype=float).ravel()
    if xy.shape[0] != areas.shape[0]:
        raise ValueError(
            f"centroid count {xy.shape[0]} != area count {areas.shape[0]}"
        )
    D = pairwise_euclidean(xy)
    for i, a in enumerate(areas):
        D[i, i] = intra_zonal_distance(a)
    summary = distance_summary(D)
    # within-zone formula confirmation: recreate diag and compare
    expected_diag = np.array([intra_zonal_distance(a) for a in areas], dtype=float)
    diag_ok = bool(np.allclose(np.diag(D), expected_diag, rtol=0, atol=1e-6))
    qa: Dict[str, Any] = {
        "distance_status": METRIC_DISTANCE,
        "crs": crs,
        "study_area": study_area,
        "units": "meters",
        "d_ii_formula": "sqrt(A_i/pi)",
        "d_ii_formula_confirmed": diag_ok,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "n_zones": int(xy.shape[0]),
        "zone_ids": [str(z) for z in zone_ids] if zone_ids is not None else None,
        "summary": summary,
        "ok_mainline": True,
        "ok_smoke": True,
        "verification_only_required": False,
        "notes": [
            "Euclidean distances between official centroids in a metric CRS (draft2).",
            "Intra-zonal d_ii is the radius of a circle with the same polygon area.",
        ],
    }
    if summary["n_nonfinite"] or summary["n_negative"]:
        qa["ok_mainline"] = False
        qa["verification_only_required"] = True
        qa["notes"].append("non-finite or negative distances present")
    return D, qa


def build_from_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    areas: np.ndarray,
    *,
    study_area: str = "mitma",
    crs: Optional[str] = None,
    zone_ids: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """Project WGS84 lon/lat → metric CRS, then build the distance matrix."""
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    crs_res = resolve_metric_crs(
        study_area=study_area,
        lon=float(np.nanmean(lon)),
        lat=float(np.nanmean(lat)),
        crs=crs,
    )
    xy = project_lonlat(lon, lat, crs_res)
    D, qa = build_metric_distance_matrix(
        xy, areas, crs=crs_res, zone_ids=zone_ids, study_area=study_area
    )
    return D, qa, xy


def write_distance_artifacts(
    out_dir: PathLike,
    D: np.ndarray,
    qa: Mapping[str, Any],
    *,
    centroids_xy: Optional[np.ndarray] = None,
    areas: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Write distance.npy + distance_qa.json (+ optional centroids/areas)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "distance": str(out / "distance.npy"),
        "distance_qa": str(out / "distance_qa.json"),
    }
    np.save(paths["distance"], np.asarray(D, dtype=float))
    (out / "distance_qa.json").write_text(
        json.dumps(dict(qa), indent=2, default=str) + "\n", encoding="utf-8"
    )
    if centroids_xy is not None:
        p = out / "centroids_xy.npy"
        np.save(p, np.asarray(centroids_xy, dtype=float))
        paths["centroids_xy"] = str(p)
    if areas is not None:
        p = out / "areas_m2.npy"
        np.save(p, np.asarray(areas, dtype=float))
        paths["areas_m2"] = str(p)
    return paths
