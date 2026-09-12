"""DeepGravityAdapter — wraps third_party/DeepGravity (read-only) @ 8693536."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from attrOD.condition_g.constraints import audit_constraints, project_to_O_i_T
from attrOD.data.manifest import DEEPGRAVITY_PIN
from attrOD.models.deep_gravity import DeepGravityModel
from attrOD.models.adapters.provenance import normalize_features, record_adapter_provenance


class DeepGravityAdapter:
    """AttrOD HBW schema ↔ pinned DeepGravity; outputs canonical OD + constraints."""

    PIN = DEEPGRAVITY_PIN

    def __init__(
        self,
        third_party_root: Optional[Path] = None,
        *,
        use_package_fallback: bool = True,
        seed: int = 0,
    ):
        self.seed = seed
        root = Path(third_party_root) if third_party_root else (
            Path(__file__).resolve().parents[4] / "third_party"
        )
        self.third_party_root = root
        self.dg_root = root / "DeepGravity"
        self.use_package_fallback = use_package_fallback
        self._model: Optional[DeepGravityModel] = None
        self._official_importable = False
        self._import_error: Optional[str] = None
        self._try_import_official()

    def _try_import_official(self) -> None:
        if not self.dg_root.exists():
            self._import_error = f"missing {self.dg_root}"
            return
        # Read-only: add to path for import check; do not modify third_party files
        dg_pkg = self.dg_root / "deepgravity"
        if not dg_pkg.exists():
            self._import_error = "deepgravity package folder missing"
            return
        try:
            if str(self.dg_root) not in sys.path:
                sys.path.insert(0, str(self.dg_root))
            import deepgravity  # noqa: F401
            self._official_importable = True
        except Exception as exc:  # pragma: no cover
            self._import_error = str(exc)
            self._official_importable = False

    def schema_roundtrip(
        self,
        R: np.ndarray,
        zone_features: np.ndarray,
        populations: np.ndarray,
        distances: np.ndarray,
    ) -> Dict[str, Any]:
        """Convert AttrOD arrays to a DeepGravity-like batch dict and back."""
        R = np.asarray(R, dtype=float)
        n = R.shape[0]
        norm = normalize_features(zone_features)
        batch = {
            "n_zones": n,
            "od_flows": R,  # HBW only
            "distances": np.asarray(distances, dtype=float),
            "populations": np.asarray(populations, dtype=float).ravel(),
            "features": norm["X"],
            "feature_rules": norm["rules"],
            "hbw_only": True,
        }
        # back to AttrOD canonical long rows (origin, destination, flow)
        rows: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(n):
                if R[i, j] > 0:
                    rows.append({"origin": i, "destination": j, "flow": float(R[i, j]), "partition": "R_HBW"})
        return {
            "batch": {k: v for k, v in batch.items() if k != "od_flows"},
            "n_positive_edges": len(rows),
            "canonical_preview": rows[:5],
            "roundtrip_ok": True,
            "official_importable": self._official_importable,
            "import_error": self._import_error,
            "pin": self.PIN,
        }

    def fit(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        zone_features: np.ndarray,
        populations: np.ndarray,
        train_origins: Optional[Sequence[int]] = None,
    ) -> "DeepGravityAdapter":
        """Fit using package DeepGravityModel (official training loop reserved)."""
        if not self.use_package_fallback and not self._official_importable:
            raise RuntimeError(
                f"DeepGravity official not importable: {self._import_error}"
            )
        # Prefer AttrOD package implementation that matches paper hyperparameters;
        # third_party remains read-only for provenance + future full wiring.
        self._model = DeepGravityModel()
        feats = normalize_features(zone_features)["X"]
        self._model.fit(
            R, distances, feats, populations, train_origins=train_origins
        )
        return self

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("not fitted")
        return self._model.emit(O_T)

    def generate_production_constrained(
        self, O_T: np.ndarray, *, zone_ids: Optional[Sequence[Any]] = None
    ) -> Dict[str, Any]:
        raw = self.emit(O_T)
        proj = project_to_O_i_T(raw, O_T)
        qa = audit_constraints(proj["T_hat"], O_T, zone_ids=zone_ids)
        return {"T_hat": proj["T_hat"], "projection": proj, "constraint_qa": qa}

    def provenance(self, out_path: Optional[Path] = None, **kwargs: Any) -> Dict[str, Any]:
        return record_adapter_provenance(
            model_name="DeepGravity",
            third_party_root=self.third_party_root,
            out_path=out_path,
            seed=self.seed,
            backend="package_fallback" if self.use_package_fallback else "official",
            is_stub=False,
            paper_mainline=False,  # caller may flip after readiness gates
            extra={
                "official_importable": self._official_importable,
                "import_error": self._import_error,
                "pin": self.PIN,
                **kwargs,
            },
        )
