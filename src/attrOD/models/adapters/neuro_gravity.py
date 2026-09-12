"""NeuroGravityAdapter — stub default; official backend reserved @ 7e29ef0.

IMPORTANT: The stub (meta-Gravity + residual MLP) must NEVER be labelled as
formal / paper-mainline NeuroGravity edge-enhanced results.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np

from attrOD.condition_g.constraints import audit_constraints, project_to_O_i_T
from attrOD.condition_g.masks import DEFAULT_TAU, build_few_shot_masks
from attrOD.data.manifest import NEUROGRAVITY_PIN
from attrOD.models.neurogravity import NeuroGravity
from attrOD.models.adapters.provenance import normalize_features, record_adapter_provenance

Backend = Literal["stub", "official"]


class NeuroGravityAdapter:
    PIN = NEUROGRAVITY_PIN

    def __init__(
        self,
        third_party_root: Optional[Path] = None,
        *,
        backend: Backend = "stub",
        seed: int = 0,
        tau: float = DEFAULT_TAU,
    ):
        if backend not in ("stub", "official"):
            raise ValueError("backend must be 'stub' or 'official'")
        self.backend = backend
        self.seed = seed
        self.tau = tau
        root = Path(third_party_root) if third_party_root else (
            Path(__file__).resolve().parents[4] / "third_party"
        )
        self.third_party_root = root
        self.ng_root = root / "NeuroGravity"
        self._model: Optional[NeuroGravity] = None
        self._official_importable = False
        self._import_error: Optional[str] = None
        if backend == "official":
            self._try_import_official()
            if not self._official_importable:
                raise RuntimeError(
                    "NeuroGravity official backend requested but not importable: "
                    f"{self._import_error}. Use backend='stub' for smoke/diagnostics only."
                )

    def _try_import_official(self) -> None:
        if not self.ng_root.exists():
            self._import_error = f"missing {self.ng_root}"
            return
        try:
            if str(self.ng_root) not in sys.path:
                sys.path.insert(0, str(self.ng_root))
            # soft import — may require torch_geometric
            import arguments  # noqa: F401
            self._official_importable = True
        except Exception as exc:  # pragma: no cover
            self._import_error = str(exc)
            self._official_importable = False

    @property
    def is_stub(self) -> bool:
        return self.backend == "stub"

    @property
    def formal_main_results(self) -> bool:
        """Always False for stub; official still gated until integration decided."""
        return False

    def fit_zeroshot(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        populations: np.ndarray,
        features: np.ndarray,
        train_origins: Sequence[int],
    ) -> "NeuroGravityAdapter":
        if self.backend != "stub":
            raise NotImplementedError(
                "official NeuroGravity fit not wired; full edge-enhanced GT pending"
            )
        feats = normalize_features(features)["X"]
        self._model = NeuroGravity(tau=self.tau, random_state=self.seed)
        self._model.fit_zeroshot(R, distances, populations, feats, train_origins)
        return self

    def fit_fewshot(
        self,
        R: np.ndarray,
        T: np.ndarray,
        distances: np.ndarray,
        populations: np.ndarray,
        features: np.ndarray,
        train_origins: Sequence[int],
        *,
        mask: str = "1pct_edges",
        holdout_origins: Optional[Sequence[int]] = None,
    ) -> "NeuroGravityAdapter":
        if self.backend != "stub":
            raise NotImplementedError("official few-shot not wired")
        # Build mask via AttrOD (no private adapter rewrite)
        manifest = build_few_shot_masks(
            R.shape[0],
            masks=[mask],  # type: ignore[list-item]
            seed=self.seed,
            holdout_origins=holdout_origins,
            tau=self.tau,
        )
        if not manifest["masks"][mask]["ok_no_holdout_leak"]:
            raise RuntimeError("few-shot mask leaked holdout origins")
        feats = normalize_features(features)["X"]
        self._model = NeuroGravity(tau=self.tau, random_state=self.seed)
        self._model.fit_fewshot(
            R, T, distances, populations, feats, train_origins,
            mask=mask,  # type: ignore[arg-type]
            holdout_origins=holdout_origins,
        )
        self._last_mask_manifest = manifest
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
        qa["model_label"] = "neuroGravity_stub" if self.is_stub else "neuroGravity_official"
        qa["formal_main_results"] = self.formal_main_results
        return {"T_hat": proj["T_hat"], "projection": proj, "constraint_qa": qa}

    def provenance(self, out_path: Optional[Path] = None, **kwargs: Any) -> Dict[str, Any]:
        return record_adapter_provenance(
            model_name="NeuroGravity",
            third_party_root=self.third_party_root,
            out_path=out_path,
            seed=self.seed,
            backend=self.backend,
            is_stub=self.is_stub,
            paper_mainline=False,
            extra={
                "pin": self.PIN,
                "formal_main_results": self.formal_main_results,
                "label_warning": (
                    "STUB — not formal edge-enhanced NeuroGravity main results"
                    if self.is_stub
                    else "official backend reserved; not yet paper-mainline"
                ),
                "official_importable": self._official_importable,
                "import_error": self._import_error,
                **kwargs,
            },
        )
