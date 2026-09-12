"""Third-party model adapters (read-only wrappers)."""
from .deep_gravity import DeepGravityAdapter
from .neuro_gravity import NeuroGravityAdapter
from .provenance import normalize_features, record_adapter_provenance

__all__ = [
    "DeepGravityAdapter",
    "NeuroGravityAdapter",
    "normalize_features",
    "record_adapter_provenance",
]
