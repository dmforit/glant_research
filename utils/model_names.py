from __future__ import annotations

from typing import Iterable, List


MODEL_NAME_ALIASES = {
    "glant": "GLANT",
    "gat": "GAT",
    "gatv2": "GATv2",
    "gcn": "GCN",
}


def canonical_model_name(model_name: object) -> str:
    """Return canonical baseline key for a user-provided model name."""
    name = str(model_name)
    return MODEL_NAME_ALIASES.get(name.lower(), name)


def canonical_model_names(model_names: Iterable[object]) -> List[str]:
    """Return canonical model names, preserving order and removing duplicates."""
    names: List[str] = []
    for model_name in model_names:
        canonical = canonical_model_name(model_name)
        if canonical not in names:
            names.append(canonical)
    return names
