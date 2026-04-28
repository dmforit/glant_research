from __future__ import annotations

from typing import Iterable, List


MODEL_NAME_ALIASES = {
    "glant": "GLANT_v1",
    "glant_v1": "GLANT_v1",
    "glant-v1": "GLANT_v1",
    "glant_v2": "GLANT_v2",
    "glant-v2": "GLANT_v2",
    "gat": "GAT",
    "gatv2": "GATv2",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "sage": "GraphSAGE",
    "mixhop": "MixHop",
    "khop1": "MixHop",
    "khop_model_1": "MixHop",
    "k-hop baseline #1": "MixHop",
    "tagconv": "TAGConv",
    "tag": "TAGConv",
    "khop2": "TAGConv",
    "khop_model_2": "TAGConv",
    "k-hop baseline #2": "TAGConv",
    "hoga": "HoGA",
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
