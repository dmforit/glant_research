from __future__ import annotations

import sys
from pathlib import Path

import torch
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import all_config
from model import HopEdgeSparsifier
from utils.model_names import canonical_model_name
from utils.model_utils import create_model


def synthetic_graph() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], ConfigDict]:
    num_nodes = 12
    x = torch.randn(num_nodes, 5)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6, 8],
        ],
        dtype=torch.long,
    )
    edge_index_2 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    edge_index_3 = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    ds_config = ConfigDict()
    ds_config.in_channels = x.size(-1)
    ds_config.out_channels = 3
    ds_config.num_nodes = num_nodes
    ds_config.name = "Synthetic"
    return x, edge_index, [edge_index, edge_index_2, edge_index_3], ds_config


def check_output(name: str, out: torch.Tensor, num_nodes: int, num_classes: int) -> None:
    assert out.shape == (num_nodes, num_classes), f"{name}: got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{name}: output contains non-finite values"


def check_hop_edge_sparsifier_keep_probability() -> None:
    edges = [torch.zeros(2, 10_000, dtype=torch.long) for _ in range(4)]
    sparsifier = HopEdgeSparsifier(alpha=0.5, cache_masks=False)

    with torch.no_grad():
        out = sparsifier(edges)

    kept = [edge.size(1) / edges[0].size(1) for edge in out]
    assert kept[0] == 1.0
    assert 0.45 <= kept[1] <= 0.55, kept
    assert 0.20 <= kept[2] <= 0.30, kept
    assert 0.10 <= kept[3] <= 0.15, kept
    print(f"ok sparsifier keep rates alpha^k {[round(value, 3) for value in kept]}")


def main() -> None:
    torch.manual_seed(0)
    config = all_config()
    x, edge_index, edge_index_list, ds_config = synthetic_graph()

    ordinary_edge_models = ["gcn", "graphsage", "gatv2", "mixhop", "tagconv"]
    for model_name in ordinary_edge_models:
        model = create_model(model_name, config, ds_config)
        model.eval()
        with torch.no_grad():
            out = model(x=x, edge_index=edge_index, edge_attr=None)
        check_output(model_name, out, x.size(0), ds_config.out_channels)
        print(f"ok {model_name} {tuple(out.shape)}")

    for model_name in [
        "hoga",
        "glant_v1",
        "glant_v2",
        "glant_v3",
        "glant_v4",
        "glant_v5",
        "glant_v6",
        "glant_v6p1",
        "glant_v7",
    ]:
        model = create_model(model_name, config, ds_config)
        model.eval()
        canonical_name = canonical_model_name(model_name)
        model_config = config.baselines[canonical_name]
        model_hops = int(getattr(model_config, "max_hops", len(edge_index_list)))
        with torch.no_grad():
            out = model(x=x, edge_index=edge_index_list[:model_hops], edge_attr=None)
        check_output(model_name, out, x.size(0), ds_config.out_channels)
        print(f"ok {model_name} K={model_hops} {tuple(out.shape)}")

    model = create_model("glant_v1", config, ds_config)
    model.eval()
    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list[:2], edge_attr=None)
    check_output("glant_v1", out, x.size(0), ds_config.out_channels)
    print(f"ok glant_v1 K=2 {tuple(out.shape)}")

    check_hop_edge_sparsifier_keep_probability()


if __name__ == "__main__":
    main()
