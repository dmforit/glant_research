from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling import get_K_adjs
from utils.khop_checks import (
    assert_edge_index_list,
    assert_no_duplicate_edges,
    assert_valid_edge_index,
)
from utils.khop_utils import BALANCED_UNIQUE_SELECT_METHOD


def make_path_graph(num_nodes: int = 8) -> torch.Tensor:
    """Undirected path graph as bidirectional edge_index."""
    edges: list[tuple[int, int]] = []

    for node in range(num_nodes - 1):
        edges.append((node, node + 1))
        edges.append((node + 1, node))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_star_graph(num_nodes: int = 6) -> torch.Tensor:
    """Star graph with no 3-hop pairs."""
    edges: list[tuple[int, int]] = []

    for node in range(1, num_nodes):
        edges.append((0, node))
        edges.append((node, 0))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_model_config(max_hops: int = 3, num_samples: int = 2) -> ConfigDict:
    cfg = ConfigDict()
    cfg.max_hops = max_hops
    cfg.sampling_method = BALANCED_UNIQUE_SELECT_METHOD
    cfg.num_samples = num_samples
    return cfg


def make_ds_config(num_nodes: int) -> ConfigDict:
    cfg = ConfigDict()
    cfg.num_nodes = num_nodes
    return cfg


def sample_edges(
    base_edge_index: torch.Tensor,
    *,
    num_nodes: int,
    seed: int,
    max_hops: int = 3,
    num_samples: int = 2,
) -> list[torch.Tensor]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    return get_K_adjs(
        base_edge_index,
        make_model_config(max_hops=max_hops, num_samples=num_samples),
        make_ds_config(num_nodes),
        feature_set=None,
        device=torch.device("cpu"),
    )


def check_shape_and_duplicates() -> None:
    num_nodes = 8
    base_edge_index = make_path_graph(num_nodes)

    edge_index_list = sample_edges(
        base_edge_index,
        num_nodes=num_nodes,
        seed=123,
        max_hops=3,
        num_samples=2,
    )

    assert len(edge_index_list) == 3

    for hop, edge_index in enumerate(edge_index_list, start=1):
        assert_valid_edge_index(edge_index, num_nodes, name=f"edge_index_{hop}")
        assert_no_duplicate_edges(edge_index, name=f"edge_index_{hop}")

    print("ok shape_and_duplicates")


def check_exact_hop_distance() -> None:
    num_nodes = 8
    base_edge_index = make_path_graph(num_nodes)

    edge_index_list = sample_edges(
        base_edge_index,
        num_nodes=num_nodes,
        seed=123,
        max_hops=3,
        num_samples=2,
    )

    assert_edge_index_list(
        edge_index_list,
        base_edge_index,
        num_nodes,
        directed=False,
        strict_duplicates=True,
    )

    print("ok exact_hop_distance")


def check_reproducibility() -> None:
    num_nodes = 8
    base_edge_index = make_path_graph(num_nodes)

    first = sample_edges(
        base_edge_index,
        num_nodes=num_nodes,
        seed=777,
        max_hops=3,
        num_samples=2,
    )
    second = sample_edges(
        base_edge_index,
        num_nodes=num_nodes,
        seed=777,
        max_hops=3,
        num_samples=2,
    )

    for hop, (left, right) in enumerate(zip(first, second), start=1):
        if not torch.equal(left, right):
            raise AssertionError(f"Sampling is not reproducible for hop {hop}")

    print("ok reproducibility")


def check_empty_hop() -> None:
    num_nodes = 6
    base_edge_index = make_star_graph(num_nodes)

    edge_index_list = sample_edges(
        base_edge_index,
        num_nodes=num_nodes,
        seed=123,
        max_hops=3,
        num_samples=2,
    )

    assert len(edge_index_list) == 3
    empty_hop = edge_index_list[2]

    assert_valid_edge_index(empty_hop, num_nodes, name="edge_index_3")
    if empty_hop.shape != (2, 0):
        raise AssertionError(f"Expected empty 3-hop edge_index with shape [2, 0], got {tuple(empty_hop.shape)}")

    print("ok empty_hop")


def check_glant_v1_forward() -> None:
    from configs.config import all_config
    from utils.model_utils import create_model

    num_nodes = 8
    x = torch.randn(num_nodes, 5)
    base_edge_index = make_path_graph(num_nodes)

    edge_index_list = sample_edges(
        base_edge_index,
        num_nodes=num_nodes,
        seed=123,
        max_hops=3,
        num_samples=2,
    )

    config = all_config()
    ds_config = ConfigDict()
    ds_config.in_channels = x.size(-1)
    ds_config.out_channels = 3
    ds_config.num_nodes = num_nodes
    ds_config.name = "SyntheticSamplerCheck"

    model = create_model("glant_v1", config, ds_config)
    model.eval()

    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list, edge_attr=None)

    expected_shape = (num_nodes, ds_config.out_channels)
    if tuple(out.shape) != expected_shape:
        raise AssertionError(f"GLANT-v1 output shape mismatch: got {tuple(out.shape)}, expected {expected_shape}")

    if not torch.isfinite(out).all():
        raise AssertionError("GLANT-v1 output contains non-finite values")

    # Check that an empty higher-order hop does not crash the forward pass.
    edge_index_list_with_empty = [
        edge_index_list[0],
        edge_index_list[1],
        torch.empty((2, 0), dtype=torch.long),
    ]

    with torch.no_grad():
        out_empty = model(x=x, edge_index=edge_index_list_with_empty, edge_attr=None)

    if tuple(out_empty.shape) != expected_shape:
        raise AssertionError(
            f"GLANT-v1 output shape mismatch with empty hop: "
            f"got {tuple(out_empty.shape)}, expected {expected_shape}"
        )

    print("ok glant_v1_forward")


def main() -> None:
    check_shape_and_duplicates()
    check_exact_hop_distance()
    check_reproducibility()
    check_empty_hop()
    check_glant_v1_forward()
    print("All k-hop sampler checks passed.")


if __name__ == "__main__":
    main()
