from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from utils.homophily import edge_homophily, two_hop_labelled_projection


def test_edge_homophily_counts_same_class_edges() -> None:
    data = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        y=torch.tensor([0, 0, 1, 1]),
        num_nodes=4,
    )

    result = edge_homophily(data)

    assert result.same_class_edges == 2
    assert result.total_edges == 4
    assert result.value == 0.5


def test_edge_homophily_filters_unlabelled_endpoints() -> None:
    data = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        y=torch.tensor([0, 0, 1, 1]),
        labeled_mask=torch.tensor([True, True, False, False]),
        num_nodes=4,
    )

    result = edge_homophily(data)

    assert result.same_class_edges == 1
    assert result.total_edges == 1
    assert result.value == 1.0


def test_edge_homophily_returns_nan_for_empty_filtered_edge_set() -> None:
    data = Data(
        edge_index=torch.tensor([[0, 2], [2, 0]]),
        y=torch.tensor([0, 0, 1]),
        labeled_mask=torch.tensor([True, True, False]),
        num_nodes=3,
    )

    result = edge_homophily(data)

    assert result.same_class_edges == 0
    assert result.total_edges == 0
    assert math.isnan(result.value)


def test_two_hop_labelled_projection_connects_labelled_nodes_via_intermediate() -> None:
    edge_index = torch.tensor(
        [
            [0, 2, 1, 2, 3, 2],
            [2, 0, 2, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    labeled_mask = torch.tensor([True, True, False, True])

    projected = two_hop_labelled_projection(edge_index, labeled_mask, num_nodes=4)
    edges = set(map(tuple, projected.t().tolist()))

    assert edges == {
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 3),
        (3, 0),
        (3, 1),
    }
