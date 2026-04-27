from collections import Counter
import sys
import types

import pytest
import torch

try:
    import ml_collections  # noqa: F401
except ModuleNotFoundError:
    ml_collections_stub = types.ModuleType("ml_collections")
    ml_collections_stub.ConfigDict = dict
sys.modules["ml_collections"] = ml_collections_stub

from adj_khop_logic import sampling_budget
from sampling_methods import balanced_unique_select


def _edge_tuples(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    return [tuple(edge) for edge in edge_index.t().tolist()]


def _small_hop_neighbours(hop: int) -> dict[tuple[int, int], list[int]]:
    return {
        (hop, 0): [1, 2, 2, 0],
        (hop, 1): [0, 2],
        (hop, 2): [0, 1, 3],
        (hop, 3): [],
    }


def _large_hop_neighbours() -> dict[tuple[int, int], list[int]]:
    return {
        (2, 0): [2, 3, 4, 4, 0],
        (2, 1): [3, 4, 5, 6, 1],
        (2, 2): [0, 4, 6, 7, 7],
        (2, 3): [0, 1, 5, 8, 3],
        (2, 4): [0, 1, 2, 6, 9],
        (2, 5): [1, 3, 7, 10, 10],
        (2, 6): [1, 2, 4, 8, 11],
        (2, 7): [2, 5, 9, 10, 7],
        (2, 8): [3, 6, 10, 11, 11],
        (2, 9): [4, 7, 0, 11, 9],
        (2, 10): [5, 7, 8, 1, 10],
        (2, 11): [6, 8, 9, 2, 11],
        (3, 0): [5, 6, 7, 8, 0],
        (3, 1): [6, 7, 8, 9, 9],
        (3, 2): [7, 8, 9, 10, 2],
        (3, 3): [8, 9, 10, 11, 11],
        (3, 4): [9, 10, 11, 0, 4],
        (3, 5): [10, 11, 0, 1, 1],
        (3, 6): [11, 0, 1, 2, 6],
        (3, 7): [0, 1, 2, 3, 3],
        (3, 8): [1, 2, 3, 4, 8],
        (3, 9): [2, 3, 4, 5, 5],
        (3, 10): [3, 4, 5, 6, 10],
        (3, 11): [4, 5, 6, 7, 7],
    }


def _valid_targets(
    hop_neighbours: dict[tuple[int, int], list[int]],
    hop: int,
    source: int,
) -> set[int]:
    return {
        int(target)
        for target in hop_neighbours[(hop, source)]
        if int(target) != source
    }


def test_balanced_unique_select_returns_unique_valid_edges() -> None:
    device = torch.device("cpu")
    hop = 2
    num_nodes = 4
    hop_neighbours = _small_hop_neighbours(hop)
    edge_index = torch.empty((2, 10), dtype=torch.int64)

    result = balanced_unique_select(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        num_samples=10,
        num_nodes=num_nodes,
        hop=hop,
        device=device,
    )

    edges = _edge_tuples(result)

    assert result.shape == (2, 7)
    assert len(edges) == len(set(edges))
    assert all(source != target for source, target in edges)
    assert all(
        target in set(hop_neighbours[(hop, source)])
        for source, target in edges
    )


def test_balanced_unique_select_respects_sample_limit() -> None:
    device = torch.device("cpu")
    hop_neighbours = {
        (2, 0): [1, 2, 3],
        (2, 1): [0, 2, 3],
        (2, 2): [0, 1, 3],
        (2, 3): [0, 1, 2],
    }
    edge_index = torch.empty((2, 5), dtype=torch.int64)

    result = balanced_unique_select(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        num_samples=5,
        num_nodes=4,
        hop=2,
        device=device,
    )

    edges = _edge_tuples(result)

    assert result.shape == (2, 5)
    assert len(edges) == len(set(edges))


def test_balanced_unique_select_uses_num_samples_not_input_width() -> None:
    device = torch.device("cpu")
    hop_neighbours = {
        (2, 0): [1, 2, 3],
        (2, 1): [0, 2, 3],
        (2, 2): [0, 1, 3],
        (2, 3): [0, 1, 2],
    }
    edge_index = torch.empty((2, 3), dtype=torch.int64)

    result = balanced_unique_select(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        num_samples=8,
        num_nodes=4,
        hop=2,
        device=device,
    )

    edges = _edge_tuples(result)

    assert result.shape == (2, 8)
    assert len(edges) == len(set(edges))


def test_balanced_unique_select_spreads_sources_when_possible() -> None:
    device = torch.device("cpu")
    hop_neighbours = {
        (2, 0): [1, 2],
        (2, 1): [0, 2],
        (2, 2): [0, 1],
        (2, 3): [0, 1],
    }
    edge_index = torch.empty((2, 4), dtype=torch.int64)

    result = balanced_unique_select(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        num_samples=4,
        num_nodes=4,
        hop=2,
        device=device,
    )

    sources = Counter(result[0].tolist())

    assert result.shape == (2, 4)
    assert set(sources) == {0, 1, 2, 3}
    assert all(count == 1 for count in sources.values())


def test_balanced_unique_select_returns_empty_for_non_positive_samples() -> None:
    device = torch.device("cpu")
    edge_index = torch.empty((2, 5), dtype=torch.int64)
    hop_neighbours = {(2, 0): [1], (2, 1): [0]}

    result = balanced_unique_select(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        num_samples=0,
        num_nodes=2,
        hop=2,
        device=device,
    )

    assert result.shape == (2, 0)


def test_balanced_unique_select_keeps_first_hop_edge_index_unchanged() -> None:
    device = torch.device("cpu")
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.int64,
    )
    original = edge_index.clone()
    hop_neighbours = _small_hop_neighbours(hop=1)

    result = balanced_unique_select(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        num_samples=2,
        num_nodes=4,
        hop=1,
        device=device,
    )

    assert result is edge_index
    assert torch.equal(edge_index, original)


def test_balanced_unique_select_budget_uses_nodes_times_config_samples() -> None:
    model_config = types.SimpleNamespace(num_samples=3)

    assert sampling_budget(
        method="balanced_unique_select",
        model_config=model_config,
        num_nodes=12,
        default_num_samples=99,
    ) == 36


def test_sampling_budget_keeps_default_for_other_methods() -> None:
    model_config = types.SimpleNamespace(num_samples=3)

    assert sampling_budget(
        method="random",
        model_config=model_config,
        num_nodes=12,
        default_num_samples=99,
    ) == 99


@pytest.mark.parametrize("hop", [2, 3])
def test_balanced_unique_select_large_graph_for_second_and_third_hops(
    hop: int,
) -> None:
    device = torch.device("cpu")
    num_nodes = 12
    num_samples = 20
    hop_neighbours = _large_hop_neighbours()
    edge_index = torch.empty((2, num_samples), dtype=torch.int64)

    result = balanced_unique_select(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        num_samples=num_samples,
        num_nodes=num_nodes,
        hop=hop,
        device=device,
    )

    edges = _edge_tuples(result)
    source_counts = Counter(result[0].tolist())

    assert result.shape == (2, num_samples)
    assert len(edges) == len(set(edges))
    assert all(source != target for source, target in edges)
    assert all(
        target in _valid_targets(hop_neighbours, hop, source)
        for source, target in edges
    )
    assert set(source_counts) == set(range(num_nodes))
    assert max(source_counts.values()) - min(source_counts.values()) <= 1
