from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import torch
from torch import Tensor


def assert_valid_edge_index(
    edge_index: Tensor,
    num_nodes: int,
    *,
    name: str = "edge_index",
) -> None:
    """Validate basic PyG edge_index invariants."""
    if not torch.is_tensor(edge_index):
        raise TypeError(f"{name} must be a torch.Tensor")

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, num_edges], got {tuple(edge_index.shape)}")

    if edge_index.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {edge_index.dtype}")

    if edge_index.numel() == 0:
        if edge_index.shape != (2, 0):
            raise ValueError(f"{name} empty tensor must have shape [2, 0], got {tuple(edge_index.shape)}")
        return

    min_idx = int(edge_index.min().item())
    max_idx = int(edge_index.max().item())

    if min_idx < 0:
        raise ValueError(f"{name} contains negative node index: {min_idx}")

    if max_idx >= num_nodes:
        raise ValueError(f"{name} contains node index {max_idx}, but num_nodes={num_nodes}")


def edge_tuples(edge_index: Tensor) -> list[tuple[int, int]]:
    """Convert edge_index to a list of edge tuples."""
    if edge_index.numel() == 0:
        return []

    return [
        (int(source), int(target))
        for source, target in edge_index.t().detach().cpu().tolist()
    ]


def duplicate_edges(edge_index: Tensor) -> list[tuple[tuple[int, int], int]]:
    """Return duplicate edges and their counts."""
    counts = Counter(edge_tuples(edge_index))
    return [(edge, count) for edge, count in counts.items() if count > 1]


def assert_no_duplicate_edges(
    edge_index: Tensor,
    *,
    name: str = "edge_index",
) -> None:
    """Assert that edge_index contains no duplicate directed edges."""
    duplicates = duplicate_edges(edge_index)

    if duplicates:
        examples = duplicates[:10]
        num_edges = int(edge_index.size(1))
        num_unique = len(set(edge_tuples(edge_index)))
        raise ValueError(
            f"{name} contains duplicate edges: "
            f"num_edges={num_edges}, num_unique={num_unique}, examples={examples}"
        )


def _adjacency(
    edge_index: Tensor,
    num_nodes: int,
    *,
    directed: bool,
) -> list[list[int]]:
    adj = [[] for _ in range(num_nodes)]

    for source, target in edge_tuples(edge_index):
        adj[source].append(target)
        if not directed:
            adj[target].append(source)

    return adj


def shortest_path_lengths(
    edge_index: Tensor,
    num_nodes: int,
    *,
    directed: bool = False,
) -> list[list[int | None]]:
    """Compute all-pairs shortest path lengths for small debug graphs."""
    assert_valid_edge_index(edge_index, num_nodes, name="base_edge_index")

    adj = _adjacency(edge_index, num_nodes, directed=directed)
    distances: list[list[int | None]] = [[None] * num_nodes for _ in range(num_nodes)]

    for source in range(num_nodes):
        distances[source][source] = 0
        queue: deque[int] = deque([source])

        while queue:
            current = queue.popleft()
            current_distance = distances[source][current]
            assert current_distance is not None

            for target in adj[current]:
                if distances[source][target] is None:
                    distances[source][target] = current_distance + 1
                    queue.append(target)

    return distances


def assert_exact_hop_edges(
    sampled_edge_index: Tensor,
    base_edge_index: Tensor,
    num_nodes: int,
    hop: int,
    *,
    directed: bool = False,
    name: str | None = None,
) -> None:
    """Assert that every sampled edge connects nodes at exact shortest-path distance hop."""
    check_name = name or f"edge_index_{hop}"
    assert_valid_edge_index(sampled_edge_index, num_nodes, name=check_name)

    if sampled_edge_index.numel() == 0:
        return

    distances = shortest_path_lengths(
        base_edge_index,
        num_nodes,
        directed=directed,
    )

    wrong: list[tuple[int, int, int | None]] = []
    for source, target in edge_tuples(sampled_edge_index):
        distance = distances[source][target]
        if distance != hop:
            wrong.append((source, target, distance))

    if wrong:
        raise ValueError(
            f"{check_name} contains edges that are not exact {hop}-hop edges. "
            f"Examples: {wrong[:10]}"
        )


def assert_edge_index_list(
    edge_index_list: Iterable[Tensor],
    base_edge_index: Tensor,
    num_nodes: int,
    *,
    directed: bool = False,
    strict_duplicates: bool = True,
) -> None:
    """Validate a full [edge_index_1, ..., edge_index_K] list."""
    edges = list(edge_index_list)

    if not edges:
        raise ValueError("edge_index_list must be non-empty")

    for idx, edge_index in enumerate(edges):
        hop = idx + 1
        name = f"edge_index_{hop}"

        assert_valid_edge_index(edge_index, num_nodes, name=name)

        if strict_duplicates:
            assert_no_duplicate_edges(edge_index, name=name)

        assert_exact_hop_edges(
            edge_index,
            base_edge_index,
            num_nodes,
            hop,
            directed=directed,
            name=name,
        )
