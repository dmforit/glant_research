from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from ml_collections import ConfigDict
from torch import Tensor
from torch.nn.functional import softmax

from utils.khop_utils import (
    GRAPH_SEARCH_METHODS,
    HopNeighbours,
    Node,
    add_bidirectional_edge,
    node_to_int,
)


def random_walk(
    edge_index: Tensor,
    hop_neighbours: HopNeighbours,
    num_samples: int,
    num_nodes: int,
    hop: int,
    device: torch.device,
) -> Tensor:
    """Sample edges by repeatedly walking through k-hop neighbours."""
    total_visited = 0
    restart = True

    for _ in range(2 * num_samples):
        if restart:
            source_node = np.random.randint(0, num_nodes)
            restart = False

        neighbours = hop_neighbours[(hop, source_node)]
        if len(neighbours) == 0:
            restart = True
            continue

        target_node = neighbours[np.random.randint(0, len(neighbours))]
        add_bidirectional_edge(
            edge_index,
            source_node,
            target_node,
            total_visited,
            device,
        )

        source_node = target_node
        total_visited += 2

        if total_visited == num_samples:
            break

    return edge_index


def random_select(
    edge_index: Tensor,
    hop_neighbours: HopNeighbours,
    num_samples: int,
    num_nodes: int,
    hop: int,
    device: torch.device,
) -> Tensor:
    """Sample random source nodes and one random k-hop neighbour for each."""
    total_samples = 0

    while total_samples < num_samples:
        source_node = np.random.randint(0, num_nodes)
        neighbours = hop_neighbours[(hop, source_node)]
        if len(neighbours) == 0:
            continue

        target_node = neighbours[np.random.randint(0, len(neighbours))]
        add_bidirectional_edge(
            edge_index,
            source_node,
            target_node,
            total_samples,
            device,
        )
        total_samples += 2

    return edge_index


def cosine_sim(source_features: Tensor, target_features: Tensor) -> Tensor:
    """Return cosine similarity between source and target vectors."""
    repeated_source = source_features.repeat(target_features.shape[0], 1)
    norm = torch.norm(repeated_source) * torch.norm(target_features)

    return torch.sum(repeated_source * target_features, dim=-1) / norm


def choose_similarity_walk_target(
    source_node: Node,
    previous_node: Node,
    candidate_nodes: Tensor,
    shortest_path_lengths: Tensor,
    feature_set: Tensor,
    weighted_sum: Tensor,
    hop: int,
    walk_config: ConfigDict,
    use_moving_average: bool,
) -> Optional[Tensor]:
    """Choose the next node for similarity-based walks."""
    neighbour_distances = shortest_path_lengths[candidate_nodes, source_node]
    valid_mask = (
        (neighbour_distances == hop)
        & (candidate_nodes != previous_node)
    )
    valid_candidates = candidate_nodes[valid_mask]

    if valid_candidates.shape[0] == 0:
        return None

    scores = 1 - cosine_sim(
        feature_set[source_node].unsqueeze(0),
        feature_set[valid_candidates],
    )
    if walk_config.use_cosine:
        scores = torch.arccos(scores)

    if use_moving_average:
        moving_scores = 1 - cosine_sim(
            weighted_sum.unsqueeze(0),
            feature_set[valid_candidates],
        )
        scores = (
            walk_config.gamma * scores # type: ignore
            + (1 - walk_config.gamma) * moving_scores
        )

    probabilities = torch.cumsum(softmax(scores, dim=-1), dim=0)
    above_threshold = probabilities > torch.randn(
        (1,),
        device=feature_set.device,
    )
    target_index = (
        len(above_threshold)
        - max(torch.sum(above_threshold).item(), 1)
        - 1
    )

    return valid_candidates[target_index]


def sim_walk(
    edge_index: Tensor,
    hop_neighbours: HopNeighbours,
    walk_config: ConfigDict,
    shortest_path_lengths: Tensor,
    feature_set: Tensor,
    hop: int,
    num_samples: int,
    device: torch.device,
    mving_avg: bool = True,
) -> Tensor:
    """Populate k-hop adjacency by walking toward feature-similar neighbours."""
    gamma = walk_config.gamma
    jump_prob = walk_config.jump_prob
    num_nodes = feature_set.shape[0]
    max_iters = 2 * num_samples

    total_visited = 0
    previous_node: Node = -1
    source_node: Node = -1
    weighted_sum: Optional[Tensor] = None
    restart = True

    walk = torch.empty((2, num_samples), device=device, dtype=torch.int64)

    for iter_idx in range(max_iters):
        if iter_idx % 1000 == 0:
            print(f'{iter_idx}/{max_iters}', walk.shape)

        if restart:
            source_node = torch.randint(
                0,
                num_nodes,
                (1,),
                device=device,
            ).item()
            neighbours = hop_neighbours[(hop, source_node)]
            weighted_sum = gamma * feature_set[source_node] # type: ignore
            restart = False

        should_restart = (
            len(neighbours) == 0
            or torch.rand(1, device=device).item() <= jump_prob
        )
        if should_restart:
            restart = True
            continue

        neighbour_tensor = torch.tensor(
            neighbours,
            device=device,
            dtype=torch.long,
        )
        assert weighted_sum is not None
        target_node = choose_similarity_walk_target(
            source_node,
            previous_node,
            neighbour_tensor,
            shortest_path_lengths,
            feature_set,
            weighted_sum,
            hop,
            walk_config,
            use_moving_average=mving_avg,
        )

        if target_node is None:
            restart = True
            continue

        add_bidirectional_edge(
            walk,
            source_node,
            target_node,
            total_visited,
            device,
        )
        total_visited += 2
        if total_visited >= num_samples:
            break

        weighted_sum = gamma * (weighted_sum + feature_set[target_node]) # type: ignore
        previous_node = source_node
        source_node = target_node
        neighbours = hop_neighbours[(hop, node_to_int(source_node))]

    return walk


def graph_search(
    edge_index: Tensor,
    hop_neighbours: HopNeighbours,
    num_samples: int,
    num_nodes: int,
    hop: int,
    method: str,
    device: torch.device,
) -> Tensor:
    """Sample edges by traversing the k-hop graph with BFS or DFS."""
    if method not in GRAPH_SEARCH_METHODS:
        raise ValueError(
            'Must be either depth first or breadth first search '
            'for this setting'
        )

    visited = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    start_node = np.random.randint(0, num_nodes)
    queue = [start_node]
    visited[start_node] = True

    previous_node = None
    total_visited = 0

    for _ in range(2 * num_samples):
        if method == 'bfs':
            source_node = queue.pop(0)
        else:
            source_node = queue.pop()

        if previous_node is not None:
            add_bidirectional_edge(
                edge_index,
                previous_node,
                source_node,
                total_visited,
                device,
            )
            total_visited += 2
        previous_node = source_node

        for target_node in hop_neighbours[(hop, source_node)]:
            if not visited[source_node, target_node]:
                visited[source_node, target_node] = True
                visited[target_node, source_node] = True
                queue.append(target_node)

        if len(queue) <= 1:
            start_node = np.random.randint(0, num_nodes)
            queue = [start_node]
            visited[start_node] = True

        if total_visited >= num_samples - 1:
            break

    return edge_index
