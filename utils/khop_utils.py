from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch import Tensor


RANDOM_METHOD = 'random'
RANDOM_WALK_METHOD = 'random_walk'
SIM_WALK_METHOD = 'sim_walk'
GREEDY_METHOD = 'greedy'
BFS_METHOD = 'bfs'
DFS_METHOD = 'dfs'
BALANCED_UNIQUE_SELECT_METHOD = 'balanced_unique_select'
SIMILARITY_WALK_METHODS = {SIM_WALK_METHOD, GREEDY_METHOD}
GRAPH_SEARCH_METHODS = {BFS_METHOD, DFS_METHOD}
SUPPORTED_METHODS = {
    RANDOM_METHOD,
    RANDOM_WALK_METHOD,
    BALANCED_UNIQUE_SELECT_METHOD,
    *SIMILARITY_WALK_METHODS,
    *GRAPH_SEARCH_METHODS,
}

Node = Union[int, Tensor]
Edge = Tuple[int, int]
ShortestPath = Dict[int, List[int]]
ShortestPaths = Iterable[Tuple[int, ShortestPath]]
ShortestPathItem = Tuple[Edge, List[int]]
PathLengths = Dict[Edge, int]
HopNeighbours = Dict[Tuple[int, int], List[int]]
ExtraPathData = Dict[str, HopNeighbours]


def create_empty_adjacencies(
    edge_index: Tensor,
    num_hops: int,
    num_samples: int,
    device: torch.device,
) -> List[Tensor]:
    """Create one adjacency tensor per hop."""
    sample_count = num_samples + num_samples % 2

    return [
        edge_index.to(device)
        if hop == 0
        else torch.empty(
            (2, sample_count),
            device=device,
            dtype=torch.int64,
        )
        for hop in range(num_hops)
    ]


def iter_shortest_paths(
    paths: ShortestPaths,
) -> Iterable[ShortestPathItem]:
    """Yield shortest paths keyed by source and target node."""
    for source_node, node_paths in paths:
        for target_node, path in node_paths.items():
            yield (source_node, target_node), path


def build_hop_neighbours(
    paths: ShortestPaths,
    max_hops: int,
    num_nodes: int,
) -> Tuple[PathLengths, ExtraPathData]:
    """Build shortest-path lengths and k-hop neighbour lists."""
    shortest_path_lengths: PathLengths = {}
    hop_neighbours = {
        (hop, node): []
        for hop, node in product(range(1, max_hops + 1), range(num_nodes))
    }

    for (source_node, target_node), path in iter_shortest_paths(paths):
        hop_count = len(path) - 1
        shortest_path_lengths[(source_node, target_node)] = hop_count

        if hop_count > 0:
            hop_neighbours[(hop_count, source_node)].append(target_node)

    return shortest_path_lengths, {'k_hop_neighbours': hop_neighbours}


def prepare_path_data(
    shortest_paths: ShortestPaths,
    cutoff: int,
    method: str,
    num_nodes: int,
) -> Tuple[Union[PathLengths, Tensor], ExtraPathData]:
    """Prepare shortest-path data for a sampling method."""
    path_lengths, extra_data = build_hop_neighbours(
        shortest_paths,
        cutoff,
        num_nodes,
    )

    if method in SIMILARITY_WALK_METHODS:
        return build_dense_distance_matrix(path_lengths, num_nodes), extra_data

    return path_lengths, extra_data


def alter_paths(
    shortest_paths: ShortestPaths,
    cutoff: int,
    method: str,
    num_nodes: int,
) -> Tuple[Union[PathLengths, Tensor], ExtraPathData]:
    """Backward-compatible alias for prepare_path_data."""
    return prepare_path_data(shortest_paths, cutoff, method, num_nodes)


def add_bidirectional_edge(
    edge_index: Tensor,
    source_node: Node,
    target_node: Node,
    column: int,
    device: torch.device,
) -> None:
    """Write a forward edge and, when there is room, its reverse edge."""
    source = node_to_int(source_node)
    target = node_to_int(target_node)
    forward_edge = torch.tensor(
        [source, target],
        device=device,
        dtype=torch.int64,
    )
    backward_edge = torch.tensor(
        [target, source],
        device=device,
        dtype=torch.int64,
    )

    edge_index[:, column] = forward_edge
    if column < edge_index.shape[-1] - 1:
        edge_index[:, column + 1] = backward_edge
    else:
        print('Only forward tensor assignment due to index out of bounds')


def add_tensor(
    set_of_tensors: Tensor,
    node_i: Node,
    node_j: Node,
    index: int,
    device: torch.device,
) -> None:
    """Backward-compatible alias for add_bidirectional_edge."""
    add_bidirectional_edge(set_of_tensors, node_i, node_j, index, device)


def build_dense_distance_matrix(
    path_lengths: PathLengths,
    num_nodes: int,
) -> Tensor:
    """Convert sparse shortest-path lengths to a dense matrix."""
    distances = torch.full((num_nodes, num_nodes), torch.inf)

    for (source_node, target_node), hop_count in path_lengths.items():
        distances[source_node, target_node] = hop_count

    return distances


def add_to_dict(target: Dict[Any, List[Any]], key: Any, value: Any) -> None:
    """Append value to a list stored under key."""
    target.setdefault(key, []).append(value)


def to_numpy(value: Tensor) -> np.ndarray:
    """Detach a tensor and move it to NumPy."""
    return value.cpu().detach().numpy()


def node_to_int(node: Node) -> int:
    """Convert a scalar node id to plain int."""
    if isinstance(node, Tensor):
        return int(node.detach().cpu().item())

    return int(node)


def create_networkx_graph(edge_index: Tensor) -> nx.Graph:
    """Create an undirected NetworkX graph from a PyG edge_index tensor."""
    end_nodes, start_nodes = edge_index
    edges: List[Edge] = []

    for start_node, end_node in zip(start_nodes, end_nodes):
        source = node_to_int(start_node)
        target = node_to_int(end_node)

        if source > target:
            continue

        edges.append((source, target))

    graph = nx.Graph()
    graph.add_edges_from(edges)

    return graph


def get_shortest_paths(
    graph: nx.Graph,
    max_hops: int,
) -> ShortestPaths:
    """Return all shortest paths up to max_hops."""
    return nx.all_pairs_shortest_path(graph, max_hops)
