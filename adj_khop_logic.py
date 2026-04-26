from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Union

import torch
from ml_collections import ConfigDict
from torch import Tensor

from sampling_methods import (
    graph_search,
    random_select,
    random_walk,
    sim_walk,
)
from utils.khop_utils import (
    BFS_METHOD,
    DFS_METHOD,
    GREEDY_METHOD,
    HopNeighbours,
    PathLengths,
    RANDOM_METHOD,
    RANDOM_WALK_METHOD,
    SIM_WALK_METHOD,
    create_empty_adjacencies,
    create_networkx_graph,
    get_shortest_paths,
    prepare_path_data,
)


SamplingHandler = Callable[['SamplingContext'], Tensor]


class SamplingContext(NamedTuple):
    """Arguments needed to sample one hop adjacency."""

    edge_index: Tensor
    hop_neighbours: HopNeighbours
    shortest_path_lengths: Union[PathLengths, Tensor]
    model_config: ConfigDict
    feature_set: Optional[Tensor]
    num_samples: int
    num_nodes: int
    hop: int
    method: str
    device: torch.device


def require_feature_set(context: SamplingContext) -> Tensor:
    """Return feature_set or fail with a method-specific error."""
    if context.feature_set is None:
        raise ValueError(f'feature_set is required for {context.method}')

    return context.feature_set


def require_distance_matrix(context: SamplingContext) -> Tensor:
    """Return shortest path lengths as a tensor."""
    if not isinstance(context.shortest_path_lengths, Tensor):
        raise TypeError('shortest_path_lengths must be a Tensor')

    return context.shortest_path_lengths.to(context.device)


def sample_similarity_hop(context: SamplingContext) -> Tensor:
    """Sample one hop with feature-similarity walk."""
    return sim_walk(
        context.edge_index,
        context.hop_neighbours,
        context.model_config.walk,
        require_distance_matrix(context),
        require_feature_set(context),
        context.hop,
        context.num_samples,
        context.device,
        mving_avg=(context.method != 'greedy'),
    )


def sample_random_hop(context: SamplingContext) -> Tensor:
    """Sample one hop by independent random neighbour choices."""
    return random_select(
        context.edge_index,
        context.hop_neighbours,
        context.num_samples,
        context.num_nodes,
        context.hop,
        context.device,
    )


def sample_random_walk_hop(context: SamplingContext) -> Tensor:
    """Sample one hop by random walk."""
    return random_walk(
        context.edge_index,
        context.hop_neighbours,
        context.num_samples,
        context.num_nodes,
        context.hop,
        context.device,
    )


def sample_graph_search_hop(context: SamplingContext) -> Tensor:
    """Sample one hop by BFS or DFS traversal."""
    return graph_search(
        context.edge_index,
        context.hop_neighbours,
        context.num_samples,
        context.num_nodes,
        context.hop,
        context.method,
        context.device,
    )


SAMPLING_HANDLERS: Dict[str, SamplingHandler] = {
    RANDOM_METHOD: sample_random_hop,
    RANDOM_WALK_METHOD: sample_random_walk_hop,
    SIM_WALK_METHOD: sample_similarity_hop,
    GREEDY_METHOD: sample_similarity_hop,
    BFS_METHOD: sample_graph_search_hop,
    DFS_METHOD: sample_graph_search_hop,
}


def get_sampling_handler(method: str) -> SamplingHandler:
    """Return a sampler for method or raise a consistent error."""
    try:
        return SAMPLING_HANDLERS[method]
    except KeyError as exc:
        raise ValueError(
            'Invalid method specified for periphery graph selection'
        ) from exc


def sample_hop_adjacency(
    edge_index: Tensor,
    hop_neighbours: HopNeighbours,
    shortest_path_lengths: Union[PathLengths, Tensor],
    model_config: ConfigDict,
    feature_set: Optional[Tensor],
    num_samples: int,
    num_nodes: int,
    hop: int,
    method: str,
    device: torch.device,
) -> Tensor:
    """Populate one k-hop adjacency tensor."""
    context = SamplingContext(
        edge_index=edge_index,
        hop_neighbours=hop_neighbours,
        shortest_path_lengths=shortest_path_lengths,
        model_config=model_config,
        feature_set=feature_set,
        num_samples=num_samples,
        num_nodes=num_nodes,
        hop=hop,
        method=method,
        device=device,
    )
    return get_sampling_handler(method)(context)


def get_K_adjs(
    adj_list: Tensor,
    model_config: ConfigDict,
    ds_config: ConfigDict,
    feature_set: Optional[Tensor] = None,
    device: torch.device = torch.device('cpu'),
) -> List[Tensor]:
    """Build sampled adjacency lists for each configured hop."""
    num_hops = model_config.max_hops
    method = model_config.sampling_method
    num_samples = adj_list.shape[-1]
    num_nodes = ds_config.num_nodes

    get_sampling_handler(method)

    print('creating networkx graph')
    graph = create_networkx_graph(adj_list)
    adj_lists = create_empty_adjacencies(
        adj_list,
        num_hops,
        num_samples,
        device,
    )

    print('finding shortest paths')
    shortest_paths = get_shortest_paths(graph, num_hops)
    shortest_path_lengths, extra_data = prepare_path_data(
        shortest_paths,
        num_hops,
        method,
        num_nodes,
    )

    print('shortest paths complete')
    hop_neighbours = extra_data['k_hop_neighbours']

    print('samapling started...')
    for hop_index in range(1, num_hops):
        print(f'sampling for {hop_index} hop')
        hop_distance = hop_index + 1
        adj_lists[hop_index] = sample_hop_adjacency(
            adj_lists[hop_index],
            hop_neighbours,
            shortest_path_lengths,
            model_config,
            feature_set,
            num_samples,
            num_nodes,
            hop_distance,
            method,
            device,
        )
    print('sampling done')

    return adj_lists
