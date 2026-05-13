from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops


@dataclass(frozen=True)
class EdgeHomophily:
    value: float
    same_class_edges: int
    total_edges: int


def labelled_node_mask(data: Data) -> Tensor:
    """Return nodes that have valid class labels for homophily calculation."""
    mask = getattr(data, "labelled_mask", None)
    if mask is None:
        mask = getattr(data, "labeled_mask", None)
    if mask is None:
        return torch.ones(int(data.num_nodes), dtype=torch.bool, device=data.y.device)
    return mask.to(device=data.y.device, dtype=torch.bool).view(-1)


def edge_homophily(
    data: Data,
    *,
    node_mask: Tensor | None = None,
    labelled_only: bool = True,
    drop_self_loops: bool = False,
    unique_edges: bool = False,
) -> EdgeHomophily:
    """
    Compute edge homophily h_edge = |{(i,j) in E | y_i = y_j}| / |E|.

    When labelled_only=True, edges are kept only if both endpoints have real
    labels according to node_mask or data.labelled_mask/data.labeled_mask.
    """
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Data object does not contain node labels in data.y")
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object does not contain edges in data.edge_index")

    edge_index = data.edge_index.to(device=data.y.device, dtype=torch.long)
    if drop_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
    if unique_edges:
        edge_index = coalesce(edge_index, num_nodes=int(data.num_nodes))

    y = data.y
    if y.dim() > 1:
        y = y.argmax(dim=-1)
    y = y.to(dtype=torch.long).view(-1)

    if labelled_only:
        if node_mask is None:
            node_mask = labelled_node_mask(data)
        else:
            node_mask = node_mask.to(device=y.device, dtype=torch.bool).view(-1)
        src, dst = edge_index
        keep = node_mask[src] & node_mask[dst]
        edge_index = edge_index[:, keep]

    total_edges = int(edge_index.size(1))
    if total_edges == 0:
        return EdgeHomophily(float("nan"), 0, 0)

    src, dst = edge_index
    same_class_edges = int((y[src] == y[dst]).sum().item())
    return EdgeHomophily(
        value=same_class_edges / total_edges,
        same_class_edges=same_class_edges,
        total_edges=total_edges,
    )


def two_hop_labelled_projection(edge_index: Tensor, node_mask: Tensor, num_nodes: int) -> Tensor:
    """
    Build directed edges between labelled nodes that share one intermediate node.

    This is useful for heterogeneous node-classification datasets where labels
    exist only on the target node type and raw edges connect different types.
    """
    device = edge_index.device
    edge_index = coalesce(edge_index.to(dtype=torch.long), num_nodes=num_nodes)
    src, dst = edge_index.cpu()
    labelled = node_mask.to(dtype=torch.bool).cpu().view(-1)

    labelled_neighbors: dict[int, list[int]] = {}
    for source, target in zip(src.tolist(), dst.tolist(), strict=True):
        if labelled[source] and not labelled[target]:
            labelled_neighbors.setdefault(target, []).append(source)

    edges: set[tuple[int, int]] = set()
    for neighbors in labelled_neighbors.values():
        unique_neighbors = sorted(set(neighbors))
        for source in unique_neighbors:
            for target in unique_neighbors:
                if source != target:
                    edges.add((source, target))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    projected = torch.tensor(sorted(edges), dtype=torch.long, device=device).t().contiguous()
    return coalesce(projected, num_nodes=num_nodes)
