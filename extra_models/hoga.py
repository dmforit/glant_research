from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch import Tensor
from torch_geometric.nn import GATConv


EdgeInput = Tensor | Sequence[Tensor]


def _cfg_get(cfg: ConfigDict, name: str, default: Any = None) -> Any:
    return cfg[name] if name in cfg else default


def _edge_list(edge_index: EdgeInput, max_hops: int) -> list[Tensor]:
    edges = [edge_index] if torch.is_tensor(edge_index) else list(edge_index)
    if not edges:
        raise ValueError("edge_index list must be non-empty")
    if len(edges) > max_hops:
        edges = edges[:max_hops]
    for idx, edge in enumerate(edges):
        if edge.dim() != 2 or edge.size(0) != 2:
            raise ValueError(f"edge_index[{idx}] must have shape [2, num_edges]")
    return edges


class HigherOrderGATLayer(nn.Module):
    """
    Project-local adapter for the original HoGA higher-order GAT layer.

    The original module applies one GAT head to each sampled hop adjacency and
    scales higher-order hops as beta_mul / (k + 1), with k counted from zero.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_config: ConfigDict,
    ) -> None:
        super().__init__()
        self.num_hops = int(_cfg_get(model_config, "K_hops", _cfg_get(model_config, "max_hops", 3)))
        self.num_heads_small = int(_cfg_get(model_config, "num_heads_small", 1))
        self.dropout = float(_cfg_get(model_config, "drop_out", _cfg_get(model_config, "dropout", 0.6)))
        self.beta_mul = float(_cfg_get(model_config, "beta_mul", 0.9))
        self.agg_type = str(_cfg_get(model_config, "agg_func", "sum")).lower()
        self.head_type = str(_cfg_get(model_config, "head_type", "gat")).lower()

        if self.num_hops < 1:
            raise ValueError("K_hops/max_hops must be positive")
        if self.num_heads_small < 1:
            raise ValueError("num_heads_small must be positive")
        if self.head_type != "gat":
            raise ValueError("This adapter currently supports original HoGA head_type='gat'")
        if self.agg_type not in {"sum", "mean", "max"}:
            raise ValueError("agg_func must be one of: sum, mean, max")

        self.attn_heads = nn.ModuleList([
            GATConv(
                in_channels,
                out_channels,
                heads=self.num_heads_small,
                concat=False,
                dropout=self.dropout,
                add_self_loops=True,
            )
            for _ in range(self.num_hops)
        ])

    def reset_parameters(self) -> None:
        for head in self.attn_heads:
            head.reset_parameters()

    def _hop_scale(self, hop_idx: int) -> float:
        return 1.0 if hop_idx == 0 else self.beta_mul / float(hop_idx + 1)

    def forward(self, x: Tensor, edge_index: EdgeInput) -> Tensor:
        edges = _edge_list(edge_index, self.num_hops)
        hop_outputs = []

        for hop_idx, head in enumerate(self.attn_heads):
            if hop_idx >= len(edges) or edges[hop_idx].size(1) == 0:
                out = x.new_zeros(x.size(0), head.out_channels)
            else:
                out = head(x, edges[hop_idx])
            hop_outputs.append(self._hop_scale(hop_idx) * out)

        stacked = torch.stack(hop_outputs, dim=-1)
        if self.agg_type == "sum":
            return stacked.sum(dim=-1)
        if self.agg_type == "mean":
            return stacked.mean(dim=-1)
        return stacked.max(dim=-1).values


class HoGA(nn.Module):
    """
    HoGA_GAT adapter based on TB862/Higher-Order-Graph-Attention-Module.

    It keeps the local project interface while matching the original core:
    sampled k-hop edge lists, GATConv per hop, beta scaling, and hop aggregation.
    The final activation is left as logits for compatibility with CrossEntropyLoss.
    """

    def __init__(self, model_config: ConfigDict, ds_config: ConfigDict) -> None:
        super().__init__()
        self.model_config = model_config
        self.num_layers = int(_cfg_get(model_config, "num_layers", 2))
        self.hidden_channels = int(_cfg_get(model_config, "hidden_channels", 64))
        self.dropout = float(_cfg_get(model_config, "drop_out", _cfg_get(model_config, "dropout", 0.6)))

        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

        in_channels = int(_cfg_get(ds_config, "in_channels"))
        out_channels = int(_cfg_get(ds_config, "out_channels", _cfg_get(ds_config, "num_classes", None)))
        dims = [in_channels]
        dims.extend([self.hidden_channels] * max(self.num_layers - 1, 0))
        dims.append(out_channels)

        self.layers = nn.ModuleList([
            HigherOrderGATLayer(dims[layer_idx], dims[layer_idx + 1], model_config)
            for layer_idx in range(self.num_layers)
        ])

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_attr: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        del edge_attr
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if layer_idx != len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
