from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch import Tensor
from torch_geometric.nn import TAGConv


EdgeInput = Tensor | Sequence[Tensor]


def _cfg_get(cfg: ConfigDict, name: str, default: Any = None) -> Any:
    return cfg[name] if name in cfg else default


def _as_edge_index(edge_index: EdgeInput) -> Tensor:
    edge = edge_index if torch.is_tensor(edge_index) else list(edge_index)[0]
    if edge.dim() != 2 or edge.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return edge


class TAGNet(nn.Module):
    """TAG baseline based on PyG TAGConv."""

    def __init__(self, model_config: ConfigDict, ds_config: ConfigDict) -> None:
        super().__init__()
        self.model_config = model_config
        self.hidden_channels = int(_cfg_get(model_config, "hidden_channels", 64))
        self.dropout = float(_cfg_get(model_config, "dropout", 0.5))
        self.K = int(_cfg_get(model_config, "K", 3))

        if self.K < 1:
            raise ValueError("K must be positive")

        in_channels = int(_cfg_get(ds_config, "in_channels"))
        out_channels = int(_cfg_get(ds_config, "out_channels"))

        self.conv1 = TAGConv(in_channels, self.hidden_channels, K=self.K)
        self.conv2 = TAGConv(self.hidden_channels, out_channels, K=self.K)

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_attr: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        del edge_attr
        edge_index = _as_edge_index(edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


KHopModel2 = TAGNet
