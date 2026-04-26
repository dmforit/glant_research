from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union, Sequence, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch import Tensor

from torch_geometric.nn import GATv2Conv


EdgeInput: TypeAlias = Union[Tensor, Sequence[Tensor]]
MaskKey: TypeAlias = tuple[int, int, int, torch.device]
MaskDict: TypeAlias = dict[MaskKey, Tensor]


class GLANTConv(nn.Module):
    """Higher-order GATv2Conv with shared lin_l and hop weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = False,
        residual: bool = False,
        max_hops: int = 1,
        **kwargs,
    ):
        super().__init__()
        if max_hops < 1:
            raise ValueError("max_hops must be positive")

        self.max_hops = max_hops
        self.convs = nn.ModuleList()

        for k in range(max_hops):
            self.convs.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops if k == 0 else False,
                    edge_dim=edge_dim if k == 0 else None,
                    fill_value=fill_value,
                    bias=bias,
                    residual=residual,
                    share_weights=False,
                    **kwargs,
                )
            )

        for k in range(1, max_hops):
            self.convs[k].lin_l = self.convs[0].lin_l

        self.theta = nn.Parameter(torch.zeros(max_hops - 1))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

        nn.init.zeros_(self.theta)

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        edges = [edge_index] if torch.is_tensor(edge_index) else list(edge_index)

        if len(edges) > self.max_hops:
            raise ValueError("len(edge_index_list) exceeds max_hops")

        out = self.convs[0](x, edges[0], edge_attr)

        for k, ei in enumerate(edges[1:], start=1):
            if ei.numel() == 0:
                continue
            out = out + self.theta[k - 1].sigmoid() * self.convs[k](x, ei)

        return out


class GLANT(nn.Module):
    """Node-classification GNN with cached structural hop-dropout."""

    def __init__(
        self,
        model_config: ConfigDict,
        ds_config: ConfigDict,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Model parameters

        self.model_config = model_config
        self.ds_config = ds_config
        self.device = device

        self.alpha = model_config.alpha
        self.max_hops = model_config.max_hops
        self.dropout = model_config.dropout

        self.pre_linear = model_config.pre_linear
        self.batchnorm = model_config.batchnorm
        self.layernorm = model_config.layernorm
        self.residual = model_config.residual

        num_layers = model_config.num_layers

        # Model layers and structures

        self._masks = {}
        self.pre_lin = torch.nn.Linear(
            ds_config.in_channels,
            model_config.hidden_channels
        )
        self.convs = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.res_proj = nn.ModuleList()

        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        
        if not self.pre_linear:
            self.convs.append(
                GLANTConv(
                    in_channels=ds_config.in_channels,
                    out_channels=model_config.hidden_channels,
                    heads=model_config.heads,
                    concat=False, # Option True -> change out_channels
                    dropout=self.dropout,
                    max_hops=self.max_hops
                )
            )
            self.res_proj.append(
                torch.nn.Linear(
                    ds_config.in_channels,
                    model_config.hidden_channels
                )
            )
            self.layernorms.append(
                torch.nn.LayerNorm(
                    model_config.hidden_channels
                )
            )
            self.batchnorms.append(
                torch.nn.BatchNorm1d(
                    model_config.hidden_channels
                )
            )
            num_layers -= 1
        
        for _ in range(num_layers):
            self.convs.append(
                GLANTConv(
                    in_channels=model_config.hidden_channels,
                    out_channels=model_config.hidden_channels,
                    heads=model_config.heads,
                    concat=False, # Option True -> change out_channels
                    dropout=self.dropout,
                    max_hops=self.max_hops
                )
            )
            self.res_proj.append(
                torch.nn.Linear(
                    model_config.hidden_channels,
                    model_config.hidden_channels
                )
            )
            self.layernorms.append(
                torch.nn.LayerNorm(
                    model_config.hidden_channels
                )
            )
            self.batchnorms.append(
                torch.nn.BatchNorm1d(
                    model_config.hidden_channels
                )
            )
        
        self.prediction_layer = torch.nn.Linear(
            model_config.hidden_channels,
            ds_config.out_channels
        )

    def reset_parameters(self) -> None:
        self._masks.clear()

        for conv in self.convs:
            conv.reset_parameters()
        for res_proj in self.res_proj:
            res_proj.reset_parameters()
        for ln in self.layernorms:
            ln.reset_parameters()
        for bn in self.batchnorms:
            bn.reset_parameters()

        self.pre_lin.reset_parameters()
        self.prediction_layer.reset_parameters()
    
    @property
    def masks(self):
        return self._masks
    
    @masks.setter
    def masks(
        self,
        mask: MaskDict
    ) -> None:
        self._masks = mask

    def create_masks(
        self,
        edge_index: EdgeInput,
        set_mask: bool = True
    ) -> MaskDict:
        masks = {}
        if torch.is_tensor(edge_index):
            return masks

        for k, ei in enumerate(edge_index[1:], start=1):
            p = (1.0 - self.alpha) ** k
            key = (k, ei.data_ptr(), ei.size(1), ei.device)

            if key not in masks:
                masks[key] = torch.rand(ei.size(1), device=ei.device) < p

        if set_mask:
            self._masks = masks
        return masks
    
    def drop_edges(
        self,
        edge_index: EdgeInput,
    ) -> EdgeInput:
        if torch.is_tensor(edge_index) or not self._masks:
            return edge_index

        out = [edge_index[0]]

        for k, ei in enumerate(edge_index[1:], start=1):
            p = (1.0 - self.alpha) ** k
            key = (k, ei.data_ptr(), ei.size(1), ei.device)

            if key not in self.masks:
                self.masks[key] = torch.rand(ei.size(1), device=ei.device) < p

            out.append(ei[:, self.masks[key]])

        return out

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pre_linear:
            x = self.pre_lin(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):

            if self.residual:
                x = conv(x, edge_index, edge_attr) + self.res_proj[i](x)
            else:
                x = conv(x, edge_index, edge_attr)

            if self.layernorm:
                x = self.layernorms[i](x)
            elif self.batchnorm:
                x = self.batchnorms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.prediction_layer(x)
        return x
