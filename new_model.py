from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, TypeAlias, Union

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
        alpha: float,
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
        share_weights: bool = False,
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
                    dropout=1 - ((1 - alpha) ** k),
                    add_self_loops=add_self_loops if k == 0 else False,
                    edge_dim=edge_dim if k == 0 else None,
                    fill_value=fill_value,
                    bias=bias,
                    residual=residual,
                    share_weights=share_weights,
                    **kwargs,
                )
            )

        # for k in range(1, max_hops):
        #     self.convs[k].lin_l = self.convs[0].lin_l

        # self.theta = nn.Parameter(torch.zeros(max_hops - 1))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

        # nn.init.zeros_(self.theta)

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
            # out = out + F.softplus(self.theta[k - 1]) * self.convs[k](x, ei)
            out = out + self.convs[k](x, ei) / (k + 1)

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
        self.act = model_config.act

        num_layers = model_config.num_layers
        conv_kwargs = {
            "concat": model_config.concat,
            "negative_slope": model_config.negative_slope,
            "add_self_loops": model_config.add_self_loops,
            "bias": model_config.bias,
            "share_weights": model_config.share_weights,
        }

        # Model layers and structures

        self._masks = {}
        self.pre_lin = (
            torch.nn.Linear(ds_config.in_channels, model_config.hidden_channels)
            if self.pre_linear else None
        )
        self.convs = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.res_proj = nn.ModuleList()

        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        hidden_input_dim = (
            model_config.hidden_channels
            if self.pre_linear
            else ds_config.in_channels
        )
        hidden_layers = max(num_layers - 1, 0)
        hidden_concat = model_config.heads > 1
        hidden_out_channels = (
            model_config.hidden_channels // model_config.heads
            if hidden_concat
            else model_config.hidden_channels
        )
        hidden_dim = (
            hidden_out_channels * model_config.heads
            if hidden_concat
            else hidden_out_channels
        )
        hidden_conv_kwargs = {
            **conv_kwargs,
            "concat": hidden_concat,
        }
        output_conv_kwargs = {
            **conv_kwargs,
            "concat": False,
        }

        if hidden_layers > 0:
            self.convs.append(
                GLANTConv(
                    in_channels=hidden_input_dim,
                    out_channels=hidden_out_channels,
                    alpha=self.alpha,
                    heads=model_config.heads,
                    dropout=self.dropout,
                    max_hops=self.max_hops,
                    **hidden_conv_kwargs,
                )
            )
            self.res_proj.append(
                torch.nn.Linear(
                    hidden_input_dim,
                    hidden_dim,
                )
            )
            self.layernorms.append(
                torch.nn.LayerNorm(
                    hidden_dim
                )
            )
            self.batchnorms.append(
                torch.nn.BatchNorm1d(
                    hidden_dim
                )
            )
        
        for _ in range(hidden_layers - 1):
            self.convs.append(
                GLANTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_out_channels,
                    alpha=self.alpha,
                    heads=model_config.heads,
                    dropout=self.dropout,
                    max_hops=self.max_hops,
                    **hidden_conv_kwargs,
                )
            )
            self.res_proj.append(
                torch.nn.Linear(
                    hidden_dim,
                    hidden_dim
                )
            )
            self.layernorms.append(
                torch.nn.LayerNorm(
                    hidden_dim
                )
            )
            self.batchnorms.append(
                torch.nn.BatchNorm1d(
                    hidden_dim
                )
            )
        
        self.convs.append(
            GLANTConv(
                in_channels=(
                    hidden_dim
                    if hidden_layers > 0
                    else hidden_input_dim
                ),
                out_channels=ds_config.out_channels,
                alpha=self.alpha,
                heads=1,
                dropout=self.dropout,
                max_hops=self.max_hops,
                **output_conv_kwargs,
            )
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

        if self.pre_lin is not None:
            self.pre_lin.reset_parameters()
    
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
            x = self.pre_lin(x) # type: ignore
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            is_last = i == len(self.convs) - 1

            if self.residual and not is_last:
                x = conv(x, edge_index, edge_attr) + self.res_proj[i](x)
            else:
                x = conv(x, edge_index, edge_attr)

            if is_last:
                return x

            if self.layernorm:
                x = self.layernorms[i](x)
            elif self.batchnorm:
                x = self.batchnorms[i](x)
            
            if self.act == "elu":
                x = F.elu(x)
            elif self.act == "relu":
                x = F.relu(x)
            else:
                raise ValueError("act must be 'elu' or 'relu'")
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
