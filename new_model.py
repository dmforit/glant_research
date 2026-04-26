from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GATv2Conv


EdgeInput = Union[Tensor, Sequence[Tensor]]


class GLANTConv(nn.Module):
    """Higher-order GATv2Conv with shared lin_l and hop weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
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
                    **kwargs,
                )
            )

        for k in range(1, max_hops):
            self.convs[k].lin_l = self.convs[0].lin_l

        self.theta = nn.Parameter(torch.zeros(max_hops - 1))

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
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 1,
        alpha: float = 0.0,
        max_hops: int = 1,
        conv_cls: Optional[Callable[..., nn.Module]] = None,
        conv_kwargs: Optional[dict] = None,
        dropout: float = 0.5,
        norm: str = "layer",
        residual: bool = True,
        feat_norm: bool = True,
        use_edge_attr: bool = False,
    ):
        super().__init__()

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        conv_cls = GLANTConv if conv_cls is None else conv_cls
        conv_kwargs = {} if conv_kwargs is None else dict(conv_kwargs)

        self.alpha = alpha
        self.max_hops = max_hops
        self.dropout = dropout
        self.residual = residual
        self.feat_norm = feat_norm
        self.use_glant = conv_cls is GLANTConv
        self.use_edge_attr = use_edge_attr
        self.masks = {}

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1)
        outs = [hidden_channels] * (num_layers - 1) + [out_channels]

        for i, (din, dout) in enumerate(zip(dims, outs)):
            last = i == num_layers - 1
            h = 1 if last else heads
            cat = False if last else True
            real_out = dout if not cat else dout * h

            if self.use_glant:
                conv = GLANTConv(
                    in_channels=din,
                    out_channels=dout,
                    heads=h,
                    concat=cat,
                    dropout=dropout,
                    max_hops=max_hops,
                    **conv_kwargs,
                )
            else:
                conv = conv_cls(din, real_out, **conv_kwargs)

            self.convs.append(conv)

            if not last:
                if norm == "batch":
                    self.norms.append(nn.BatchNorm1d(real_out))
                elif norm == "layer":
                    self.norms.append(nn.LayerNorm(real_out))
                elif norm in {None, "none"}:
                    self.norms.append(nn.Identity())
                else:
                    raise ValueError("norm must be 'batch', 'layer', or 'none'")

            if residual and not last:
                self.proj.append(
                    nn.Identity()
                    if din == real_out
                    else nn.Linear(din, real_out)
                )

    def drop_edges(
        self,
        edge_index: EdgeInput,
        resample: bool = False,
    ) -> EdgeInput:
        if torch.is_tensor(edge_index):
            return edge_index

        out = [edge_index[0]]

        for k, ei in enumerate(edge_index[1:], start=1):
            p = (1.0 - self.alpha) ** k
            key = (k, ei.data_ptr(), ei.size(1), ei.device)

            if resample or key not in self.masks:
                self.masks[key] = torch.rand(ei.size(1), device=ei.device) < p

            out.append(ei[:, self.masks[key]])

        return out

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_attr: Optional[Tensor] = None,
        resample: bool = False,
    ) -> Tensor:
        if self.feat_norm:
            x = F.normalize(x, p=2, dim=-1)

        edges = self.drop_edges(edge_index, resample=resample)

        for i, conv in enumerate(self.convs):
            h = x

            if self.use_glant:
                x = conv(x, edges, edge_attr=edge_attr)
            else:
                ei = edges[0] if not torch.is_tensor(edges) else edges
                if self.use_edge_attr:
                    x = conv(x, ei, edge_attr=edge_attr)
                else:
                    x = conv(x, ei)

            if i == len(self.convs) - 1:
                return x

            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.residual:
                x = x + self.proj[i](h)

        return x