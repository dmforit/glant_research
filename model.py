from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from ml_collections import ConfigDict
from torch import Tensor
from torch_geometric.utils import softmax


GAT_HEAD = 'gat'
GATV2_HEAD = 'gatv2'
TRANSFORMER_HEAD = 'transformer'


class GLANT(nn.Module):
    """GAT wrapper that can use regular or higher-order attention heads."""

    def __init__(
        self,
        model_config: ConfigDict,
        ds_config: ConfigDict,
        device: torch.device,
        layer_type: str = 'normal',
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.model_config = model_config
        self.ds_config = ds_config
        self.device = device
        self.num_layers = model_config.num_layers
        self.layer_type = layer_type
        self.num_heads = model_config.num_heads

        self.heads = nn.ModuleList(
            self._build_layers(model_config, ds_config, layer_type)
        )
        self.act_prelim = nn.ELU()
        self.act_final = nn.LogSoftmax(dim=-1)
        self.reset_parameters()

    def _layer_dimensions(
        self,
        layer_idx: int,
        ds_config: ConfigDict,
        layer_type: str,
    ) -> Tuple[int, int]:
        """Return input/output dimensions for one layer."""
        input_dim = ds_config.in_channels
        output_dim = ds_config.num_classes
        hidden_dim = ds_config.hidden_channels
        num_heads = self.num_heads

        previous_heads = num_heads[layer_idx - 1]
        current_heads = num_heads[layer_idx]
        if layer_type == 'multi_hop':
            previous_heads = current_heads = 1

        is_only_layer = layer_idx == self.num_layers - 1 == 0 # type: ignore
        if is_only_layer:
            return input_dim, output_dim

        if layer_idx == 0:
            return input_dim, hidden_dim

        if layer_idx == self.num_layers - 1:
            return previous_heads * hidden_dim, output_dim

        return previous_heads * hidden_dim, hidden_dim

    def _build_layer(
        self,
        in_channels: int,
        out_channels: int,
        layer_idx: int,
        do_concat: bool,
        model_config: ConfigDict,
        layer_type: str,
    ) -> nn.Module:
        """Build one attention layer."""
        if layer_type == 'normal':
            return tgnn.GATConv(
                in_channels,
                out_channels,
                heads=self.num_heads[layer_idx],
                dropout=model_config.drop_out,
                concat=do_concat,
                add_self_loops=True,
            )

        if layer_type == 'multi_hop':
            return HigherOrderGATHead(
                in_channels,
                out_channels,
                self.num_heads[layer_idx],
                model_config,
            )

        raise ValueError(f'Invalid layer type: {layer_type}')

    def _build_layers(
        self,
        model_config: ConfigDict,
        ds_config: ConfigDict,
        layer_type: str,
    ) -> List[nn.Module]:
        """Build all model layers."""
        layers: List[nn.Module] = []

        for layer_idx in range(self.num_layers):
            in_channels, out_channels = self._layer_dimensions(
                layer_idx,
                ds_config,
                layer_type,
            )
            do_concat = layer_idx != self.num_layers - 1
            layers.append(
                self._build_layer(
                    in_channels,
                    out_channels,
                    layer_idx,
                    do_concat,
                    model_config,
                    layer_type,
                )
            )

        return layers

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        """Run a forward pass."""
        edge_indices = [edge_index for _ in self.heads]

        for layer_idx, (head, edges) in enumerate(
            zip(self.heads, edge_indices)
        ):
            x = head(x, edges)
            if layer_idx != len(self.heads) - 1:
                x = self.act_prelim(x)

        return self.act_final(x)

    def reset_parameters(self) -> None:
        """Reset all layer parameters."""
        for head in self.heads:
            head.reset_parameters()



class HigherOrderGATHead(nn.Module):
    """Aggregate several higher-order GAT heads."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        model_config: ConfigDict,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.agg_type = model_config.agg_func
        self.heads = nn.ModuleList([
            HigherOrderGATLayer(
                in_features,
                out_features,
                model_config,
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        x: Tensor,
        edge_index_list: List[Tensor],
    ) -> Tensor:
        """Apply heads and aggregate their predictions."""
        predictions = [
            head(x, edge_index_list).unsqueeze(-1)
            for head in self.heads
        ]
        stacked = torch.concat(predictions, dim=-1)

        if self.agg_type == 'max':
            return torch.max(stacked, dim=-1, keepdim=False).values

        if self.agg_type == 'mean':
            return torch.mean(stacked, dim=-1, keepdim=False)

        if self.agg_type == 'sum':
            return torch.sum(stacked, dim=-1, keepdim=False)

        raise ValueError(f'Invalid aggregation type: {self.agg_type}')

    def reset_parameters(self) -> None:
        """Reset all head parameters."""
        for head in self.heads:
            head.reset_parameters()


class TransConv(nn.Module):
    """Sparse transformer-style attention layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        opt: ConfigDict,
        concat: bool = True,
        edge_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['leaky_relu_slope']
        self.concat = concat
        self.opt = opt
        self.h = int(opt['heads'])
        self.edge_weights = edge_weights
        self.attention_dim = out_features

        if self.attention_dim % self.h != 0:
            self.h = 1

        self.d_k = self.attention_dim // self.h
        self.Q = nn.Linear(in_features, self.attention_dim)
        self.V = nn.Linear(in_features, self.attention_dim)
        self.K = nn.Linear(in_features, self.attention_dim)
        self.activation = nn.Sigmoid()
        self.Wout = nn.Linear(self.d_k, in_features)
        self.init_weights()

    def reset_parameters(self) -> None:
        """Reset projection matrices."""
        self.Q.reset_parameters()
        self.V.reset_parameters()
        self.K.reset_parameters()

    def init_weights(self) -> None:
        """Initialize transformer projections."""
        for layer in (self.Q, self.V, self.K, self.Wout):
            nn.init.constant_(layer.weight, 1e-5)

    def forward(self, x: Tensor, edge: Tensor) -> Tensor:
        """Apply sparse transformer attention."""
        try:
            import torch_sparse
        except ImportError as exc:
            raise ImportError(
                'torch_sparse is required for transformer attention heads'
            ) from exc

        query = self.Q(x).view(-1, self.h, self.d_k).transpose(1, 2)
        key = self.K(x).view(-1, self.h, self.d_k).transpose(1, 2)
        value = self.V(x).view(-1, self.h, self.d_k).transpose(1, 2)

        source_query = query[edge[0, :], :, :]
        target_key = key[edge[1, :], :, :]
        products = torch.sum(source_query * target_key, dim=1) / np.sqrt(
            self.d_k
        )
        attention = softmax(products, edge[0])

        outputs: List[Tensor] = []
        for head_idx in range(self.h):
            value_head = value[:, :, head_idx]
            attention_head = attention[:, head_idx]
            outputs.append(
                torch_sparse.spmm(
                    edge,
                    attention_head,
                    value_head.shape[0],
                    value_head.shape[0],
                    value_head,
                )
            )

        return torch.cat(outputs, dim=-1)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} '
            f'({self.in_features} -> {self.out_features})'
        )


class HigherOrderGATLayer(nn.Module):
    """Apply one attention layer to each hop and aggregate hop outputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        model_config: ConfigDict,
    ) -> None:
        super().__init__()
        self.beta_mul = model_config.beta_mul
        self.agg_type = model_config.agg_func
        self.head_type = model_config.head_type
        self.attn_heads = nn.ModuleList([
            self._build_attention_head(
                in_features,
                out_features,
                model_config,
            )
            for _ in range(model_config.K_hops)
        ])

    def _build_attention_head(
        self,
        in_features: int,
        out_features: int,
        model_config: ConfigDict,
    ) -> nn.Module:
        """Build one hop-level attention head."""
        if self.head_type == GAT_HEAD:
            return tgnn.GATConv(
                in_features,
                out_features,
                dropout=model_config.drop_out,
                heads=model_config.num_heads_small,
                add_self_loops=True,
                concat=False,
            )

        if self.head_type == GATV2_HEAD:
            return tgnn.GATv2Conv(
                in_features,
                out_features,
                dropout=model_config.drop_out,
                heads=model_config.num_heads_small,
                add_self_loops=True,
                concat=False,
            )

        if self.head_type == TRANSFORMER_HEAD:
            return TransConv(in_features, out_features, model_config)

        raise ValueError(f'Invalid head type: {self.head_type}')

    def forward(
        self,
        features: Tensor,
        edge_index_list: List[Tensor],
    ) -> Tensor:
        """Apply hop attention and aggregate hop outputs."""
        hop_outputs = []
        for hop_idx, (head, edge_index) in enumerate(
            zip(self.attn_heads, edge_index_list)
        ):
            beta = 1 if hop_idx == 0 else self.beta_mul / (hop_idx + 1)
            hop_outputs.append(beta * head(features, edge_index))

        stacked = torch.stack(hop_outputs, dim=-1)

        if self.agg_type == 'sum':
            return torch.sum(stacked, dim=-1)

        if self.agg_type == 'mean':
            return torch.mean(stacked, dim=-1)

        if self.agg_type == 'max':
            return torch.max(stacked, dim=-1).values

        if self.agg_type == 'identity':
            return stacked

        raise ValueError(f'Invalid aggregation type: {self.agg_type}')

    def reset_parameters(self) -> None:
        """Reset all hop-level attention heads."""
        for head in self.attn_heads:
            head.reset_parameters()
