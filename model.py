from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, TypeAlias, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch import Tensor

from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv


EdgeInput: TypeAlias = Union[Tensor, Sequence[Tensor]]
MaskKey: TypeAlias = tuple[int, int, int, torch.device]
MaskDict: TypeAlias = dict[MaskKey, Tensor]


def cfg_get(cfg: ConfigDict, name: str, default: Any = None) -> Any:
    """Read a ConfigDict field with a default value."""
    return cfg[name] if name in cfg else default


def as_edge_list(edge_index: EdgeInput) -> list[Tensor]:
    """Convert a single edge_index or a sequence of edge_index tensors to a non-empty list."""
    edges = [edge_index] if torch.is_tensor(edge_index) else list(edge_index)

    if not edges:
        raise ValueError("edge_index list must be non-empty")

    for i, ei in enumerate(edges):
        if not torch.is_tensor(ei):
            raise TypeError(f"edge_index[{i}] must be a Tensor")
        if ei.dim() != 2 or ei.size(0) != 2:
            raise ValueError(f"edge_index[{i}] must have shape [2, num_edges]")

    return edges


class HopEdgeSparsifier(nn.Module):
    """
    Sparsifies higher-order hop edge sets.

    Input can be either:

        edge_index = E_1

    or:

        edge_index = [E_1, E_2, ..., E_K]

    The first edge set E_1 is never changed. It corresponds to k=0.

    For all higher-order edge sets, the module drops edges with probability:

        p_drop(k) = (1 - alpha) ** k,

    where k starts from 1 for E_2, 2 for E_3, and so on.

    Therefore:

        alpha = 0.0 -> all higher-order edges are removed.
        alpha = 1.0 -> all higher-order edges are preserved.

    The keep probability is:

        p_keep(k) = 1 - (1 - alpha) ** k.
    """

    def __init__(
        self,
        alpha: float,
        cache_masks: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            alpha:
                Higher-hop preservation parameter in [0, 1].
            cache_masks:
                If True, sampled masks are cached by edge_index identity.
                If False, masks are resampled on every forward call.
            enabled:
                If False, the module returns edge_index unchanged.
        """
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.alpha = float(alpha)
        self.cache_masks = bool(cache_masks)
        self.enabled = bool(enabled)
        self._masks: MaskDict = {}

    @property
    def masks(self) -> MaskDict:
        """Cached keep masks."""
        return self._masks

    def clear_cache(self) -> None:
        """Clear cached masks."""
        self._masks.clear()

    def reset_parameters(self) -> None:
        """Compatibility method. Clears cached masks."""
        self.clear_cache()

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, "
            f"cache_masks={self.cache_masks}, "
            f"enabled={self.enabled}"
        )

    @staticmethod
    def _validate_edge_index(edge_index: Tensor, name: str) -> None:
        """Validate edge_index shape."""
        if not torch.is_tensor(edge_index):
            raise TypeError(f"{name} must be a Tensor")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"{name} must have shape [2, num_edges]")

    def _drop_prob(self, k: int) -> float:
        """
        Return drop probability for higher-hop index k.

        k=1 corresponds to E_2.
        k=2 corresponds to E_3.
        """
        if k < 1:
            raise ValueError("k must be >= 1 for higher-order hops")
        return (1.0 - self.alpha) ** k

    def _mask_key(self, k: int, edge_index: Tensor) -> MaskKey:
        """Build cache key for a higher-hop edge_index."""
        return (k, edge_index.data_ptr(), edge_index.size(1), edge_index.device)

    def _make_mask(self, edge_index: Tensor, k: int) -> Tensor:
        """
        Create boolean keep mask.

        True means keep the edge.
        False means drop the edge.
        """
        num_edges = edge_index.size(1)

        if num_edges == 0:
            return torch.empty(0, device=edge_index.device, dtype=torch.bool)

        p_drop = self._drop_prob(k)

        if p_drop <= 0.0:
            return torch.ones(num_edges, device=edge_index.device, dtype=torch.bool)

        if p_drop >= 1.0:
            return torch.zeros(num_edges, device=edge_index.device, dtype=torch.bool)

        return torch.rand(num_edges, device=edge_index.device) >= p_drop

    def _get_mask(self, edge_index: Tensor, k: int) -> Tensor:
        """Return cached or newly sampled keep mask."""
        if not self.cache_masks:
            return self._make_mask(edge_index, k)

        key = self._mask_key(k, edge_index)

        if key not in self._masks:
            self._masks[key] = self._make_mask(edge_index, k)

        return self._masks[key]

    def forward(self, edge_index: EdgeInput) -> EdgeInput:
        """
        Apply higher-hop sparsification.

        Args:
            edge_index:
                Either a Tensor [2, E], or a sequence:
                [edge_index_1, edge_index_2, ..., edge_index_K].

        Returns:
            Tensor input -> Tensor output.
            Sequence input -> list[Tensor] output.
        """
        if not self.enabled:
            return edge_index

        if torch.is_tensor(edge_index):
            self._validate_edge_index(edge_index, "edge_index")
            return edge_index

        edges = list(edge_index)

        if not edges:
            raise ValueError("edge_index list must be non-empty")

        for i, ei in enumerate(edges):
            self._validate_edge_index(ei, f"edge_index[{i}]")

        out = [edges[0]]

        for k, ei in enumerate(edges[1:], start=1):
            mask = self._get_mask(ei, k)
            out.append(ei[:, mask])

        return out


class HopGatedGATv2Conv(nn.Module):
    """
    Higher-order GATv2 layer with node-wise adaptive hop gating.

    The layer receives either a single edge_index or a list of edge_index tensors:

        edge_index = E_1

    or

        edge_index = [E_1, E_2, ..., E_K],

    where E_k contains edges for the k-hop graph.

    For each hop k, the layer computes a hop-specific GATv2 message:

        M_k = GATv2_k(X, E_k).

    Then, for each node i, it computes adaptive hop weights:

        pi_i = softmax(W_hops x_i),

    and mixes hop messages node-wise:

        x_i' = sum_k pi_{ik} M_{k,i}.

    Parameter sharing policy:
        - W_left is shared across all hop-specific GATv2Conv modules.
        - W_right is shared across all hop-specific GATv2Conv modules.
        - W_left and W_right are still different matrices.
        - Attention vectors are not shared across hops.
        - Edge attributes are used only for the first hop.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_hops: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        residual: bool = False,
        gate_hidden: Optional[int] = None,
        gate_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Args:
            in_channels:
                Input node feature dimension.
            out_channels:
                Output feature dimension per head.
            max_hops:
                Maximum number of hop edge sets.
            heads:
                Number of GATv2 attention heads.
            concat:
                If True, concatenate heads. Otherwise, average heads.
            negative_slope:
                Negative slope for LeakyReLU in GATv2 attention.
            dropout:
                Dropout probability inside GATv2 attention.
            add_self_loops:
                Whether to add self-loops for the first hop.
                Higher hops do not receive self-loops.
            edge_dim:
                Edge feature dimension. Edge features are used only for the first hop.
            fill_value:
                Fill value for self-loop edge attributes in PyG GATv2Conv.
            bias:
                Whether to use bias in GATv2Conv.
            residual:
                Whether to use PyG GATv2Conv internal residual connection.
            gate_hidden:
                Hidden dimension for the hop gate MLP.
                If None, the hop gate is a single Linear layer.
            gate_dropout:
                Dropout used inside the hop gate MLP.
            **kwargs:
                Additional arguments passed to PyG GATv2Conv.
        """
        super().__init__()

        if max_hops < 1:
            raise ValueError("max_hops must be positive")
        if gate_hidden is not None and gate_hidden < 1:
            raise ValueError("gate_hidden must be positive or None")

        self.max_hops = int(max_hops)
        self.edge_dim = edge_dim
        self.out_dim = out_channels * heads if concat else out_channels

        self.convs = nn.ModuleList([
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
            for k in range(self.max_hops)
        ])

        self._share_left_right_projections()

        self.hop_gate = (
            nn.Linear(in_channels, self.max_hops)
            if gate_hidden is None
            else nn.Sequential(
                nn.Linear(in_channels, gate_hidden),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden, self.max_hops),
            )
        )

    def _share_left_right_projections(self) -> None:
        """
        Shares W_left and W_right across all hop-specific GATv2Conv modules.

        Attention vectors remain hop-specific.
        """
        base = self.convs[0]

        for conv in self.convs[1:]:
            conv.lin_l = base.lin_l
            conv.lin_r = base.lin_r

    def reset_parameters(self) -> None:
        """
        Resets trainable parameters.

        Shared W_left/W_right are reset through the first GATv2Conv.
        Hop-specific attention vectors are reset through their own GATv2Conv modules.
        """
        for conv in self.convs:
            conv.reset_parameters()

        self._share_left_right_projections()
        self._reset_hop_gate()

    def _reset_hop_gate(self) -> None:
        """Resets hop-gating network parameters."""
        if isinstance(self.hop_gate, nn.Linear):
            self.hop_gate.reset_parameters()
            return

        for module in self.hop_gate:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_attr: Optional[Tensor] = None,
        return_hop_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Applies hop-gated higher-order GATv2 convolution.

        Args:
            x:
                Node features of shape [num_nodes, in_channels].
            edge_index:
                Either a single edge_index tensor of shape [2, num_edges],
                or a sequence [edge_index_1, ..., edge_index_K].
            edge_attr:
                Optional edge features for the first hop only.
            return_hop_weights:
                If True, returns both output features and hop weights.

        Returns:
            If return_hop_weights is False:
                Tensor of shape [num_nodes, out_dim].
            If return_hop_weights is True:
                Tuple:
                    - output tensor of shape [num_nodes, out_dim]
                    - hop weights of shape [num_nodes, num_hops]
        """
        if edge_attr is not None and self.edge_dim is None:
            raise ValueError("edge_attr was provided, but edge_dim is None")

        edges = [edge_index] if torch.is_tensor(edge_index) else list(edge_index)

        if not edges:
            raise ValueError("edge_index_list must be non-empty")
        if len(edges) > self.max_hops:
            raise ValueError("len(edge_index_list) exceeds max_hops")

        num_nodes = x.size(0)
        num_hops = len(edges)

        messages: list[Tensor] = []
        empty_hops: list[bool] = []

        for k, ei in enumerate(edges):
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(f"edge_index at hop {k} must have shape [2, num_edges]")

            is_empty = ei.size(1) == 0

            # For the first hop we still call GATv2Conv, because it may add
            # self-loops and can therefore produce a non-empty message.
            skip_hop = is_empty and k > 0
            empty_hops.append(skip_hop)

            if skip_hop:
                msg = x.new_zeros(num_nodes, self.out_dim)
            else:
                msg = self.convs[k](
                    x,
                    ei,
                    edge_attr=edge_attr if k == 0 else None,
                )

            messages.append(msg)

        messages_t = torch.stack(messages, dim=1)

        logits = self.hop_gate(x)[:, :num_hops]

        empty_t = torch.tensor(
            empty_hops,
            device=x.device,
            dtype=torch.bool,
        )

        if empty_t.all():
            out = x.new_zeros(num_nodes, self.out_dim)
            weights = x.new_zeros(num_nodes, num_hops)
            return (out, weights) if return_hop_weights else out

        if empty_t.any():
            logits = logits.masked_fill(empty_t.unsqueeze(0), float("-inf"))

        weights = torch.softmax(logits, dim=-1)
        out = (messages_t * weights.unsqueeze(-1)).sum(dim=1)

        return (out, weights) if return_hop_weights else out


class GLANT(nn.Module):
    """
    General node-classification GNN wrapper.

    Supports both ordinary PyG message-passing layers and hop-aware layers.

    Ordinary layers receive only the first edge set:

        edge_index = E_1

    Hop-aware layers receive the full list:

        edge_index = [E_1, E_2, ..., E_K]

    This wrapper does not sparsify higher-hop edges. If higher-hop sparsification
    is needed, it should be done before calling the model:

        edge_index = sparsifier(edge_index, alpha)
        logits = model(x, edge_index, edge_attr)

    Supported conv_type values:
        - "hop_gated_gatv2"
        - "gatv2"
        - "gat"
        - "sage"
        - "gcn"
    """

    HOP_AWARE_CONVS = {"hop_gated_gatv2"}
    EDGE_ATTR_CONVS = {"hop_gated_gatv2", "gatv2", "gat"}

    def __init__(
        self,
        model_config: ConfigDict,
        ds_config: ConfigDict,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.ds_config = ds_config
        self.device = device

        self.conv_type = str(cfg_get(model_config, "conv_type", "hop_gated_gatv2")).lower()

        self.alpha = cfg_get(model_config, "alpha", None)
        self.max_hops = int(cfg_get(model_config, "max_hops", 1))

        self.num_layers = int(cfg_get(model_config, "num_layers", 2))
        self.hidden_channels = self._resolve_hidden_channels(model_config)

        self.dropout = float(cfg_get(model_config, "dropout", 0.0))
        self.attn_dropout = float(cfg_get(model_config, "attn_dropout", self.dropout))

        self.pre_linear = bool(cfg_get(model_config, "pre_linear", False))
        self.residual = bool(cfg_get(model_config, "residual", False))
        self.act = str(cfg_get(model_config, "act", "relu")).lower()

        self.norm_type = self._resolve_norm_type(model_config)

        self.in_channels = self._resolve_in_channels(ds_config)
        self.out_channels = self._resolve_out_channels(ds_config)

        self.edge_dim = cfg_get(model_config, "edge_dim", None)

        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.max_hops < 1:
            raise ValueError("max_hops must be positive")
        if self.alpha is not None and not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.act not in {"relu", "elu"}:
            raise ValueError("act must be 'relu' or 'elu'")
        if self.norm_type not in {"none", "batchnorm", "layernorm"}:
            raise ValueError("norm must be 'none', 'batchnorm', or 'layernorm'")

        self.use_hops = self.conv_type in self.HOP_AWARE_CONVS
        self.use_edge_attr = (
            self.conv_type in self.EDGE_ATTR_CONVS
            and self.edge_dim is not None
        )

        self.pre_lin = (
            nn.Linear(self.in_channels, self.hidden_channels)
            if self.pre_linear
            else None
        )

        input_dim = self.hidden_channels if self.pre_linear else self.in_channels
        hidden_layers = self.num_layers - 1

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_proj = nn.ModuleList()

        if hidden_layers == 0:
            self.convs.append(
                self._make_conv(
                    in_channels=input_dim,
                    out_dim=self.out_channels,
                    is_last=True,
                )
            )
            return

        hidden_dim = self.hidden_channels

        self.convs.append(
            self._make_conv(
                in_channels=input_dim,
                out_dim=hidden_dim,
                is_last=False,
            )
        )
        self.res_proj.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(self._make_norm(hidden_dim))

        for _ in range(hidden_layers - 1):
            self.convs.append(
                self._make_conv(
                    in_channels=hidden_dim,
                    out_dim=hidden_dim,
                    is_last=False,
                )
            )
            self.res_proj.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(self._make_norm(hidden_dim))

        self.convs.append(
            self._make_conv(
                in_channels=hidden_dim,
                out_dim=self.out_channels,
                is_last=True,
            )
        )

    @staticmethod
    def _resolve_hidden_channels(model_config: ConfigDict) -> int:
        """Resolve hidden dimension from model_config."""
        hidden_channels = cfg_get(model_config, "hidden_channels", None)
        if hidden_channels is None:
            raise ValueError("model_config must contain hidden_channels")
        return int(hidden_channels)

    @staticmethod
    def _resolve_in_channels(ds_config: ConfigDict) -> int:
        """Resolve input dimension from ds_config."""
        in_channels = cfg_get(ds_config, "in_channels", None)
        if in_channels is None:
            raise ValueError("ds_config must contain in_channels")
        return int(in_channels)

    @staticmethod
    def _resolve_out_channels(ds_config: ConfigDict) -> int:
        """Resolve output dimension from ds_config."""
        out_channels = cfg_get(ds_config, "out_channels", None)
        if out_channels is None:
            out_channels = cfg_get(ds_config, "num_classes", None)
        if out_channels is None:
            raise ValueError("ds_config must contain out_channels or num_classes")
        return int(out_channels)

    @staticmethod
    def _resolve_norm_type(model_config: ConfigDict) -> str:
        """
        Resolve normalization type.

        Supports new style:

            norm = "none" | "batchnorm" | "layernorm"

        and old style:

            batchnorm = True
            layernorm = True
        """
        if "norm" in model_config:
            return str(model_config.norm).lower()

        batchnorm = bool(cfg_get(model_config, "batchnorm", False))
        layernorm = bool(cfg_get(model_config, "layernorm", False))

        if batchnorm and layernorm:
            raise ValueError("Only one of batchnorm/layernorm can be True")

        if layernorm:
            return "layernorm"
        if batchnorm:
            return "batchnorm"
        return "none"

    def _attention_args(self, out_dim: int, is_last: bool) -> dict[str, Any]:
        """
        Build dimensional arguments for attention layers.

        Hidden layers may use multi-head concatenation.
        Output layer always uses heads=1 and concat=False.
        """
        if is_last:
            return {
                "out_channels": out_dim,
                "heads": 1,
                "concat": False,
            }

        heads = int(cfg_get(self.model_config, "heads", 1))
        concat = bool(cfg_get(self.model_config, "concat", heads > 1))

        if heads < 1:
            raise ValueError("heads must be positive")

        if concat:
            if out_dim % heads != 0:
                raise ValueError("hidden_channels must be divisible by heads when concat=True")
            conv_out = out_dim // heads
        else:
            conv_out = out_dim

        return {
            "out_channels": conv_out,
            "heads": heads,
            "concat": concat,
        }

    def _make_conv(
        self,
        in_channels: int,
        out_dim: int,
        is_last: bool,
    ) -> nn.Module:
        """Create one convolution layer."""
        if self.conv_type == "hop_gated_gatv2":
            return self._make_hop_gated_gatv2(in_channels, out_dim, is_last)

        if self.conv_type == "gatv2":
            return self._make_gatv2(in_channels, out_dim, is_last)

        if self.conv_type == "gat":
            return self._make_gat(in_channels, out_dim, is_last)

        if self.conv_type == "sage":
            return SAGEConv(
                in_channels=in_channels,
                out_channels=out_dim,
            )

        if self.conv_type == "gcn":
            return GCNConv(
                in_channels=in_channels,
                out_channels=out_dim,
                add_self_loops=bool(cfg_get(self.model_config, "add_self_loops", True)),
                bias=bool(cfg_get(self.model_config, "bias", True)),
            )

        raise ValueError(f"Unknown conv_type: {self.conv_type}")

    def _make_hop_gated_gatv2(
        self,
        in_channels: int,
        out_dim: int,
        is_last: bool,
    ) -> nn.Module:
        """
        Create HopGatedGATv2Conv.

        Requires HopGatedGATv2Conv to be defined or imported.
        """
        return HopGatedGATv2Conv(
            in_channels=in_channels,
            max_hops=self.max_hops,
            negative_slope=float(cfg_get(self.model_config, "negative_slope", 0.2)),
            dropout=self.attn_dropout,
            add_self_loops=bool(cfg_get(self.model_config, "add_self_loops", True)),
            edge_dim=self.edge_dim,
            fill_value=cfg_get(self.model_config, "fill_value", "mean"),
            bias=bool(cfg_get(self.model_config, "bias", True)),
            residual=bool(cfg_get(self.model_config, "conv_residual", False)),
            gate_hidden=cfg_get(self.model_config, "gate_hidden", None),
            gate_dropout=float(cfg_get(self.model_config, "gate_dropout", 0.0)),
            **self._attention_args(out_dim, is_last),
        )

    def _make_gatv2(
        self,
        in_channels: int,
        out_dim: int,
        is_last: bool,
    ) -> nn.Module:
        """Create ordinary PyG GATv2Conv."""
        return GATv2Conv(
            in_channels=in_channels,
            negative_slope=float(cfg_get(self.model_config, "negative_slope", 0.2)),
            dropout=self.attn_dropout,
            add_self_loops=bool(cfg_get(self.model_config, "add_self_loops", True)),
            edge_dim=self.edge_dim,
            fill_value=cfg_get(self.model_config, "fill_value", "mean"),
            bias=bool(cfg_get(self.model_config, "bias", True)),
            share_weights=bool(cfg_get(self.model_config, "share_weights", False)),
            **self._attention_args(out_dim, is_last),
        )

    def _make_gat(
        self,
        in_channels: int,
        out_dim: int,
        is_last: bool,
    ) -> nn.Module:
        """Create ordinary PyG GATConv."""
        return GATConv(
            in_channels=in_channels,
            negative_slope=float(cfg_get(self.model_config, "negative_slope", 0.2)),
            dropout=self.attn_dropout,
            add_self_loops=bool(cfg_get(self.model_config, "add_self_loops", True)),
            edge_dim=self.edge_dim,
            fill_value=cfg_get(self.model_config, "fill_value", "mean"),
            bias=bool(cfg_get(self.model_config, "bias", True)),
            **self._attention_args(out_dim, is_last),
        )

    def _make_norm(self, dim: int) -> nn.Module:
        """Create hidden normalization layer."""
        if self.norm_type == "batchnorm":
            return nn.BatchNorm1d(dim)
        if self.norm_type == "layernorm":
            return nn.LayerNorm(dim)
        return nn.Identity()

    def reset_parameters(self) -> None:
        """Reset all trainable parameters."""
        if self.pre_lin is not None:
            self.pre_lin.reset_parameters()

        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

        for proj in self.res_proj:
            proj.reset_parameters()

    def _validate_edge_attr(
        self,
        edges: list[Tensor],
        edge_attr: Optional[Tensor],
    ) -> None:
        """
        Validate edge attributes.

        edge_attr must correspond to the first edge set E_1.
        For hop-aware layers, edge_attr is passed into the first hop only.
        """
        if edge_attr is None:
            return

        if self.edge_dim is None:
            raise ValueError("edge_attr was provided, but model_config.edge_dim is None")

        if edge_attr.dim() != 2:
            raise ValueError("edge_attr must have shape [num_edges, edge_dim]")

        if edge_attr.size(-1) != self.edge_dim:
            raise ValueError(
                f"edge_attr.size(-1)={edge_attr.size(-1)} does not match edge_dim={self.edge_dim}"
            )

        if edge_attr.size(0) != edges[0].size(1):
            raise ValueError(
                "edge_attr must correspond to the first edge_index: "
                f"edge_attr.size(0)={edge_attr.size(0)}, edges[0].size(1)={edges[0].size(1)}"
            )

    def _activate(self, x: Tensor) -> Tensor:
        """Apply configured activation."""
        if self.act == "relu":
            return F.relu(x)
        if self.act == "elu":
            return F.elu(x)
        raise ValueError("act must be 'relu' or 'elu'")

    def _call_conv(
        self,
        conv: nn.Module,
        x: Tensor,
        edges: list[Tensor],
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """
        Call a hop-aware or ordinary convolution.

        Hop-aware conv receives all hop edge sets.
        Ordinary conv receives only the first edge set.
        """
        ei: EdgeInput = edges if self.use_hops else edges[0]

        if edge_attr is not None and self.use_edge_attr:
            return conv(x, ei, edge_attr=edge_attr)

        return conv(x, ei)

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x:
                Node features of shape [num_nodes, in_channels].
            edge_index:
                Either ordinary edge_index of shape [2, num_edges] or a list:
                [edge_index_1, ..., edge_index_K].
            edge_attr:
                Optional edge attributes for the first edge set E_1.

        Returns:
            Node logits of shape [num_nodes, out_channels].
        """
        edges = as_edge_list(edge_index)
        self._validate_edge_attr(edges, edge_attr)

        if self.pre_lin is not None:
            x = self.pre_lin(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            is_last = i == len(self.convs) - 1

            h = self._call_conv(
                conv=conv,
                x=x,
                edges=edges,
                edge_attr=edge_attr,
            )

            if is_last:
                return h

            if self.residual:
                h = h + self.res_proj[i](x)

            h = self.norms[i](h)
            h = self._activate(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            x = h

        return x
