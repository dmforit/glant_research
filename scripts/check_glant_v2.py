from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import all_config
from model import GLANT, LambdaHopGatedGATv2Conv
from utils.model_utils import create_model


def make_ds_config(num_nodes: int = 12, in_channels: int = 5, out_channels: int = 3) -> ConfigDict:
    cfg = ConfigDict()
    cfg.in_channels = in_channels
    cfg.out_channels = out_channels
    cfg.num_classes = out_channels
    cfg.num_nodes = num_nodes
    cfg.name = "SyntheticGLANTv2Check"
    return cfg


def make_edges() -> tuple[torch.Tensor, list[torch.Tensor]]:
    edge_index_1 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6, 8],
        ],
        dtype=torch.long,
    )

    edge_index_2 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )

    edge_index_3 = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    return edge_index_1, [edge_index_1, edge_index_2, edge_index_3]


def assert_output(name: str, out: torch.Tensor, num_nodes: int, out_channels: int) -> None:
    expected = (num_nodes, out_channels)

    if tuple(out.shape) != expected:
        raise AssertionError(f"{name}: output shape {tuple(out.shape)} != {expected}")

    if not torch.isfinite(out).all():
        raise AssertionError(f"{name}: output contains NaN/inf")


def make_config(lambda_higher: float, max_hops: int = 3) -> ConfigDict:
    cfg = all_config()
    cfg.device = torch.device("cpu")
    cfg.baselines.GLANT_v2.lambda_higher = lambda_higher
    cfg.baselines.GLANT_v2.learn_lambda_higher = False
    cfg.baselines.GLANT_v2.max_hops = max_hops
    cfg.baselines.GLANT_v2.edge_dim = None
    return cfg


def create_glant_v2(lambda_higher: float, max_hops: int, ds_config: ConfigDict) -> GLANT:
    cfg = make_config(lambda_higher=lambda_higher, max_hops=max_hops)
    model = create_model("glant_v2", cfg, ds_config)

    if not isinstance(model, GLANT):
        raise AssertionError(f"glant_v2 should create GLANT wrapper, got {type(model)}")

    return model


def glant_v2_layers(model: GLANT) -> list[LambdaHopGatedGATv2Conv]:
    layers = [conv for conv in model.convs if isinstance(conv, LambdaHopGatedGATv2Conv)]

    if not layers:
        raise AssertionError("GLANT-v2 model has no LambdaHopGatedGATv2Conv layers")

    return layers


def check_model_factory() -> None:
    ds_config = make_ds_config()
    model = create_glant_v2(lambda_higher=0.5, max_hops=3, ds_config=ds_config)
    glant_v2_layers(model)
    print("ok model factory creates glant_v2")


def check_forward_k_values() -> None:
    torch.manual_seed(0)

    ds_config = make_ds_config()
    x = torch.randn(ds_config.num_nodes, ds_config.in_channels)
    edge_index_1, edge_index_list = make_edges()

    for max_hops, edges in [
        (1, [edge_index_1]),
        (2, edge_index_list[:2]),
        (3, edge_index_list),
    ]:
        model = create_glant_v2(lambda_higher=0.5, max_hops=max_hops, ds_config=ds_config)
        model.eval()

        with torch.no_grad():
            out = model(x=x, edge_index=edges, edge_attr=None)

        assert_output(f"forward K={max_hops}", out, ds_config.num_nodes, ds_config.out_channels)

    print("ok forward K=1/K=2/K=3")


def check_empty_higher_hop() -> None:
    torch.manual_seed(0)

    ds_config = make_ds_config()
    x = torch.randn(ds_config.num_nodes, ds_config.in_channels)
    edge_index_1, edge_index_list = make_edges()

    edges = [
        edge_index_1,
        edge_index_list[1],
        torch.empty((2, 0), dtype=torch.long),
    ]

    model = create_glant_v2(lambda_higher=0.5, max_hops=3, ds_config=ds_config)
    model.eval()

    with torch.no_grad():
        out = model(x=x, edge_index=edges, edge_attr=None)

    assert_output("empty higher hop", out, ds_config.num_nodes, ds_config.out_channels)
    print("ok empty higher-order hop")


def check_lambda_zero_independence() -> None:
    torch.manual_seed(0)

    ds_config = make_ds_config()
    x = torch.randn(ds_config.num_nodes, ds_config.in_channels)
    edge_index_1, edge_index_list = make_edges()

    model = create_glant_v2(lambda_higher=0.0, max_hops=3, ds_config=ds_config)
    model.eval()

    empty_higher_edges = [
        edge_index_1,
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
    ]

    changed_higher_edges = [
        edge_index_1,
        edge_index_list[2],
        edge_index_list[1],
    ]

    with torch.no_grad():
        out_normal = model(x=x, edge_index=edge_index_list, edge_attr=None)
        out_empty = model(x=x, edge_index=empty_higher_edges, edge_attr=None)
        out_changed = model(x=x, edge_index=changed_higher_edges, edge_attr=None)

    if not torch.allclose(out_normal, out_empty, atol=1e-6):
        max_diff = float((out_normal - out_empty).abs().max())
        raise AssertionError(f"lambda=0 depends on empty higher-order edges, max_diff={max_diff}")

    if not torch.allclose(out_normal, out_changed, atol=1e-6):
        max_diff = float((out_normal - out_changed).abs().max())
        raise AssertionError(f"lambda=0 depends on changed higher-order edges, max_diff={max_diff}")

    print("ok lambda=0 disables higher-order part")


def check_beta_normalization() -> None:
    torch.manual_seed(0)

    ds_config = make_ds_config()
    x = torch.randn(ds_config.num_nodes, ds_config.in_channels)
    _, edge_index_list = make_edges()

    model = create_glant_v2(lambda_higher=0.5, max_hops=3, ds_config=ds_config)
    model.eval()

    first_layer = glant_v2_layers(model)[0]

    with torch.no_grad():
        _, diagnostics = first_layer(
            x,
            edge_index_list,
            edge_attr=None,
            return_hop_diagnostics=True,
            return_attention_weights=True,
        )

    beta = diagnostics["weights"]

    if tuple(beta.shape) != (ds_config.num_nodes, 2):
        raise AssertionError(f"beta shape {tuple(beta.shape)} != {(ds_config.num_nodes, 2)}")

    if not torch.isfinite(beta).all():
        raise AssertionError("beta contains NaN/inf")

    if (beta < 0).any():
        raise AssertionError("beta contains negative values")

    sums = beta.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise AssertionError("beta does not sum to 1 over higher-order hops")

    if diagnostics["lambda_higher"] != 0.5:
        raise AssertionError("diagnostics lambda_higher mismatch")

    if diagnostics["one_hop_weight"] != 0.5:
        raise AssertionError("diagnostics one_hop_weight mismatch")

    if diagnostics["higher_order_weight"] != 0.5:
        raise AssertionError("diagnostics higher_order_weight mismatch")

    print("ok beta normalization and lambda diagnostics")


def check_lambda_one_formula() -> None:
    torch.manual_seed(0)

    ds_config = make_ds_config()
    x = torch.randn(ds_config.num_nodes, ds_config.in_channels)
    _, edge_index_list = make_edges()

    model = create_glant_v2(lambda_higher=1.0, max_hops=3, ds_config=ds_config)
    model.eval()

    first_layer = glant_v2_layers(model)[0]

    with torch.no_grad():
        _, diagnostics = first_layer(
            x,
            edge_index_list,
            edge_attr=None,
            return_hop_diagnostics=True,
            return_attention_weights=False,
        )

    if diagnostics["one_hop_weight"] != 0.0:
        raise AssertionError("lambda=1 should set one_hop_weight=0")

    if diagnostics["higher_order_weight"] != 1.0:
        raise AssertionError("lambda=1 should set higher_order_weight=1")

    print("ok lambda=1 disables 1-hop coefficient")


def check_learned_lambda_parameter() -> None:
    torch.manual_seed(0)

    ds_config = make_ds_config()
    x = torch.randn(ds_config.num_nodes, ds_config.in_channels)
    _, edge_index_list = make_edges()
    cfg = make_config(lambda_higher=0.0, max_hops=3)
    cfg.baselines.GLANT_v2.learn_lambda_higher = True

    model = GLANT(cfg.baselines.GLANT_v2, ds_config)
    first_layer = glant_v2_layers(model)[0]

    if first_layer.lambda_logit is None:
        raise AssertionError("learn_lambda_higher=True did not create lambda_logit")

    lambda_value = torch.sigmoid(first_layer.lambda_logit.detach()).item()
    if not 0.0 < lambda_value < 1.0:
        raise AssertionError(f"learned lambda should be inside (0, 1), got {lambda_value}")

    out = model(x=x, edge_index=edge_index_list, edge_attr=None)
    loss = out.pow(2).mean()
    loss.backward()

    if first_layer.lambda_logit.grad is None:
        raise AssertionError("lambda_logit did not receive a gradient")

    print("ok learned lambda parameter receives gradient")


def check_edge_attr_first_hop_only() -> None:
    torch.manual_seed(0)

    ds_config = make_ds_config()
    x = torch.randn(ds_config.num_nodes, ds_config.in_channels)
    _, edge_index_list = make_edges()

    cfg = make_config(lambda_higher=0.5, max_hops=3)
    cfg.baselines.GLANT_v2.edge_dim = 4

    edge_attr = torch.randn(edge_index_list[0].size(1), 4)

    model = create_model("glant_v2", cfg, ds_config)
    model.eval()

    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list, edge_attr=edge_attr)

    assert_output("edge_attr first hop only", out, ds_config.num_nodes, ds_config.out_channels)
    print("ok edge_attr is accepted for first hop")


def check_parameter_sharing() -> None:
    ds_config = make_ds_config()
    model = create_glant_v2(lambda_higher=0.5, max_hops=3, ds_config=ds_config)

    for layer_id, conv in enumerate(glant_v2_layers(model)):
        conv.assert_hop_invariants()
        print(f"ok shared W_l/W_r and separate attention vectors, layer={layer_id}")


def main() -> None:
    check_model_factory()
    check_parameter_sharing()
    check_forward_k_values()
    check_empty_higher_hop()
    check_lambda_zero_independence()
    check_beta_normalization()
    check_lambda_one_formula()
    check_learned_lambda_parameter()
    check_edge_attr_first_hop_only()

    print("All GLANT-v2 checks passed.")


if __name__ == "__main__":
    main()
