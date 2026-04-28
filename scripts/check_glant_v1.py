from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import copy

import torch
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import all_config
from model import GLANT, HopGatedGATv2Conv
from utils.model_utils import create_model


def make_synthetic_inputs(
    *,
    num_nodes: int = 12,
    in_channels: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], ConfigDict]:
    x = torch.randn(num_nodes, in_channels)

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

    ds_config = ConfigDict()
    ds_config.in_channels = in_channels
    ds_config.out_channels = 3
    ds_config.num_classes = 3
    ds_config.num_nodes = num_nodes
    ds_config.name = "SyntheticGLANTCheck"

    return x, edge_index_1, [edge_index_1, edge_index_2, edge_index_3], ds_config


def assert_output(name: str, out: torch.Tensor, num_nodes: int, out_channels: int) -> None:
    expected = (num_nodes, out_channels)

    if tuple(out.shape) != expected:
        raise AssertionError(f"{name}: output shape {tuple(out.shape)} != {expected}")

    if not torch.isfinite(out).all():
        raise AssertionError(f"{name}: output contains NaN/inf")


def assert_hop_weights(weights: torch.Tensor, num_nodes: int, num_hops: int) -> None:
    expected = (num_nodes, num_hops)

    if tuple(weights.shape) != expected:
        raise AssertionError(f"hop weights shape {tuple(weights.shape)} != {expected}")

    if not torch.isfinite(weights).all():
        raise AssertionError("hop weights contain NaN/inf")

    if (weights < 0).any():
        raise AssertionError("hop weights contain negative values after softmax")

    sums = weights.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise AssertionError(
            f"hop weights do not sum to 1 per node: "
            f"min={float(sums.min())}, max={float(sums.max())}"
        )


def hop_layers(model: GLANT) -> list[HopGatedGATv2Conv]:
    layers = [conv for conv in model.convs if isinstance(conv, HopGatedGATv2Conv)]

    if not layers:
        raise AssertionError("GLANT model has no HopGatedGATv2Conv layers")

    return layers


def check_parameter_sharing(model: GLANT) -> None:
    for layer_id, conv in enumerate(hop_layers(model)):
        conv.assert_hop_invariants()
        print(f"ok shared W_l/W_r and separate attention vectors, layer={layer_id}")


def check_direct_conv_diagnostics(
    model: GLANT,
    x: torch.Tensor,
    edge_index_list: list[torch.Tensor],
) -> None:
    first_hop_layer = hop_layers(model)[0]
    first_hop_layer.eval()

    with torch.no_grad():
        out, diagnostics = first_hop_layer(
            x,
            edge_index_list,
            edge_attr=None,
            return_hop_diagnostics=True,
            return_attention_weights=True,
        )

    assert_output(
        "direct HopGatedGATv2Conv",
        out,
        x.size(0),
        first_hop_layer.out_dim,
    )

    weights = diagnostics["weights"]
    assert_hop_weights(weights, x.size(0), len(edge_index_list))

    hop_logits = diagnostics.get("hop_logits")
    if hop_logits is None:
        raise AssertionError("diagnostics does not contain hop_logits")

    if tuple(hop_logits.shape) != (x.size(0), len(edge_index_list)):
        raise AssertionError(
            f"hop_logits shape {tuple(hop_logits.shape)} != "
            f"{(x.size(0), len(edge_index_list))}"
        )

    messages_shape = diagnostics.get("messages_shape")
    expected_messages_shape = [x.size(0), len(edge_index_list), first_hop_layer.out_dim]
    if messages_shape != expected_messages_shape:
        raise AssertionError(
            f"messages_shape {messages_shape} != {expected_messages_shape}"
        )

    attention = diagnostics.get("attention", [])
    if len(attention) != len(edge_index_list):
        raise AssertionError(
            f"attention diagnostics length {len(attention)} != {len(edge_index_list)}"
        )

    print("ok direct hop diagnostics")


def check_forward_modes(
    model: GLANT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_index_list: list[torch.Tensor],
    out_channels: int,
) -> None:
    model.eval()

    with torch.no_grad():
        out = model(x=x, edge_index=edge_index, edge_attr=None)
    assert_output("GLANT edge_index", out, x.size(0), out_channels)
    print("ok GLANT forward edge_index")

    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list[:1], edge_attr=None)
    assert_output("GLANT edge_index_list K=1", out, x.size(0), out_channels)
    print("ok GLANT forward edge_index_list K=1")

    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list[:2], edge_attr=None)
    assert_output("GLANT edge_index_list K=2", out, x.size(0), out_channels)
    print("ok GLANT forward edge_index_list K=2")

    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list, edge_attr=None)
    assert_output("GLANT edge_index_list K=3", out, x.size(0), out_channels)
    print("ok GLANT forward edge_index_list K=3")


def check_empty_higher_hop(
    model: GLANT,
    x: torch.Tensor,
    edge_index_list: list[torch.Tensor],
    out_channels: int,
) -> None:
    edge_index_list_with_empty = [
        edge_index_list[0],
        edge_index_list[1],
        torch.empty((2, 0), dtype=torch.long),
    ]

    model.eval()
    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list_with_empty, edge_attr=None)

    assert_output("GLANT empty higher hop", out, x.size(0), out_channels)
    print("ok GLANT empty higher-order hop")


def check_edge_attr_first_hop(
    config: ConfigDict,
    x: torch.Tensor,
    edge_index_list: list[torch.Tensor],
    ds_config: ConfigDict,
) -> None:
    edge_dim = 4
    edge_attr = torch.randn(edge_index_list[0].size(1), edge_dim)

    config.baselines.GLANT_v1.edge_dim = edge_dim
    model = create_model("glant_v1", config, ds_config)
    model.eval()

    with torch.no_grad():
        out = model(x=x, edge_index=edge_index_list, edge_attr=edge_attr)

    assert_output("GLANT edge_attr first hop", out, x.size(0), ds_config.out_channels)
    print("ok GLANT edge_attr for first hop only")


def check_logging_call(
    model: GLANT,
    x: torch.Tensor,
    edge_index_list: list[torch.Tensor],
    out_channels: int,
) -> None:
    log_path = Path("results/debug/check_glant_v1/hop_weights.csv")

    if log_path.exists():
        log_path.unlink()

    summary_path = log_path.with_name(f"{log_path.stem}_summary.csv")
    if summary_path.exists():
        summary_path.unlink()

    model.eval()
    with torch.no_grad():
        out = model(
            x=x,
            edge_index=edge_index_list,
            edge_attr=None,
            log_hop_diagnostics=True,
            hop_log_path=str(log_path),
            epoch=0,
            phase="val",
            lr=0.0,
            log_only_layer=0,
        )

    assert_output("GLANT logging call", out, x.size(0), out_channels)

    # Current GLANT behavior:
    # - if hop_log_path has .csv suffix, diagnostics are written directly to that file;
    # - otherwise GLANT writes {stem}_summary.csv.
    existing_paths = [path for path in (log_path, summary_path) if path.exists()]

    if not existing_paths:
        raise AssertionError(
            "expected diagnostics file was not created. "
            f"Checked: {log_path} and {summary_path}"
        )

    for path in existing_paths:
        if path.stat().st_size == 0:
            raise AssertionError(f"diagnostics file is empty: {path}")

    print(f"ok GLANT diagnostics logging: {existing_paths[0]}")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        if lowered == "auto":
            raise AssertionError("logging policy left value as 'auto'")

    return bool(value)


def check_hpo_vs_final_flags(config: ConfigDict) -> None:
    from utils.result_logging import resolve_logging_policy

    hpo_config = copy.deepcopy(config.baselines.GLANT_v1)
    final_config = copy.deepcopy(config.baselines.GLANT_v1)

    resolve_logging_policy("glant_v1", hpo_config, "hpo")
    resolve_logging_policy("glant_v1", final_config, "final")

    hpo_attention_scores = _as_bool(hpo_config.log_attention_scores)
    hpo_attention_statistics = _as_bool(hpo_config.log_attention_statistics)

    final_attention_scores = _as_bool(final_config.log_attention_scores)
    final_attention_statistics = _as_bool(final_config.log_attention_statistics)

    if hpo_attention_scores:
        raise AssertionError("hpo mode should disable log_attention_scores")

    if hpo_attention_statistics:
        raise AssertionError("hpo mode should disable log_attention_statistics")

    if not final_attention_scores:
        raise AssertionError("final mode should enable log_attention_scores")

    if not final_attention_statistics:
        raise AssertionError("final mode should enable log_attention_statistics")

    print("ok GLANT hpo/final logging policy")


def main() -> None:
    torch.manual_seed(0)

    config = all_config()
    config.device = torch.device("cpu")

    x, edge_index, edge_index_list, ds_config = make_synthetic_inputs()

    model = create_model("glant_v1", config, ds_config)
    if not isinstance(model, GLANT):
        raise AssertionError(f"glant_v1 should create GLANT, got {type(model)}")

    check_parameter_sharing(model)
    check_forward_modes(model, x, edge_index, edge_index_list, ds_config.out_channels)
    check_empty_higher_hop(model, x, edge_index_list, ds_config.out_channels)
    check_direct_conv_diagnostics(model, x, edge_index_list)
    check_logging_call(model, x, edge_index_list, ds_config.out_channels)

    # Build a fresh config/model because edge_dim mutates the config.
    config_edge = all_config()
    config_edge.device = torch.device("cpu")
    check_edge_attr_first_hop(config_edge, x, edge_index_list, ds_config)

    check_hpo_vs_final_flags(all_config())

    print("All GLANT-v1 checks passed.")


if __name__ == "__main__":
    main()
