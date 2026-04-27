from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from ml_collections import ConfigDict
from torch_geometric.nn.models import GAT, GCN

from new_model import GLANT


Metrics = Dict[str, Dict[str, List[float]]]
ModelRegistry = Dict[str, nn.Module]

GAT_MODEL = 'GAT'
GATV2_MODEL = 'GATv2'
GCN_MODEL = 'GCN'
GLANT_MODEL = 'GLANT'


def load_from_checkpoint(
    config: ConfigDict,
    ds_name: str,
    nrepeats: int,
) -> Metrics:
    """Load saved metrics for configured models."""
    model_names = config.baselines.names
    metric_names = config[ds_name.lower()].metrics
    metric_dict: Metrics = {
        model_name: {metric: [] for metric in metric_names}
        for model_name in model_names
    }

    for model_name, repeat_idx in product(model_names, range(nrepeats)):
        metrics_path = (
            Path('checkpoints')
            / ds_name
            / f'{model_name}{repeat_idx}'
            / 'metrics.txt'
        )

        if not metrics_path.exists():
            print(f'Metric file path {metrics_path} not found')
            continue

        with metrics_path.open('r', encoding='utf-8') as reader:
            for metric_line in reader:
                metric_name, metric_value = metric_line.split()
                metric_dict[model_name][metric_name].append(
                    float(metric_value)
                )

    return metric_dict


def save_to_checkpoint(model: nn.Module, save_dir: str) -> None:
    """Save a model object to checkpoint directory."""
    checkpoint_dir = Path(save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, checkpoint_dir / 'model.pt')


def assign_to_config(
    config: ConfigDict,
    opt: ConfigDict,
    training: bool = True,
) -> None:
    """Copy scalar options into a config object."""
    for key, best_param in opt.items():
        if isinstance(best_param, ConfigDict):
            continue

        if key in config and config[key] is not None:
            del config[key]

        config[key] = best_param

    if hasattr(opt, 'decay') and training:
        config.training.weight_decay = opt['decay']
        config.training.lr = opt['lr']
        config.dropout = opt['dropout']


def create_gat_model(
    config: ConfigDict,
    ds_config: ConfigDict,
) -> nn.Module:
    """Create a PyG GAT model from config."""
    model_config = config.baselines.GAT
    return GAT(
        in_channels=ds_config.in_channels,
        hidden_channels=model_config.hidden_channels,
        num_layers=model_config.num_layers,
        out_channels=ds_config.out_channels,
        heads=model_config.heads,
        dropout=model_config.dropout,
        act=getattr(model_config, 'act', 'relu'),
    )


def create_gatv2_model(
    config: ConfigDict,
    ds_config: ConfigDict,
) -> nn.Module:
    """Create a PyG GATv2 model from config."""
    model_config = config.baselines.GATv2
    return GAT(
        in_channels=ds_config.in_channels,
        hidden_channels=model_config.hidden_channels,
        num_layers=model_config.num_layers,
        out_channels=ds_config.out_channels,
        heads=model_config.heads,
        dropout=model_config.dropout,
        act=getattr(model_config, 'act', 'relu'),
        v2=True,
        share_weights=getattr(model_config, 'share_weights', False),
    )


def create_gcn_model(
    config: ConfigDict,
    ds_config: ConfigDict,
) -> nn.Module:
    """Create a PyG GCN model from config."""
    model_config = config.baselines.GCN
    return GCN(
        in_channels=ds_config.in_channels,
        hidden_channels=model_config.hidden_channels,
        num_layers=model_config.num_layers,
        out_channels=ds_config.out_channels,
        dropout=model_config.dropout,
    )


def raise_unavailable_model(model_name: str) -> None:
    """Fail clearly for custom models missing from this codebase."""
    raise NotImplementedError(
        f'Model {model_name} is configured but is not implemented in this '
        'repository. Missing legacy symbols were GNN and best_params_dict.'
    )


def get_baseline_config(config: ConfigDict, model_name: str) -> ConfigDict:
    """Return baseline config or fail with a clear message."""
    if model_name not in config.baselines:
        raise ValueError(f'Missing config for model: {model_name}')

    return config.baselines[model_name]


def create_multihop_gat_model(
    model_name: str,
    config: ConfigDict,
    ds_config: ConfigDict,
) -> nn.Module:
    """Create GLANT model backed by GenericGAT."""
    model_config = get_baseline_config(config, model_name)
    return GLANT(
        model_config,
        ds_config,
        device=config.device,
    )


def create_model(
    model_name: str,
    config: ConfigDict,
    ds_config: ConfigDict,
    data_dict: ConfigDict,
) -> nn.Module:
    """Create one model by name."""
    del data_dict

    print(f"Ds config ({model_name})\n: {ds_config}")

    if model_name == GAT_MODEL:
        return create_gat_model(config, ds_config)

    if model_name == GATV2_MODEL:
        return create_gatv2_model(config, ds_config)

    if model_name == GCN_MODEL:
        return create_gcn_model(config, ds_config)

    if model_name == GLANT_MODEL:
        return create_multihop_gat_model(model_name, config, ds_config)

    raise_unavailable_model(model_name)


def create_models(
    config: ConfigDict,
    ds_config: ConfigDict,
    data_dict: ConfigDict,
) -> ModelRegistry:
    """Create configured baseline models."""
    return {
        model_name: create_model(model_name, config, ds_config, data_dict)
        for model_name in config.baselines.names
    }
