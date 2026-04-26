import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from configs.config import all_config

from train import meta_train
from utils.model_utils import (
    create_models,
    load_from_checkpoint
)
from utils.data_utils import (
    fetch_dataset, 
    ds_cfg
)

SUPPORTED_GPU_IDS = {0, 1}


def json_list(string: str) -> List[Any]:
    try:
        value = json.loads(string)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError('Input must be a JSON list') from exc

    if not isinstance(value, list):
        raise argparse.ArgumentTypeError('Input must be a JSON list')

    return value


def configure_device(config: Any, gpu: Optional[int]) -> None:
    if gpu is None:
        return

    if gpu == -1:
        config.device = torch.device('cpu')
        return

    if gpu not in SUPPORTED_GPU_IDS:
        raise ValueError(
            f'Unsupported GPU id: {gpu}. '
            f'Supported values: -1, {sorted(SUPPORTED_GPU_IDS)}'
        )

    config.device = torch.device(f'cuda:{gpu}')


def apply_cli_overrides(config: Any, pargs: argparse.Namespace) -> None:
    if pargs.runs is not None:
        config.experiments.runs = pargs.runs

    if pargs.method is not None:
        print('Starting run with sampling method')
        config.baselines.names = ['GLANT']
        config.baselines.GLANT.load_samples = False
        config.baselines.GLANT.select_method = pargs.method
        config.baselines.GLANT.K_hops = 3
        config.baselines.GLANT.num_heads = [8, 1]

    if pargs.khop is not None:
        config.baselines.GLANT.K_hops = pargs.khop
        config.baselines.GLANT.load_samples = False

    if pargs.model is not None:
        config.baselines.names = [pargs.model]

    configure_device(config, pargs.gpu)


def execute_run(
    config: Any,
    ds_config: Any,
    data_dict: Any,
    loss: nn.Module,
    pargs: argparse.Namespace,
) -> Any:
    if pargs.train == pargs.test:
        raise ValueError('Exactly one mode must be selected: --train or --test')

    if pargs.train:
        models = create_models(config, ds_config, data_dict)
        return meta_train(config, ds_config, models, data_dict, loss)

    runs = config.experiments.runs
    return load_from_checkpoint(config, pargs.dataset, runs)


def get_selected_method(config: Any) -> Optional[str]:
    model_name = config.baselines.names[0]
    model_config = config.baselines.get(model_name)
    if model_config is None:
        return None

    return getattr(model_config, 'select_method', None)


def print_metrics(
    config: Any,
    ds_config: Any,
    all_metrics: Dict[str, Dict[str, List[float]]],
) -> None:
    method = get_selected_method(config)
    print(all_metrics)
    for model_name, metric_dict in all_metrics.items():
        acc = metric_dict['Accuracy']
        print(f'{ds_config.name} result with method {method}')
        print(model_name, np.mean(acc), f"+-{np.std(acc)}")


def run_experiment(pargs: argparse.Namespace) -> Any:
    config = all_config()
    apply_cli_overrides(config, pargs)

    ds_config = ds_cfg(config, pargs.dataset)
    print(f"Fetching Dataset: {pargs.dataset}")
    data_dict = fetch_dataset(config, pargs.dataset)
    print(f"Fetching Dataset - done")

    loss = nn.CrossEntropyLoss()
    print('\nStarting run\n')
    all_metrics = execute_run(config, ds_config, data_dict, loss, pargs)
    print_metrics(config, ds_config, all_metrics)

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='The name of the dataset to run',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='Override GPU from the config file; use -1 for CPU',
    )
    parser.add_argument(
        '--khop',
        type=int,
        default=None,
        help='Override GLANT K-hop value',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints',
        help='Checkpoint directory for loading models',
    )
    parser.add_argument(
        '--load',
        type=json_list,
        nargs='+',
        default=None,
        help='Load models from a JSON list',
    )
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help='Sampling method ablation',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override model from the config file',
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=None,
        help='Override number of experiment repeats',
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true', help='Train the model')
    mode.add_argument('--test', action='store_true', help='Test the model')

    args = parser.parse_args()
    run_experiment(args)
