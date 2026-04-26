from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from torch.optim.lr_scheduler import ExponentialLR

from utils.model_utils import save_to_checkpoint


Masks = Dict[str, torch.Tensor]
ModelArgs = Dict[str, torch.Tensor]
MetricCallables = Dict[str, Callable[[nn.Module, Any], float]]
MetricHistory = Dict[str, Dict[str, List[float]]]

MULTIHOP_MODEL_NAMES = {'GLANT'}


def get_args(
    data: Any,
    device: torch.device,
) -> Tuple[ModelArgs, torch.Tensor, Masks]:
    """Prepare model kwargs, labels and masks for a PyG dataset."""
    graph = data[0]
    args = {
        'x': graph.x.to(device),
        'edge_index': data.edge_index,
    }
    labels = graph.y.to(device)
    masks = {
        'train': graph.train_mask.to(device),
        'val': graph.val_mask.to(device),
        'test': graph.test_mask.to(device),
    }

    return args, labels, masks


def accuracy(model: nn.Module, data: Any, device: torch.device) -> float:
    """Compute test accuracy."""
    model.eval()
    args, labels, masks = get_args(data, device)

    with torch.no_grad():
        pred = model(**args).argmax(dim=-1)

    test_mask = masks['test']
    return int((pred[test_mask] == labels[test_mask]).sum()) / int(
        test_mask.sum()
    )


def get_metric_functions(
    ds_config: ConfigDict,
    device: torch.device,
) -> MetricCallables:
    """Return metric callables requested by dataset config."""
    metrics: MetricCallables = {}

    for metric_name in ds_config.metrics: # type: ignore
        if metric_name == 'Accuracy':
            metrics[metric_name] = (
                lambda model, data: accuracy(model, data, device)
            )
        else:
            raise ValueError(f'Unsupported metric: {metric_name}')

    return metrics


def collect_metrics(
    model: nn.Module,
    data: Any,
    metric_callables: MetricCallables,
) -> Dict[str, float]:
    """Evaluate all configured metrics."""
    return {
        name: metric(model, data)
        for name, metric in metric_callables.items()
    }


def join_metrics(
    history: MetricHistory,
    metrics: Dict[str, float],
    model_name: str,
) -> MetricHistory:
    """Append one run of metrics to the accumulated history."""
    model_metrics = history.setdefault(model_name, {})

    for metric_name, metric_value in metrics.items():
        model_metrics.setdefault(metric_name, []).append(metric_value)

    return history


def get_val_loss(
    model: nn.Module,
    data: Any,
    loss: nn.Module,
    device: torch.device,
) -> float:
    """Return validation loss for model."""
    model.eval()
    args, labels, masks = get_args(data, device)

    with torch.no_grad():
        pred = model(**args)
        val_loss = loss(pred[masks['val']], labels[masks['val']]).item()

    return val_loss


def train_step_normal(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    data: Any,
    device: torch.device,
    loss_func: nn.Module,
) -> float:
    """Run one standard supervised training step."""
    model.train()
    optimiser.zero_grad()

    args, labels, masks = get_args(data, device)
    pred = model(**args)
    train_loss = loss_func(pred[masks['train']], labels[masks['train']])

    train_loss.backward()
    optimiser.step()

    return train_loss.item()


def test(
    model: nn.Module,
    data: Any,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return train, validation and test accuracy."""
    model.eval()
    args, labels, masks = get_args(data, device)

    with torch.no_grad():
        pred = model(**args).argmax(dim=-1)

    accs = []
    for mask in masks.values():
        correct = int((pred[mask] == labels[mask]).sum())
        accs.append(correct / int(mask.sum()))

    return accs[0], accs[1], accs[2]


def create_optimizer(
    model: nn.Module,
    model_config: ConfigDict,
) -> torch.optim.Optimizer:
    """Create optimizer from model config."""
    training = model_config.training

    if training.optimizer == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            weight_decay=training.weight_decay,
            lr=training.lr,
        )

    raise ValueError(
        f'Invalid optimizer name {training.optimizer} in configuration file'
    )


def print_epoch_summary(
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    test_acc: float,
    best_val_acc: float,
    best_test_acc: float,
) -> None:
    """Print compact training progress."""
    print(
        'Epoch:',
        epoch,
        'Train Loss',
        round(train_loss, 4),
        'Val Loss',
        round(val_loss, 4),
        'Train Acc:',
        round(train_acc, 4),
        'Val Acc:',
        round(val_acc, 4),
        'Test Acc:',
        round(test_acc, 4),
        'Best Val Acc:',
        round(best_val_acc, 4),
        'Best Test Acc:',
        round(best_test_acc, 4),
    )


def train_model(
    model_config: ConfigDict,
    model: nn.Module,
    data: Any,
    loss: nn.Module,
    save_dir: Path,
    device: torch.device,
    do_save: bool = True,
) -> Tuple[List[float], List[float]]:
    """Train model and return train/validation loss curves."""
    optimiser = create_optimizer(model, model_config)
    scheduler = ExponentialLR(optimiser, gamma=model_config.training.decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    train_losses: List[float] = []
    val_losses: List[float] = []

    model = model.to(device)
    for epoch in range(model_config.training.num_epochs):
        train_loss = train_step_normal(
            model,
            optimiser,
            data,
            device=device,
            loss_func=loss,
        )
        val_loss = get_val_loss(model, data, loss, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        train_acc, val_acc, test_acc = test(model, data, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            if do_save:
                save_to_checkpoint(model, str(save_dir))

        should_log = (
            epoch % model_config.training.save_freq == 0
            or epoch == 0
        )
        if should_log:
            print_epoch_summary(
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                test_acc,
                best_val_acc,
                best_test_acc,
            )

    return train_losses, val_losses


def select_dataset_for_model(model_name: str, dataset: ConfigDict) -> Any:
    """Return regular or multihop dataset for model."""
    if model_name in MULTIHOP_MODEL_NAMES:
        return dataset['multihop_dataset']

    return dataset['dataset']


def reset_model(model: nn.Module) -> None:
    """Reset parameters when supported by the model."""
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


def save_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    save_path: Path,
) -> None:
    """Save train/validation loss curves."""
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks([
        idx + 1
        for idx in range(len(train_loss))
        if (idx + 1) % 10 == 0 or idx == 0
    ])
    plt.xlim([1, len(train_loss)])
    plt.title('Training Curves')
    plt.legend(['train', 'val'])
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics: Dict[str, float], save_path: Path) -> None:
    """Save metric dictionary as two-column text file."""
    rows = [[str(name), str(value)] for name, value in metrics.items()]
    np.savetxt(save_path, rows, fmt='%s')


def save_config(config: ConfigDict, save_path: Path) -> None:
    """Save config to YAML without non-serializable device."""
    config_dict = config.to_dict()
    config_dict.pop('device', None)

    with save_path.open('w', encoding='utf-8') as writer:
        yaml.dump(config_dict, writer)


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a full model object saved by this project."""
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def meta_train(
    config: ConfigDict,
    ds_config: ConfigDict,
    models: Dict[str, nn.Module],
    dataset: ConfigDict,
    loss: nn.Module,
) -> MetricHistory:
    """Run repeated training experiments for all configured models."""
    num_repeats = config.experiments.runs
    device = config.device
    metric_callables = get_metric_functions(ds_config, device)
    meta_metrics: MetricHistory = {model_name: {} for model_name in models}
    glant_masks = None

    for repeat_idx, (model_name, model) in product(
        range(num_repeats),
        models.items(),
    ):
        data = select_dataset_for_model(model_name, dataset)
        reset_model(model)
        if model_name in MULTIHOP_MODEL_NAMES:
            if not glant_masks:
                print("Creating masks for GLANT")
                print(f"Length: {len(data.edge_index)}")
                glant_masks = model.create_masks(
                    edge_index=data.edge_index,
                    set_mask=True
                )
                data.edge_index = model.drop_edges(data.edge_index)


        model = model.to(device)

        run_dir = (
            Path('checkpoints')
            / ds_config.name
            / f'{model_name}{repeat_idx}'
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        
        model_config = config.baselines[model_name]
        print(f"Run: {repeat_idx}, Training model {model_name}")
        train_loss, val_loss = train_model(
            model_config,
            model,
            data,
            loss,
            run_dir,
            device,
        )
        print(f"Run: {repeat_idx}, Training completed")
        print("Loading for evaluation")
        model = load_model_checkpoint(run_dir / 'model.pt', device)
        print(f"Collecting all metrics")
        metrics = collect_metrics(model, data, metric_callables)

        print(f"Saving metrics")
        save_loss_curves(train_loss, val_loss, run_dir / 'loss_curves.pdf')
        save_metrics(metrics, run_dir / 'metrics.txt')
        save_config(config, run_dir / 'config.yml')

        meta_metrics = join_metrics(meta_metrics, metrics, model_name)

    return meta_metrics
