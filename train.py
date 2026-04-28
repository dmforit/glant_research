from __future__ import annotations

from copy import copy
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
)

from model import HopEdgeSparsifier
from utils.logger import logger
from utils.model_names import canonical_model_name
from utils.model_utils import save_to_checkpoint
from utils.result_logging import (
    config_bool,
    export_glant_diagnostics,
    is_glant_model,
    raw_run_dir,
    resolve_logging_policy,
    run_seed,
    set_random_seed,
    write_config_json,
    write_metrics_csv,
)


Masks = Dict[str, torch.Tensor]
ModelArgs = Dict[str, torch.Tensor]
MetricCallables = Dict[str, Callable[[nn.Module, Any], float]]
MetricHistory = Dict[str, Dict[str, List[float]]]

HOP_AWARE_CONVS = {"hop_gated_gatv2", "lambda_hop_gated_gatv2", "hoga"}


def select_mask_column(mask: torch.Tensor, split_idx: int = 0) -> torch.Tensor:
    if mask.dim() <= 1:
        return mask

    return mask[:, split_idx]


def is_hop_aware_config(model_config: ConfigDict) -> bool:
    return str(getattr(model_config, "conv_type", "")).lower() in HOP_AWARE_CONVS


def edge_counts(edge_index: Any) -> List[int]:
    if torch.is_tensor(edge_index):
        return [int(edge_index.size(1))]

    return [int(edge.size(1)) for edge in edge_index]


def log_sparsification_progress(
    before: List[int],
    after: List[int],
    alpha: float,
    enabled: bool,
) -> None:
    if len(before) <= 1:
        logger.info("Edge sparsification skipped: no higher-hop edge sets")
        return

    total_steps = len(before) - 1
    logger.info(
        "Edge sparsification started: alpha=%s enabled=%s, 0/%s higher-hop sets complete",
        alpha,
        enabled,
        total_steps,
    )

    for step, hop_idx in enumerate(range(1, len(before)), start=1):
        before_count = before[hop_idx]
        after_count = after[hop_idx]
        kept_ratio = 0.0 if before_count == 0 else after_count / before_count
        logger.info(
            (
                "Edge sparsification progress: %s/%s complete for hop E_%s "
                "(kept %s/%s, %.2f%%)"
            ),
            step,
            total_steps,
            hop_idx + 1,
            after_count,
            before_count,
            kept_ratio * 100,
        )

    logger.info(
        "Edge sparsification done: kept %s/%s higher-hop edges",
        sum(after[1:]),
        sum(before[1:]),
    )


def select_dataset_for_model(
    model_name: str,
    dataset: ConfigDict,
    config: ConfigDict,
) -> Any:
    model_name = canonical_model_name(model_name)
    model_config = config.baselines[model_name]

    if is_hop_aware_config(model_config):
        if "multihop_dataset" not in dataset:
            raise ValueError(
                f"{model_name} requires multihop_dataset, but it was not built"
            )
        return dataset["multihop_dataset"]

    return dataset["dataset"]


def sparsify_dataset_edges(data: Any, model_config: ConfigDict) -> Any:
    if not is_hop_aware_config(model_config):
        return data

    edge_index = data.edge_index

    if torch.is_tensor(edge_index):
        return data

    before_counts = edge_counts(edge_index)
    alpha = float(getattr(model_config, "alpha", 1.0))
    enabled = bool(getattr(model_config, "sparsify_hops", True))
    sparsifier = HopEdgeSparsifier(
        alpha=alpha,
        cache_masks=bool(getattr(model_config, "sparsifier_cache_masks", True)),
        enabled=enabled,
    )

    out = copy(data)
    out.edge_index = sparsifier(edge_index)
    log_sparsification_progress(
        before=before_counts,
        after=edge_counts(out.edge_index),
        alpha=alpha,
        enabled=enabled,
    )

    return out


def get_args(
    data: Any,
    device: torch.device,
) -> Tuple[ModelArgs, torch.Tensor, Masks]:
    graph = data[0]
    args = {
        "x": graph.x.to(device),
        "edge_index": data.edge_index,
    }
    labels = graph.y.to(device)
    masks = {
        "train": select_mask_column(graph.train_mask).to(device),
        "val": select_mask_column(graph.val_mask).to(device),
        "test": select_mask_column(graph.test_mask).to(device),
    }

    return args, labels, masks


def accuracy(model: nn.Module, data: Any, device: torch.device) -> float:
    model.eval()
    args, labels, masks = get_args(data, device)

    with torch.no_grad():
        pred = model(**args).argmax(dim=-1)

    test_mask = masks["test"]
    return int((pred[test_mask] == labels[test_mask]).sum()) / int(test_mask.sum())


def get_metric_functions(
    ds_config: ConfigDict,
    device: torch.device,
) -> MetricCallables:
    metrics: MetricCallables = {}

    for metric_name in ds_config.metrics:  # type: ignore
        if metric_name == "Accuracy":
            metrics[metric_name] = (
                lambda model, data: accuracy(model, data, device)
            )
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    return metrics


def collect_metrics(
    model: nn.Module,
    data: Any,
    metric_callables: MetricCallables,
) -> Dict[str, float]:
    return {
        name: metric(model, data)
        for name, metric in metric_callables.items()
    }


def join_metrics(
    history: MetricHistory,
    metrics: Dict[str, float],
    model_name: str,
) -> MetricHistory:
    model_metrics = history.setdefault(model_name, {})

    for metric_name, metric_value in metrics.items():
        model_metrics.setdefault(metric_name, []).append(metric_value)

    return history


def get_val_loss(
    model: nn.Module,
    data: Any,
    loss: nn.Module,
    device: torch.device,
    *,
    log_hop_diagnostics: bool = False,
    hop_log_path: str = "",
    epoch: Optional[int] = None,
    lr: Optional[float] = None,
    log_only_layer: Optional[int] = None,
) -> float:
    model.eval()
    args, labels, masks = get_args(data, device)

    with torch.no_grad():
        pred = model(
            **args,
            log_hop_diagnostics=log_hop_diagnostics,
            hop_log_path=hop_log_path,
            epoch=epoch,
            phase="val",
            lr=lr,
            log_only_layer=log_only_layer,
        )
        val_loss = loss(pred[masks["val"]], labels[masks["val"]]).item()

    return val_loss


def get_split_losses(
    model: nn.Module,
    data: Any,
    loss: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    args, labels, masks = get_args(data, device)

    with torch.no_grad():
        pred = model(**args)
        losses = []
        for mask in masks.values():
            if int(mask.sum()) == 0:
                losses.append(float("nan"))
            else:
                losses.append(loss(pred[mask], labels[mask]).item())

    return losses[0], losses[1], losses[2]


def train_step_normal(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    data: Any,
    device: torch.device,
    loss_func: nn.Module,
    *,
    log_hop_diagnostics: bool = False,
    hop_log_path: str = "",
    epoch: Optional[int] = None,
    lr: Optional[float] = None,
    log_only_layer: Optional[int] = None,
) -> float:
    model.train()
    optimiser.zero_grad()

    args, labels, masks = get_args(data, device)
    lr = float(optimiser.param_groups[0]["lr"])
    pred = model(
        **args,
        log_hop_diagnostics=log_hop_diagnostics,
        hop_log_path=hop_log_path,
        epoch=epoch,
        phase="train",
        lr=lr,
        log_only_layer=log_only_layer,
    )
    train_loss = loss_func(pred[masks["train"]], labels[masks["train"]])

    train_loss.backward()
    if log_hop_diagnostics and hasattr(model, "log_hop_gate_gradients"):
        model.log_hop_gate_gradients(
            hop_log_path,
            epoch=epoch,
            phase="train",
            optimizer=optimiser,
        )
    optimiser.step()

    return train_loss.item()


def test(
    model: nn.Module,
    data: Any,
    device: torch.device,
    *,
    log_hop_diagnostics: bool = False,
    hop_log_path: str = "",
    epoch: Optional[int] = None,
    lr: Optional[float] = None,
    log_only_layer: Optional[int] = None,
) -> Tuple[float, float, float]:
    model.eval()
    args, labels, masks = get_args(data, device)

    with torch.no_grad():
        pred = model(
            **args,
            log_hop_diagnostics=log_hop_diagnostics,
            hop_log_path=hop_log_path,
            epoch=epoch,
            phase="test",
            lr=lr,
            log_only_layer=log_only_layer,
        ).argmax(dim=-1)

    accs = []
    for mask in masks.values():
        correct = int((pred[mask] == labels[mask]).sum())
        accs.append(correct / int(mask.sum()))

    return accs[0], accs[1], accs[2]


def create_optimizer(
    model: nn.Module,
    model_config: ConfigDict,
) -> torch.optim.Optimizer:
    training = model_config.training

    if training.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            weight_decay=training.weight_decay,
            lr=training.lr,
        )

    raise ValueError(
        f"Invalid optimizer name {training.optimizer} in configuration file"
    )


def create_scheduler(
    optimiser: torch.optim.Optimizer,
    model_config: ConfigDict,
) -> Optional[Any]:
    training = model_config.training
    scheduler_config = getattr(training, "scheduler", None)
    scheduler_name = "exponential" if scheduler_config is None else scheduler_config.name

    if scheduler_name in {None, "none"}:
        return None

    if scheduler_name == "exponential":
        gamma = getattr(
            scheduler_config,
            "gamma",
            getattr(training, "decay", 1.0),
        )
        return ExponentialLR(optimiser, gamma=gamma)

    if scheduler_name == "step":
        return StepLR(
            optimiser,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma,
        )

    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimiser,
            T_max=training.num_epochs,
            eta_min=scheduler_config.eta_min,
        )

    if scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimiser,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            threshold=scheduler_config.threshold,
            min_lr=scheduler_config.min_lr,
        )

    raise ValueError(
        f"Invalid scheduler name {scheduler_name} in configuration file"
    )


def step_scheduler(
    scheduler: Optional[Any],
    val_loss: float,
) -> None:
    if scheduler is None:
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
        return

    scheduler.step()


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
    logger.info(
        (
            "Epoch: %s Train Loss %.4f Val Loss %.4f "
            "Train Acc: %.4f Val Acc: %.4f Test Acc: %.4f "
            "Best Val Acc: %.4f Best Test Acc: %.4f"
        ),
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        test_acc,
        best_val_acc,
        best_test_acc,
    )


def hop_log_enabled(model_config: ConfigDict, epoch: int) -> bool:
    every = max(1, int(getattr(model_config, "hop_log_every", 10)))
    return hop_logging_configured(model_config) and epoch % every == 0


def hop_logging_configured(model_config: ConfigDict) -> bool:
    return (
        bool(getattr(model_config, "log_hop_diagnostics", False))
        and is_hop_aware_config(model_config)
    )


def write_hop_run_start(
    model_config: ConfigDict,
    path: str,
) -> None:
    if not hop_logging_configured(model_config):
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def train_model(
    model_config: ConfigDict,
    model: nn.Module,
    data: Any,
    loss: nn.Module,
    save_dir: Path,
    device: torch.device,
    *,
    model_name: str,
    dataset_name: str,
    repeat_idx: int,
    do_save: bool = True,
    full_config: Optional[ConfigDict] = None,
    raw_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    run_mode: str = "final",
) -> Tuple[List[float], List[float]]:
    optimiser = create_optimizer(model, model_config)
    scheduler = create_scheduler(optimiser, model_config)
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        write_config_json(
            config=full_config if full_config is not None else ConfigDict(),
            model_config=model_config,
            model_name=model_name,
            dataset_name=dataset_name,
            seed=int(seed if seed is not None else repeat_idx),
            run_mode=run_mode,
            path=raw_dir / "config.json",
        )
    path = ""
    if hop_logging_configured(model_config):
        if raw_dir is None:
            raise ValueError(
                "GLANT hop diagnostics require raw_dir. "
                "Pass raw_dir from meta_train/train loop."
            )

        path = str(raw_dir / "hop_diagnostics.csv")
        write_hop_run_start(model_config, path)

    best_val_acc = float("-inf")
    best_test_acc = 0.0
    best_epoch = None
    train_losses: List[float] = []
    val_losses: List[float] = []
    metrics_rows: List[Dict[str, Any]] = []

    model = model.to(device)
    for epoch in range(model_config.training.num_epochs):
        log_hops = hop_log_enabled(model_config, epoch)
        current_lr = float(optimiser.param_groups[0]["lr"])
        only_layer = getattr(model_config, "hop_log_only_layer", None)

        train_loss = train_step_normal(
            model,
            optimiser,
            data,
            device=device,
            loss_func=loss,
            log_hop_diagnostics=log_hops,
            hop_log_path=path,
            epoch=epoch,
            lr=current_lr,
            log_only_layer=only_layer,
        )
        val_loss = get_val_loss(
            model,
            data,
            loss,
            device,
            log_hop_diagnostics=log_hops,
            hop_log_path=path,
            epoch=epoch,
            lr=current_lr,
            log_only_layer=only_layer,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        step_scheduler(scheduler, val_loss)

        train_acc, val_acc, test_acc = test(
            model,
            data,
            device,
            log_hop_diagnostics=log_hops,
            hop_log_path=path,
            epoch=epoch,
            lr=current_lr,
            log_only_layer=only_layer,
        )
        train_eval_loss, val_eval_loss, test_loss = get_split_losses(
            model,
            data,
            loss,
            device,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            if do_save:
                save_to_checkpoint(model, str(save_dir))
            if raw_dir is not None and bool(getattr(full_config, "save_best_model", False)):
                torch.save(model, raw_dir / "best_model.pt")

        metrics_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_metric": train_acc,
            "val_metric": val_acc,
            "test_metric": test_acc,
            "lr": current_lr,
            "best_val_metric": best_val_acc,
            "best_test_metric": best_test_acc,
            "best_epoch": best_epoch,
            "metric_name": "Accuracy",
            "metric_direction": "higher",
            "run_mode": run_mode,
            "seed": seed if seed is not None else repeat_idx,
        })

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

    if raw_dir is not None:
        write_metrics_csv(metrics_rows, raw_dir / "metrics.csv")

        if is_glant_model(model_name, model_config):
            export_glant_diagnostics(
                hop_summary_path=path,
                raw_dir=raw_dir,
                write_attention=config_bool(
                    getattr(model_config, "log_attention_statistics", False)
                ),
            )

    return train_losses, val_losses


def reset_model(model: nn.Module) -> None:
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()


def save_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    save_path: Path,
) -> None:
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks([
        idx + 1
        for idx in range(len(train_loss))
        if (idx + 1) % 10 == 0 or idx == 0
    ])
    plt.xlim([1, len(train_loss)])
    plt.title("Training Curves")
    plt.legend(["train", "val"])
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics: Dict[str, float], save_path: Path) -> None:
    rows = [[str(name), str(value)] for name, value in metrics.items()]
    np.savetxt(save_path, rows, fmt="%s")


def save_config(config: ConfigDict, save_path: Path) -> None:
    config_dict = config.to_dict()
    config_dict.pop("device", None)

    with save_path.open("w", encoding="utf-8") as writer:
        yaml.dump(config_dict, writer)


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
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
    num_repeats = config.experiments.runs
    device = config.device
    metric_callables = get_metric_functions(ds_config, device)
    meta_metrics: MetricHistory = {model_name: {} for model_name in models}

    for repeat_idx, (model_name, model) in product(
        range(num_repeats),
        models.items(),
    ):
        model_config = config.baselines[model_name]
        current_seed = run_seed(config, repeat_idx)
        set_random_seed(current_seed)
        run_mode = str(getattr(config, "run_mode", "final")).lower()
        resolve_logging_policy(model_name, model_config, run_mode)

        data = select_dataset_for_model(
            model_name=model_name,
            dataset=dataset,
            config=config,
        )
        data = sparsify_dataset_edges(data, model_config)

        reset_model(model)
        model = model.to(device)

        run_dir = (
            Path("checkpoints")
            / ds_config.name
            / f"{model_name}{repeat_idx}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Run: %s, Training model %s", repeat_idx, model_name)
        raw_dir = raw_run_dir(
            config,
            model_name,
            ds_config.name,
            current_seed,
            repeat_idx,
        )
        train_loss, val_loss = train_model(
            model_config,
            model,
            data,
            loss,
            run_dir,
            device,
            model_name=model_name,
            dataset_name=ds_config.name,
            repeat_idx=repeat_idx,
            full_config=config,
            raw_dir=raw_dir,
            seed=current_seed,
            run_mode=run_mode,
        )
        logger.info("Run: %s, Training completed", repeat_idx)

        logger.info("Loading for evaluation")
        model = load_model_checkpoint(run_dir / "model.pt", device)

        logger.info("Collecting all metrics")
        metrics = collect_metrics(model, data, metric_callables)

        logger.info("Saving metrics")
        save_loss_curves(train_loss, val_loss, run_dir / "loss_curves.pdf")
        save_metrics(metrics, run_dir / "metrics.txt")
        save_config(config, run_dir / "config.yml")

        meta_metrics = join_metrics(meta_metrics, metrics, model_name)

    return meta_metrics
