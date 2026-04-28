from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from ml_collections import ConfigDict

from extra_models import HoGA, MixHopNet, TAGNet
from model import GLANT
from utils.logger import logger
from utils.model_names import canonical_model_name, canonical_model_names


Metrics = Dict[str, Dict[str, List[float]]]
ModelRegistry = Dict[str, nn.Module]
EXTRA_MODEL_TYPES = {
    "MixHop": MixHopNet,
    "TAGConv": TAGNet,
    "HoGA": HoGA,
}


def load_from_checkpoint(
    config: ConfigDict,
    ds_name: str,
    nrepeats: int,
    checkpoint_root: Path = Path("checkpoints"),
) -> Metrics:
    """Load saved metrics for configured models."""
    model_names = canonical_model_names(config.baselines.names)
    metric_names = config[ds_name.lower()].metrics
    metric_dict: Metrics = {
        model_name: {metric: [] for metric in metric_names}
        for model_name in model_names
    }

    for model_name, repeat_idx in product(model_names, range(nrepeats)):
        metrics_path = (
            checkpoint_root
            / ds_name
            / f"{model_name}{repeat_idx}"
            / "metrics.txt"
        )

        if not metrics_path.exists():
            logger.warning("Metric file path %s not found", metrics_path)
            continue

        with metrics_path.open("r", encoding="utf-8") as reader:
            for metric_line in reader:
                metric_name, metric_value = metric_line.split()
                metric_dict[model_name][metric_name].append(float(metric_value))

    return metric_dict


def save_to_checkpoint(model: nn.Module, save_dir: str) -> None:
    """Save a model object to checkpoint directory."""
    checkpoint_dir = Path(save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, checkpoint_dir / "model.pt")


def get_baseline_config(config: ConfigDict, model_name: str) -> ConfigDict:
    """Return baseline config or fail with a clear message."""
    model_name = canonical_model_name(model_name)

    if model_name not in config.baselines:
        raise ValueError(f"Missing config for model: {model_name}")

    return config.baselines[model_name]


def create_wrapped_model(
    model_name: str,
    config: ConfigDict,
    ds_config: ConfigDict,
) -> nn.Module:
    """Create any configured model through the shared GLANT wrapper."""
    model_config = get_baseline_config(config, model_name)

    return GLANT(
        model_config=model_config,
        ds_config=ds_config,
    )


def create_model(
    model_name: str,
    config: ConfigDict,
    ds_config: ConfigDict,
) -> nn.Module:
    """Create one model by name."""
    model_name = canonical_model_name(model_name)
    logger.info("Ds config (%s):\n%s", model_name, ds_config)
    if model_name in EXTRA_MODEL_TYPES:
        return EXTRA_MODEL_TYPES[model_name](
            model_config=get_baseline_config(config, model_name),
            ds_config=ds_config,
        )
    return create_wrapped_model(model_name, config, ds_config)


def create_models(
    config: ConfigDict,
    ds_config: ConfigDict,
) -> ModelRegistry:
    """Create configured baseline models."""
    return {
        model_name: create_model(model_name, config, ds_config)
        for model_name in canonical_model_names(config.baselines.names)
    }
