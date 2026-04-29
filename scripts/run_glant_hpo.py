from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import all_config
from train import meta_train
from utils.data_utils import ds_cfg, fetch_dataset
from utils.logger import logger
from utils.model_names import canonical_model_name
from utils.model_utils import create_models
from utils.result_logging import best_metric_row
from utils.run_paths import make_launch_id


DEFAULT_DATASETS = ["Cora", "Citeseer", "Texas", "AIFB", "IMDB", "ACM"]
DEFAULT_MODELS = ["glant_v1", "glant_v2"]
DEFAULT_TRIALS = {
    "GLANT_v1": 10,
    "GLANT_v2": 20,
}


SEARCH_SPACE = {
    "GLANT_v1": {
        "max_hops": [2, 3],
        "alpha": [0.03, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0],
        "num_edges": [10000, 25000, 50000, 75000],
        "num_layers": [2, 3],
        "hidden_channels": [32, 64],
        "heads": [4, 8],
        "dropout": [0.3, 0.5, 0.7],
        "norm": ["none", "layernorm"],
    },
    "GLANT_v2": {
        "max_hops": [2, 3],
        "alpha": [0.03, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0],
        "num_edges": [10000, 25000, 50000, 75000],
        "num_layers": [2, 3],
        "hidden_channels": [32, 64],
        "heads": [4, 8],
        "dropout": [0.3, 0.5, 0.7],
        "norm": ["none", "layernorm"],
        "lambda_higher": [0.0, 0.1, 0.25, 0.5, 1.0],
    },
}


def max_search_hops(model_names: list[str]) -> int:
    return max(
        max(int(value) for value in SEARCH_SPACE[model_name]["max_hops"])
        for model_name in model_names
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small GLANT HPO grid.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--trials-v1", type=int, default=DEFAULT_TRIALS["GLANT_v1"])
    parser.add_argument("--trials-v2", type=int, default=DEFAULT_TRIALS["GLANT_v2"])
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_device(config: Any, gpu: int) -> None:
    if gpu == -1:
        config.device = torch.device("cpu")
    else:
        config.device = torch.device(f"cuda:{gpu}")


def trial_count(model_name: str, args: argparse.Namespace) -> int:
    if args.trial_limit is not None:
        return int(args.trial_limit)
    if model_name == "GLANT_v1":
        return int(args.trials_v1)
    if model_name == "GLANT_v2":
        return int(args.trials_v2)
    raise ValueError(f"HPO supports only GLANT_v1 and GLANT_v2, got {model_name}")


def all_param_combinations(model_name: str) -> list[dict[str, Any]]:
    space = SEARCH_SPACE[model_name]
    keys = list(space)
    values = [space[key] for key in keys]
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]


def selected_trials(model_name: str, count: int, seed: int) -> list[dict[str, Any]]:
    combinations = all_param_combinations(model_name)
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(combinations), generator=generator).tolist()
    return [combinations[idx] for idx in order[: min(count, len(combinations))]]


def apply_trial_params(config: Any, model_name: str, params: dict[str, Any]) -> None:
    model_config = config.baselines[model_name]

    model_config.max_hops = int(params["max_hops"])
    model_config.alpha = float(params["alpha"])
    model_config.num_edges = int(params["num_edges"])
    model_config.num_layers = int(params["num_layers"])
    model_config.hidden_channels = int(params["hidden_channels"])
    model_config.heads = int(params["heads"])
    model_config.dropout = float(params["dropout"])
    model_config.attn_dropout = float(params["dropout"])
    model_config.norm = str(params["norm"])

    if model_name == "GLANT_v2":
        model_config.lambda_higher = float(params["lambda_higher"])
        model_config.learn_lambda_higher = False


def hpo_summary_dir(results_dir: str | Path, launch_id: str) -> Path:
    return Path(results_dir) / "launches" / launch_id / "summary"


def write_hpo_results(rows: list[dict[str, Any]], results_dir: str | Path, launch_id: str) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    summary_dir = hpo_summary_dir(results_dir, launch_id)
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / "hpo_results.csv", index=False)

    root_summary = Path(results_dir) / "summary"
    root_summary.mkdir(parents=True, exist_ok=True)
    root_path = root_summary / "hpo_results.csv"
    if root_path.exists():
        existing = pd.read_csv(root_path)
        df = pd.concat(
            [existing[existing.get("launch_id") != launch_id], df],
            ignore_index=True,
        )
    df.to_csv(root_path, index=False)


def write_best_configs(rows: list[dict[str, Any]], results_dir: str | Path, launch_id: str) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    best: dict[str, dict[str, Any]] = {}
    for (dataset_name, model_name), group in df.groupby(["dataset_name", "model_name"]):
        group = group.copy()
        group["best_val_metric"] = pd.to_numeric(group["best_val_metric"], errors="coerce")
        group = group.dropna(subset=["best_val_metric"])
        if group.empty:
            continue

        direction = str(group["metric_direction"].iloc[0])
        idx = group["best_val_metric"].idxmin() if direction == "lower" else group["best_val_metric"].idxmax()
        row = group.loc[idx].to_dict()
        best[f"{dataset_name}/{model_name}"] = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "trial_id": int(row["trial_id"]),
            "seed": int(row["seed"]),
            "params": json.loads(row["params"]),
            "best_val_metric": row["best_val_metric"],
            "final_test_metric": row["final_test_metric"],
            "best_epoch": row["best_epoch"],
            "run_dir": row["run_dir"],
        }

    summary_dir = hpo_summary_dir(results_dir, launch_id)
    with (summary_dir / "best_hpo_configs.json").open("w", encoding="utf-8") as writer:
        json.dump(best, writer, indent=2, ensure_ascii=False)


def read_trial_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"{metrics_path} is empty")

    metric_name = str(df.get("metric_name", pd.Series(["Accuracy"])).iloc[-1])
    direction = str(df.get("metric_direction", pd.Series(["higher"])).iloc[-1])
    best = best_metric_row(df.to_dict("records"), metric_name)
    if best is None:
        raise ValueError(f"{metrics_path} has no metric rows")

    final = df.iloc[-1].to_dict()
    return {
        "best_val_metric": best.get("val_metric"),
        "final_test_metric": final.get("test_metric"),
        "best_epoch": best.get("epoch"),
        "metric_name": metric_name,
        "metric_direction": direction,
    }


def max_search_num_edges(model_names: list[str]) -> int:
    return max(
        max(int(value) for value in SEARCH_SPACE[model_name]["num_edges"])
        for model_name in model_names
    )


def prepare_dataset_cache(
    base_config: Any,
    dataset_name: str,
    model_names: list[str],
) -> tuple[Any, Any]:
    """Load one dataset and sample the largest required hop/edge budget once."""
    config = copy.deepcopy(base_config)
    config.baselines.names = [model_names[0]]

    max_hops = max_search_hops(model_names)
    max_num_edges = max_search_num_edges(model_names)
    for model_name in model_names:
        model_config = config.baselines[model_name]
        model_config.max_hops = max_hops
        model_config.num_edges = max_num_edges
        model_config.load_samples = False

    # fetch_dataset builds multihop_dataset once using mh_cfg(config).
    ds_config = ds_cfg(config, dataset_name)
    data_dict = fetch_dataset(config, dataset_name)
    return ds_config, data_dict


def limit_edges_per_hop(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    num_edges: int | None,
    num_samples: int | None,
) -> torch.Tensor:
    edge_budget = (
        int(num_edges)
        if num_edges is not None
        else int(num_nodes) * int(num_samples or 1)
    )
    if edge_index.size(1) <= edge_budget:
        return edge_index
    return edge_index[:, :edge_budget]


def dataset_for_trial(
    data_dict: Any,
    max_hops: int,
    *,
    num_edges: int | None,
    num_samples: int | None,
) -> Any:
    """Return a trial copy with max_hops and edge budget cut from the shared sample."""
    trial_data = copy.copy(data_dict)
    multihop_dataset = copy.copy(data_dict.multihop_dataset)
    num_nodes = int(data_dict.num_nodes)
    edge_index = [multihop_dataset.edge_index[0]]
    edge_index.extend(
        limit_edges_per_hop(
            edge,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_samples=num_samples,
        )
        for edge in list(multihop_dataset.edge_index)[1: int(max_hops)]
    )

    multihop_dataset.edge_index = edge_index
    if hasattr(multihop_dataset, "graph"):
        multihop_dataset.graph = copy.copy(multihop_dataset.graph)
        multihop_dataset.graph.edge_index = edge_index

    trial_data.multihop_dataset = multihop_dataset
    return trial_data


def run_trial(
    *,
    base_config: Any,
    ds_config: Any,
    data_dict: Any,
    model_name: str,
    trial_id: int,
    params: dict[str, Any],
    seed: int,
    launch_id: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.baselines.names = [model_name]
    config.seed = int(seed)
    config.run_mode = "hpo"
    config.launch_id = launch_id
    config.ablation_name = f"hpo_trial_{trial_id:03d}"
    config.experiments.runs = 1

    apply_trial_params(config, model_name, params)

    data_dict = dataset_for_trial(
        data_dict,
        int(params["max_hops"]),
        num_edges=int(params["num_edges"]) if "num_edges" in params else None,
        num_samples=int(params["num_samples"]) if "num_samples" in params else None,
    )
    models = create_models(config, ds_config)
    meta_train(config, ds_config, models, data_dict, nn.CrossEntropyLoss())

    run_dir = (
        Path(config.results_dir)
        / "launches"
        / launch_id
        / "raw"
        / ds_config.name
        / model_name
        / config.ablation_name
        / f"seed_{seed}"
        / "run_0"
    )
    metrics = read_trial_metrics(run_dir)

    return {
        "launch_id": launch_id,
        "dataset_name": ds_config.name,
        "model_name": model_name,
        "trial_id": trial_id,
        "seed": seed,
        "params": json.dumps(params, sort_keys=True),
        "best_val_metric": metrics["best_val_metric"],
        "final_test_metric": metrics["final_test_metric"],
        "best_epoch": metrics["best_epoch"],
        "metric_name": metrics["metric_name"],
        "metric_direction": metrics["metric_direction"],
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    base_config = all_config()
    configure_device(base_config, args.gpu)
    base_config.results_dir = args.results_dir
    base_config.run_mode = "hpo"

    if args.epochs is not None:
        for model_name in ("GLANT_v1", "GLANT_v2"):
            getattr(base_config.baselines, model_name).training.num_epochs = int(args.epochs)

    launch_id = args.launch_id or make_launch_id(
        dataset_names=args.datasets,
        mode="hpo",
        ablation_name="glant",
    )

    model_names = [canonical_model_name(model_name) for model_name in args.models]
    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        if not args.dry_run:
            ds_config, data_dict = prepare_dataset_cache(
                base_config=base_config,
                dataset_name=dataset_name,
                model_names=model_names,
            )

        for model_name in model_names:
            if model_name not in {"GLANT_v1", "GLANT_v2"}:
                raise ValueError(f"HPO supports only GLANT_v1/GLANT_v2, got {model_name}")

            trials = selected_trials(
                model_name=model_name,
                count=trial_count(model_name, args),
                seed=args.seed + len(rows),
            )

            for trial_id, params in enumerate(trials):
                seed = args.seed + trial_id
                logger.info(
                    "HPO trial dataset=%s model=%s trial=%s seed=%s params=%s",
                    dataset_name,
                    model_name,
                    trial_id,
                    seed,
                    params,
                )

                if args.dry_run:
                    continue

                row = run_trial(
                    base_config=base_config,
                    ds_config=ds_config,
                    data_dict=data_dict,
                    model_name=model_name,
                    trial_id=trial_id,
                    params=params,
                    seed=seed,
                    launch_id=launch_id,
                )
                rows.append(row)
                write_hpo_results(rows, args.results_dir, launch_id)
                write_best_configs(rows, args.results_dir, launch_id)

    if args.dry_run:
        logger.info("Dry run complete. launch_id would be: %s", launch_id)
        return

    write_hpo_results(rows, args.results_dir, launch_id)
    write_best_configs(rows, args.results_dir, launch_id)
    logger.info("Saved HPO results under %s", hpo_summary_dir(args.results_dir, launch_id))


if __name__ == "__main__":
    main()
