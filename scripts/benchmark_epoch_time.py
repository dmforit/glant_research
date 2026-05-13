from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import all_config
from train import (
    create_optimizer,
    create_scheduler,
    get_split_losses,
    get_val_loss,
    reset_model,
    select_dataset_for_model,
    sparsify_dataset_edges,
    step_scheduler,
    test,
    train_step_normal,
)
from utils.data_utils import ds_cfg, fetch_dataset
from utils.logger import logger
from utils.model_names import canonical_model_name
from utils.model_utils import create_model
from utils.result_logging import set_random_seed, write_csv_and_xlsx


DEFAULT_DATASETS = ["ACM"]
DEFAULT_MODELS = [
    "GCN",
    "GraphSAGE",
    "GATv2",
    "GLANT_v8",
    "MixHop",
    "TAGConv",
    "HoGA",
]
DEFAULT_GLANT_HPO_GLOB = "results/launches/*/summary/best_hpo_configs.json"
DISPLAY_NAMES = {
    "GLANT_v8": "GLANT",
    "HoGA": "HoGA-GAT",
}
GLANT_MODEL_KEY = "GLANT_v8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure total training time and average epoch time for graph models."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for every model. By default uses model config epochs.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Use -1 for CPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/timing")
    parser.add_argument(
        "--hpo-config",
        default=None,
        help=(
            "Optional best_hpo_configs.json. If omitted, the script scans "
            "results/launches/*/summary/best_hpo_configs.json."
        ),
    )
    parser.add_argument("--glant-k", type=int, default=2)
    parser.add_argument("--glant-alpha", type=float, default=0.1)
    parser.add_argument("--hoga-k", type=int, default=2)
    parser.add_argument("--tagconv-k", type=int, default=2)
    parser.add_argument(
        "--mixhop-powers",
        type=int,
        nargs="+",
        default=None,
        help="Override MixHop powers, for example: --mixhop-powers 0 1.",
    )
    parser.add_argument(
        "--mixhop-hidden-channels",
        type=int,
        default=None,
        help="Override MixHop hidden_channels to reduce memory usage.",
    )
    parser.add_argument(
        "--train-step-only",
        action="store_true",
        help=(
            "Measure only forward/backward/optimizer step. By default the "
            "timed epoch mirrors the project loop: train step + val/test eval."
        ),
    )
    return parser.parse_args()


def configure_device(config: Any, gpu: int) -> None:
    config.device = torch.device("cpu" if gpu == -1 else f"cuda:{gpu}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def display_model_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name)


def benchmark_model_name(model_name: object) -> str:
    name = str(model_name)
    if name.lower() == "glant":
        return GLANT_MODEL_KEY
    return canonical_model_name(name)


def benchmark_model_names(model_names: list[object]) -> list[str]:
    names: list[str] = []
    for model_name in model_names:
        canonical = benchmark_model_name(model_name)
        if canonical not in names:
            names.append(canonical)
    if "HoGA" in names:
        names = [name for name in names if name != "HoGA"] + ["HoGA"]
    return names


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "cuda out of memory" in text
        or "cublas_status_alloc_failed" in text
    )


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def fallback_attempts(model_name: str) -> list[dict[str, Any]]:
    if model_name == "MixHop":
        return [
            {
                "fallback_name": "default",
            },
            {
                "fallback_name": "mixhop_powers_0_1_hidden_32",
                "mixhop_powers": [0, 1],
                "mixhop_hidden_channels": 32,
            },
            {
                "fallback_name": "mixhop_powers_0_1_hidden_16",
                "mixhop_powers": [0, 1],
                "mixhop_hidden_channels": 16,
            },
            {
                "fallback_name": "mixhop_power_1_hidden_16",
                "mixhop_powers": [1],
                "mixhop_hidden_channels": 16,
            },
        ]

    if model_name == "TAGConv":
        return [
            {
                "fallback_name": "default",
            },
            {
                "fallback_name": "tagconv_k_1_hidden_32",
                "tagconv_k": 1,
                "tagconv_hidden_channels": 32,
            },
            {
                "fallback_name": "tagconv_k_1_hidden_16",
                "tagconv_k": 1,
                "tagconv_hidden_channels": 16,
            },
        ]

    return [{"fallback_name": "default"}]


def apply_hpo_params(model_config: Any, params: dict[str, Any]) -> None:
    for name, value in params.items():
        if name in {"max_hops", "num_edges", "num_layers", "hidden_channels", "heads"}:
            setattr(model_config, name, int(value))
        elif name in {"alpha", "dropout", "attn_dropout", "lambda_higher"}:
            setattr(model_config, name, float(value))
        elif name in {"use_zero_hop", "v7_input_skip"}:
            setattr(model_config, name, bool(value))
        else:
            setattr(model_config, name, value)

    if "dropout" in params and "attn_dropout" not in params:
        model_config.attn_dropout = float(params["dropout"])
    if hasattr(model_config, "lambda_higher"):
        model_config.learn_lambda_higher = False


def best_glant_hpo_entry(
    dataset_name: str,
    model_name: str,
    hpo_config: str | None,
) -> dict[str, Any] | None:
    paths = [Path(hpo_config)] if hpo_config else sorted(Path().glob(DEFAULT_GLANT_HPO_GLOB))
    candidates: list[dict[str, Any]] = []
    key = f"{dataset_name}/{model_name}"

    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as reader:
            payload = json.load(reader)
        if key in payload:
            entry = dict(payload[key])
            entry["_source"] = str(path)
            candidates.append(entry)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            float(item.get("best_val_metric", float("-inf"))),
            float(item.get("final_test_metric", float("-inf"))),
        ),
    )


def configure_model(
    config: Any,
    dataset_name: str,
    model_name: str,
    args: argparse.Namespace,
    attempt: dict[str, Any],
) -> dict[str, Any]:
    model_config = config.baselines[model_name]
    notes: dict[str, Any] = {
        "fallback_attempt": str(attempt.get("fallback_name", "default")),
    }

    if args.epochs is not None:
        model_config.training.num_epochs = int(args.epochs)

    if model_name == GLANT_MODEL_KEY:
        entry = best_glant_hpo_entry(dataset_name, model_name, args.hpo_config)
        if entry is not None:
            apply_hpo_params(model_config, dict(entry["params"]))
            notes["glant_hpo_source"] = entry["_source"]
            notes["glant_hpo_trial_id"] = entry.get("trial_id")
        else:
            notes["glant_hpo_source"] = "not_found_default_config_used"

        model_config.max_hops = int(args.glant_k)
        model_config.alpha = float(args.glant_alpha)
        model_config.load_samples = False
        model_config.log_hop_diagnostics = False
        model_config.log_hop_weights = False
        model_config.log_attention_scores = "false"
        model_config.log_attention_statistics = "false"

    if model_name == "HoGA":
        model_config.max_hops = int(args.hoga_k)
        model_config.K_hops = int(args.hoga_k)
        model_config.load_samples = False

    if model_name == "MixHop":
        mixhop_powers = attempt.get("mixhop_powers", args.mixhop_powers)
        mixhop_hidden_channels = attempt.get(
            "mixhop_hidden_channels",
            args.mixhop_hidden_channels,
        )
        if mixhop_powers is not None:
            model_config.powers = [int(power) for power in mixhop_powers]
            notes["mixhop_powers_override"] = ",".join(map(str, model_config.powers))
        if mixhop_hidden_channels is not None:
            model_config.hidden_channels = int(mixhop_hidden_channels)
            notes["mixhop_hidden_channels_override"] = model_config.hidden_channels

    if model_name == "TAGConv":
        model_config.K = int(attempt.get("tagconv_k", args.tagconv_k))
        if "tagconv_hidden_channels" in attempt:
            model_config.hidden_channels = int(attempt["tagconv_hidden_channels"])
            notes["tagconv_hidden_channels_override"] = model_config.hidden_channels

    return notes


def timed_epoch(
    *,
    model: nn.Module,
    data: Any,
    device: torch.device,
    loss_func: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: Any,
    train_step_only: bool,
) -> tuple[float, dict[str, float]]:
    sync_device(device)
    start = time.perf_counter()

    train_loss = train_step_normal(
        model,
        optimiser,
        data,
        device=device,
        loss_func=loss_func,
    )

    metrics = {"train_loss": float(train_loss)}
    if not train_step_only:
        val_loss = get_val_loss(model, data, loss_func, device)
        step_scheduler(scheduler, val_loss)
        train_acc, val_acc, test_acc = test(model, data, device)
        _, _, test_loss = get_split_losses(model, data, loss_func, device)
        metrics.update(
            {
                "val_loss": float(val_loss),
                "test_loss": float(test_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
            }
        )

    sync_device(device)
    return time.perf_counter() - start, metrics


def benchmark_one(
    *,
    dataset_name: str,
    model_name: str,
    repeat_idx: int,
    args: argparse.Namespace,
    attempt: dict[str, Any],
) -> dict[str, Any]:
    config = all_config()
    configure_device(config, args.gpu)
    config.baselines.names = [model_name]
    seed = int(args.seed) + repeat_idx
    config.seed = seed
    config.run_mode = "timing"
    set_random_seed(seed)

    notes = configure_model(config, dataset_name, model_name, args, attempt)
    ds_config = ds_cfg(config, dataset_name)

    logger.info(
        "Preparing dataset=%s model=%s run=%s seed=%s",
        dataset_name,
        display_model_name(model_name),
        repeat_idx,
        seed,
    )
    dataset = fetch_dataset(config, dataset_name)
    data = select_dataset_for_model(
        model_name=model_name,
        dataset=dataset,
        config=config,
    )
    data = sparsify_dataset_edges(data, config.baselines[model_name])

    set_random_seed(seed)
    model = create_model(model_name, config, ds_config)
    reset_model(model)
    model = model.to(config.device)
    loss_func = nn.CrossEntropyLoss()
    optimiser = create_optimizer(model, config.baselines[model_name])
    scheduler = create_scheduler(optimiser, config.baselines[model_name])

    epoch_times: list[float] = []
    num_epochs = int(config.baselines[model_name].training.num_epochs)

    logger.info(
        "Timing started: dataset=%s model=%s epochs=%s train_step_only=%s",
        dataset_name,
        display_model_name(model_name),
        num_epochs,
        args.train_step_only,
    )
    for epoch in range(num_epochs):
        elapsed, metrics = timed_epoch(
            model=model,
            data=data,
            device=config.device,
            loss_func=loss_func,
            optimiser=optimiser,
            scheduler=scheduler,
            train_step_only=bool(args.train_step_only),
        )
        epoch_times.append(elapsed)

        if args.train_step_only:
            logger.info(
                "dataset=%s model=%s run=%s epoch=%s/%s time=%.6fs train_loss=%.6f",
                dataset_name,
                display_model_name(model_name),
                repeat_idx,
                epoch + 1,
                num_epochs,
                elapsed,
                metrics["train_loss"],
            )
        else:
            logger.info(
                (
                    "dataset=%s model=%s run=%s epoch=%s/%s time=%.6fs "
                    "train_loss=%.6f val_loss=%.6f train_acc=%.4f "
                    "val_acc=%.4f test_acc=%.4f"
                ),
                dataset_name,
                display_model_name(model_name),
                repeat_idx,
                epoch + 1,
                num_epochs,
                elapsed,
                metrics["train_loss"],
                metrics["val_loss"],
                metrics["train_acc"],
                metrics["val_acc"],
                metrics["test_acc"],
            )

    total_time = float(np.sum(epoch_times))
    return {
        "dataset": dataset_name,
        "model": display_model_name(model_name),
        "model_key": model_name,
        "run": repeat_idx,
        "seed": seed,
        "epochs": num_epochs,
        "total_training_time_sec": total_time,
        "mean_epoch_time_sec": float(np.mean(epoch_times)),
        "std_epoch_time_sec": float(np.std(epoch_times, ddof=0)),
        "min_epoch_time_sec": float(np.min(epoch_times)),
        "max_epoch_time_sec": float(np.max(epoch_times)),
        "device": str(config.device),
        "train_step_only": bool(args.train_step_only),
        "glant_k": int(args.glant_k) if model_name == GLANT_MODEL_KEY else None,
        "glant_alpha": float(args.glant_alpha) if model_name == GLANT_MODEL_KEY else None,
        "hoga_k": int(args.hoga_k) if model_name == "HoGA" else None,
        "mixhop_powers": ",".join(map(str, config.baselines[model_name].powers))
        if model_name == "MixHop"
        else None,
        "mixhop_hidden_channels": int(config.baselines[model_name].hidden_channels)
        if model_name == "MixHop"
        else None,
        "tagconv_k": int(config.baselines[model_name].K)
        if model_name == "TAGConv"
        else None,
        "tagconv_hidden_channels": int(config.baselines[model_name].hidden_channels)
        if model_name == "TAGConv"
        else None,
        **notes,
    }


def benchmark_with_oom_fallback(
    *,
    dataset_name: str,
    model_name: str,
    repeat_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    attempts = fallback_attempts(model_name)
    last_error: Exception | None = None

    for attempt_idx, attempt in enumerate(attempts, start=1):
        try:
            if attempt_idx > 1:
                logger.warning(
                    (
                        "Retry after CUDA OOM: dataset=%s model=%s run=%s "
                        "attempt=%s params=%s"
                    ),
                    dataset_name,
                    display_model_name(model_name),
                    repeat_idx,
                    attempt_idx,
                    attempt,
                )
            return benchmark_one(
                dataset_name=dataset_name,
                model_name=model_name,
                repeat_idx=repeat_idx,
                args=args,
                attempt=attempt,
            )
        except Exception as exc:
            if not is_cuda_oom(exc) or attempt_idx == len(attempts):
                raise
            last_error = exc
            logger.warning(
                (
                    "CUDA OOM: dataset=%s model=%s run=%s attempt=%s "
                    "failed, clearing cache"
                ),
                dataset_name,
                display_model_name(model_name),
                repeat_idx,
                attempt_idx,
            )
            cleanup_memory()

    raise RuntimeError(
        f"OOM fallback exhausted for {dataset_name}/{model_name}"
    ) from last_error


def aggregate_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    group_cols = ["dataset", "model", "model_key", "epochs", "device", "train_step_only"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("run", "count"),
            total_training_time_sec=("total_training_time_sec", "mean"),
            mean_epoch_time_sec=("mean_epoch_time_sec", "mean"),
            std_epoch_time_sec=("mean_epoch_time_sec", "std"),
            min_epoch_time_sec=("mean_epoch_time_sec", "min"),
            max_epoch_time_sec=("mean_epoch_time_sec", "max"),
        )
        .reset_index()
    )
    agg["std_epoch_time_sec"] = agg["std_epoch_time_sec"].fillna(0.0)
    order = {display_model_name(name): idx for idx, name in enumerate(DEFAULT_MODELS)}
    agg["_order"] = agg["model"].map(order).fillna(len(order))
    return agg.sort_values(["dataset", "_order", "model"]).drop(columns=["_order"])


def save_scatterplot(df: pd.DataFrame, output_path: Path) -> None:
    datasets = list(dict.fromkeys(df["dataset"].tolist()))
    models = list(dict.fromkeys(df["model"].tolist()))
    x_positions = {model: idx for idx, model in enumerate(models)}
    offsets = [0.0] if len(datasets) == 1 else np.linspace(-0.22, 0.22, len(datasets))

    plt.figure(figsize=(max(8, len(models) * 1.2), 5))
    for dataset_idx, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        x = [
            x_positions[model] + offsets[dataset_idx]
            for model in subset["model"].tolist()
        ]
        y = subset["mean_epoch_time_sec"].tolist()
        plt.scatter(x, y, s=90, label=dataset)
        for point_x, point_y, model in zip(x, y, subset["model"].tolist()):
            plt.annotate(
                model,
                (point_x, point_y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    plt.xticks(range(len(models)), models, rotation=25, ha="right")
    plt.ylabel("Mean epoch time, sec")
    plt.xlabel("Model")
    plt.title("Model training time by dataset")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(title="Dataset")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    model_names = benchmark_model_names(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        for model_name in model_names:
            canonical = benchmark_model_name(model_name)
            for repeat_idx in range(int(args.runs)):
                rows.append(
                    benchmark_with_oom_fallback(
                        dataset_name=dataset_name,
                        model_name=canonical,
                        repeat_idx=repeat_idx,
                        args=args,
                    )
                )

    raw_df = pd.DataFrame(rows)
    summary_df = aggregate_results(rows)

    raw_path = output_dir / "epoch_time_raw.csv"
    summary_path = output_dir / "epoch_time_summary.csv"
    plot_path = output_dir / "epoch_time_scatter.png"
    write_csv_and_xlsx(raw_df, raw_path)
    write_csv_and_xlsx(summary_df, summary_path)
    save_scatterplot(summary_df, plot_path)

    columns = [
        "dataset",
        "model",
        "runs",
        "epochs",
        "total_training_time_sec",
        "mean_epoch_time_sec",
        "std_epoch_time_sec",
    ]
    print("\nTiming summary:")
    print(summary_df[columns].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved raw timings: {raw_path}")
    print(f"Saved summary table: {summary_path}")
    print(f"Saved scatterplot: {plot_path}")


if __name__ == "__main__":
    main()
