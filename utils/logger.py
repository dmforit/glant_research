from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None  # type: ignore[assignment]


LOGGER_NAME = "glant"


def get_logger() -> logging.Logger:
    """Return the shared project logger configured with Rich."""
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    if RichHandler is None:
        handler: logging.Handler = logging.StreamHandler()
    else:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


logger = get_logger()


def _value(config: object, name: str, default: object = None) -> object:
    return getattr(config, name, default)


def _section(title: str, rows: list[tuple[str, object]]) -> None:
    width = max(len(name) for name, _ in rows)
    body = "\n".join(f"  {name:<{width}} : {value}" for name, value in rows)
    logger.info("%s\n%s", title, body)


def log_dataset_config(ds_config: object) -> None:
    """Log a compact dataset configuration summary."""
    _section(
        "Dataset config",
        [
            ("name", _value(ds_config, "name")),
            ("num_nodes", _value(ds_config, "num_nodes")),
            ("in_channels", _value(ds_config, "in_channels")),
            ("out_channels", _value(ds_config, "out_channels")),
            ("metrics", _value(ds_config, "metrics")),
            ("split_idx", _value(ds_config, "split_idx")),
            ("save_path", _value(ds_config, "save_path")),
        ],
    )


def log_model_config(model_name: str, model_config: object) -> None:
    """Log a compact model configuration summary."""
    training = _value(model_config, "training")
    _section(
        f"Model config: {model_name}",
        [
            ("model_name", _value(model_config, "model_name")),
            ("conv_type", _value(model_config, "conv_type")),
            ("num_layers", _value(model_config, "num_layers")),
            ("hidden_channels", _value(model_config, "hidden_channels")),
            ("heads", _value(model_config, "heads")),
            ("dropout", _value(model_config, "dropout")),
            ("attn_dropout", _value(model_config, "attn_dropout")),
            ("max_hops", _value(model_config, "max_hops")),
            ("alpha", _value(model_config, "alpha")),
            ("sparsify_hops", _value(model_config, "sparsify_hops")),
            ("sampling_method", _value(model_config, "sampling_method")),
            ("num_samples", _value(model_config, "num_samples")),
            ("load_samples", _value(model_config, "load_samples")),
            ("optimizer", _value(training, "optimizer")),
            ("lr", _value(training, "lr")),
            ("weight_decay", _value(training, "weight_decay")),
            ("num_epochs", _value(training, "num_epochs")),
            ("scheduler", _value(_value(training, "scheduler"), "name")),
        ],
    )


def log_selected_configs(config: object, ds_config: object) -> None:
    """Log dataset and selected model configuration summaries."""
    log_dataset_config(ds_config)
    baselines = _value(config, "baselines")
    for model_name in _value(baselines, "names", []):
        log_model_config(str(model_name), getattr(baselines, str(model_name)))
