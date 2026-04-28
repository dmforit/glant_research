import argparse
import copy
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

import numpy as np
import torch
from torch import nn

from configs.config import all_config
from configs.ablation_config import apply_ablation
from train import meta_train
from utils.data_utils import ds_cfg, fetch_dataset
from utils.logger import logger
from utils.model_names import canonical_model_name, canonical_model_names
from utils.model_utils import create_models, load_from_checkpoint
from utils.run_paths import make_launch_id



SUPPORTED_GPU_IDS = {0, 1}
DEFAULT_RESULTS_DIR = Path("results")


def configure_device(config: Any, gpu: Optional[int]) -> None:
    if gpu is None:
        return

    if gpu == -1:
        config.device = torch.device("cpu")
        return

    if gpu not in SUPPORTED_GPU_IDS:
        raise ValueError(
            f"Unsupported GPU id: {gpu}. "
            f"Supported values: -1, {sorted(SUPPORTED_GPU_IDS)}"
        )

    config.device = torch.device(f"cuda:{gpu}")


def configure_seed(config: Any, seed: Optional[int]) -> None:
    if seed is None:
        return

    config.seed = int(seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def model_args(values: List[str]) -> List[str]:
    """Flatten CLI model arguments, supporting spaces and comma-separated lists."""
    names: List[str] = []
    for value in values:
        names.extend(name.strip() for name in value.split(",") if name.strip())
    return names


def apply_cli_overrides(config: Any, pargs: argparse.Namespace) -> None:
    if pargs.runs is not None:
        config.experiments.runs = pargs.runs

    configure_seed(config, getattr(pargs, "seed", None))

    if getattr(pargs, "run_mode", None) is not None:
        config.run_mode = pargs.run_mode

    if getattr(pargs, "results_dir", None) is not None:
        config.results_dir = pargs.results_dir
    
    if getattr(pargs, "ablation", None) is not None:
        apply_ablation(config, pargs.ablation)

    if getattr(pargs, "lambda_higher", None) is not None:
        config.baselines.GLANT_v2.lambda_higher = float(pargs.lambda_higher)
        if "GLANT_v2" in config.baselines:
            config.baselines.GLANT_v2.ablation_name = getattr(
                config,
                "ablation_name",
                f"glant_v2_lambda_{str(pargs.lambda_higher).replace('.', '_')}",
            )

    if getattr(pargs, "save_best_model", False):
        config.save_best_model = True
    
    if getattr(pargs, "launch_id", None) is not None:
        config.launch_id = pargs.launch_id

    if pargs.model is not None:
        config.baselines.names = canonical_model_names(model_args(pargs.model))

    if pargs.method is not None:
        logger.info("Starting run with sampling method")
        config.baselines.names = ["GLANT"]
        config.baselines.GLANT.load_samples = False
        config.baselines.GLANT.sampling_method = pargs.method

    if pargs.khop is not None:
        config.baselines.GLANT.max_hops = pargs.khop
        config.baselines.GLANT.load_samples = False

    if pargs.alpha is not None:
        config.baselines.GLANT.alpha = pargs.alpha

    if pargs.num_samples is not None:
        config.baselines.GLANT.num_samples = pargs.num_samples
        config.baselines.GLANT.load_samples = False

    if pargs.load_samples:
        config.baselines.GLANT.load_samples = True

    if pargs.conv_type is not None:
        config.baselines.GLANT.conv_type = pargs.conv_type
        if pargs.conv_type != "hop_gated_gatv2":
            config.baselines.GLANT.max_hops = 1

    if pargs.heads is not None:
        for model_name in config.baselines.names:
            model_name = canonical_model_name(model_name)
            model_config = config.baselines.get(model_name)
            if model_config is not None and hasattr(model_config, "heads"):
                model_config.heads = pargs.heads

    configure_device(config, pargs.gpu)


def execute_run(
    config: Any,
    ds_config: Any,
    data_dict: Any,
    loss: nn.Module,
    pargs: argparse.Namespace,
) -> Any:
    if pargs.train == pargs.test:
        raise ValueError("Exactly one mode must be selected: --train or --test")

    if pargs.train:
        models = create_models(config, ds_config)
        return meta_train(config, ds_config, models, data_dict, loss)

    runs = config.experiments.runs
    return load_from_checkpoint(config, pargs.dataset, runs, Path(pargs.checkpoint))


def get_selected_method(config: Any) -> Optional[str]:
    model_name = canonical_model_name(config.baselines.names[0])
    model_config = config.baselines.get(model_name)
    if model_config is None:
        return None

    return getattr(model_config, "sampling_method", None)


def print_metrics(
    config: Any,
    ds_config: Any,
    all_metrics: Dict[str, Dict[str, List[float]]],
) -> None:
    method = get_selected_method(config)
    logger.info("%s", all_metrics)
    for model_name, metric_dict in all_metrics.items():
        acc = metric_dict["Accuracy"]
        logger.info("%s result with method %s", ds_config.name, method)
        logger.info("%s %s +-%s", model_name, np.mean(acc), np.std(acc))


def run_experiment(pargs: argparse.Namespace) -> Any:
    config = all_config()
    apply_cli_overrides(config, pargs)

    if not hasattr(config, "launch_id") or config.launch_id is None:
        config.launch_id = make_launch_id(
            dataset_names=getattr(pargs, "dataset", None) or list(config.datasets.names),
            mode="train" if pargs.train else "eval",
            ablation_name=getattr(config, "ablation_name", None),
        )

    logger.info("launch_id: %s", config.launch_id)

    ds_config = ds_cfg(config, pargs.dataset)
    logger.info("Fetching Dataset: %s", pargs.dataset)
    data_dict = fetch_dataset(config, pargs.dataset)
    logger.info("Fetching Dataset - done")

    loss = nn.CrossEntropyLoss()
    logger.info("Starting run")
    all_metrics = execute_run(config, ds_config, data_dict, loss, pargs)
    print_metrics(config, ds_config, all_metrics)

    return all_metrics


def selected_datasets(pargs: argparse.Namespace) -> List[str]:
    if pargs.datasets is not None:
        return pargs.datasets

    return [pargs.dataset]


def selected_model_names(pargs: argparse.Namespace) -> List[str]:
    if pargs.method is not None:
        return ["GLANT"]

    if pargs.model is not None:
        return canonical_model_names(model_args(pargs.model))

    return canonical_model_names(all_config().baselines.names)


def slug(value: object) -> str:
    text = str(value)
    chars = [
        char.lower() if char.isalnum() else "-"
        for char in text
    ]
    out = "-".join(part for part in "".join(chars).split("-") if part)
    return out or "none"


def results_xlsx_filename(pargs: argparse.Namespace) -> str:
    datasets = "-".join(slug(dataset) for dataset in selected_datasets(pargs))
    model_names = selected_model_names(pargs)
    models = "-".join(slug(model) for model in model_names)
    ablation = ""
    if getattr(pargs, "ablation", None) is not None:
        ablation = f"_ablation-{slug(pargs.ablation)}"
    glant_config = all_config().baselines.GLANT
    if pargs.conv_type is not None:
        glant_config.conv_type = pargs.conv_type

    architecture = ""
    if any(model in {"GLANT", "GLANT_v1"} for model in model_names) and glant_config.conv_type == "hop_gated_gatv2":
        architecture = f"_architecture-{slug(glant_config.architecture)}"

    mode = "train" if pargs.train else "test"
    runs = slug(pargs.runs if pargs.runs is not None else "config")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = (
        f"{mode}_datasets-{datasets}_models-{models}{architecture}{ablation}_"
        f"runs-{runs}_{timestamp}.xlsx"
    )
    return filename


def results_xlsx_path(pargs: argparse.Namespace) -> Path:
    filename = results_xlsx_filename(pargs)

    if pargs.results_xlsx is None:
        return DEFAULT_RESULTS_DIR / filename

    path = Path(pargs.results_xlsx)
    output_dir = path.parent if path.suffix.lower() == ".xlsx" else path
    return output_dir / filename


def run_experiments(pargs: argparse.Namespace) -> Dict[str, Any]:
    results = {}

    for dataset in selected_datasets(pargs):
        dataset_args = copy.copy(pargs)
        dataset_args.dataset = dataset
        logger.info("Running dataset: %s", dataset)
        results[dataset] = run_experiment(dataset_args)

    save_results_xlsx(results, results_xlsx_path(pargs))

    return results


def excel_column(index: int) -> str:
    column = ""

    while index:
        index, remainder = divmod(index - 1, 26)
        column = chr(ord("A") + remainder) + column

    return column


def xlsx_cell(row: int, column: int, value: Any) -> str:
    reference = f"{excel_column(column)}{row}"

    if value is None:
        return f'<c r="{reference}"/>'

    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f'<c r="{reference}"><v>{float(value)}</v></c>'

    text = escape(str(value))
    return (
        f'<c r="{reference}" t="inlineStr">'
        f"<is><t>{text}</t></is>"
        f"</c>"
    )


def mean_accuracy(metric_dict: Dict[str, List[float]]) -> Optional[float]:
    accuracy = metric_dict.get("Accuracy", [])
    if not accuracy:
        return None

    return float(np.mean(accuracy))


def results_table(results: Dict[str, Any]) -> List[List[Any]]:
    model_names = []
    for dataset_metrics in results.values():
        for model_name in dataset_metrics:
            if model_name not in model_names:
                model_names.append(model_name)

    header = ["Model", *results.keys()]
    rows = [header]

    for model_name in model_names:
        row = [model_name]
        for dataset_name in results:
            metric_dict = results[dataset_name].get(model_name, {})
            row.append(mean_accuracy(metric_dict))
        rows.append(row)

    return rows


def write_xlsx(rows: List[List[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(
            xlsx_cell(row_idx, col_idx, value)
            for col_idx, value in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main">'
        "<sheetData>"
        f'{"".join(sheet_rows)}'
        "</sheetData>"
        "</worksheet>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.'
            'openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.spreadsheetml.'
            'worksheet+xml"/>'
            "</Types>",
        )
        workbook.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.'
            'org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Results" sheetId="1" r:id="rId1"/>'
            "</sheets></workbook>",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def save_results_xlsx(results: Dict[str, Any], path: Path) -> None:
    if not results:
        return

    write_xlsx(results_table(results), path)
    logger.info("Saved summary results to %s", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset",
        type=str,
        help="The name of the dataset to run",
    )
    dataset_group.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset names to run sequentially",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Override GPU from the config file; use -1 for CPU",
    )
    parser.add_argument(
        "--khop",
        type=int,
        default=None,
        help="Override GLANT max-hops value",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override GLANT higher-hop sparsification alpha",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of sampled edges per node/hop",
    )
    parser.add_argument(
        "--load-samples",
        action="store_true",
        help="Load cached sampled hop edges from disk when available",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        default=None,
        choices=["hop_gated_gatv2", "gatv2", "gat", "sage", "gcn"],
        help="Override GLANT wrapper conv_type",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help="Override attention heads",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help="Checkpoint directory for loading models",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Sampling method ablation",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="Override model list from the config file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override number of experiment repeats",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set random seed for a run",
    )
    parser.add_argument(
        "--results-xlsx",
        type=str,
        default=None,
        help="Directory for generated summary XLSX files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Root directory for run-level results",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["hpo", "final", "baseline", "debug"],
        default=None,
        help="Logging mode for run-level results",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Run one predefined GLANT ablation preset",
    )
    parser.add_argument(
        "--lambda-higher",
        type=float,
        default=None,
        help="Override GLANT-v2 lambda_higher",
    )
    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="Also save best_model.pt under results/raw",
    )
    parser.add_argument(
        "--launch-id",
        type=str,
        default=None,
        help="Optional launch id for grouping all runs from one program launch",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train the model")
    mode.add_argument("--test", action="store_true", help="Test the model")

    args = parser.parse_args()
    run_experiments(args)
