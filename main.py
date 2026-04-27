import argparse
import copy
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from new_configs.config import all_config

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
DEFAULT_RESULTS_XLSX = 'model_runs/results.xlsx'


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

    if pargs.model is not None:
        config.baselines.names = [pargs.model]

    if pargs.method is not None:
        print('Starting run with sampling method')
        config.baselines.names = ['GLANT']
        config.baselines.GLANT.load_samples = False
        config.baselines.GLANT.sampling_method = pargs.method

    if pargs.khop is not None:
        config.baselines.GLANT.max_hops = pargs.khop
        config.baselines.GLANT.load_samples = False

    if pargs.heads is not None:
        for model_name in config.baselines.names:
            model_config = config.baselines.get(model_name)
            if model_config is not None and hasattr(model_config, 'heads'):
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

    return getattr(model_config, 'sampling_method', None)


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


def selected_datasets(pargs: argparse.Namespace) -> List[str]:
    if pargs.datasets is not None:
        return pargs.datasets

    return [pargs.dataset]


def run_experiments(pargs: argparse.Namespace) -> Dict[str, Any]:
    results = {}

    for dataset in selected_datasets(pargs):
        dataset_args = copy.copy(pargs)
        dataset_args.dataset = dataset
        print(f'\nRunning dataset: {dataset}\n')
        results[dataset] = run_experiment(dataset_args)

    save_results_xlsx(results, Path(pargs.results_xlsx))

    return results


def excel_column(index: int) -> str:
    column = ''

    while index:
        index, remainder = divmod(index - 1, 26)
        column = chr(ord('A') + remainder) + column

    return column


def xlsx_cell(row: int, column: int, value: Any) -> str:
    reference = f'{excel_column(column)}{row}'

    if value is None:
        return f'<c r="{reference}"/>'

    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return f'<c r="{reference}"><v>{float(value)}</v></c>'

    text = escape(str(value))
    return (
        f'<c r="{reference}" t="inlineStr">'
        f'<is><t>{text}</t></is>'
        f'</c>'
    )


def mean_accuracy(metric_dict: Dict[str, List[float]]) -> Optional[float]:
    accuracy = metric_dict.get('Accuracy', [])
    if not accuracy:
        return None

    return float(np.mean(accuracy))


def results_table(results: Dict[str, Any]) -> List[List[Any]]:
    model_names = []
    for dataset_metrics in results.values():
        for model_name in dataset_metrics:
            if model_name not in model_names:
                model_names.append(model_name)

    header = ['Model', *results.keys()]
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
        cells = ''.join(
            xlsx_cell(row_idx, col_idx, value)
            for col_idx, value in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main">'
        '<sheetData>'
        f'{"".join(sheet_rows)}'
        '</sheetData>'
        '</worksheet>'
    )

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr(
            '[Content_Types].xml',
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
            '</Types>',
        )
        workbook.writestr(
            '_rels/.rels',
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        workbook.writestr(
            'xl/workbook.xml',
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.'
            'org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Results" sheetId="1" r:id="rId1"/>'
            '</sheets></workbook>',
        )
        workbook.writestr(
            'xl/_rels/workbook.xml.rels',
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '</Relationships>',
        )
        workbook.writestr('xl/worksheets/sheet1.xml', sheet_xml)


def save_results_xlsx(results: Dict[str, Any], path: Path) -> None:
    if not results:
        return

    write_xlsx(results_table(results), path)
    print(f'\nSaved summary results to {path}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        '--dataset',
        type=str,
        help='The name of the dataset to run',
    )
    dataset_group.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Dataset names to run sequentially',
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
        help='Override GLANT max-hops value',
    )
    parser.add_argument(
        '--heads',
        type=int,
        default=None,
        help='Override GLANT attention heads',
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
    parser.add_argument(
        '--results-xlsx',
        type=str,
        default=DEFAULT_RESULTS_XLSX,
        help='Path for the summary XLSX file',
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true', help='Train the model')
    mode.add_argument('--test', action='store_true', help='Test the model')

    args = parser.parse_args()
    run_experiments(args)
