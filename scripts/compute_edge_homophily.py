from __future__ import annotations

import argparse
import csv
import glob
import logging
import math
import sys
from pathlib import Path

import fsspec
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import all_config
from utils.data_utils import (
    ds_cfg,
    ensure_masks,
    load_ds,
    mask_paths,
    select_mask_split,
    transform,
)
from utils.homophily import edge_homophily, labelled_node_mask, two_hop_labelled_projection


DEFAULT_DATASETS = ["Cora", "Citeseer", "Texas", "AIFB", "IMDB", "ACM"]
LOG = logging.getLogger("edge_homophily")


def patch_fsspec_open_for_glob_chars() -> None:
    """
    PyG loads cached datasets through fsspec. On local Windows paths containing
    square brackets, fsspec treats the current working directory as a glob
    pattern and can miss existing files. Retry local paths with escaped glob
    metacharacters.
    """
    original_open = fsspec.open

    def open_with_escaped_local_path(urlpath, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return original_open(urlpath, *args, **kwargs)
        except FileNotFoundError:
            if not isinstance(urlpath, str) or "://" in urlpath:
                raise
            escaped = glob.escape(str(Path(urlpath).resolve()))
            return original_open(escaped, *args, **kwargs)

    fsspec.open = open_with_escaped_local_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute edge homophily h_edge = |{(i,j) in E: y_i = y_j}| / |E| "
            "for graph datasets."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, or -1 for CPU.")
    parser.add_argument(
        "--labelled-only",
        action="store_true",
        help=(
            "Keep only edges whose endpoints have real labels according to "
            "labeled_mask. By default, the metric is computed on every edge "
            "of the preprocessed homogeneous edge_index."
        ),
    )
    parser.add_argument(
        "--two-hop-labelled-projection",
        action="store_true",
        help=(
            "For target-only heterogeneous datasets, replace E with directed "
            "2-hop edges between labelled nodes that share one intermediate node."
        ),
    )
    parser.add_argument(
        "--drop-self-loops",
        action="store_true",
        help="Exclude self-loops from E before computing homophily.",
    )
    parser.add_argument(
        "--unique-edges",
        action="store_true",
        help="Coalesce duplicate directed edges before computing homophily.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for the computed table.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logs.",
    )
    return parser.parse_args()


def format_value(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def load_graph(config, dataset_name: str):  # type: ignore[no-untyped-def]
    root = Path("Datasets") / dataset_name
    paths = mask_paths(root)
    cfg = ds_cfg(config, dataset_name)
    dataset = load_ds(dataset_name, root, transform(paths), config.device, cfg)
    ensure_masks(dataset, paths, cfg)
    select_mask_split(dataset, getattr(cfg, "split_idx", 0))
    return dataset[0]


def main() -> None:
    patch_fsspec_open_for_glob_chars()
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = all_config()
    if args.gpu >= 0 and torch.cuda.is_available():
        config.device = torch.device(f"cuda:{args.gpu}")
    else:
        config.device = torch.device("cpu")

    rows = []
    for idx, dataset_name in enumerate(args.datasets, start=1):
        LOG.info(
            "[%s/%s] Loading dataset %s",
            idx,
            len(args.datasets),
            dataset_name,
        )
        graph = load_graph(config, dataset_name)
        if args.two_hop_labelled_projection:
            mask = labelled_node_mask(graph)
            graph = graph.clone()
            graph.edge_index = two_hop_labelled_projection(
                graph.edge_index,
                mask,
                int(graph.num_nodes),
            )
        LOG.info(
            "[%s/%s] Computing homophily for %s: nodes=%s edges=%s mode=%s",
            idx,
            len(args.datasets),
            dataset_name,
            int(graph.num_nodes),
            int(graph.edge_index.size(1)),
            "2-hop-labelled-projection"
            if args.two_hop_labelled_projection
            else "preprocessed-edge-index"
            if not args.labelled_only
            else "labelled-only",
        )
        result = edge_homophily(
            graph,
            labelled_only=args.labelled_only and not args.two_hop_labelled_projection,
            drop_self_loops=args.drop_self_loops,
            unique_edges=args.unique_edges,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "h_edge": format_value(result.value),
                "same_class_edges": result.same_class_edges,
                "total_edges": result.total_edges,
            }
        )
        LOG.info(
            "[%s/%s] Done %s: h_edge=%s same=%s total=%s",
            idx,
            len(args.datasets),
            dataset_name,
            format_value(result.value),
            result.same_class_edges,
            result.total_edges,
        )

    fieldnames = ["dataset", "h_edge", "same_class_edges", "total_edges"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            file_writer = csv.DictWriter(file, fieldnames=fieldnames)
            file_writer.writeheader()
            file_writer.writerows(rows)
        LOG.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
