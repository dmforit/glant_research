from __future__ import annotations

from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, TypeAlias

import torch
import torch_geometric.datasets as pygds
from ml_collections import ConfigDict
from torch import Tensor
from torch_geometric.transforms import Compose, NormalizeFeatures

from adj_khop_logic import get_K_adjs


Dataset: TypeAlias = Any
Graph: TypeAlias = Any
Loader: TypeAlias = Callable[[str, Path, Compose, torch.device], Dataset]
Edges: TypeAlias = list[Tensor]
Masks: TypeAlias = dict[str, Tensor]
Paths: TypeAlias = dict[str, Path]
FetchResult: TypeAlias = ConfigDict | list[Any]

SPLITS = ("train", "val", "test")
WEBKB = frozenset({"Texas", "Wisconsin"})
MULTIHOP = frozenset({"HoGA_GAT", "HoGA_GRAND", "GLANT"})

DS_CFG = {
    "Cora": "cora",
    "Pubmed": "pubmed",
    "Citeseer": "citeseer",
    "Computers": "computers",
    "Photo": "photo",
    "Actor": "actor",
    "Wisconsin": "wisconsin",
    "Texas": "texas",
}


def mask_name(split: str) -> str:
    return f"{split}_mask"


def set_masks(data: Graph, **masks: Tensor) -> Graph:
    for split, mask in masks.items():
        setattr(data, mask_name(split), mask)
    return data


def sync_masks(ds: Dataset) -> None:
    graph = ds[0]
    for split in SPLITS:
        setattr(ds, mask_name(split), getattr(graph, mask_name(split)))


def pick_webkb_split(ds: Dataset, idx: int = 0) -> None:
    graph = ds[0]
    for split in SPLITS:
        mask = getattr(graph, mask_name(split))[:, idx]
        setattr(graph, mask_name(split), mask)
        setattr(ds, mask_name(split), mask)


def mask_paths(root: Path) -> Paths:
    return {split: root / mask_name(split) for split in SPLITS}


def load_masks(paths: Paths) -> Optional[Masks]:
    if not all(path.exists() for path in paths.values()):
        return None
    return {split: torch.load(path) for split, path in paths.items()}


def save_masks(graph: Graph, paths: Paths) -> None:
    for split, path in paths.items():
        torch.save(getattr(graph, mask_name(split)), path)


def transform(paths: Paths) -> Compose:
    steps: list[Any] = [NormalizeFeatures()]
    masks = load_masks(paths)
    if masks is not None:
        steps.append(partial(set_masks, **masks))
    return Compose(steps)


def planetoid(name: str, root: Path, _: Compose, device: torch.device) -> Dataset:
    return pygds.Planetoid(
        root=str(root),
        name=name,
        transform=NormalizeFeatures(),
    ).to(device)


def amazon(name: str, root: Path, tr: Compose, device: torch.device) -> Dataset:
    return pygds.Amazon(root=str(root), name=name, transform=tr).to(device)


def actor(_: str, root: Path, tr: Compose, device: torch.device) -> Dataset:
    return pygds.Actor(root=str(root), transform=tr).to(device)


def webkb(name: str, root: Path, _: Compose, device: torch.device) -> Dataset:
    ds = pygds.WebKB(root=str(root), name=name).to(device)
    pick_webkb_split(ds)
    return ds


LOADERS: dict[str, Loader] = {
    "Cora": planetoid,
    "Pubmed": planetoid,
    "Citeseer": planetoid,
    "Computers": amazon,
    "Photo": amazon,
    "Actor": actor,
    "Texas": webkb,
    "Wisconsin": webkb,
}


def load_ds(name: str, root: Path, tr: Compose, device: torch.device) -> Dataset:
    try:
        return LOADERS[name](name, root, tr, device)
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc


def ds_cfg(config: ConfigDict, name: str) -> ConfigDict:
    try:
        key = DS_CFG[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc

    if not hasattr(config, key):
        raise ValueError(f"Missing config.{key}")

    return getattr(config, key)


def pack(ds: Dataset, cfg: ConfigDict) -> ConfigDict:
    graph = ds[0]
    return ConfigDict(
        {
            "dataset": ds,
            "graph": graph.edge_index,
            "train": graph.train_mask,
            "val": graph.val_mask,
            "test": graph.test_mask,
            "num_classes": cfg.num_classes,
            "num_features": graph.num_features,
            "num_nodes": cfg.num_nodes,
            "name": cfg.name,
        }
    )


def mh_cfg(config: ConfigDict) -> Optional[ConfigDict]:
    names = set(config.baselines.names)
    if names.isdisjoint(MULTIHOP):
        return None
    return getattr(config.baselines, "GLANT", None)


def edge_dir(cfg: ConfigDict, model: ConfigDict) -> Path:
    return Path(cfg.save_path) / model.select_method / "shared"


def load_edges(ds: Dataset, model: ConfigDict, path: Path, device: torch.device) -> Edges:
    edges = [ds.edge_index.to(device=device, dtype=torch.int64)]

    for hop in range(1, model.K_hops):
        edge = torch.load(path / str(hop)).to(device=device, dtype=torch.int64)
        edges.append(edge)

    return edges


def make_edges(
    ds: Dataset,
    model: ConfigDict,
    cfg: ConfigDict,
    device: torch.device,
) -> Edges:
    edges = get_K_adjs(
        ds.edge_index,
        model,
        cfg,
        feature_set=ds.x.to(device),
        device=device,
    )
    return [edge.to(device=device, dtype=torch.int64) for edge in edges]


def save_edges(edges: Edges, path: Path) -> None:
    for hop, edge in enumerate(edges[1:], start=1):
        torch.save(edge.cpu().to(torch.int64), path / str(hop))


def show_edges(edges: Edges) -> None:
    print("\nEdges for each hop\n")
    for hop, edge in enumerate(edges, start=1):
        print(f"{hop} hop: {edge.shape[1]}")
    print("\nEdges for each hop ended")


def edges(
    ds: Dataset,
    model: ConfigDict,
    cfg: ConfigDict,
    device: torch.device,
) -> Edges:
    path = edge_dir(cfg, model)
    path.mkdir(parents=True, exist_ok=True)

    if model.load_samples:
        out = load_edges(ds, model, path, device)
    else:
        out = make_edges(ds, model, cfg, device)
        save_edges(out, path)

    show_edges(out)
    return out


def add_mh(data: ConfigDict, model: ConfigDict, cfg: ConfigDict, device: torch.device) -> None:
    ds = copy(data.dataset)
    ds.edge_index = edges(data.dataset, model, cfg, device)
    data.multihop_dataset = ds


def maybe_add_mh(data: ConfigDict, config: ConfigDict, cfg: ConfigDict) -> None:
    model = mh_cfg(config)
    if model is None:
        return

    print(f"\nMultihop config:\n{model}")

    print("\nBuilding multihop edges...\n")
    add_mh(data, model, cfg, config.device)
    print("\nBuilding multihop edges - done.")


def unpack(data: ConfigDict) -> list[Any]:
    return [data[key] for key in data]


def fetch_dataset(
    config: ConfigDict,
    ds_name: str,
    unpack_: bool = False,
) -> FetchResult:
    root = Path("Datasets") / ds_name
    paths = mask_paths(root)

    print("Dataset loading...")
    ds = load_ds(ds_name, root, transform(paths), config.device)
    print("Dataset has been loaded successfully.")

    if ds_name not in WEBKB:
        sync_masks(ds)

    save_masks(ds[0], paths)

    cfg = ds_cfg(config, ds_name)
    data = pack(ds, cfg)

    maybe_add_mh(data, config, cfg)

    return unpack(data) if unpack_ else data
