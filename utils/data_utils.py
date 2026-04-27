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

from sampling import get_K_adjs
from utils.logger import logger


Dataset: TypeAlias = Any
Graph: TypeAlias = Any
Loader: TypeAlias = Callable[[str, Path, Compose, torch.device], Dataset]
Edges: TypeAlias = list[Tensor]
Masks: TypeAlias = dict[str, Tensor]
Paths: TypeAlias = dict[str, Path]
FetchResult: TypeAlias = ConfigDict | list[Any]

SPLITS = ("train", "val", "test")
WEBKB = frozenset({"Texas", "Wisconsin"})
HOP_AWARE_CONVS = frozenset({"hop_gated_gatv2"})

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


def stored_graph(ds: Dataset) -> Optional[Graph]:
    """Return the underlying PyG in-memory graph object when available."""
    return getattr(ds, "_data", None)


def set_mask(ds: Dataset, graph: Graph, split: str, mask: Tensor) -> None:
    """Set one split mask on all graph holders used by PyG datasets."""
    name = mask_name(split)
    setattr(graph, name, mask)
    base_graph = stored_graph(ds)
    if base_graph is not None:
        setattr(base_graph, name, mask)
    setattr(ds, name, mask)


def has_masks(graph: Graph) -> bool:
    """Return True if graph has all required split masks."""
    return all(hasattr(graph, mask_name(split)) for split in SPLITS)


def apply_masks(ds: Dataset, masks: Masks) -> None:
    """Attach split masks to both the graph and dataset objects."""
    graph = ds[0]
    for split, mask in masks.items():
        mask = mask.to(graph.y.device)
        set_mask(ds, graph, split, mask)


def random_masks(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Masks:
    """Create reproducible train/validation/test masks for datasets without splits."""
    if num_nodes < 3:
        raise ValueError("At least 3 nodes are required to create train/val/test masks")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_count = max(1, int(num_nodes * train_ratio))
    val_count = max(1, int(num_nodes * val_ratio))
    if train_count + val_count >= num_nodes:
        val_count = max(1, num_nodes - train_count - 1)

    train_idx = perm[:train_count]
    val_idx = perm[train_count:train_count + val_count]
    test_idx = perm[train_count + val_count:]

    masks = {
        split: torch.zeros(num_nodes, dtype=torch.bool)
        for split in SPLITS
    }
    masks["train"][train_idx] = True
    masks["val"][val_idx] = True
    masks["test"][test_idx] = True

    return masks


def ensure_masks(ds: Dataset, paths: Paths, cfg: ConfigDict) -> None:
    """Ensure graph has train/validation/test masks, creating cached ones if needed."""
    graph = ds[0]

    if has_masks(graph):
        sync_masks(ds)
        logger.info("Using masks provided by dataset")
        return

    cached_masks = load_masks(paths)
    if cached_masks is not None:
        apply_masks(ds, cached_masks)
        logger.info("Using cached masks from %s", paths["train"].parent)
        return

    logger.info("Dataset has no masks; creating reproducible random split")
    masks = random_masks(
        num_nodes=graph.num_nodes,
        train_ratio=float(getattr(cfg, "train_ratio", 0.6)),
        val_ratio=float(getattr(cfg, "val_ratio", 0.2)),
        seed=int(getattr(cfg, "split_seed", 0)),
    )
    apply_masks(ds, masks)
    save_masks(ds[0], paths)
    logger.info(
        "Created masks: train=%s val=%s test=%s",
        int(masks["train"].sum()),
        int(masks["val"].sum()),
        int(masks["test"].sum()),
    )


def pick_webkb_split(ds: Dataset, idx: int = 0) -> None:
    graph = ds[0]
    for split in SPLITS:
        mask = getattr(graph, mask_name(split))[:, idx]
        set_mask(ds, graph, split, mask)


def select_mask_split(ds: Dataset, idx: int = 0) -> None:
    """Convert multi-split masks to a single split mask."""
    graph = ds[0]
    for split in SPLITS:
        name = mask_name(split)
        mask = getattr(graph, name)
        if mask.dim() > 1:
            mask = mask[:, idx]
        set_mask(ds, graph, split, mask)


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
            "out_channels": cfg.out_channels,
            "num_features": graph.num_features,
            "num_nodes": cfg.num_nodes,
            "name": cfg.name,
        }
    )


def needs_multihop(model: ConfigDict) -> bool:
    """Return True if model config requires a list of hop edge_index tensors."""
    return str(getattr(model, "conv_type", "")).lower() in HOP_AWARE_CONVS


def mh_cfg(config: ConfigDict) -> Optional[ConfigDict]:
    """
    Return the first selected model config that requires multihop edges.

    Current pipeline supports one shared multihop edge list per experiment.
    """
    for model_name in config.baselines.names:
        if model_name not in config.baselines:
            continue

        model = config.baselines[model_name]
        if needs_multihop(model):
            return model

    return None


def edge_dir(cfg: ConfigDict, model: ConfigDict) -> Path:
    """Directory for cached sampled higher-hop edge sets."""
    num_samples = getattr(model, "num_samples", "default")
    name = (
        f"{model.sampling_method}"
        f"_K{model.max_hops}"
        f"_S{num_samples}"
    )
    return Path(cfg.save_path) / cfg.name / name / "shared"


def load_edges(ds: Dataset, model: ConfigDict, path: Path, device: torch.device) -> Edges:
    logger.info("Loading sampled hop edges from %s", path)
    edges = [ds.edge_index.to(device=device, dtype=torch.int64)]

    for hop in range(1, model.max_hops):
        logger.info(
            "Loading sampled hop edges progress: %s/%s started",
            hop,
            model.max_hops - 1,
        )
        edge = torch.load(path / str(hop)).to(device=device, dtype=torch.int64)
        edges.append(edge)
        logger.info(
            "Loading sampled hop edges progress: %s/%s complete (%s edges)",
            hop,
            model.max_hops - 1,
            edge.shape[1],
        )

    return edges


def make_edges(
    ds: Dataset,
    model: ConfigDict,
    cfg: ConfigDict,
    device: torch.device,
) -> Edges:
    logger.info(
        "Generating sampled hop edges: method=%s max_hops=%s num_samples=%s",
        model.sampling_method,
        model.max_hops,
        getattr(model, "num_samples", "default"),
    )
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
    logger.info("Edges for each hop")
    for hop, edge in enumerate(edges, start=1):
        logger.info("%s hop: %s", hop, edge.shape[1])
    logger.info("Edges for each hop ended")


def edges(
    ds: Dataset,
    model: ConfigDict,
    cfg: ConfigDict,
    device: torch.device,
) -> Edges:
    path = edge_dir(cfg, model)
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Shared sampled hop edge cache: %s", path)

    if model.load_samples:
        out = load_edges(ds, model, path, device)
    else:
        out = make_edges(ds, model, cfg, device)
        logger.info("Saving sampled hop edges to %s", path)
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

    logger.info("Multihop config:\n%s", model)

    logger.info("Building multihop edges...")
    add_mh(data, model, cfg, config.device)
    logger.info("Building multihop edges - done.")


def unpack(data: ConfigDict) -> list[Any]:
    return [data[key] for key in data]


def fetch_dataset(
    config: ConfigDict,
    ds_name: str,
    unpack_: bool = False,
) -> FetchResult:
    root = Path("Datasets") / ds_name
    paths = mask_paths(root)
    cfg = ds_cfg(config, ds_name)

    logger.info("Dataset loading...")
    ds = load_ds(ds_name, root, transform(paths), config.device)
    logger.info("Dataset has been loaded successfully.")

    ensure_masks(ds, paths, cfg)
    select_mask_split(ds, getattr(cfg, "split_idx", 0))
    save_masks(ds[0], paths)

    data = pack(ds, cfg)
    maybe_add_mh(data, config, cfg)

    return unpack(data) if unpack_ else data
