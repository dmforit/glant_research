from __future__ import annotations

from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, TypeAlias

import torch
import torch_geometric.datasets as pygds
from ml_collections import ConfigDict
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, NormalizeFeatures

from sampling import get_K_adjs
from utils.logger import logger
from utils.model_names import canonical_model_name


Dataset: TypeAlias = Any
Graph: TypeAlias = Any
Loader: TypeAlias = Callable[[str, Path, Compose, torch.device, ConfigDict], Dataset]
Edges: TypeAlias = list[Tensor]
Masks: TypeAlias = dict[str, Tensor]
Paths: TypeAlias = dict[str, Path]

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
    "AIFB": "aifb",
    "MUTAG": "mutag",
    "BGS": "bgs",
    "DBLP": "dblp",
    "IMDB": "imdb",
    "ACM": "acm",
}


class SingleGraphDataset:
    def __init__(self, graph: Data) -> None:
        self.graph = graph

    def __getitem__(self, index: int) -> Data:
        if index != 0:
            raise IndexError(index)
        return self.graph

    def __len__(self) -> int:
        return 1

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.graph, name)

    def __copy__(self) -> SingleGraphDataset:
        return SingleGraphDataset(copy(self.graph))

    def to(self, device: torch.device) -> SingleGraphDataset:
        self.graph = self.graph.to(device)
        return self


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


def planetoid(
    name: str,
    root: Path,
    _: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    del cfg
    return pygds.Planetoid(
        root=str(root),
        name=name,
        transform=NormalizeFeatures(),
    ).to(device)


def amazon(
    name: str,
    root: Path,
    tr: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    del cfg
    return pygds.Amazon(root=str(root), name=name, transform=tr).to(device)


def actor(
    _: str,
    root: Path,
    tr: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    del cfg
    return pygds.Actor(root=str(root), transform=tr).to(device)


def webkb(
    name: str,
    root: Path,
    _: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    del cfg
    ds = pygds.WebKB(root=str(root), name=name).to(device)
    pick_webkb_split(ds)
    return ds


def split_train_val_mask(
    train_mask: Tensor,
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    train_idx = train_mask.nonzero(as_tuple=False).view(-1).cpu()
    if train_idx.numel() < 2:
        return train_mask, torch.zeros_like(train_mask)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
    val_count = max(1, int(train_idx.numel() * val_ratio))
    val_idx = train_idx[:val_count]
    keep_idx = train_idx[val_count:]

    new_train_mask = torch.zeros_like(train_mask)
    val_mask = torch.zeros_like(train_mask)
    new_train_mask[keep_idx.to(train_mask.device)] = True
    val_mask[val_idx.to(train_mask.device)] = True
    return new_train_mask, val_mask


def select_split_mask(mask: Tensor, cfg: ConfigDict) -> Tensor:
    if mask.dim() > 1:
        mask = mask[:, int(getattr(cfg, "split_idx", 0))]
    return mask


def set_dynamic_cfg(cfg: ConfigDict, graph: Data) -> None:
    cfg.in_channels = int(graph.x.size(-1))
    cfg.out_channels = int(graph.y.max().item()) + 1
    cfg.num_nodes = int(graph.num_nodes)


def make_index_masks(
    num_nodes: int,
    train_idx: Tensor,
    test_idx: Tensor,
    *,
    device: torch.device,
    seed: int,
) -> Masks:
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx.to(device=device, dtype=torch.long)] = True
    test_mask[test_idx.to(device=device, dtype=torch.long)] = True
    train_mask, val_mask = split_train_val_mask(train_mask, seed=seed)
    return {"train": train_mask, "val": val_mask, "test": test_mask}


def entity_labels_and_masks(graph: Data, cfg: ConfigDict, device: torch.device) -> tuple[Tensor, Masks]:
    num_nodes = int(graph.num_nodes)
    seed = int(getattr(cfg, "split_seed", 0))

    if hasattr(graph, "y") and graph.y is not None and graph.y.numel() == num_nodes:
        y = graph.y.to(device=device, dtype=torch.long).view(-1)
        if hasattr(graph, "train_mask") and hasattr(graph, "test_mask"):
            train_mask = select_split_mask(graph.train_mask, cfg).to(
                device=device,
                dtype=torch.bool,
            ).view(-1)
            test_mask = select_split_mask(graph.test_mask, cfg).to(
                device=device,
                dtype=torch.bool,
            ).view(-1)
            train_mask, val_mask = split_train_val_mask(train_mask, seed=seed)
            return y, {"train": train_mask, "val": val_mask, "test": test_mask}

    if all(hasattr(graph, name) for name in ("train_idx", "test_idx", "train_y", "test_y")):
        train_idx = graph.train_idx.view(-1)
        test_idx = graph.test_idx.view(-1)
        y = torch.zeros(num_nodes, dtype=torch.long, device=device)
        y[train_idx.to(device=device, dtype=torch.long)] = graph.train_y.to(device=device, dtype=torch.long).view(-1)
        y[test_idx.to(device=device, dtype=torch.long)] = graph.test_y.to(device=device, dtype=torch.long).view(-1)
        masks = make_index_masks(
            num_nodes,
            train_idx,
            test_idx,
            device=device,
            seed=seed,
        )
        return y, masks

    raise ValueError(f"{cfg.name} does not expose node labels and train/test splits")


def entity_dataset(
    name: str,
    root: Path,
    _: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    raw = pygds.Entities(root=str(root), name=name)
    graph = raw[0]
    graph.num_nodes = int(graph.num_nodes)
    graph.edge_index = graph.edge_index.to(device=device, dtype=torch.int64)
    graph.x = torch.ones((graph.num_nodes, 1), dtype=torch.float32, device=device)

    graph.y, masks = entity_labels_and_masks(graph, cfg, device)
    for split, mask in masks.items():
        setattr(graph, mask_name(split), mask)

    set_dynamic_cfg(cfg, graph)
    return SingleGraphDataset(graph)


def target_node_type(data: Any, cfg: ConfigDict) -> str:
    configured = getattr(cfg, "target_node_type", None)
    if configured is not None and str(configured) in data.node_types:
        return str(configured)

    for node_type in data.node_types:
        store = data[node_type]
        if hasattr(store, "y") and hasattr(store, "train_mask"):
            return str(node_type)

    raise ValueError(f"{cfg.name} does not expose a labelled target node type")


def node_offsets(data: Any) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for node_type in data.node_types:
        offsets[str(node_type)] = offset
        offset += int(data[node_type].num_nodes)
    return offsets


def hetero_node_features(data: Any, offsets: dict[str, int]) -> Tensor:
    max_feature_dim = 0
    for node_type in data.node_types:
        x = getattr(data[node_type], "x", None)
        if torch.is_tensor(x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            max_feature_dim = max(max_feature_dim, int(x.size(-1)))

    num_types = len(data.node_types)
    total_nodes = sum(int(data[node_type].num_nodes) for node_type in data.node_types)
    x = torch.zeros((total_nodes, max_feature_dim + num_types), dtype=torch.float32)
    type_offset = max_feature_dim

    for type_idx, node_type in enumerate(data.node_types):
        store = data[node_type]
        start = offsets[str(node_type)]
        stop = start + int(store.num_nodes)
        node_x = getattr(store, "x", None)

        if torch.is_tensor(node_x):
            if node_x.dim() == 1:
                node_x = node_x.unsqueeze(-1)
            node_x = node_x.to(dtype=torch.float32)
            x[start:stop, :node_x.size(-1)] = node_x

        x[start:stop, type_offset + type_idx] = 1.0

    return x


def target_masks(
    total_nodes: int,
    target_start: int,
    target_stop: int,
    target_store: Any,
    cfg: ConfigDict,
) -> Masks:
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    if hasattr(target_store, "train_mask") and hasattr(target_store, "test_mask"):
        target_train = select_split_mask(target_store.train_mask, cfg).to(dtype=torch.bool).view(-1)
        target_test = select_split_mask(target_store.test_mask, cfg).to(dtype=torch.bool).view(-1)

        if hasattr(target_store, "val_mask"):
            target_val = select_split_mask(target_store.val_mask, cfg).to(dtype=torch.bool).view(-1)
        else:
            target_train, target_val = split_train_val_mask(
                target_train,
                seed=int(getattr(cfg, "split_seed", 0)),
            )

        train_mask[target_start:target_stop] = target_train
        val_mask[target_start:target_stop] = target_val
        test_mask[target_start:target_stop] = target_test
        return {"train": train_mask, "val": val_mask, "test": test_mask}

    target_node_count = target_stop - target_start
    generated = random_masks(
        target_node_count,
        train_ratio=float(getattr(cfg, "train_ratio", 0.6)),
        val_ratio=float(getattr(cfg, "val_ratio", 0.2)),
        seed=int(getattr(cfg, "split_seed", 0)),
    )
    train_mask[target_start:target_stop] = generated["train"]
    val_mask[target_start:target_stop] = generated["val"]
    test_mask[target_start:target_stop] = generated["test"]
    return {"train": train_mask, "val": val_mask, "test": test_mask}


def hetero_to_homogeneous(data: Any, cfg: ConfigDict, device: torch.device) -> Data:
    target = target_node_type(data, cfg)
    offsets = node_offsets(data)
    total_nodes = sum(int(data[node_type].num_nodes) for node_type in data.node_types)
    x = hetero_node_features(data, offsets)

    edge_indices = []
    make_undirected = bool(getattr(cfg, "hetero_to_homo_undirected", True))
    for src_type, _, dst_type in data.edge_types:
        edge_index = data[(src_type, _, dst_type)].edge_index
        edge_index = edge_index.clone()
        edge_index[0] += offsets[str(src_type)]
        edge_index[1] += offsets[str(dst_type)]
        edge_indices.append(edge_index)
        if make_undirected:
            edge_indices.append(edge_index.flip(0))

    if not edge_indices:
        raise ValueError(f"{cfg.name} does not contain edges")

    target_store = data[target]
    y_target = target_store.y
    if y_target.dim() > 1:
        y_target = y_target.argmax(dim=-1)
    y_target = y_target.to(dtype=torch.long).view(-1)
    y = torch.zeros(total_nodes, dtype=torch.long)
    target_start = offsets[target]
    target_stop = target_start + int(target_store.num_nodes)
    y[target_start:target_stop] = y_target

    masks = target_masks(
        total_nodes,
        target_start,
        target_stop,
        target_store,
        cfg,
    )
    graph = Data(
        x=x,
        edge_index=torch.cat(edge_indices, dim=1).to(dtype=torch.int64),
        y=y,
        train_mask=masks["train"],
        val_mask=masks["val"],
        test_mask=masks["test"],
        num_nodes=total_nodes,
    ).to(device)
    set_dynamic_cfg(cfg, graph)
    return graph


def hgb_dataset(
    name: str,
    root: Path,
    _: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    raw = pygds.HGBDataset(root=str(root), name=name)
    graph = hetero_to_homogeneous(raw[0], cfg, device)
    return SingleGraphDataset(graph)


def imdb_dataset(
    name: str,
    root: Path,
    _: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    del name
    raw = pygds.IMDB(root=str(root))
    graph = hetero_to_homogeneous(raw[0], cfg, device)
    return SingleGraphDataset(graph)


LOADERS: dict[str, Loader] = {
    "Cora": planetoid,
    "Pubmed": planetoid,
    "Citeseer": planetoid,
    "Computers": amazon,
    "Photo": amazon,
    "Actor": actor,
    "Texas": webkb,
    "Wisconsin": webkb,
    "AIFB": entity_dataset,
    "MUTAG": entity_dataset,
    "BGS": entity_dataset,
    "DBLP": hgb_dataset,
    "IMDB": imdb_dataset,
    "ACM": hgb_dataset,
}


def load_ds(
    name: str,
    root: Path,
    tr: Compose,
    device: torch.device,
    cfg: ConfigDict,
) -> Dataset:
    try:
        return LOADERS[name](name, root, tr, device, cfg)
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
        model_name = canonical_model_name(model_name)
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
    if isinstance(ds, SingleGraphDataset):
        ds.graph.edge_index = ds.edge_index
    data.multihop_dataset = ds


def maybe_add_mh(data: ConfigDict, config: ConfigDict, cfg: ConfigDict) -> None:
    model = mh_cfg(config)
    if model is None:
        return

    logger.info("Multihop config:\n%s", model)

    logger.info("Building multihop edges...")
    add_mh(data, model, cfg, config.device)
    logger.info("Building multihop edges - done.")


def fetch_dataset(
    config: ConfigDict,
    ds_name: str,
) -> ConfigDict:
    root = Path("Datasets") / ds_name
    paths = mask_paths(root)
    cfg = ds_cfg(config, ds_name)

    logger.info("Dataset loading...")
    ds = load_ds(ds_name, root, transform(paths), config.device, cfg)
    logger.info("Dataset has been loaded successfully.")

    ensure_masks(ds, paths, cfg)
    select_mask_split(ds, getattr(cfg, "split_idx", 0))
    save_masks(ds[0], paths)

    data = pack(ds, cfg)
    maybe_add_mh(data, config, cfg)

    return data
