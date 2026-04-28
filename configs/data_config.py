from ml_collections import ConfigDict


def base_data_config(
    name: str,
    in_channels: int | None = None,
    out_channels: int | None = None,
    num_nodes: int | None = None,
) -> ConfigDict:
    config = ConfigDict()
    config.name = name
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    if in_channels is not None:
        config.in_channels = in_channels
    if out_channels is not None:
        config.out_channels = out_channels
    if num_nodes is not None:
        config.num_nodes = num_nodes

    return config


def cora_config() -> ConfigDict:
    return base_data_config("Cora", 1433, 7, 2708)


def pubmed_config() -> ConfigDict:
    return base_data_config("PubMed", 500, 3, 19717)


def citeseer_config() -> ConfigDict:
    return base_data_config("Citeseer", 3703, 6, 3327)


def computers_config() -> ConfigDict:
    return base_data_config("Computers", 767, 10, 13752)


def photo_config() -> ConfigDict:
    return base_data_config("Photo", 745, 8, 7650)


def actor_config() -> ConfigDict:
    return base_data_config("Actor", 932, 5, 7600)


def wisconsin_config() -> ConfigDict:
    return base_data_config("Wisconsin", 1703, 5, 251)


def texas_config() -> ConfigDict:
    return base_data_config("Texas", 1703, 5, 183)


def aifb_config() -> ConfigDict:
    return base_data_config("AIFB", 1, 4, 8285)


def mutag_config() -> ConfigDict:
    return base_data_config("MUTAG", 1, 2, 23644)


def bgs_config() -> ConfigDict:
    return base_data_config("BGS", 1, 2, 333845)


def dblp_config() -> ConfigDict:
    config = base_data_config("DBLP")
    config.target_node_type = "author"
    config.hetero_source = "hgb"
    config.hetero_to_homo_undirected = True
    return config


def imdb_config() -> ConfigDict:
    config = base_data_config("IMDB")
    config.target_node_type = "movie"
    config.hetero_source = "pyg"
    config.hetero_to_homo_undirected = True
    return config


def acm_config() -> ConfigDict:
    config = base_data_config("ACM")
    config.target_node_type = "paper"
    config.hetero_source = "hgb"
    config.hetero_to_homo_undirected = True
    return config
