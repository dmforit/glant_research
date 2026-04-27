from ml_collections import ConfigDict


def cora_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "Cora"
    config.in_channels = 1433
    config.out_channels = 7
    config.num_nodes = 2708
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config


def pubmed_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "PubMed"
    config.in_channels = 500
    config.out_channels = 3
    config.num_nodes = 19717
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config


def citeseer_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "Citeseer"
    config.in_channels = 3703
    config.out_channels = 6
    config.num_nodes = 3327
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config


def computers_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "Computers"
    config.in_channels = 767
    config.out_channels = 10
    config.num_nodes = 13752
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config


def photo_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "Photo"
    config.in_channels = 745
    config.out_channels = 8
    config.num_nodes = 7650
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config


def actor_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "Actor"
    config.in_channels = 932
    config.out_channels = 5
    config.num_nodes = 7600
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config


def wisconsin_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "Wisconsin"
    config.in_channels = 1703
    config.out_channels = 5
    config.num_nodes = 251
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config


def texas_config() -> ConfigDict:
    config = ConfigDict()
    config.name = "Texas"
    config.in_channels = 1703
    config.out_channels = 5
    config.num_nodes = 183
    config.save_path = "model_runs"
    config.metrics = ["Accuracy"]
    config.split_idx = 0

    return config
