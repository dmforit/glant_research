from ml_collections import ConfigDict


def cora_config() -> ConfigDict:
    config = ConfigDict()
    config.name = 'Cora'
    config.in_channels = 1433
    config.num_classes = 7
    config.hidden_channels = 64
    config.num_nodes = 2708
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def pubmed_config() -> ConfigDict:
    config = ConfigDict()
    config.name = 'PubMed'
    config.in_channels = 500
    config.num_classes = 3
    config.hidden_channels = 64
    config.num_nodes = 19717
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def citeseer_config() -> ConfigDict:
    config = ConfigDict()
    config.name = 'Citeseer'
    config.in_channels = 3703
    config.num_classes = 6
    config.hidden_channels = 64
    config.num_nodes = 3327
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def computers_config() -> ConfigDict:
    config = ConfigDict()
    config.name = 'Computers'
    config.in_channels = 767
    config.num_classes = 10
    config.hidden_channels = 128
    config.num_nodes = 13752
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def photo_config() -> ConfigDict:
    config = ConfigDict()
    config.name = 'Photo'
    config.in_channels = 745
    config.num_classes = 8
    config.hidden_channels = 128
    config.num_nodes = 7650
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def wisconsin_config() -> ConfigDict:
    config = ConfigDict()
    config.name = 'Wisconsin'
    config.in_channels = 1703
    config.num_classes = 5
    config.hidden_channels = 64
    config.num_nodes = 251
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config


def texas_config() -> ConfigDict:
    config = ConfigDict()
    config.name = 'Texas'
    config.in_channels = 1703
    config.num_classes = 5
    config.hidden_channels = 64
    config.num_nodes = 183
    config.save_path = 'model_runs'
    config.metrics = ['Accuracy']

    return config
