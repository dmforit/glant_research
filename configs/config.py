import torch
from ml_collections import ConfigDict

from configs.data_config import (
    cora_config,
    pubmed_config,
    citeseer_config,
    computers_config,
    photo_config,
    wisconsin_config,
    texas_config
)
from configs.model_config import glant_config


def all_config() -> ConfigDict:
    config = ConfigDict()
    config.device = torch.device('cuda:0')

    config.cora = cora_config()
    config.pubmed = pubmed_config()
    config.citeseer = citeseer_config()
    config.computers = computers_config()
    config.photo = photo_config()
    config.wisconsin = wisconsin_config()
    config.texas = texas_config()

    config.experiments = ConfigDict()
    config.experiments.runs = 20

    config.baselines = ConfigDict()
    config.baselines.names = ['GLANT']
    config.baselines.GLANT = glant_config()

    return config
