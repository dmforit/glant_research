import torch
from ml_collections import ConfigDict

from configs.data_config import (
    actor_config,
    citeseer_config,
    computers_config,
    cora_config,
    photo_config,
    pubmed_config,
    texas_config,
    wisconsin_config,
)
from configs.model_config import (
    gat_config,
    gatv2_config,
    gcn_config,
    glant_config,
)


def all_config() -> ConfigDict:
    config = ConfigDict()
    config.device = torch.device("cuda:0")

    config.cora = cora_config()
    config.pubmed = pubmed_config()
    config.citeseer = citeseer_config()
    config.computers = computers_config()
    config.photo = photo_config()
    config.actor = actor_config()
    config.wisconsin = wisconsin_config()
    config.texas = texas_config()

    config.experiments = ConfigDict()
    config.experiments.runs = 20

    config.baselines = ConfigDict()
    config.baselines.names = ["GLANT"] # ["GLANT", "GAT", "GATv2"]
    config.baselines.GLANT = glant_config()
    config.baselines.GAT = gat_config()
    config.baselines.GATv2 = gatv2_config()
    config.baselines.GCN = gcn_config()

    return config
