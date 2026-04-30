import torch
from ml_collections import ConfigDict

from configs.data_config import (
    acm_config,
    aifb_config,
    actor_config,
    bgs_config,
    citeseer_config,
    computers_config,
    cora_config,
    dblp_config,
    imdb_config,
    mutag_config,
    photo_config,
    pubmed_config,
    texas_config,
    wisconsin_config,
)
from configs.model_config import (
    gat_config,
    gatv2_config,
    gcn_config,
    graphsage_config,
    glant_config,
    glant_v2_config,
    glant_v3_config,
    glant_v4_config,
    glant_v5_config,
    glant_v6_config,
    glant_v6p1_config,
    glant_v7_config,
    glant_v8_config,
    hoga_config,
    mixhop_config,
    tagconv_config,
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
    config.aifb = aifb_config()
    config.mutag = mutag_config()
    config.bgs = bgs_config()
    config.dblp = dblp_config()
    config.imdb = imdb_config()
    config.acm = acm_config()

    config.experiments = ConfigDict()
    config.experiments.runs = 20

    config.results_dir = "results"
    config.save_best_model = False
    config.collect_summary_after_training = False
    config.run_mode = "final"
    config.seed = 0

    config.baselines = ConfigDict()
    config.baselines.names = [
        "GLANT_v1",
        "GLANT_v2",
        "GLANT_v3",
        "GLANT_v4",
        "GLANT_v5",
        "GLANT_v6",
        "GLANT_v6p1",
        "GLANT_v7",
        "GLANT_v8",
        "GATv2",
    ]
    config.baselines.GLANT_v1 = glant_config()
    config.baselines.GLANT = config.baselines.GLANT_v1
    config.baselines.GLANT_v2 = glant_v2_config()
    config.baselines.GLANT_v3 = glant_v3_config()
    config.baselines.GLANT_v4 = glant_v4_config()
    config.baselines.GLANT_v5 = glant_v5_config()
    config.baselines.GLANT_v6 = glant_v6_config()
    config.baselines.GLANT_v6p1 = glant_v6p1_config()
    config.baselines.GLANT_v7 = glant_v7_config()
    config.baselines.GLANT_v8 = glant_v8_config()
    config.baselines.GAT = gat_config()
    config.baselines.GATv2 = gatv2_config()
    config.baselines.GCN = gcn_config()
    config.baselines.GraphSAGE = graphsage_config()
    config.baselines.MixHop = mixhop_config()
    config.baselines.TAGConv = tagconv_config()
    config.baselines.khop_model_1 = config.baselines.MixHop
    config.baselines.khop_model_2 = config.baselines.TAGConv
    config.baselines.HoGA = hoga_config()

    return config
