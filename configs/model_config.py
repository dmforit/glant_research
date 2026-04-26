from ml_collections import ConfigDict


def glant_config() -> ConfigDict:
    config = ConfigDict()

    # architecture settings
    config.model_name = 'GLANT'
    config.num_layers = 2
    config.K_hops = 3
    config.num_heads = [8, 1]
    config.num_heads_small = 1
    config.layer_type = 'multi_hop'
    config.head_type = 'gatv2'
    config.agg_func = 'sum'
    config.beta_mul = 0.9
    config.hidden_channels = 64
    config.load_samples = False
    config.select_method = 'random_walk'
    config.loops = True
    config.drop_out = 0.6

    # heuristic walk
    config.walk = ConfigDict()
    config.walk.gamma = 0.9
    config.walk.jump_prob = 0.05
    config.walk.use_cosine = True

    # training parameters
    config.training = ConfigDict()
    config.training.weight_decay = 5e-4
    config.training.lr = 0.05
    config.training.optimizer = 'adam'
    config.training.num_epochs = 100
    config.training.drop_out = 0.6
    config.training.decay = 0.95
    config.training.early_stop_patience = 100
    config.training.save_freq = 50

    config.device = 'cuda:0'
    config.save_path = './checkpoints/glant.pt'
    config.log_interval = 20

    return config
