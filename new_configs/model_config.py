from ml_collections import ConfigDict


def glant_config() -> ConfigDict:
    config = ConfigDict()

    config.model_name = "GLANT"
    config.num_layers = 2
    config.heads = 1
    config.alpha = 1.0
    config.max_hops = 1
    config.hidden_channels = 64
    config.dropout = 0.3
    config.pre_linear = False
    config.batchnorm = False
    config.layernorm = False
    config.residual = True
    config.load_samples = False
    config.sampling_method = "sim_walk"

    config.walk = ConfigDict()
    config.walk.gamma = 0.9
    config.walk.jump_prob = 0.05
    config.walk.use_cosine = True

    config.training = ConfigDict()
    config.training.weight_decay = 5e-4
    config.training.lr = 0.001
    config.training.optimizer = "adam"
    config.training.num_epochs = 101
    config.training.decay = 0.95
    config.training.early_stop_patience = 100
    config.training.save_freq = 50

    config.save_path = "./checkpoints/new_glant.pt"
    config.log_interval = 20

    return config
