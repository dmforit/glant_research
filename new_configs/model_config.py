from ml_collections import ConfigDict


def training_config(
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    num_epochs: int = 101,
    scheduler_name: str = "plateau",
) -> ConfigDict:
    config = ConfigDict()
    config.weight_decay = weight_decay
    config.lr = lr
    config.optimizer = "adam"
    config.num_epochs = num_epochs
    config.decay = 0.95
    config.early_stop_patience = 100
    config.save_freq = 50

    config.scheduler = ConfigDict()
    config.scheduler.name = scheduler_name
    config.scheduler.mode = "min"
    config.scheduler.factor = 0.5
    config.scheduler.patience = 10
    config.scheduler.threshold = 1e-4
    config.scheduler.min_lr = 1e-6

    return config


def glant_config() -> ConfigDict:
    config = ConfigDict()

    config.model_name = "GLANT"
    config.num_layers = 2
    config.heads = 8
    config.alpha = 0.3
    config.max_hops = 3
    config.hidden_channels = 64
    config.dropout = 0.7
    config.act = "elu"
    config.concat = False
    config.negative_slope = 0.2
    config.add_self_loops = True
    config.bias = False
    config.share_weights = False
    config.pre_linear = False
    config.batchnorm = False
    config.layernorm = False
    config.residual = False
    config.load_samples = False
    config.sampling_method = "balanced_unique_select"

    config.walk = ConfigDict()
    config.walk.gamma = 0.9
    config.walk.jump_prob = 0.05
    config.walk.use_cosine = True

    config.training = training_config(
        lr=0.005,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    config.save_path = "./checkpoints/new_glant.pt"
    config.log_interval = 20

    return config


def gat_config() -> ConfigDict:
    config = ConfigDict()

    config.model_name = "GAT"
    config.num_layers = 2
    config.heads = 8
    config.hidden_channels = 64
    config.dropout = 0.6
    config.act = "elu"
    config.training = training_config(
        lr=0.005,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    config.save_path = "./checkpoints/gat.pt"
    config.log_interval = 20

    return config


def gatv2_config() -> ConfigDict:
    config = ConfigDict()

    config.model_name = "GATv2"
    config.num_layers = 2
    config.heads = 8
    config.hidden_channels = 64
    config.dropout = 0.7
    config.act = "elu"
    config.share_weights = False
    config.training = training_config(
        lr=0.005,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    config.save_path = "./checkpoints/gatv2.pt"
    config.log_interval = 20

    return config


def gcn_config() -> ConfigDict:
    config = ConfigDict()

    config.model_name = "GCN"
    config.num_layers = 2
    config.hidden_channels = 64
    config.dropout = 0.3
    config.training = training_config()

    config.save_path = "./checkpoints/gcn.pt"
    config.log_interval = 20

    return config
