from ml_collections import ConfigDict


def training_config(
    lr: float = 0.05,
    weight_decay: float = 5e-4,
    num_epochs: int = 300,
    scheduler_name: str = "plateau",
) -> ConfigDict:
    config = ConfigDict()

    config.optimizer = "adam"
    config.lr = lr
    config.weight_decay = weight_decay
    config.num_epochs = num_epochs
    config.decay = 0.95
    config.save_freq = 50

    config.scheduler = ConfigDict()
    config.scheduler.name = scheduler_name
    config.scheduler.mode = "min"
    config.scheduler.factor = 0.6
    config.scheduler.patience = 10
    config.scheduler.threshold = 1e-4
    config.scheduler.min_lr = 1e-5

    return config


def base_gnn_config() -> ConfigDict:
    config = ConfigDict()

    config.num_layers = 2
    config.hidden_channels = 64

    config.dropout = 0.5
    config.attn_dropout = 0.5

    config.pre_linear = False
    config.residual = False
    config.norm = "none"
    config.act = "elu"

    config.heads = 8
    config.concat = False
    config.negative_slope = 0.2
    config.add_self_loops = True
    config.bias = True
    config.edge_dim = None
    config.fill_value = "mean"

    config.conv_residual = False
    config.share_weights = False

    config.max_hops = 1
    config.alpha = None

    config.training = training_config(
        lr=0.005,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="plateau",
    )
    return config


def glant_config() -> ConfigDict:
    config = base_gnn_config()

    config.architecture = "mixture of all hops"
    config.conv_type = "hop_gated_gatv2"

    config.num_layers = 2
    config.hidden_channels = 64
    config.heads = 8
    config.concat = False

    config.max_hops = 3
    config.alpha = 0.05

    config.dropout = 0.7
    config.attn_dropout = 0.7
    config.gate_hidden = None
    config.gate_dropout = 0.0

    config.pre_linear = False
    config.residual = False
    config.norm = "none"
    config.act = "elu"

    config.negative_slope = 0.2
    config.add_self_loops = True
    config.bias = False
    config.edge_dim = None
    config.fill_value = "mean"

    config.sparsify_hops = True
    config.sparsifier_cache_masks = True

    config.load_samples = False
    config.sampling_method = "balanced_unique_select"
    config.num_samples = 15

    config.log_hop_diagnostics = True
    config.hop_log_every = 50
    config.hop_log_only_layer = None

    config.walk = ConfigDict()
    config.walk.gamma = 0.9
    config.walk.jump_prob = 0.05
    config.walk.use_cosine = True

    config.training = training_config(
        lr=0.05,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="plateau",
    )

    return config


def gat_config() -> ConfigDict:
    config = glant_config()

    config.conv_type = "gat"
    config.max_hops = 1
    config.alpha = None
    config.sparsify_hops = False
    config.log_hop_diagnostics = False

    return config


def gatv2_config() -> ConfigDict:
    config = glant_config()

    config.conv_type = "gatv2"
    config.max_hops = 1
    config.alpha = None
    config.sparsify_hops = False
    config.share_weights = False
    config.log_hop_diagnostics = False

    return config


def gcn_config() -> ConfigDict:
    config = base_gnn_config()

    config.conv_type = "gcn"

    config.num_layers = 2
    config.hidden_channels = 64
    config.dropout = 0.3
    config.attn_dropout = 0.0
    config.act = "relu"

    config.training = training_config(
        lr=0.01,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    return config
