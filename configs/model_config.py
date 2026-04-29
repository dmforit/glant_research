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

    config.log_hop_weights = False
    config.log_attention_scores = "auto"
    config.log_attention_statistics = "auto"

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
    config.glant_version = "v1"
    config.conv_type = "hop_gated_gatv2"

    config.num_layers = 2
    config.hidden_channels = 64
    config.heads = 8
    config.concat = False

    config.max_hops = 4
    config.alpha = 1.0

    config.dropout = 0.7
    config.attn_dropout = 0.7
    config.gate_hidden = None
    config.gate_dropout = 0.3

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
    config.num_samples = None
    config.num_edges = 10000

    config.log_hop_diagnostics = True
    config.log_hop_weights = True
    config.log_attention_scores = "auto"
    config.log_attention_statistics = "auto"
    config.hop_log_every = 50
    config.hop_log_only_layer = None

    config.walk = ConfigDict()
    config.walk.gamma = 0.9
    config.walk.jump_prob = 0.05
    config.walk.use_cosine = True

    config.training = training_config(
        lr=0.01,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="plateau",
    )

    return config


def glant_v2_config() -> ConfigDict:
    config = glant_config()

    config.architecture = "lambda interpolation between 1-hop and higher-order hops"
    config.glant_version = "v2"
    config.conv_type = "lambda_hop_gated_gatv2"

    # Main GLANT-v2 parameter:
    # out = (1 - lambda_higher) * H_1 + lambda_higher * H_higher
    config.lambda_higher = 0.5
    config.learn_lambda_higher = True
    config.lambda_init_epsilon = 1e-3

    # Keep the same basic setup as GLANT-v1 for fair comparison.
    config.max_hops = 3
    config.alpha = 0.05
    config.sparsify_hops = True

    config.log_hop_diagnostics = True
    config.log_hop_weights = True
    config.log_attention_scores = "auto"
    config.log_attention_statistics = "auto"

    return config


def glant_v3_config() -> ConfigDict:
    config = glant_config()

    config.architecture = "mixture of all hops with shared left projections"
    config.glant_version = "v3"
    config.conv_type = "glant_v3"

    return config


def glant_v4_config() -> ConfigDict:
    config = glant_config()

    config.architecture = "mixture of all hops with shared right projections"
    config.glant_version = "v4"
    config.conv_type = "glant_v4"

    return config


def glant_v5_config() -> ConfigDict:
    config = glant_config()

    config.architecture = "mixture of independent hop GATv2 convolutions"
    config.glant_version = "v5"
    config.conv_type = "glant_v5"

    return config


def glant_v6_config() -> ConfigDict:
    config = glant_config()

    config.architecture = "one-hop plus sigmoid-gated higher hops with shared left projections"
    config.glant_version = "v6"
    config.conv_type = "glant_v6"

    config.num_layers = 2
    config.hidden_channels = 64
    config.heads = 4
    config.concat = False

    config.max_hops = 3
    config.alpha = 0.4
    config.sparsify_hops = True
    config.sparsifier_cache_masks = True

    config.sampling_method = "balanced_unique_select"
    config.num_samples = None
    config.num_edges = 1000

    config.dropout = 0.5
    config.attn_dropout = 0.5
    config.gate_hidden = None
    config.gate_dropout = 0.2

    config.pre_linear = False
    config.residual = False
    config.norm = "none"
    config.act = "elu"

    config.training = training_config(
        lr=0.005,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="plateau",
    )

    return config


def glant_v6p1_config() -> ConfigDict:
    config = glant_v6_config()

    config.architecture = (
        "learnable bounded hop scalars with shared left projections"
    )
    config.glant_version = "v6p1"
    config.conv_type = "glant_v6p1"

    config.num_layers = 2
    config.hidden_channels = 64
    config.heads = 4
    config.concat = False

    config.max_hops = 3
    config.alpha = 0.5
    config.sparsify_hops = True
    config.sparsifier_cache_masks = True

    config.sampling_method = "balanced_unique_select"
    config.num_samples = None
    config.num_edges = 2500

    config.dropout = 0.4
    config.attn_dropout = 0.4
    config.gate_hidden = None
    config.gate_dropout = 0.0

    config.pre_linear = False
    config.residual = False
    config.norm = "none"
    config.act = "elu"

    config.hop0_scalar_init = 0.35
    config.higher_hop_scalar_init = 0.2

    config.training = training_config(
        lr=0.005,
        weight_decay=1e-3,
        num_epochs=300,
        scheduler_name="plateau",
    )

    return config


def glant_v7_config() -> ConfigDict:
    config = glant_config()

    config.architecture = (
        "MixHop-style k-hop GATv2 attention bank with root branch and late classifier"
    )
    config.glant_version = "v7"
    config.conv_type = "glant_v7"

    # GLANT-v7 keeps sampled k-hop GATv2 branches separate and concatenates
    # them before classification. A persistent input skip preserves the raw
    # feature signal after stacked hop-attention banks.
    config.num_layers = 2
    config.hidden_channels = 32
    config.heads = 8
    config.concat = False

    config.max_hops = 2
    config.alpha = 1.0
    config.sparsify_hops = True
    config.sparsifier_cache_masks = True

    config.sampling_method = "balanced_unique_select"
    config.num_samples = None
    config.num_edges = 15000
    config.sample_pool_edges = 40000

    config.dropout = 0.7
    config.attn_dropout = 0.45

    config.include_root = True
    config.hop_mode = "edge_hop"
    config.v7_num_banks = 2
    config.branch_norm = "none"
    config.root_scalar_init = 0.9
    config.hop_scalar_init = [0.95, 0.5]
    config.v7_gate_mode = "node"
    config.v7_input_skip = False
    config.v7_input_skip_dim = 32
    config.classifier_bias = True

    config.pre_linear = False
    config.residual = False
    config.norm = "layernorm"
    config.act = "elu"

    config.negative_slope = 0.2
    config.add_self_loops = True
    config.bias = True
    config.edge_dim = None
    config.fill_value = "mean"

    config.log_hop_diagnostics = True
    config.log_hop_weights = True
    config.log_attention_scores = "auto"
    config.log_attention_statistics = "auto"
    config.hop_log_every = 50
    config.hop_log_only_layer = 0

    config.training = training_config(
        lr=0.002,
        weight_decay=2e-3,
        num_epochs=500,
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
    config.dropout = 0.5
    config.attn_dropout = 0.0
    config.act = "relu"

    config.training = training_config(
        lr=0.01,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    return config


def graphsage_config() -> ConfigDict:
    config = base_gnn_config()

    config.conv_type = "sage"
    config.num_layers = 2
    config.hidden_channels = 64
    config.dropout = 0.5
    config.attn_dropout = 0.0
    config.act = "relu"

    config.training = training_config(
        lr=0.01,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    return config


def mixhop_config() -> ConfigDict:
    config = base_gnn_config()

    config.conv_type = "mixhop"
    config.powers = [0, 1, 2]

    config.training = training_config(
        lr=0.01,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    return config


def tagconv_config() -> ConfigDict:
    config = base_gnn_config()

    config.conv_type = "tagconv"
    config.K = 3

    config.training = training_config(
        lr=0.01,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="none",
    )

    return config


def khop_model_1_config() -> ConfigDict:
    return mixhop_config()


def khop_model_2_config() -> ConfigDict:
    return tagconv_config()


def hoga_config() -> ConfigDict:
    config = base_gnn_config()

    config.conv_type = "hoga"
    config.model_name = "HoGA GAT"
    config.max_hops = 3
    config.alpha = 1.0
    config.K_hops = 3
    config.layer_type = "multi_hop"
    config.head_type = "gat"
    config.agg_func = "sum"
    config.beta_mul = 0.9
    config.heads = 8
    config.num_heads = [8, 1]
    config.num_heads_small = 1
    config.concat = False
    config.dropout = 0.6
    config.drop_out = 0.6
    config.attn_dropout = 0.6
    config.load_samples = False
    config.select_method = "sim_walk"
    config.sampling_method = "sim_walk"
    config.num_samples = 15
    config.num_edges = None
    config.sparsify_hops = False
    config.sparsifier_cache_masks = True

    config.walk = ConfigDict()
    config.walk.gamma = 0.9
    config.walk.jump_prob = 0.05
    config.walk.use_cosine = True

    config.training = training_config(
        lr=0.005,
        weight_decay=5e-4,
        num_epochs=300,
        scheduler_name="plateau",
    )

    return config
