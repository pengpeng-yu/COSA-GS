from dataclasses import dataclass, field

from utils.simple_config import SimpleConfig


@dataclass
class DataConfig(SimpleConfig):
    source_path: str = "datasets/mipnerf360/bicycle"
    scaffold_model: str = "outputs/scaffold15k_mipnerf360/bicycle/point_cloud/iteration_15000"
    model_path: str = "outputs/bicycle_scaffold_codec"
    images: str = "images"
    resolution: int = -1
    data_device: str = "cpu"
    eval: bool = True
    lod: int = 0
    white_background: bool = False


@dataclass
class ModelConfig(SimpleConfig):
    anchor_feat_channels: int = 32
    n_offsets: int = 10
    latent_context_channels: int = 24
    anchor_context_channels: int = 24
    expand_offset_entropy_hidden: bool = True
    offset_coding_scale_init: float = 25.0
    position_scaling_coding_scale_init: float = 32.0
    gaussian_scaling_coding_scale_init: float = 32.0
    feat_coding_scale_init: float = 4.0
    physical_voxel_size: float = 0.001
    anchor_grid_size: float = 0.002
    anchor_coord_codec: str = "octree"
    octree_base_node_threshold: int = 512
    rans_block_count: int = 4
    anchor_latent_channels: int = 4
    anchor_latent_coding_scale_init: float = 4.0
    offset_mask_threshold: float = 0.4
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False


@dataclass
class TrainConfig(SimpleConfig):
    iterations: int = 45_000
    seed: int = 0
    lambda_dssim: float = 0.2
    lambda_rate: float = 0.001
    scaling_regularization: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    codec_network_beta2: float = 0.999
    codec_network_eps: float = 1.0e-8
    feat_lr_init: float = 0.0075
    feat_lr_final: float = 0.0075
    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0004
    scaling_lr_init: float = 0.007
    scaling_lr_final: float = 0.007
    codec_lr_init: float = 0.001
    codec_lr_final: float = 0.0004
    codec_lr_warmup_steps: int = 2_000
    anchor_latent_lr_init: float = 0.01
    anchor_latent_lr_final: float = 0.008
    offset_mask_lr_init: float = 0.0025
    offset_mask_lr_final: float = 0.0005
    coding_scale_lr_init: float = 0.01
    coding_scale_lr_final: float = 0.001
    coding_scale_grad_clip: float = 0.01
    codec_grad_clip: float = 0.1
    mlp_opacity_lr_init: float = 0.0002
    mlp_opacity_lr_final: float = 0.00002
    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_color_lr_init: float = 0.00065
    mlp_color_lr_final: float = 0.00005
    rd_start_iteration: int = 0
    position_scaling_ste_start: int = 10_000
    anchor_latent_rate_warmup_steps: int = 2_000
    geom_attr_rate_warmup_steps: int = 3_000
    feat_rate_warmup_steps: int = 2_000
    log_interval: int = 500
    eval_interval: int = 5_000
    checkpoint_interval: int = 0
    periodic_test_views: int | None = None
    offset_mask_prune_iterations: list[int] = field(default_factory=lambda: [35_000])
    preserve_optimizer_after_mask_pruning: list[int] = field(default_factory=lambda: [1])
    resume: str | None = None
    stop_iteration: int | None = None


@dataclass
class ExperimentConfig(SimpleConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path, overrides=()):
    cfg = ExperimentConfig()
    cfg.merge_with_yaml(path)
    cfg.merge_with_dotlist(overrides)
    cfg.check()
    return cfg
