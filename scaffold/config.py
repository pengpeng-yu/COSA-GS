from dataclasses import dataclass, field

from utils.simple_config import SimpleConfig


@dataclass
class DataConfig(SimpleConfig):
    root: str = "datasets/mipnerf360"
    scene: str = "bicycle"
    output_root: str = "outputs/scaffold15k_mipnerf360"
    model_path: str | None = None
    images: str = "images"
    resolution: int = -1
    data_device: str = "cuda"
    eval: bool = True
    lod: int = 0
    white_background: bool = False


@dataclass
class ModelConfig(SimpleConfig):
    anchor_feat_channels: int = 32
    n_offsets: int = 10
    physical_voxel_size: float = 0.001
    anchor_grid_size: float = 0.002
    grow_physical_size_factors: list[int] = field(default_factory=lambda: [16, 4, 1])
    grow_grid_size_factors: list[int] = field(default_factory=lambda: [1, 1, 1])
    grow_threshold_factors: list[int] = field(default_factory=lambda: [1, 2, 4])
    use_feat_bank: bool = False
    appearance_dim: int = 0
    point_ratio: int = 1
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False


@dataclass
class TrainConfig(SimpleConfig):
    iterations: int = 15_000
    seed: int = 0
    lambda_dssim: float = 0.2
    scaling_regularization: float = 0.01
    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.001
    feat_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002
    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.0002
    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00065
    appearance_lr_init: float = 0.05
    appearance_lr_final: float = 0.005
    percent_dense: float = 0.01
    start_stat: int = 500
    update_from: int = 1_500
    update_interval: int = 100
    update_until: int = 15_000
    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002
    log_interval: int = 500
    test_iterations: list[int] = field(default_factory=lambda: [15_000])


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
