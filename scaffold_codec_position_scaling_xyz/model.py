import numpy as np
import torch
import torch.nn as nn

from scaffold_codec.config import ModelConfig
from scaffold_codec.model import CompressedGaussianModel as MainlineCompressedGaussianModel


class CompressedGaussianModel(MainlineCompressedGaussianModel):
    """Mainline codec retaining three-axis log position scaling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        channels = cfg.anchor_context_channels
        self.position_scaling_encoder = nn.Sequential(
            nn.Linear(3, channels),
            nn.GELU(),
        )
        self.position_scaling_entropy = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, 6),
        )
        self.log_position_scaling = nn.Parameter(torch.empty(0, 3))

    def prepare_scaffold_position_scaling(
        self,
        offset: np.ndarray,
        log_position_scaling_xyz: np.ndarray,
    ):
        return offset, log_position_scaling_xyz
