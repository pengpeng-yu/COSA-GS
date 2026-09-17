import torch
import torch.nn as nn

from scaffold_codec.config import ModelConfig
from scaffold_codec.model import CompressedGaussianModel as MainlineCompressedGaussianModel


class GlobalPriorEntropy(nn.Sequential):
    def forward(self, context):
        return super().forward(context[:1]).expand(context.shape[0], -1)


class CompressedGaussianModel(MainlineCompressedGaussianModel):
    """Anchor attributes conditioned only on a scene-level global prior."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        del self.anchor_coord_encoder
        self.anchor_latent_entropy = nn.ModuleList()
        del self.anchor_latent_context
        del self.anchor_latent_fusion
        self.anchor_global_context = nn.Parameter(torch.zeros(
            1, cfg.anchor_context_channels))
        self.anchor_latents = nn.Parameter(torch.empty(0, 0), requires_grad=False)
        self.position_scaling_entropy = GlobalPriorEntropy(*self.position_scaling_entropy)
        self.feat_context_entropy = GlobalPriorEntropy(*self.feat_context_entropy)

    @torch.no_grad()
    def load_scaffold_model(self, path):
        super().load_scaffold_model(path)
        self.anchor_latents = nn.Parameter(
            self.anchor.new_empty((self.anchor.shape[0], 0)),
            requires_grad=False,
        )

    def construct_anchor_geometry_context(self):
        return self.anchor_global_context.expand(self.anchor.shape[0], -1)

    def construct_anchor_context(self):
        context = self.construct_anchor_geometry_context()
        return context, context.new_zeros(self.anchor.shape[0]), ()

    def decode_anchor_latent(self, _context, _streams, _gaussian_coder):
        return self.anchor.new_empty((self.anchor.shape[0], 0))

    def fuse_anchor_latent(self, context, _latent):
        return context
