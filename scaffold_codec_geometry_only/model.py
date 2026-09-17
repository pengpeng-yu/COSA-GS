import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffold_codec.config import ModelConfig
from scaffold_codec.model import CompressedGaussianModel as MainlineCompressedGaussianModel


class CompressedGaussianModel(MainlineCompressedGaussianModel):
    """Anchor attributes conditioned only on anchor geometry context."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.anchor_latent_entropy = nn.ModuleList()
        del self.anchor_latent_context
        del self.anchor_latent_fusion
        self.anchor_context_fusion = nn.Linear(
            cfg.anchor_context_channels, cfg.anchor_context_channels)
        self.anchor_latents = nn.Parameter(torch.empty(0, 0), requires_grad=False)

    @torch.no_grad()
    def load_scaffold_model(self, path):
        super().load_scaffold_model(path)
        self.anchor_latents = nn.Parameter(
            self.anchor.new_empty((self.anchor.shape[0], 0)),
            requires_grad=False,
        )

    def construct_anchor_context(self):
        context = F.gelu(self.anchor_context_fusion(
            self.construct_anchor_geometry_context()))
        return context, context.new_zeros(context.shape[0]), ()

    def decode_anchor_latent(self, context, _streams, _gaussian_coder):
        return context.new_empty((context.shape[0], 0))

    def fuse_anchor_latent(self, context, _latent):
        return F.gelu(self.anchor_context_fusion(context))
