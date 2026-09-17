import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffold_codec.config import ModelConfig
from scaffold_codec.model import CompressedGaussianModel as MainlineCompressedGaussianModel


class CompressedGaussianModel(MainlineCompressedGaussianModel):
    """Anchor attributes conditioned only on a coded anchor latent."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        del self.anchor_coord_encoder
        self.anchor_global_context = nn.Parameter(torch.zeros(
            1, cfg.anchor_context_channels))
        self.anchor_latent_fusion = nn.Linear(
            cfg.latent_context_channels, cfg.anchor_context_channels)

    def construct_anchor_geometry_context(self):
        return None

    def predict_anchor_latent_channel(self, _context, decoded_channels, stage):
        count = self.anchor.shape[0]
        if stage == 0:
            context = self.anchor_global_context
        else:
            context = torch.cat((
                self.anchor_global_context.expand(count, -1),
                *decoded_channels,
            ), 1)
        mean, scale_index = self.anchor_latent_entropy[stage](context).chunk(2, 1)
        if stage == 0:
            mean = mean.expand(count, -1)
            scale_index = scale_index.expand(count, -1)
        return mean, scale_index

    def decode_anchor_latent(self, context, streams, gaussian_coder):
        decoded_channels = []
        coding_scale = self.log_anchor_latent_coding_scale.exp() + 1e-8
        for stage, stream in enumerate(streams):
            mean, scale_index = self.predict_anchor_latent_channel(
                context, decoded_channels, stage)
            entropy_start = time.perf_counter()
            decoded_channels.append(gaussian_coder.decode(
                stream, mean, scale_index, coding_scale, mean.shape))
            self.codec_times["entropy_decode_seconds"] = (
                self.codec_times.get("entropy_decode_seconds", 0.0)
                + time.perf_counter() - entropy_start)
        return torch.cat(decoded_channels, 1)

    def fuse_anchor_latent(self, _context, latent):
        latent_context = self.anchor_latent_context(latent)
        return F.gelu(self.anchor_latent_fusion(latent_context))
