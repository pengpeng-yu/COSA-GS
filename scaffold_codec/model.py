import io
import math
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement

from utils.general_utils import get_expon_lr_func

from . import octree
from .config import ModelConfig, TrainConfig
from .entropy import (
    IndexedGaussianCoder,
    RansDecoder,
    RansEncoder,
    gaussian_log_prob,
    gaussian_scale_index_for_scale,
    indexed_gaussian_scale,
    quantize_categorical,
)
from .gpcc import gpcc_decode, gpcc_encode, write_xyz_ply
from .space_filling_curves import morton_encode_magicbits


@dataclass(frozen=True)
class RateSummary:
    """Encoded coordinate size and predicted rates for anchor latents and attributes."""

    geom_bits: torch.Tensor
    anchor_latent_bits: torch.Tensor
    offset_mask_bits: torch.Tensor
    offset_bits: torch.Tensor
    position_scaling_bits: torch.Tensor
    gaussian_scaling_bits: torch.Tensor
    feat_bits: torch.Tensor


class CompressedGaussianModel(nn.Module):

    gaussian_coder_class = IndexedGaussianCoder

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        torch.set_float32_matmul_precision("highest")
        self.cfg = cfg
        self.physical_voxel_size = cfg.physical_voxel_size
        self.anchor_grid_size = cfg.anchor_grid_size
        self.distance_scale = 1.0
        assert cfg.latent_context_channels > 0
        assert cfg.anchor_context_channels > 0
        assert cfg.anchor_latent_channels > 0
        assert cfg.anchor_coord_codec in ("gpcc", "octree")
        assert cfg.octree_base_node_threshold > 1
        assert cfg.rans_block_count > 0
        assert 0.0 < cfg.offset_mask_threshold < 1.0

        self.anchor_latent_entropy = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.anchor_context_channels + stage, cfg.anchor_context_channels),
                nn.GELU(),
                nn.Linear(cfg.anchor_context_channels, 2),
            )
            for stage in range(cfg.anchor_latent_channels)
        ])
        self.anchor_latent_context = nn.Sequential(
            nn.Linear(cfg.anchor_latent_channels, cfg.latent_context_channels),
            nn.GELU(),
        )
        self.anchor_latent_fusion = nn.Sequential(
            nn.Linear(
                cfg.anchor_context_channels + cfg.latent_context_channels,
                cfg.anchor_context_channels,
            ),
            nn.GELU(),
        )

        self.anchor_coord_encoder = nn.Sequential(
            nn.Linear(3, cfg.anchor_context_channels),
            nn.GELU(),
            nn.Linear(cfg.anchor_context_channels, cfg.anchor_context_channels),
            nn.GELU(),
        )
        self.position_scaling_encoder = nn.Sequential(
            nn.Linear(1, cfg.anchor_context_channels),
            nn.GELU(),
        )
        self.offset_prior_fusion = nn.Sequential(
            nn.Linear(cfg.anchor_context_channels, cfg.anchor_context_channels),
            nn.GELU(),
            nn.Linear(cfg.anchor_context_channels, cfg.anchor_context_channels),
        )
        self.gaussian_scaling_prior_fusion = nn.Sequential(
            nn.Linear(cfg.anchor_context_channels, cfg.anchor_context_channels),
            nn.GELU(),
            nn.Linear(cfg.anchor_context_channels, cfg.anchor_context_channels),
        )
        offset_entropy_output_channels = 6 * cfg.n_offsets
        offset_entropy_hidden_channels = (
            max(cfg.anchor_context_channels, offset_entropy_output_channels)
            if cfg.expand_offset_entropy_hidden
            else cfg.anchor_context_channels
        )
        self.offset_entropy = nn.Sequential(
            nn.Linear(cfg.anchor_context_channels, offset_entropy_hidden_channels),
            nn.GELU(),
            nn.Linear(offset_entropy_hidden_channels, offset_entropy_output_channels),
        )
        self.position_scaling_entropy = nn.Sequential(
            nn.Linear(cfg.anchor_context_channels, cfg.anchor_context_channels),
            nn.GELU(),
            nn.Linear(cfg.anchor_context_channels, 2),
        )
        self.gaussian_scaling_entropy = nn.Sequential(
            nn.Linear(cfg.anchor_context_channels, cfg.anchor_context_channels),
            nn.GELU(),
            nn.Linear(cfg.anchor_context_channels, 6),
        )

        self.log_offset_coding_scale = nn.Parameter(
            torch.tensor(math.log(cfg.offset_coding_scale_init)))
        self.log_position_scaling_coding_scale = nn.Parameter(
            torch.tensor(math.log(cfg.position_scaling_coding_scale_init)))
        self.log_gaussian_scaling_coding_scale = nn.Parameter(
            torch.tensor(math.log(cfg.gaussian_scaling_coding_scale_init)))
        self.log_feat_coding_scale = nn.Parameter(
            torch.tensor(math.log(cfg.feat_coding_scale_init)))
        self.log_anchor_latent_coding_scale = nn.Parameter(
            torch.tensor(math.log(cfg.anchor_latent_coding_scale_init)))

        feat_entropy_hidden_channels = 2 * max(
            cfg.anchor_feat_channels, cfg.anchor_context_channels)
        self.feat_context_entropy = nn.Sequential(
            nn.Linear(cfg.anchor_context_channels, feat_entropy_hidden_channels),
            nn.GELU(),
            nn.Linear(feat_entropy_hidden_channels, 2 * cfg.anchor_feat_channels),
        )

        opacity_channels = cfg.anchor_feat_channels + 3 + int(cfg.add_opacity_dist)
        cov_channels = cfg.anchor_feat_channels + 3 + int(cfg.add_cov_dist)
        color_channels = cfg.anchor_feat_channels + 3 + int(cfg.add_color_dist)
        self.mlp_opacity = self.make_scaffold_mlp(
            opacity_channels, cfg.n_offsets, nn.Tanh())
        self.mlp_cov = self.make_scaffold_mlp(
            cov_channels, 7 * cfg.n_offsets)
        self.mlp_color = self.make_scaffold_mlp(
            color_channels, 3 * cfg.n_offsets, nn.Sigmoid())

        self.anchor = nn.Parameter(torch.empty(0, 3), requires_grad=False)
        self.anchor_feat = nn.Parameter(torch.empty(0, cfg.anchor_feat_channels))
        self.offset = nn.Parameter(torch.empty(0, cfg.n_offsets, 3))
        self.offset_mask_logits = nn.Parameter(torch.empty(0, cfg.n_offsets))
        self.register_buffer("hard_offset_mask", None)
        self.register_buffer(
            "anchor_coords", torch.empty(0, 3, dtype=torch.int32),
            persistent=False)
        self.anchor_coord_bitstream = b""
        self.codec_times = {}
        self.train_seconds = 0.0
        self.log_position_scaling = nn.Parameter(torch.empty(0, 1))
        self.log_gaussian_scaling = nn.Parameter(torch.empty(0, 3))
        self.anchor_latents = nn.Parameter(torch.empty(0, cfg.anchor_latent_channels))
        self.register_buffer("rotation", torch.empty(0, 4), persistent=False)
        self.register_buffer("opacity", torch.empty(0, 1), persistent=False)

        self.register_buffer(
            "coord_shape", torch.empty(0, dtype=torch.int32), persistent=False)

        self.origin: torch.Tensor | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_schedules = {}
        self.codec_parameters: list[nn.Parameter] = []

    # Scaffold rendering state

    def make_scaffold_mlp(
        self, input_channels: int, output_channels: int,
        output: nn.Module | None = None,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(input_channels, self.cfg.anchor_feat_channels),
            nn.ReLU(True),
            nn.Linear(self.cfg.anchor_feat_channels, output_channels),
        ]
        if output is not None:
            layers.append(output)
        return nn.Sequential(*layers)

    def initialize_entropy_heads(self):
        with torch.no_grad():
            for head in self.anchor_latent_entropy:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                head[-1].bias[1] = gaussian_scale_index_for_scale(1.0)

            position_scaling_head = self.position_scaling_entropy[-1]
            position_scaling_channels = position_scaling_head.out_features // 2
            nn.init.zeros_(position_scaling_head.weight)
            position_scaling_head.bias[:position_scaling_channels].fill_(
                math.log(self.physical_voxel_size * 4.0))
            position_scaling_head.bias[position_scaling_channels:].fill_(
                gaussian_scale_index_for_scale(16.0))

            offset_head = self.offset_entropy[-1]
            offset_channels = offset_head.out_features // 2
            nn.init.zeros_(offset_head.weight)
            offset_head.bias[:offset_channels].zero_()
            offset_head.bias[offset_channels:].fill_(
                gaussian_scale_index_for_scale(16.0))

            gaussian_scaling_head = self.gaussian_scaling_entropy[-1]
            gaussian_scaling_channels = gaussian_scaling_head.out_features // 2
            base_log_scaling = math.log(self.physical_voxel_size * 4.0)
            nn.init.zeros_(gaussian_scaling_head.weight)
            gaussian_scaling_head.bias[:gaussian_scaling_channels].fill_(base_log_scaling)
            gaussian_scaling_head.bias[gaussian_scaling_channels:].fill_(
                gaussian_scale_index_for_scale(16.0))

            feat_head = self.feat_context_entropy[-1]
            feat_channels = feat_head.out_features // 2
            nn.init.zeros_(feat_head.weight)
            feat_head.bias[:feat_channels].zero_()
            feat_head.bias[feat_channels:].fill_(
                gaussian_scale_index_for_scale(1.0))

    def prepare_scaffold_position_scaling(
        self,
        offset: np.ndarray,
        log_position_scaling_xyz: np.ndarray,
    ):
        log_position_scaling = log_position_scaling_xyz.max(1, keepdims=True)
        offset *= np.exp(
            log_position_scaling_xyz[:, None]
            - log_position_scaling[:, None]
        )
        return offset, log_position_scaling

    @torch.no_grad()
    def load_scaffold_model(self, path: str | Path):
        path = Path(path)
        if self.cfg.physical_voxel_size == 0:
            assert (path / "extra_params.pt").is_file(), path / "extra_params.pt"
            extra_params = torch.load(path / "extra_params.pt", map_location="cpu", weights_only=True)
            self.physical_voxel_size = extra_params["physical_voxel_size"]
            self.anchor_grid_size = extra_params["anchor_grid_size"]
        else:
            self.physical_voxel_size = self.cfg.physical_voxel_size
            self.anchor_grid_size = self.cfg.anchor_grid_size
        vertex = PlyData.read(path / "point_cloud.ply")["vertex"]
        properties = [prop.name for prop in vertex.properties]

        def columns(prefix):
            names = sorted(
                (name for name in properties if name.startswith(prefix)),
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )
            return np.column_stack([vertex[name] for name in names]).astype(
                np.float32, copy=False)

        anchor = np.column_stack((
            vertex["x"], vertex["y"], vertex["z"],
        )).astype(np.float32, copy=False)
        offset = columns("f_offset_")
        feat = columns("f_anchor_feat_")
        log_scaling = columns("scale_")

        expected_offsets = 3 * self.cfg.n_offsets
        if offset.shape[1] != expected_offsets:
            raise ValueError(
                f"Scaffold model has {offset.shape[1] // 3} offsets, "
                f"expected {self.cfg.n_offsets}.")
        if feat.shape[1] != self.cfg.anchor_feat_channels:
            raise ValueError(
                f"Scaffold model has {feat.shape[1]} feat channels, "
                f"expected {self.cfg.anchor_feat_channels}.")
        if log_scaling.shape[1] != 6:
            raise ValueError("Scaffold model must contain six scaling values.")

        offset = offset.reshape(-1, 3, self.cfg.n_offsets).transpose(0, 2, 1)
        offset, log_position_scaling = self.prepare_scaffold_position_scaling(
            offset, log_scaling[:, :3]
        )

        device = self.anchor.device
        anchor = torch.from_numpy(anchor).to(device)
        feat = torch.from_numpy(feat).to(device)
        offset = torch.from_numpy(offset.copy()).to(device)
        log_position_scaling = torch.from_numpy(log_position_scaling).to(device)
        log_gaussian_scaling = torch.from_numpy(log_scaling[:, 3:].copy()).to(device)

        for name, module in (
            ("opacity", self.mlp_opacity),
            ("cov", self.mlp_cov),
            ("color", self.mlp_color),
        ):
            source = torch.jit.load(str(path / f"{name}_mlp.pt"), map_location=device)
            module.load_state_dict(source.state_dict())

        anchor_coords = torch.round(
            anchor / self.anchor_grid_size).to(torch.int32)
        origin = anchor_coords.amin(0)
        anchor_coords.sub_(origin)
        order = torch.argsort(morton_encode_magicbits(anchor_coords, inverse=True))

        self.set_anchor(anchor[order])
        self.anchor_coords = anchor_coords[order].contiguous()
        self.origin = origin
        self.coord_shape = anchor_coords.amax(0) + 1
        self.anchor_feat = nn.Parameter(feat[order].contiguous())
        self.offset = nn.Parameter(offset[order].contiguous())
        self.offset_mask_logits = nn.Parameter(
            torch.ones(anchor.shape[0], self.cfg.n_offsets, device=device))
        self.log_position_scaling = nn.Parameter(
            log_position_scaling[order].contiguous())
        self.log_gaussian_scaling = nn.Parameter(
            log_gaussian_scaling[order].contiguous())

        self.anchor_coord_bitstream = self.encode_anchor_coord()
        self.initialize_anchor_buffers()
        self.anchor_latents = nn.Parameter(torch.zeros(
            self.anchor.shape[0], self.cfg.anchor_latent_channels,
            device=device,
        ))
        self.initialize_entropy_heads()

    def encode_anchor_coord(self) -> bytes:
        if self.cfg.anchor_coord_codec == "octree":
            return octree.encode_anchor_coord(self.anchor_coords, self.cfg.octree_base_node_threshold)
        tmc3 = Path(__file__).resolve().parents[1] / "bin" / "tmc3"
        with tempfile.TemporaryDirectory() as directory:
            ply_path = Path(directory) / "anchor.ply"
            stream_path = Path(directory) / "anchor.bin"
            write_xyz_ply(self.anchor_coords, ply_path)
            gpcc_encode(ply_path, stream_path, tmc3)
            return stream_path.read_bytes()

    def decode_anchor_coord(self, stream: bytes) -> torch.Tensor:
        if self.cfg.anchor_coord_codec == "octree":
            return octree.decode_anchor_coord(stream, self.anchor.device)
        tmc3 = Path(__file__).resolve().parents[1] / "bin" / "tmc3"
        with tempfile.TemporaryDirectory() as directory:
            stream_path = Path(directory) / "anchor.bin"
            ply_path = Path(directory) / "anchor.ply"
            stream_path.write_bytes(stream)
            gpcc_decode(stream_path, ply_path, tmc3)
            vertex = PlyData.read(ply_path)["vertex"]
            coords = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
        coords = torch.from_numpy(np.rint(coords).astype(np.int32)).to(self.anchor.device)
        order = torch.argsort(morton_encode_magicbits(coords, inverse=True))
        return coords[order].contiguous()

    def set_anchor(self, anchor: torch.Tensor):
        if anchor.ndim != 2 or anchor.shape[0] == 0 or anchor.shape[1] != 3:
            raise ValueError(f"Anchors must be nonempty [N, 3], got {anchor.shape}.")
        self.anchor = nn.Parameter(anchor.detach().contiguous(), requires_grad=False)

    def initialize_anchor_buffers(self):
        count = self.anchor.shape[0]
        self.rotation = torch.zeros(count, 4, device=self.anchor.device)
        self.rotation[:, 0] = 1.0
        self.opacity = torch.logit(
            torch.full((count, 1), 0.1, device=self.anchor.device))

    def offset_mask(self) -> torch.Tensor:
        if self.hard_offset_mask is not None:
            return self.hard_offset_mask
        prob = self.offset_mask_logits.sigmoid()
        hard = prob > self.cfg.offset_mask_threshold
        return prob + (hard.to(prob.dtype) - prob).detach()

    def quantize_position_scaling(
        self, log_scaling: torch.Tensor, mean: torch.Tensor,
        rd_iteration: int | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coding_scale = self.log_position_scaling_coding_scale.exp() + 1e-8
        residual_symbols = (log_scaling - mean) * coding_scale
        if rd_iteration is None:
            residual_symbols = residual_symbols.round()
        elif rd_iteration >= self.train_cfg.position_scaling_ste_start:
            residual_symbols = residual_symbols + (
                residual_symbols.round() - residual_symbols).detach()
        elif rd_iteration > 0:
            residual_symbols = residual_symbols + noise
        return residual_symbols, mean + residual_symbols / coding_scale

    def raw_attrs(self):
        return (
            self.anchor_feat,
            self.offset,
            self.log_position_scaling,
            self.log_gaussian_scaling,
        )

    def prefilter_scaling(self, attrs):
        _, _, log_position_scaling, _ = attrs
        return log_position_scaling.exp().expand(-1, 3)

    def training_attrs(
        self,
        rd_iteration: int,
        attr_quantization,
    ):
        if rd_iteration == 0:
            return self.raw_attrs()

        (
            feat_noise,
            offset_noise,
            log_position_scaling_mean,
            position_scaling_noise,
            log_gaussian_scaling_noise,
        ) = attr_quantization
        _, log_position_scaling = self.quantize_position_scaling(
            self.log_position_scaling,
            log_position_scaling_mean,
            rd_iteration,
            position_scaling_noise,
        )

        return (
            self.anchor_feat + feat_noise / (self.log_feat_coding_scale.exp() + 1e-8),
            self.offset + offset_noise / (self.log_offset_coding_scale.exp() + 1e-8),
            log_position_scaling,
            self.log_gaussian_scaling
            + log_gaussian_scaling_noise
            / (self.log_gaussian_scaling_coding_scale.exp() + 1e-8),
        )

    def predict_anchor_latent_channel(self, context, decoded_channels, stage):
        if stage != 0:
            context = torch.cat((context, *decoded_channels), 1)
        mean, scale_index = self.anchor_latent_entropy[stage](context).chunk(2, 1)
        return mean, scale_index

    def encode_anchor_latent(self, distributions, gaussian_coder):
        latent = self.anchor_latents
        coding_scale = self.log_anchor_latent_coding_scale.exp() + 1e-8
        return tuple(
            gaussian_coder.encode(
                latent[:, stage:stage + 1], mean, scale_index, coding_scale)
            for stage, (mean, scale_index) in enumerate(distributions)
        )

    def decode_anchor_latent(self, context, streams, gaussian_coder):
        decoded_channels = []
        coding_scale = self.log_anchor_latent_coding_scale.exp() + 1e-8
        for stage, stream in enumerate(streams):
            mean, scale_index = self.predict_anchor_latent_channel(
                context, decoded_channels, stage)
            entropy_start = time.perf_counter()
            decoded_channels.append(gaussian_coder.decode(
                stream, mean, scale_index, coding_scale,
                torch.Size((context.shape[0], 1))))
            self.codec_times["entropy_decode_seconds"] = (
                self.codec_times.get("entropy_decode_seconds", 0.0)
                + time.perf_counter() - entropy_start)
        return torch.cat(decoded_channels, 1)

    def fuse_anchor_latent(self, context, latent):
        latent_context = self.anchor_latent_context(latent)
        return self.anchor_latent_fusion(
            torch.cat((context, latent_context), 1))

    def quantize_anchor_latent(self, context):
        latent = self.anchor_latents
        coding_scale = self.log_anchor_latent_coding_scale.exp() + 1e-8
        decoded_channels = []
        distributions = []
        bits = latent.new_zeros(latent.shape[0])

        for stage in range(self.cfg.anchor_latent_channels):
            mean, scale_index = self.predict_anchor_latent_channel(
                context, decoded_channels, stage)
            latent_channel = latent[:, stage:stage + 1]
            if self.training:
                noise = torch.rand_like(latent_channel) - 0.5
                latent_channel_hat = latent_channel + noise / coding_scale
                symbols = (latent_channel - mean) * coding_scale + noise
                bits = bits - (
                    gaussian_log_prob(
                        symbols,
                        indexed_gaussian_scale(scale_index),
                    )[:, 0] / math.log(2.0)
                )
            else:
                latent_channel_hat = mean + torch.round(
                    (latent_channel - mean) * coding_scale) / coding_scale

            decoded_channels.append(latent_channel_hat)
            distributions.append((mean, scale_index))

        latent_hat = torch.cat(decoded_channels, 1)
        context = self.fuse_anchor_latent(context, latent_hat)
        return context, bits, tuple(distributions)

    def construct_anchor_geometry_context(self):
        return self.anchor_coord_encoder(self.normalize_coord(self.anchor))

    def construct_anchor_context(self):
        context = self.construct_anchor_geometry_context()
        return self.quantize_anchor_latent(context)

    def normalize_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return 2.0 * (coord / self.anchor_grid_size - self.origin) / (self.coord_shape - 1) - 1.0

    def predict_offset_and_log_gaussian_scaling(
        self,
        context: torch.Tensor,
        log_position_scaling_hat: torch.Tensor,
    ):
        count = context.shape[0]
        prior = self.position_scaling_encoder(log_position_scaling_hat)
        offset_context = context + self.offset_prior_fusion(prior)
        gaussian_scaling_context = context + self.gaussian_scaling_prior_fusion(prior)

        offset_mean, offset_scale_index = self.offset_entropy(offset_context).chunk(2, 1)
        offset_mean = offset_mean.reshape(count, self.cfg.n_offsets, 3)
        offset_scale_index = offset_scale_index.reshape(count, self.cfg.n_offsets, 3)

        log_gaussian_scaling_mean, gaussian_scaling_scale_index = \
            self.gaussian_scaling_entropy(gaussian_scaling_context).chunk(2, 1)

        return (
            offset_mean, offset_scale_index,
            log_gaussian_scaling_mean, gaussian_scaling_scale_index,
        )

    def compute_attr_bits(
        self,
        position_scaling_residual_symbols: torch.Tensor,
        position_scaling_scale_index: torch.Tensor,
        offset_mean: torch.Tensor,
        offset_scale_index: torch.Tensor,
        log_gaussian_scaling_mean: torch.Tensor,
        gaussian_scaling_scale_index: torch.Tensor,
        feat_mean: torch.Tensor,
        feat_scale_index: torch.Tensor,
        offset_mask: torch.Tensor,
        anchor_mask: torch.Tensor | float,
        offset_coding_scale: torch.Tensor,
        gaussian_scaling_coding_scale: torch.Tensor,
        feat_noise: torch.Tensor | None,
        offset_noise: torch.Tensor | None,
        log_gaussian_scaling_noise: torch.Tensor | None,
    ):
        feat = self.anchor_feat
        offset = self.offset
        log_gaussian_scaling = self.log_gaussian_scaling

        log2 = math.log(2.0)

        if self.hard_offset_mask is None:
            offset_symbols = (offset - offset_mean) * offset_coding_scale
            if self.training:
                offset_symbols = offset_symbols + offset_noise
            else:
                offset_symbols = offset_symbols.round()
                offset_scale_index = offset_scale_index.round()

            offset_bits = -gaussian_log_prob(
                offset_symbols, indexed_gaussian_scale(offset_scale_index)) / log2
            offset_bits = (offset_bits * offset_mask[:, :, None]).sum()
        else:
            active_offset_mask = offset_mask[:, :, None].expand_as(offset)
            offset_symbols = (
                offset[active_offset_mask] - offset_mean[active_offset_mask]
            ) * offset_coding_scale
            active_scale_index = offset_scale_index[active_offset_mask]
            if self.training:
                offset_symbols = offset_symbols + offset_noise[active_offset_mask]
            else:
                offset_symbols = offset_symbols.round()
                active_scale_index = active_scale_index.round()

            offset_bits = -gaussian_log_prob(
                offset_symbols,
                indexed_gaussian_scale(active_scale_index),
            ).sum() / log2

        gaussian_scaling_symbols = (
            log_gaussian_scaling - log_gaussian_scaling_mean
        ) * gaussian_scaling_coding_scale
        if self.training:
            gaussian_scaling_symbols = (
                gaussian_scaling_symbols + log_gaussian_scaling_noise)
        else:
            gaussian_scaling_symbols = gaussian_scaling_symbols.round()
            gaussian_scaling_scale_index = gaussian_scaling_scale_index.round()

        gaussian_scaling_bits = -gaussian_log_prob(
            gaussian_scaling_symbols,
            indexed_gaussian_scale(gaussian_scaling_scale_index)) / log2
        gaussian_scaling_bits = (gaussian_scaling_bits.sum(1) * anchor_mask).sum()

        feat_coding_scale = self.log_feat_coding_scale.exp() + 1e-8
        if self.training:
            feat_symbols = feat_noise + (feat - feat_mean) * feat_coding_scale
        else:
            feat_symbols = ((feat - feat_mean) * feat_coding_scale).round()
            feat_scale_index = feat_scale_index.round()

        feat_bits = -gaussian_log_prob(
            feat_symbols, indexed_gaussian_scale(feat_scale_index)) / log2
        feat_bits = (feat_bits.sum(1) * anchor_mask).sum()

        if not self.training:
            position_scaling_scale_index = position_scaling_scale_index.round()
        position_scaling_bits = -gaussian_log_prob(
            position_scaling_residual_symbols,
            indexed_gaussian_scale(position_scaling_scale_index),
        ).sum(1) / log2
        position_scaling_bits = (position_scaling_bits * anchor_mask).sum()

        return offset_bits, position_scaling_bits, gaussian_scaling_bits, feat_bits

    def forward(self, rd_iteration: int):
        context, anchor_latent_bits_per_anchor, _ = self.construct_anchor_context()

        log_position_scaling_mean, position_scaling_scale_index = (
            self.position_scaling_entropy(context).chunk(2, 1))
        position_scaling_noise = (
            torch.rand_like(self.log_position_scaling) - 0.5
            if self.training
            and 0 < rd_iteration < self.train_cfg.position_scaling_ste_start
            else None
        )
        position_scaling_residual_symbols, log_position_scaling_hat = self.quantize_position_scaling(
            self.log_position_scaling,
            log_position_scaling_mean,
            rd_iteration if self.training else None,
            position_scaling_noise,
        )
        (
            offset_mean,
            offset_scale_index,
            log_gaussian_scaling_mean,
            gaussian_scaling_scale_index,
        ) = self.predict_offset_and_log_gaussian_scaling(
            context, log_position_scaling_hat)

        offset_coding_scale = self.log_offset_coding_scale.exp() + 1e-8
        gaussian_scaling_coding_scale = self.log_gaussian_scaling_coding_scale.exp() + 1e-8
        if self.hard_offset_mask is None:
            mask_prob = self.offset_mask_logits.sigmoid()
            hard_offset_mask = mask_prob > self.cfg.offset_mask_threshold
            offset_mask = mask_prob + (
                hard_offset_mask.to(mask_prob.dtype) - mask_prob).detach()
            anchor_prob = 1.0 - (1.0 - mask_prob).prod(1)
            hard_anchor_mask = hard_offset_mask.any(1)
            anchor_mask = anchor_prob + (
                hard_anchor_mask.to(anchor_prob.dtype) - anchor_prob).detach()
        else:
            offset_mask = self.hard_offset_mask
            anchor_mask = 1.0

        anchor_latent_bits = (anchor_latent_bits_per_anchor * anchor_mask).sum()

        feat_mean, feat_scale_index = self.feat_context_entropy(context).chunk(2, 1)
        if self.training:
            feat_noise = torch.rand_like(self.anchor_feat) - 0.5
            offset_noise = torch.rand_like(self.offset) - 0.5
            log_gaussian_scaling_noise = (
                torch.rand_like(self.log_gaussian_scaling) - 0.5)
        else:
            feat_noise = None
            offset_noise = None
            log_gaussian_scaling_noise = None

        (
            offset_bits,
            position_scaling_bits,
            gaussian_scaling_bits,
            feat_bits,
        ) = self.compute_attr_bits(
            position_scaling_residual_symbols,
            position_scaling_scale_index,
            offset_mean,
            offset_scale_index,
            log_gaussian_scaling_mean,
            gaussian_scaling_scale_index,
            feat_mean,
            feat_scale_index,
            offset_mask,
            anchor_mask,
            offset_coding_scale,
            gaussian_scaling_coding_scale,
            feat_noise,
            offset_noise,
            log_gaussian_scaling_noise,
        )

        geom_bits = context.new_tensor(float(len(self.anchor_coord_bitstream) * 8))
        offset_mask_bits = context.new_zeros(())

        return RateSummary(
            geom_bits=geom_bits,
            anchor_latent_bits=anchor_latent_bits,
            offset_mask_bits=offset_mask_bits,
            offset_bits=offset_bits,
            position_scaling_bits=position_scaling_bits,
            gaussian_scaling_bits=gaussian_scaling_bits,
            feat_bits=feat_bits,
        ), (
            feat_noise,
            offset_noise,
            log_position_scaling_mean.detach(),
            position_scaling_noise,
            log_gaussian_scaling_noise,
        )

    # Entropy coding

    def coded_model_parameters(self):
        direct = {
            "anchor", "anchor_feat", "offset",
            "offset_mask_logits", "log_position_scaling",
            "log_gaussian_scaling", "anchor_latents",
        }
        for name, parameter in self.named_parameters():
            if name not in direct:
                yield name, parameter

    def coded_model_bits(self) -> int:
        return sum(
            parameter.numel() for _, parameter in self.coded_model_parameters()
        ) * 16

    @torch.no_grad()
    def coded_model_state(self) -> bytes:
        return torch.cat(tuple(
            parameter.reshape(-1) for _, parameter in self.coded_model_parameters()
        )).to(device="cpu", dtype=torch.float16).numpy().tobytes()

    @torch.no_grad()
    def load_coded_model_state(self, state: bytes):
        fp16_state = torch.tensor(
            np.frombuffer(state, dtype=np.float16),
            device=self.anchor.device, dtype=self.anchor.dtype)

        start = 0
        for _, parameter in self.coded_model_parameters():
            end = start + parameter.numel()
            parameter.copy_(fp16_state[start:end].reshape_as(parameter))
            start = end

    def encode_offset_mask(self, symbols, frequencies):
        cdf = frequencies.cumsum(1, dtype=torch.int32)[:, :-1].to(
            device="cpu", dtype=torch.uint16).contiguous()
        symbols = symbols.to(device="cpu", dtype=torch.int16).contiguous()
        encoder = RansEncoder()
        encoder.encode_categorical_shared(symbols.numpy(), cdf.numpy().reshape(-1))
        return encoder.flush(), cdf

    def decode_offset_mask(self, stream, cdf, count):
        decoder = RansDecoder()
        decoder.set_stream(stream)
        return decoder.decode_categorical_shared(cdf.reshape(-1), count)

    @contextmanager
    def entropy_model_timer(self, direction):
        timings = []
        is_cuda = self.anchor.is_cuda
        stream = torch.cuda.current_stream(self.anchor.device) if is_cuda else None

        def record_start(module, inputs):
            if is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                timings.append([start, end])
            else:
                timings.append([time.perf_counter(), None])

        def record_end(module, inputs, output):
            if is_cuda:
                timings[-1][1].record(stream)
            else:
                timings[-1][1] = time.perf_counter()

        # Time non-nested network blocks, including each latent channel head.
        hooks = []
        for child in self.children():
            for module in (child if isinstance(child, nn.ModuleList) else (child,)):
                hooks.append(module.register_forward_pre_hook(record_start))
                hooks.append(module.register_forward_hook(record_end))
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()

        # compress/decompress already synchronize before leaving this scope.
        if is_cuda:
            seconds = sum(start.elapsed_time(end) for start, end in timings) / 1000.0
        else:
            seconds = sum(end - start for start, end in timings)
        self.codec_times[f"entropy_model_{direction}_seconds"] = seconds

    @torch.no_grad()
    def compress(self) -> bytes:
        with self.entropy_model_timer("encode"):
            if self.anchor.is_cuda:
                torch.cuda.synchronize(self.anchor.device)
            compress_start = time.perf_counter()
            self.codec_times.update(compress_seconds=0.0, entropy_encode_seconds=0.0,
                                    anchor_coord_encode_seconds=0.0)
            self.eval()
            coord_start = time.perf_counter()
            self.anchor_coord_bitstream = self.encode_anchor_coord()
            self.codec_times["anchor_coord_encode_seconds"] = time.perf_counter() - coord_start
            model_state = self.coded_model_state()
            self.load_coded_model_state(model_state)
            context, _, anchor_latent_distributions = self.construct_anchor_context()
            geom_stream = self.anchor_coord_bitstream

            log_position_scaling_mean, position_scaling_scale_index = (
                self.position_scaling_entropy(context).chunk(2, 1))
            position_scaling_coding_scale = self.log_position_scaling_coding_scale.exp() + 1e-8
            _, log_position_scaling_hat = self.quantize_position_scaling(
                self.log_position_scaling, log_position_scaling_mean)
            (
                offset_mean,
                offset_scale_index,
                log_gaussian_scaling_mean,
                gaussian_scaling_scale_index,
            ) = self.predict_offset_and_log_gaussian_scaling(
                context, log_position_scaling_hat)

            gaussian_coder = self.gaussian_coder_class(self.cfg.rans_block_count)
            entropy_start = time.perf_counter()
            anchor_latent_streams = self.encode_anchor_latent(
                anchor_latent_distributions, gaussian_coder)
            self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
            del anchor_latent_distributions

            entropy_start = time.perf_counter()
            position_scaling_stream = gaussian_coder.encode(
                self.log_position_scaling,
                log_position_scaling_mean,
                position_scaling_scale_index,
                position_scaling_coding_scale,
            )
            self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
            del log_position_scaling_mean, position_scaling_scale_index
            del log_position_scaling_hat

            entropy_start = time.perf_counter()
            offset = self.offset
            offset_mask = self.offset_mask().bool()
            shifts = torch.arange(
                self.cfg.n_offsets - 1, -1, -1, device=offset.device)
            mask_symbols = (offset_mask.long() << shifts).sum(1)
            mask_count = 1 << self.cfg.n_offsets
            mask_counts = torch.bincount(
                mask_symbols, minlength=mask_count).float()
            mask_frequencies = quantize_categorical(
                mask_counts.clamp_min(1.0).log()[None])
            offset_mask_stream, mask_cdf = self.encode_offset_mask(mask_symbols, mask_frequencies)
            self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
            del mask_symbols, mask_counts, mask_frequencies

            log_gaussian_scaling = self.log_gaussian_scaling
            offset_coding_scale = self.log_offset_coding_scale.exp() + 1e-8
            gaussian_scaling_coding_scale = (
                self.log_gaussian_scaling_coding_scale.exp() + 1e-8)
            active_offset_mask = offset_mask[:, :, None].expand_as(offset)

            entropy_start = time.perf_counter()
            offset_stream = gaussian_coder.encode(
                offset[active_offset_mask], offset_mean[active_offset_mask],
                offset_scale_index[active_offset_mask], offset_coding_scale)
            self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
            del active_offset_mask, offset_mask, offset_mean, offset_scale_index

            entropy_start = time.perf_counter()
            gaussian_scaling_stream = gaussian_coder.encode(
                log_gaussian_scaling,
                log_gaussian_scaling_mean,
                gaussian_scaling_scale_index,
                gaussian_scaling_coding_scale)
            self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
            del log_gaussian_scaling_mean, gaussian_scaling_scale_index

            feat_mean, feat_scale_index = self.feat_context_entropy(context).chunk(2, 1)

            del context
            feat = self.anchor_feat
            feat_coding_scale = self.log_feat_coding_scale.exp() + 1e-8
            entropy_start = time.perf_counter()
            feat_stream = gaussian_coder.encode(
                feat, feat_mean, feat_scale_index, feat_coding_scale)
            self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
            del feat_mean, feat_scale_index

            streams = (
                geom_stream,
                *anchor_latent_streams,
                offset_mask_stream,
                position_scaling_stream,
                offset_stream,
                gaussian_scaling_stream,
                feat_stream,
            )
            with io.BytesIO() as bitstream:
                bitstream.write(
                    self.origin.to(device="cpu", dtype=torch.int32).numpy().tobytes())
                bitstream.write(
                    self.coord_shape.to(device="cpu", dtype=torch.int32).numpy().tobytes())
                if self.cfg.physical_voxel_size == 0:
                    bitstream.write(np.array(
                        [self.physical_voxel_size, self.anchor_grid_size], dtype="<f8").tobytes())
                if self.cfg.add_opacity_dist or self.cfg.add_cov_dist or self.cfg.add_color_dist:
                    bitstream.write(np.float32(self.distance_scale).tobytes())
                bitstream.write(model_state)
                bitstream.write(mask_cdf.numpy().tobytes())
                for stream in streams:
                    bitstream.write(len(stream).to_bytes(4, "little"))
                    bitstream.write(stream)
                encoded = bitstream.getvalue()
            if self.anchor.is_cuda:
                torch.cuda.synchronize(self.anchor.device)
            self.codec_times["compress_seconds"] = time.perf_counter() - compress_start

            return encoded

    @torch.no_grad()
    def decompress(self, bitstream: bytes):
        with self.entropy_model_timer("decode"):
            device = self.anchor.device
            if self.anchor.is_cuda:
                torch.cuda.synchronize(device)
            decompress_start = time.perf_counter()
            self.codec_times.update(decompress_seconds=0.0, entropy_decode_seconds=0.0,
                                    anchor_coord_decode_seconds=0.0)
            with io.BytesIO(bitstream) as bitstream_reader:
                origin = torch.tensor(
                    np.frombuffer(bitstream_reader.read(12), dtype=np.int32),
                    device=device, dtype=torch.int32)
                coord_shape = torch.tensor(
                    np.frombuffer(bitstream_reader.read(12), dtype=np.int32),
                    device=device, dtype=torch.int32)
                if self.cfg.physical_voxel_size == 0:
                    self.physical_voxel_size, self.anchor_grid_size = np.frombuffer(
                        bitstream_reader.read(16), dtype="<f8").tolist()
                if self.cfg.add_opacity_dist or self.cfg.add_cov_dist or self.cfg.add_color_dist:
                    self.distance_scale = np.frombuffer(
                        bitstream_reader.read(4), dtype=np.float32).item()

                model_state_bytes = self.coded_model_bits() // 8
                self.load_coded_model_state(bitstream_reader.read(model_state_bytes))

                mask_cdf_count = (1 << self.cfg.n_offsets) - 1
                offset_mask_cdf = np.frombuffer(
                    bitstream_reader.read(mask_cdf_count * 2),
                    dtype=np.uint16).reshape(1, -1)

                streams = []
                anchor_latent_stream_count = len(self.anchor_latent_entropy)
                for _ in range(anchor_latent_stream_count + 6):
                    stream_bytes = int.from_bytes(
                        bitstream_reader.read(4), "little")
                    streams.append(bitstream_reader.read(stream_bytes))

            anchor_latent_end = anchor_latent_stream_count + 1
            geom_stream = streams[0]
            anchor_latent_streams = tuple(streams[1:anchor_latent_end])
            (
                offset_mask_stream,
                position_scaling_stream,
                offset_stream,
                gaussian_scaling_stream,
                feat_stream,
            ) = streams[anchor_latent_end:]
            self.eval()

            coord_start = time.perf_counter()
            self.anchor_coords = self.decode_anchor_coord(geom_stream)
            self.codec_times["anchor_coord_decode_seconds"] = time.perf_counter() - coord_start
            anchor = (self.anchor_coords + origin).to(torch.float32)
            self.set_anchor(anchor * self.anchor_grid_size)
            self.origin = origin
            self.coord_shape = coord_shape
            self.anchor_coord_bitstream = geom_stream
            context = self.construct_anchor_geometry_context()
            gaussian_coder = self.gaussian_coder_class(self.cfg.rans_block_count)
            anchor_latents = self.decode_anchor_latent(
                context, anchor_latent_streams, gaussian_coder)
            self.anchor_latents = nn.Parameter(anchor_latents, requires_grad=False)
            context = self.fuse_anchor_latent(context, anchor_latents)
            del anchor_latents
            count = anchor.shape[0]

            log_position_scaling_mean, position_scaling_scale_index = (
                self.position_scaling_entropy(context).chunk(2, 1))
            entropy_start = time.perf_counter()
            log_position_scaling = gaussian_coder.decode(
                position_scaling_stream,
                log_position_scaling_mean,
                position_scaling_scale_index,
                self.log_position_scaling_coding_scale.exp() + 1e-8,
                log_position_scaling_mean.shape,
            )
            self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start
            del log_position_scaling_mean, position_scaling_scale_index
            (
                offset_mean,
                offset_scale_index,
                log_gaussian_scaling_mean,
                gaussian_scaling_scale_index,
            ) = self.predict_offset_and_log_gaussian_scaling(
                context, log_position_scaling)

            entropy_start = time.perf_counter()
            mask_symbols = self.decode_offset_mask(offset_mask_stream, offset_mask_cdf, count)
            mask_symbols = torch.from_numpy(mask_symbols).to(self.anchor.device)
            shifts = torch.arange(
                self.cfg.n_offsets - 1, -1, -1,
                device=self.anchor.device,
            )
            offset_mask = ((mask_symbols[:, None] >> shifts) & 1).bool()
            self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start
            del mask_symbols, shifts
            active_offset_mask = offset_mask[:, :, None].expand(-1, -1, 3)

            offset = torch.zeros_like(offset_mean)
            active_offset_mean = offset_mean[active_offset_mask]
            entropy_start = time.perf_counter()
            offset[active_offset_mask] = gaussian_coder.decode(
                offset_stream, active_offset_mean,
                offset_scale_index[active_offset_mask],
                self.log_offset_coding_scale.exp() + 1e-8,
                active_offset_mean.shape)
            self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start
            del active_offset_mask, active_offset_mean, offset_mean, offset_scale_index

            entropy_start = time.perf_counter()
            log_gaussian_scaling = gaussian_coder.decode(
                gaussian_scaling_stream,
                log_gaussian_scaling_mean,
                gaussian_scaling_scale_index,
                self.log_gaussian_scaling_coding_scale.exp() + 1e-8,
                torch.Size((count, 3)),
            )
            self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start
            del log_gaussian_scaling_mean, gaussian_scaling_scale_index

            feat_mean, feat_scale_index = self.feat_context_entropy(context).chunk(2, 1)

            del context
            feat_coding_scale = self.log_feat_coding_scale.exp() + 1e-8
            entropy_start = time.perf_counter()
            feat = gaussian_coder.decode(
                feat_stream, feat_mean, feat_scale_index,
                feat_coding_scale,
                torch.Size((count, self.cfg.anchor_feat_channels)),
            )
            self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start
            del feat_mean, feat_scale_index

            self.offset = nn.Parameter(offset, requires_grad=False)
            self.offset_mask_logits.requires_grad_(False)
            self.hard_offset_mask = offset_mask.contiguous()
            self.log_position_scaling = nn.Parameter(
                log_position_scaling, requires_grad=False)
            self.log_gaussian_scaling = nn.Parameter(
                log_gaussian_scaling, requires_grad=False)
            self.anchor_feat = nn.Parameter(feat, requires_grad=False)

            self.initialize_anchor_buffers()
            attrs = self.raw_attrs()
            if self.anchor.is_cuda:
                torch.cuda.synchronize(device)
            self.codec_times["decompress_seconds"] = time.perf_counter() - decompress_start
            return attrs

    # Training

    def training_setup(self, train_cfg: TrainConfig):
        self.train_cfg = train_cfg
        coding_scales = [
            self.log_feat_coding_scale,
            self.log_offset_coding_scale,
            self.log_position_scaling_coding_scale,
            self.log_gaussian_scaling_coding_scale,
            self.log_anchor_latent_coding_scale,
        ]

        codec_network_parameters = [
            parameter for name, parameter in self.named_parameters()
            if name not in {
                "anchor", "anchor_feat", "offset",
                "offset_mask_logits",
                "log_position_scaling", "log_gaussian_scaling",
                "log_offset_coding_scale",
                "log_position_scaling_coding_scale",
                "log_gaussian_scaling_coding_scale",
                "log_feat_coding_scale",
                "log_anchor_latent_coding_scale",
            }
            and name != "anchor_latents"
            and not name.startswith(("mlp_opacity.", "mlp_cov.", "mlp_color."))
        ]
        self.codec_parameters = [*codec_network_parameters, self.anchor_latents]

        groups = [
            {"params": [self.anchor_feat], "lr": self.train_cfg.feat_lr_init,
             "name": "feat"},
            {"params": [self.offset], "lr": self.train_cfg.offset_lr_init,
             "name": "offset"},
            {"params": [self.log_position_scaling],
             "lr": self.train_cfg.scaling_lr_init,
             "name": "position_scaling"},
            {"params": [self.log_gaussian_scaling],
             "lr": self.train_cfg.scaling_lr_init,
             "name": "gaussian_scaling"},
            {"params": codec_network_parameters,
             "lr": self.train_cfg.codec_lr_init,
             "betas": (self.train_cfg.adam_beta1,
                       self.train_cfg.codec_network_beta2),
             "eps": self.train_cfg.codec_network_eps, "name": "codec"},
            {"params": self.mlp_opacity.parameters(),
             "lr": self.train_cfg.mlp_opacity_lr_init,
             "name": "mlp_opacity"},
            {"params": self.mlp_cov.parameters(),
             "lr": self.train_cfg.mlp_cov_lr_init,
             "name": "mlp_cov"},
            {"params": self.mlp_color.parameters(),
             "lr": self.train_cfg.mlp_color_lr_init,
             "name": "mlp_color"},
            {"params": coding_scales,
             "lr": self.train_cfg.coding_scale_lr_init,
             "name": "coding_scale"},
            {"params": [self.anchor_latents],
             "lr": 0.0,
             "name": "anchor_latent"},
        ]
        if self.hard_offset_mask is None:
            groups.insert(2, {
                "params": [self.offset_mask_logits],
                "lr": self.train_cfg.offset_mask_lr_init,
                "name": "offset_mask",
            })
        self.optimizer = torch.optim.Adam(
            groups, lr=0.0,
            betas=(self.train_cfg.adam_beta1, self.train_cfg.adam_beta2),
            eps=1.0e-15,
        )
        codec_steps = max(
            self.train_cfg.iterations - self.train_cfg.rd_start_iteration - 1,
            1,
        )

        def schedule(initial, final, steps=self.train_cfg.iterations):
            if initial == final:
                return lambda _: initial
            return get_expon_lr_func(initial, final, max_steps=steps)

        self.lr_schedules = {
            "feat": schedule(
                self.train_cfg.feat_lr_init, self.train_cfg.feat_lr_final),
            "offset": schedule(
                self.train_cfg.offset_lr_init, self.train_cfg.offset_lr_final),
            "position_scaling": schedule(
                self.train_cfg.scaling_lr_init, self.train_cfg.scaling_lr_final),
            "gaussian_scaling": schedule(
                self.train_cfg.scaling_lr_init, self.train_cfg.scaling_lr_final),
            "codec": schedule(
                self.train_cfg.codec_lr_init, self.train_cfg.codec_lr_final,
                codec_steps),
            "coding_scale": schedule(
                self.train_cfg.coding_scale_lr_init,
                self.train_cfg.coding_scale_lr_final,
                codec_steps),
            "anchor_latent": schedule(
                self.train_cfg.anchor_latent_lr_init,
                self.train_cfg.anchor_latent_lr_final,
                codec_steps),
            "mlp_opacity": schedule(
                self.train_cfg.mlp_opacity_lr_init,
                self.train_cfg.mlp_opacity_lr_final),
            "mlp_cov": schedule(
                self.train_cfg.mlp_cov_lr_init, self.train_cfg.mlp_cov_lr_final),
            "mlp_color": schedule(
                self.train_cfg.mlp_color_lr_init,
                self.train_cfg.mlp_color_lr_final),
        }
        if self.hard_offset_mask is None:
            self.lr_schedules["offset_mask"] = schedule(
                self.train_cfg.offset_mask_lr_init,
                self.train_cfg.offset_mask_lr_final,
                codec_steps)

    def update_learning_rate(
        self,
        iteration: int,
        rd_iteration: int,
    ):
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name in {"codec", "coding_scale", "anchor_latent", "offset_mask"}:
                codec_step = max(rd_iteration - 1, 0)
                learning_rate = self.lr_schedules[name](codec_step)
                if name == "codec" and self.train_cfg.codec_lr_warmup_steps > 0:
                    progress = rd_iteration / self.train_cfg.codec_lr_warmup_steps
                    learning_rate *= min(progress, 1.0)
                group["lr"] = learning_rate
            elif name in self.lr_schedules:
                group["lr"] = self.lr_schedules[name](iteration)

    @torch.no_grad()
    def apply_offset_mask(
        self,
        finalize: bool = True,
        preserve_optimizer: int = 0,
    ) -> dict[str, float | int | bool]:
        old_optimizer = self.optimizer
        old_groups = {
            group["name"]: tuple(group["params"])
            for group in old_optimizer.param_groups
        } if preserve_optimizer else {}

        hard_mask = self.offset_mask_logits.sigmoid() > self.cfg.offset_mask_threshold
        keep = hard_mask.any(1)
        old_anchor_count = self.anchor.shape[0]
        active_offsets = int(hard_mask.sum())

        old_anchor_latents = self.anchor_latents.detach()

        self.set_anchor(self.anchor[keep])
        self.anchor_feat = nn.Parameter(
            self.anchor_feat[keep].detach().contiguous())
        self.offset = nn.Parameter(self.offset[keep].detach().contiguous())
        self.offset_mask_logits = nn.Parameter(
            self.offset_mask_logits[keep].detach().contiguous(),
            requires_grad=not finalize)
        self.hard_offset_mask = hard_mask[keep].contiguous() if finalize else None
        self.log_position_scaling = nn.Parameter(
            self.log_position_scaling[keep].detach().contiguous())
        self.log_gaussian_scaling = nn.Parameter(
            self.log_gaussian_scaling[keep].detach().contiguous())
        self.anchor_latents = nn.Parameter(old_anchor_latents[keep].contiguous())

        self.anchor_coords = self.anchor_coords[keep].contiguous()
        self.anchor_coord_bitstream = self.encode_anchor_coord()
        self.initialize_anchor_buffers()

        if preserve_optimizer:
            self.training_setup(self.train_cfg)
            row_groups = {
                "feat", "offset", "offset_mask",
                "position_scaling", "gaussian_scaling", "anchor_latent",
            }
            for group in self.optimizer.param_groups:
                name = group["name"]
                if preserve_optimizer == 2 and name == "codec":
                    continue
                old_parameters = old_groups[name]
                new_parameters = tuple(group["params"])
                assert len(old_parameters) == len(new_parameters)
                for index, (old_parameter, new_parameter) in enumerate(zip(
                    old_parameters, new_parameters,
                )):
                    old_state = old_optimizer.state.get(old_parameter)
                    if old_state is None:
                        continue
                    indices = keep if name in row_groups else None
                    if indices is None:
                        self.optimizer.state[new_parameter] = old_state
                    else:
                        self.optimizer.state[new_parameter] = {
                            key: value[indices].contiguous()
                            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
                            else value
                            for key, value in old_state.items()
                        }

        return {
            "old_anchor_count": old_anchor_count,
            "anchor_count": int(keep.sum()),
            "removed_anchor_count": int((~keep).sum()),
            "active_offset_count": active_offsets,
            "active_offset_fraction": active_offsets / hard_mask.numel(),
            "mask_finalized": finalize,
            "optimizer_state_preserved": preserve_optimizer != 0,
            "optimizer_state_mode": preserve_optimizer,
        }

    # Checkpoints and export

    def checkpoint_dict(self, iteration: int) -> dict[str, Any]:
        checkpoint = {
            "iteration": iteration,
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "origin": self.origin.detach().cpu(),
            "coord_shape": self.coord_shape.detach().cpu(),
        }
        if self.train_seconds is not None:
            checkpoint["train_seconds"] = self.train_seconds
        if self.cfg.physical_voxel_size == 0:
            checkpoint["physical_voxel_size"] = self.physical_voxel_size
            checkpoint["anchor_grid_size"] = self.anchor_grid_size
        return checkpoint

    @torch.no_grad()
    def restore_model_state(self, checkpoint: dict[str, Any]):
        self.train_seconds = checkpoint.get("train_seconds")
        if self.cfg.physical_voxel_size == 0:
            self.physical_voxel_size = checkpoint["physical_voxel_size"]
            self.anchor_grid_size = checkpoint["anchor_grid_size"]
        device = self.anchor.device
        model_state = checkpoint["model_state"]
        anchor = model_state["anchor"].to(device)
        self.set_anchor(anchor)

        hard_offset_mask = model_state.get("hard_offset_mask")
        self.hard_offset_mask = (
            None if hard_offset_mask is None else hard_offset_mask.to(device))
        self.anchor_feat = nn.Parameter(
            model_state["anchor_feat"].to(device))
        self.offset = nn.Parameter(model_state["offset"].to(device))
        self.offset_mask_logits = nn.Parameter(
            model_state["offset_mask_logits"].to(device),
            requires_grad=self.hard_offset_mask is None)
        self.log_position_scaling = nn.Parameter(
            model_state["log_position_scaling"].to(device))
        self.log_gaussian_scaling = nn.Parameter(
            model_state["log_gaussian_scaling"].to(device))
        self.anchor_latents = nn.Parameter(model_state["anchor_latents"].to(device))

        self.origin = checkpoint["origin"].to(device)
        self.coord_shape = checkpoint["coord_shape"].to(device)
        self.anchor_coords = torch.round(
            self.anchor / self.anchor_grid_size).to(torch.int32)
        self.anchor_coords.sub_(self.origin)
        for name, parameter in self.coded_model_parameters():
            parameter.copy_(model_state[name].to(device=device, dtype=parameter.dtype))
        self.anchor_coord_bitstream = self.encode_anchor_coord()
        self.initialize_anchor_buffers()

    def save_checkpoint(self, path: str | Path, iteration: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(iteration), path)

    @torch.no_grad()
    def save_ply(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        anchor = self.anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        offset = self.offset.transpose(1, 2).flatten(1).detach().cpu().numpy()
        feats = self.anchor_feat.detach().cpu().numpy()
        opacity = self.opacity.detach().cpu().numpy()
        scaling = torch.cat((
            self.log_position_scaling.expand(-1, 3),
            self.log_gaussian_scaling,
        ), 1)
        scaling = scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()
        columns = anchor, normals, offset, feats, opacity, scaling, rotation
        values = np.concatenate(columns, axis=1)
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [f"f_offset_{index}" for index in range(offset.shape[1])]
        names += [f"f_anchor_feat_{index}" for index in range(feats.shape[1])]
        names += ["opacity"]
        names += [f"scale_{index}" for index in range(scaling.shape[1])]
        names += [f"rot_{index}" for index in range(rotation.shape[1])]
        elements = np.empty(anchor.shape[0], dtype=[(name, "f4") for name in names])
        elements[:] = list(map(tuple, values))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
        torch.save(self.mlp_opacity.state_dict(), path.with_name(f"{path.stem}_opacity_mlp.pt"))
        torch.save(self.mlp_cov.state_dict(), path.with_name(f"{path.stem}_cov_mlp.pt"))
        torch.save(self.mlp_color.state_dict(), path.with_name(f"{path.stem}_color_mlp.pt"))
        torch.save(self.hard_offset_mask.cpu(), path.with_name(f"{path.stem}_mask.pt"))
