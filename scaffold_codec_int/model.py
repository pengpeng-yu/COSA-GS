import io
import math
import time

import numpy as np
import torch
import torch.nn as nn

from scaffold_codec.entropy import quantize_categorical
from scaffold_codec.model import CompressedGaussianModel as FloatModel

from .entropy import IntegerGaussianCoder, quantize_residual, \
    reconstruct_fixed, reconstruct_float
from .int_nn_ops import ONE
from .qat import QATLinear, replace_linear
from .quantization import checked_int32, export_sequential, round_divide


ENTROPY_MODULES = (
    "anchor_latent_entropy", "anchor_latent_context", "anchor_latent_fusion",
    "anchor_coord_encoder", "position_scaling_encoder", "offset_prior_fusion",
    "gaussian_scaling_prior_fusion", "offset_entropy", "position_scaling_entropy",
    "gaussian_scaling_entropy", "feat_context_entropy",
)
RENDER_MODULES = ("mlp_opacity", "mlp_cov", "mlp_color")
CODING_SCALES = ("anchor_latent", "position_scaling", "offset", "gaussian_scaling", "feat")


def normalize_coord(coord, shape):
    assert coord.dtype == torch.int32 and shape.dtype == torch.int32
    denominator = shape.to(torch.int64) - 1
    assert (denominator > 0).all()
    return (round_divide(coord.to(torch.int64) * (2 * ONE), denominator) - ONE).to(torch.int32)


class CompressedGaussianModel(FloatModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        for name in ENTROPY_MODULES:
            module = getattr(self, name)
            if isinstance(module, nn.ModuleList):
                for sequence in module:
                    replace_linear(sequence)
            else:
                replace_linear(module)

    def update_learning_rate(self, iteration, rd_iteration):
        super().update_learning_rate(iteration, rd_iteration)
        for module in self.modules():
            if isinstance(module, QATLinear):
                for quantizer in (module.activation_fake_quant, module.weight_fake_quant):
                    quantizer.enable_fake_quant(iteration >= self.cfg.qat_start_iteration)
                    quantizer.enable_observer(iteration < self.cfg.qat_observer_freeze_iteration)

    def restore_model_state(self, checkpoint):
        super().restore_model_state(checkpoint)
        state = checkpoint["model_state"]
        for name, module in self.named_modules():
            if isinstance(module, QATLinear):
                prefix = name + "."
                module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                        if key.startswith(prefix)})

    def make_integer_modules(self, import_parameters):
        modules = nn.ModuleDict()
        for name in ENTROPY_MODULES:
            module = getattr(self, name)
            if isinstance(module, nn.ModuleList):
                modules[name] = nn.ModuleList([export_sequential(seq, import_parameters) for seq in module])
            else:
                modules[name] = export_sequential(module, import_parameters)
        return modules

    def coded_model_bits(self):
        modules = self.make_integer_modules(False)
        size = sum(t.numel() * t.element_size() for t in modules.state_dict().values())
        size += sum(p.numel() * 2 for name in RENDER_MODULES for p in getattr(self, name).parameters())
        size += len(CODING_SCALES) * 12
        return size * 8

    def integer_coord_context(self, modules, coords):
        return modules["anchor_coord_encoder"](normalize_coord(coords, self.coord_shape))

    def integer_latent_distribution(self, modules, context, channels, stage):
        inputs = torch.cat((context, *channels), 1) if channels else context
        return modules["anchor_latent_entropy"][stage](inputs).chunk(2, 1)

    def integer_fuse_latent(self, modules, context, latent):
        latent_context = modules["anchor_latent_context"](latent)
        return modules["anchor_latent_fusion"](torch.cat((context, latent_context), 1))

    def integer_attr_distributions(self, modules, context, position):
        prior = modules["position_scaling_encoder"](position)
        offset_context = checked_int32(context.to(torch.int64) + modules["offset_prior_fusion"](prior).to(torch.int64))
        scaling_context = checked_int32(context.to(torch.int64) + modules["gaussian_scaling_prior_fusion"](prior).to(torch.int64))
        offset_mean, offset_index = modules["offset_entropy"](offset_context).chunk(2, 1)
        scaling_mean, scaling_index = modules["gaussian_scaling_entropy"](scaling_context).chunk(2, 1)
        return (offset_mean.reshape(-1, self.cfg.n_offsets, 3),
                offset_index.reshape(-1, self.cfg.n_offsets, 3), scaling_mean, scaling_index)

    @torch.no_grad()
    def compress(self):
        device = self.anchor.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        self.codec_times.update(compress_seconds=0.0, entropy_encode_seconds=0.0,
                                anchor_coord_encode_seconds=0.0)
        modules = self.make_integer_modules(True).to(device)
        steps = {}
        # int16 residuals times a 47-bit multiplier fit in int64 with room for the mean.
        for name in CODING_SCALES:
            scale = getattr(self, "log_" + name + "_coding_scale").detach().cpu().double().exp() + 1e-8
            ratio = ONE / scale.item()
            assert math.isfinite(ratio) and ratio > 0
            shift = min(62, math.floor(math.log2(((1 << 47) - 1) / ratio)))
            assert shift >= 0, "Reconstruction step is too large"
            mul = round(ratio * (1 << shift))
            assert 0 < mul < (1 << 47)
            steps[name] = (mul, shift)
        coder = IntegerGaussianCoder(self.cfg.rans_block_count)

        coord_start = time.perf_counter()
        geom_stream = self.encode_anchor_coord()
        self.codec_times["anchor_coord_encode_seconds"] = time.perf_counter() - coord_start
        context = self.integer_coord_context(modules, self.anchor_coords)
        channels, latent_streams = [], []
        for stage in range(self.cfg.anchor_latent_channels):
            mean, index = self.integer_latent_distribution(modules, context, channels, stage)
            entropy_start = time.perf_counter()
            symbols = quantize_residual(self.anchor_latents[:, stage:stage + 1], mean, steps["anchor_latent"])
            latent_streams.append(coder.encode(symbols, index))
            self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
            latent = reconstruct_fixed(mean, symbols, *steps["anchor_latent"])
            channels.append(latent)

        context = self.integer_fuse_latent(modules, context, torch.cat(channels, 1))
        mean, index = modules["position_scaling_entropy"](context).chunk(2, 1)
        entropy_start = time.perf_counter()
        symbols = quantize_residual(self.log_position_scaling, mean, steps["position_scaling"])
        position_stream = coder.encode(symbols, index)
        self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
        position = reconstruct_fixed(mean, symbols, *steps["position_scaling"])
        offset_mean, offset_index, scaling_mean, scaling_index = \
            self.integer_attr_distributions(modules, context, position)

        entropy_start = time.perf_counter()
        mask = self.offset_mask().bool()
        shifts = torch.arange(self.cfg.n_offsets - 1, -1, -1, dtype=torch.int64, device=device)
        mask_symbols = (mask.long() << shifts).sum(1)
        counts = torch.bincount(mask_symbols, minlength=1 << self.cfg.n_offsets).float()
        frequencies = quantize_categorical(counts.clamp_min(1).log()[None])
        mask_stream, mask_cdf = self.encode_offset_mask(mask_symbols, frequencies)
        self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start
        active = mask[:, :, None].expand_as(self.offset)

        entropy_start = time.perf_counter()
        symbols = quantize_residual(self.offset[active], offset_mean[active], steps["offset"])
        offset_stream = coder.encode(symbols, offset_index[active])
        self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start

        entropy_start = time.perf_counter()
        symbols = quantize_residual(self.log_gaussian_scaling, scaling_mean, steps["gaussian_scaling"])
        scaling_stream = coder.encode(symbols, scaling_index)
        self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start

        mean, index = modules["feat_context_entropy"](context).chunk(2, 1)
        entropy_start = time.perf_counter()
        symbols = quantize_residual(self.anchor_feat, mean, steps["feat"])
        feat_stream = coder.encode(symbols, index)
        self.codec_times["entropy_encode_seconds"] += time.perf_counter() - entropy_start

        with io.BytesIO() as writer:
            writer.write(self.origin.cpu().numpy().astype("<i4").tobytes())
            writer.write(self.coord_shape.cpu().numpy().astype("<i4").tobytes())
            if self.cfg.physical_voxel_size == 0:
                writer.write(np.array([self.physical_voxel_size, self.anchor_grid_size], dtype="<f8").tobytes())
            if self.cfg.add_opacity_dist or self.cfg.add_cov_dist or self.cfg.add_color_dist:
                writer.write(np.array([self.distance_scale], dtype="<f4").tobytes())

            for tensor in modules.state_dict().values():
                array = tensor.cpu().numpy()
                writer.write(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())

            for name in RENDER_MODULES:
                for parameter in getattr(self, name).parameters():
                    writer.write(parameter.detach().cpu().numpy().astype("<f2").tobytes())

            for mul, shift in steps.values():
                writer.write(int(mul).to_bytes(8, "little"))
                writer.write(int(shift).to_bytes(4, "little"))

            writer.write(mask_cdf.numpy().astype("<u2").tobytes())

            for stream in (geom_stream, *latent_streams, mask_stream, position_stream,
                           offset_stream, scaling_stream, feat_stream):
                writer.write(len(stream).to_bytes(4, "little"))
                writer.write(stream)

            encoded = writer.getvalue()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.codec_times["compress_seconds"] = time.perf_counter() - start
        return encoded

    def decode_integer_attributes(self, modules, coder, streams, steps):
        latent_streams = streams[:self.cfg.anchor_latent_channels]
        mask_stream, position_stream, offset_stream, scaling_stream, feat_stream = \
            streams[self.cfg.anchor_latent_channels:]

        context = self.integer_coord_context(modules, self.anchor_coords)
        channels = []
        latent_symbols = {}
        for stage, stream in enumerate(latent_streams):
            mean, index = self.integer_latent_distribution(modules, context, channels, stage)
            start = time.perf_counter()
            symbols = coder.decode(stream, index)
            latent_symbols[f"anchor_latent_symbols_{stage}"] = symbols
            channels.append(reconstruct_fixed(mean, symbols, *steps["anchor_latent"]))
            self.codec_times["entropy_decode_seconds"] += time.perf_counter() - start
        latent = torch.cat(channels, 1)

        context = self.integer_fuse_latent(modules, context, latent)
        mean, index = modules["position_scaling_entropy"](context).chunk(2, 1)
        start = time.perf_counter()
        position_symbols = coder.decode(position_stream, index)
        position = reconstruct_fixed(mean, position_symbols, *steps["position_scaling"])
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - start

        offset_mean, offset_index, scaling_mean, scaling_index = \
            self.integer_attr_distributions(modules, context, position)

        mask_cdf, mask_stream = mask_stream
        start = time.perf_counter()
        mask_symbols = torch.from_numpy(self.decode_offset_mask(mask_stream, mask_cdf, context.shape[0]))
        mask_values = mask_symbols.to(device=context.device, dtype=torch.int64)
        shifts = torch.arange(self.cfg.n_offsets - 1, -1, -1, dtype=torch.int64, device=context.device)
        mask = ((mask_values[:, None] >> shifts) & 1).bool()
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - start

        active = mask[:, :, None].expand_as(offset_mean)
        offset_mean = offset_mean[active]
        start = time.perf_counter()
        offset_symbols = coder.decode(offset_stream, offset_index[active])
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        scaling_symbols = coder.decode(scaling_stream, scaling_index)
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - start

        feat_mean, feat_index = modules["feat_context_entropy"](context).chunk(2, 1)
        start = time.perf_counter()
        feat_symbols = coder.decode(feat_stream, feat_index)
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - start

        return {**latent_symbols, "position_scaling_symbols": position_symbols,
                "offset_mask_symbols": mask_symbols,
                "anchor_latents": latent, "log_position_scaling": position,
                "offset_mean": offset_mean, "offset_symbols": offset_symbols,
                "scaling_mean": scaling_mean, "scaling_symbols": scaling_symbols,
                "feat_mean": feat_mean, "feat_symbols": feat_symbols,
                "hard_offset_mask": mask}

    @torch.no_grad()
    def decompress(self, encoded):
        modules = self.make_integer_modules(False)
        device = self.anchor.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        self.codec_times.update(decompress_seconds=0.0, entropy_decode_seconds=0.0,
                                anchor_coord_decode_seconds=0.0)

        with io.BytesIO(encoded) as reader:
            self.origin = torch.from_numpy(np.frombuffer(reader.read(12), dtype="<i4").copy()).to(device)
            self.coord_shape = torch.from_numpy(np.frombuffer(reader.read(12), dtype="<i4").copy()).to(device)
            if self.cfg.physical_voxel_size == 0:
                self.physical_voxel_size, self.anchor_grid_size = np.frombuffer(reader.read(16), dtype="<f8").tolist()
            if self.cfg.add_opacity_dist or self.cfg.add_cov_dist or self.cfg.add_color_dist:
                self.distance_scale = np.frombuffer(reader.read(4), dtype="<f4").item()

            for tensor in modules.state_dict().values():
                dtype = tensor.numpy().dtype.newbyteorder("<")
                array = np.frombuffer(reader.read(tensor.numel() * tensor.element_size()), dtype=dtype).copy()
                tensor.copy_(torch.from_numpy(array).reshape(tensor.shape))

            for name in RENDER_MODULES:
                for parameter in getattr(self, name).parameters():
                    array = np.frombuffer(reader.read(parameter.numel() * 2), dtype="<f2").copy()
                    parameter.copy_(torch.from_numpy(array).reshape(parameter.shape).to(device=device, dtype=parameter.dtype))

            steps = {name: (int.from_bytes(reader.read(8), "little"), int.from_bytes(reader.read(4), "little"))
                     for name in CODING_SCALES}

            count = (1 << self.cfg.n_offsets) - 1
            mask_cdf = np.frombuffer(reader.read(count * 2), dtype="<u2")

            streams = [reader.read(int.from_bytes(reader.read(4), "little"))
                       for _ in range(self.cfg.anchor_latent_channels + 6)]
            assert not reader.read(1), "Trailing integer scene bytes"

        modules.to(device)
        coord_start = time.perf_counter()
        self.anchor_coords = self.decode_anchor_coord(streams[0])
        self.codec_times["anchor_coord_decode_seconds"] = time.perf_counter() - coord_start
        self.anchor_coord_bitstream = streams[0]

        streams = streams[1:]
        mask_index = self.cfg.anchor_latent_channels
        streams[mask_index] = (mask_cdf, streams[mask_index])
        decoded = self.decode_integer_attributes(
            modules, IntegerGaussianCoder(self.cfg.rans_block_count), streams, steps)

        self.set_anchor((self.anchor_coords.to(torch.int64) + self.origin.to(torch.int64)).float() * self.anchor_grid_size)
        entropy_start = time.perf_counter()
        anchor_latents = decoded["anchor_latents"].float() / ONE
        log_position_scaling = decoded["log_position_scaling"].float() / ONE
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start
        self.anchor_latents = nn.Parameter(anchor_latents, requires_grad=False)
        self.log_position_scaling = nn.Parameter(log_position_scaling, requires_grad=False)
        self.hard_offset_mask = decoded["hard_offset_mask"].contiguous()

        offset = torch.zeros((*self.hard_offset_mask.shape, 3), dtype=torch.float32, device=device)
        active = self.hard_offset_mask[:, :, None].expand_as(offset)
        entropy_start = time.perf_counter()
        offset[active] = reconstruct_float(decoded["offset_mean"], decoded["offset_symbols"], *steps["offset"])
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start

        entropy_start = time.perf_counter()
        log_gaussian_scaling = reconstruct_float(decoded["scaling_mean"], decoded["scaling_symbols"], *steps["gaussian_scaling"])
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start

        entropy_start = time.perf_counter()
        anchor_feat = reconstruct_float(decoded["feat_mean"], decoded["feat_symbols"], *steps["feat"])
        self.codec_times["entropy_decode_seconds"] += time.perf_counter() - entropy_start

        self.offset = nn.Parameter(offset, requires_grad=False)
        self.log_gaussian_scaling = nn.Parameter(log_gaussian_scaling, requires_grad=False)
        self.anchor_feat = nn.Parameter(anchor_feat, requires_grad=False)

        self.offset_mask_logits.requires_grad_(False)
        self.initialize_anchor_buffers()
        self.eval()
        result = dict(zip(("anchor_feat", "offset", "log_position_scaling", "log_gaussian_scaling"),
                          self.raw_attrs()))
        result["anchor_coords"] = self.anchor_coords
        result.update((name, value) for name, value in decoded.items() if "symbols" in name)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.codec_times["decompress_seconds"] = time.perf_counter() - start
        return result
