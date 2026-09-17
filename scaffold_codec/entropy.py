import io
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import torch

from .rans_coder import (
    RansDecoder,
    RansEncoder,
    build_inverse_cdfs,
    quantize_pmfs,
)


CDF_PRECISION = 16
CDF_TOTAL = 1 << CDF_PRECISION
GAUSSIAN_SCALE_MIN = 0.1
GAUSSIAN_SCALE_MAX = 256.0
GAUSSIAN_SCALE_LEVELS = 128
GAUSSIAN_SYMBOL_BOUND = 1024
GAUSSIAN_PROB_SWITCH = 1.0e-8
ONE_DIV_SQRT2 = 1.0 / math.sqrt(2.0)
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class LowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, bound):
        ctx.save_for_backward(value, bound)
        return torch.maximum(value, bound)

    @staticmethod
    def backward(ctx, gradient):
        value, bound = ctx.saved_tensors
        return gradient * ((value >= bound) | (gradient < 0)), None


class UpperBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, bound):
        ctx.save_for_backward(value, bound)
        return torch.minimum(value, bound)

    @staticmethod
    def backward(ctx, gradient):
        value, bound = ctx.saved_tensors
        return gradient * ((value <= bound) | (gradient > 0)), None


def bound_scale_index(index: torch.Tensor, levels: int) -> torch.Tensor:
    lower = index.new_tensor(0.0)
    upper = index.new_tensor(levels - 1.0)
    return UpperBound.apply(LowerBound.apply(index, lower), upper)


def indexed_gaussian_scale(index: torch.Tensor) -> torch.Tensor:
    index = bound_scale_index(index, GAUSSIAN_SCALE_LEVELS)
    step = math.log(GAUSSIAN_SCALE_MAX / GAUSSIAN_SCALE_MIN)
    step /= GAUSSIAN_SCALE_LEVELS - 1
    return torch.exp(index * step + math.log(GAUSSIAN_SCALE_MIN))


def gaussian_scale_index_for_scale(scale: float) -> float:
    scale = min(max(scale, GAUSSIAN_SCALE_MIN), GAUSSIAN_SCALE_MAX)
    numerator = math.log(scale / GAUSSIAN_SCALE_MIN)
    denominator = math.log(GAUSSIAN_SCALE_MAX / GAUSSIAN_SCALE_MIN)
    return numerator / denominator * (GAUSSIAN_SCALE_LEVELS - 1)


@torch.autocast("cuda", enabled=False)
def gaussian_log_prob(
    value: torch.Tensor,
    scale: torch.Tensor,
    prob_switch: float = GAUSSIAN_PROB_SWITCH,
) -> torch.Tensor:
    value, scale = value.float(), scale.float()
    reciprocal = scale.reciprocal() * ONE_DIV_SQRT2
    distance = value.abs()
    prob = 0.5 * (
        ((distance - 0.5) * reciprocal).erfc()
        - ((distance + 0.5) * reciprocal).erfc()
    )
    exact = prob.clamp_min(1.0e-8).log()
    density = -(distance * reciprocal).square()
    density = density - scale.log() - HALF_LOG_2PI
    return torch.where(prob > prob_switch, exact, density)


def compute_gaussian_bits(
    value: torch.Tensor,
    mean: torch.Tensor,
    scale_index: torch.Tensor,
    coding_scale: torch.Tensor | float,
    training: bool,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if training: assert noise is not None
    symbols = (value - mean) * coding_scale
    symbols = symbols + noise if training else torch.round(symbols)
    scale_index = scale_index if training else scale_index.round()
    log_prob = gaussian_log_prob(
        symbols, indexed_gaussian_scale(scale_index))
    return (-log_prob / math.log(2.0)).sum()


def gaussian_scale_table() -> torch.Tensor:
    return torch.exp(torch.linspace(
        math.log(GAUSSIAN_SCALE_MIN),
        math.log(GAUSSIAN_SCALE_MAX),
        GAUSSIAN_SCALE_LEVELS,
        dtype=torch.float64,
    ))


@lru_cache(maxsize=1)
def build_gaussian_tables():
    scales = gaussian_scale_table()
    symbol_bound = GAUSSIAN_SYMBOL_BOUND
    symbols = torch.arange(-symbol_bound, symbol_bound + 1, dtype=torch.float64)
    lower = (symbols[None] - 0.5) / scales[:, None]
    upper = (symbols[None] + 0.5) / scales[:, None]
    left = 0.5 * (torch.erfc(-upper * ONE_DIV_SQRT2)
                  - torch.erfc(-lower * ONE_DIV_SQRT2))
    right = 0.5 * (torch.erfc(lower * ONE_DIV_SQRT2)
                   - torch.erfc(upper * ONE_DIV_SQRT2))
    pmfs = torch.where(symbols[None] < 0, left, right)
    offsets = np.full(
        GAUSSIAN_SCALE_LEVELS, -symbol_bound, dtype=np.int32)
    tables = quantize_pmfs(pmfs.numpy(), offsets)
    build_inverse_cdfs(tables)
    return tables


@lru_cache(maxsize=None)
def rans_executor(block_count):
    return ThreadPoolExecutor(max_workers=block_count, thread_name_prefix="rans")


def encode_block(symbols, indices, tables):
    encoder = RansEncoder()
    encoder.encode_indexed(symbols, indices, tables)
    return encoder.flush()


def decode_block(stream, indices, tables, symbols):
    decoder = RansDecoder()
    decoder.set_stream(stream)
    decoder.decode_indexed_lookup_into(indices, tables, symbols)


class IndexedSymbolCoder:
    def __init__(self, tables, scale_levels: int, block_count: int = 4):
        assert block_count > 0
        self.tables = tables
        self.scale_levels = scale_levels
        self.block_count = block_count

    def encode_symbols(self, symbols, indices):
        assert symbols.dtype == np.int16 and indices.dtype == np.uint16
        count = symbols.size
        blocks = self.block_count
        chunks = [slice(count * i // blocks, count * (i + 1) // blocks) for i in range(blocks)]
        if blocks == 1:
            streams = [encode_block(symbols, indices, self.tables)]
        else:
            executor = rans_executor(blocks)
            futures = [executor.submit(encode_block, symbols[chunk], indices[chunk], self.tables)
                       for chunk in chunks]
            streams = [future.result() for future in futures]
        with io.BytesIO() as bitstream:
            bitstream.write(blocks.to_bytes(4, "little"))
            for stream in streams:
                bitstream.write(len(stream).to_bytes(4, "little"))
            for stream in streams:
                bitstream.write(stream)
            return bitstream.getvalue()

    def decode_symbols_into(self, stream, indices, symbols):
        assert indices.dtype == np.uint16 and symbols.dtype == np.int16
        with io.BytesIO(stream) as bitstream:
            blocks = int.from_bytes(bitstream.read(4), "little")
            lengths = [int.from_bytes(bitstream.read(4), "little") for _ in range(blocks)]
            count = indices.size
            futures = []
            for i, length in enumerate(lengths):
                chunk = slice(count * i // blocks, count * (i + 1) // blocks)
                coded = bitstream.read(length)
                if blocks == 1 or self.block_count == 1:
                    decode_block(coded, indices[chunk], self.tables, symbols[chunk])
                else:
                    futures.append(rans_executor(self.block_count).submit(
                        decode_block, coded, indices[chunk], self.tables, symbols[chunk]))
            for future in futures:
                future.result()

    def scale_indices(
        self, scale_index: torch.Tensor, shape: torch.Size,
    ) -> np.ndarray:
        bounded = scale_index.round().clamp_(0, self.scale_levels - 1)
        bounded = bounded.to(torch.uint16)
        bounded = torch.broadcast_to(bounded, shape)
        return bounded.reshape(-1).cpu().contiguous().numpy()

    def encode(
        self,
        value: torch.Tensor,
        mean: torch.Tensor,
        scale_index: torch.Tensor,
        coding_scale: torch.Tensor | float,
    ) -> bytes:
        symbols = torch.round((value - mean) * coding_scale)
        symbols = symbols.to(torch.int16).reshape(-1).cpu().contiguous().numpy()
        indices = self.scale_indices(scale_index, value.shape)
        return self.encode_symbols(symbols, indices)

    def decode(
        self,
        stream: bytes,
        mean: torch.Tensor,
        scale_index: torch.Tensor,
        coding_scale: torch.Tensor | float,
        shape: torch.Size,
    ) -> torch.Tensor:
        indices = self.scale_indices(scale_index, shape)
        symbols = torch.empty(indices.size, dtype=torch.int16, pin_memory=mean.is_cuda)
        self.decode_symbols_into(stream, indices, symbols.numpy())
        symbols = symbols.to(mean.device, non_blocking=True).reshape(shape)
        return torch.broadcast_to(mean, shape) + symbols / coding_scale


class IndexedGaussianCoder(IndexedSymbolCoder):
    def __init__(self, block_count: int = 4):
        super().__init__(build_gaussian_tables(), GAUSSIAN_SCALE_LEVELS, block_count)


def quantize_categorical(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(-1)
    symbol_count = probabilities.shape[1]
    frequencies = (probabilities * (CDF_TOTAL - symbol_count)).to(torch.int32)
    frequencies.add_(1)
    remainder = CDF_TOTAL - frequencies.sum(1, dtype=torch.int32)
    maxima = probabilities.argmax(1, keepdim=True)
    frequencies.scatter_add_(1, maxima, remainder[:, None])
    return frequencies


def encode_categorical(
    encoder: RansEncoder,
    logits: torch.Tensor,
    symbols: torch.Tensor,
) -> None:
    frequencies = quantize_categorical(logits)
    indexes = symbols.reshape(-1, 1)
    selected = frequencies.gather(1, indexes)
    ends = frequencies.cumsum(1, dtype=torch.int32).gather(1, indexes)
    intervals = torch.cat((ends - selected, selected), 1).to(
        device="cpu", dtype=torch.uint16, memory_format=torch.contiguous_format)
    encoder.encode_categorical(intervals.numpy())


def decode_categorical(
    decoder: RansDecoder,
    logits: torch.Tensor,
) -> torch.Tensor:
    cdfs = quantize_categorical(logits).cumsum(1, dtype=torch.int32)[:, :-1].to(
        device="cpu", dtype=torch.uint16, memory_format=torch.contiguous_format
    ).numpy()
    symbols = decoder.decode_categorical(cdfs, logits.shape[0])
    return torch.from_numpy(symbols).to(device=logits.device)
