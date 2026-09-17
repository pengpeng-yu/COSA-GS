from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from scaffold_codec.entropy import GAUSSIAN_SCALE_LEVELS, IndexedSymbolCoder
from scaffold_codec.rans_coder import CdfTables, build_inverse_cdfs

from .int_nn_ops import ONE
from .quantization import checked_int32, round_divide


def reconstruct_fixed(mean, symbols, requant_mul, requant_shift):
    assert mean.dtype == torch.int32 and symbols.dtype == torch.int16
    assert type(requant_mul) is int and type(requant_shift) is int
    delta = round_divide(symbols.to(torch.int64) * requant_mul, 1 << requant_shift)
    return checked_int32(mean.to(torch.int64) + delta)


def reconstruct_float(mean, symbols, requant_mul, requant_shift):
    assert mean.dtype == torch.int32 and symbols.dtype == torch.int16
    assert type(requant_mul) is int and type(requant_shift) is int
    step = requant_mul / (ONE * (1 << requant_shift))
    return mean.float() / ONE + symbols.float() * step


@lru_cache(maxsize=1)
def load_gaussian_tables():
    cdfs, offsets = [], []
    with Path(__file__).with_name("gaussian_cdfs.bin").open("rb") as reader:
        count = int.from_bytes(reader.read(2), "little")
        assert count == GAUSSIAN_SCALE_LEVELS
        for _ in range(count):
            offsets.append(int.from_bytes(reader.read(4), "little", signed=True))
            length = int.from_bytes(reader.read(2), "little")
            cdfs.append(np.frombuffer(reader.read(length * 2), dtype="<u2").tolist())
    tables = CdfTables()
    tables.cdfs = cdfs
    tables.offsets = offsets
    build_inverse_cdfs(tables)
    return tables


def quantize_residual(value, mean, step):
    assert value.dtype == torch.float32 and mean.dtype == torch.int32
    mul, shift = step
    assert type(mul) is int and type(shift) is int
    coding_scale = ONE * (1 << shift) / mul
    symbols = ((value - mean.float() / ONE) * coding_scale).round()
    assert ((symbols >= -32768) & (symbols <= 32767)).all()
    return symbols.to(torch.int16)


class IntegerGaussianCoder(IndexedSymbolCoder):
    def __init__(self, block_count=4):
        super().__init__(load_gaussian_tables(), GAUSSIAN_SCALE_LEVELS, block_count)

    def indices(self, index):
        assert index.dtype == torch.int32
        indices = round_divide(index.to(torch.int64), ONE).clamp(0, self.scale_levels - 1).to(torch.uint16)
        return indices.reshape(-1).cpu().contiguous().numpy()

    def encode(self, symbols, index):
        assert symbols.dtype == torch.int16 and symbols.shape == index.shape
        return self.encode_symbols(symbols.reshape(-1).cpu().contiguous().numpy(), self.indices(index))

    def decode(self, stream, index):
        indices = self.indices(index)
        symbols = torch.empty(indices.size, dtype=torch.int16, pin_memory=index.is_cuda)
        self.decode_symbols_into(stream, indices, symbols.numpy())
        return symbols.to(index.device, non_blocking=True).reshape(index.shape)
