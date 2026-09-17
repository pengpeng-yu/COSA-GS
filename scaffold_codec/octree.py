import io

import numpy as np
import torch

from .entropy import quantize_categorical
from .rans_coder import RansDecoder, RansEncoder


def encode_anchor_coord(coords: torch.Tensor, point_threshold: int) -> bytes:
    assert point_threshold > 1
    encoder = RansEncoder()
    cdfs = []
    while True:
        coarse = coords.shape[0] < point_threshold
        if coarse:
            assert ((coords >= 0) & (coords <= 32767)).all(), \
                "Coarse coordinates must be in [0, 32767]."
            symbols = coords.reshape(-1)
        else:
            parents, parent_indices = torch.unique_consecutive(
                coords >> 1, dim=0, return_inverse=True,
            )
            child_indices = (
                ((coords[:, 0] & 1) << 2)
                | ((coords[:, 1] & 1) << 1)
                | (coords[:, 2] & 1)
            )
            symbols = coords.new_zeros(parents.shape[0])
            symbols.scatter_add_(0, parent_indices, 1 << child_indices)
            coords = parents

        counts = torch.bincount(symbols, minlength=1 if coarse else 256)
        frequencies = quantize_categorical(counts.float().log()[None])
        cdf = frequencies.cumsum(1, dtype=torch.int32)[0, :-1].to(
            device="cpu", dtype=torch.uint16,
        ).numpy()
        symbols = symbols.to(device="cpu", dtype=torch.int16).numpy()
        encoder.encode_categorical_shared(symbols, cdf)
        cdfs.append(cdf)
        if coarse:
            break

    with io.BytesIO() as bitstream:
        bitstream.write((len(cdfs) - 1).to_bytes(4, "little"))
        bitstream.write(coords.shape[0].to_bytes(4, "little"))
        bitstream.write(cdf.size.to_bytes(2, "little"))
        for cdf in reversed(cdfs):
            bitstream.write(cdf.astype("<u2", copy=False).tobytes())
        bitstream.write(encoder.flush())
        return bitstream.getvalue()


def decode_anchor_coord(bitstream: bytes, device="cpu") -> torch.Tensor:
    with io.BytesIO(bitstream) as bitstream_reader:
        levels = int.from_bytes(bitstream_reader.read(4), "little")
        coarse_count = int.from_bytes(bitstream_reader.read(4), "little")
        coarse_cdf_size = int.from_bytes(bitstream_reader.read(2), "little")
        cdfs = np.frombuffer(
            bitstream_reader.read((coarse_cdf_size + 255 * levels) * 2), dtype="<u2",
        )
        decoder = RansDecoder()
        decoder.set_stream(bitstream_reader.read())
    coords = torch.from_numpy(decoder.decode_categorical_shared(
        cdfs[:coarse_cdf_size], coarse_count * 3,
    )).to(device=device, dtype=torch.int32).reshape(-1, 3)

    octants = torch.arange(8, device=device, dtype=torch.int32)
    child_offsets = (octants[:, None] >> coords.new_tensor([2, 1, 0])) & 1
    for cdf in cdfs[coarse_cdf_size:].reshape(levels, 255):
        occupancy = torch.from_numpy(decoder.decode_categorical_shared(
            cdf, coords.shape[0],
        )).to(device=device, dtype=torch.int32)
        parent_indices, child_indices = (
            (occupancy[:, None] >> octants) & 1
        ).nonzero(as_tuple=True)
        coords = (coords[parent_indices] << 1) + child_offsets[child_indices]
    return coords
