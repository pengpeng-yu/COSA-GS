"""Generate the fixed Gaussian CDF resource offline, not during codec execution."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scaffold_codec.entropy import build_gaussian_tables


def main():
    tables = build_gaussian_tables()
    path = PROJECT_ROOT / "scaffold_codec_int/gaussian_cdfs.bin"
    with path.open("wb") as writer:
        writer.write(len(tables.cdfs).to_bytes(2, "little"))
        for cdf, offset in zip(tables.cdfs, tables.offsets):
            writer.write(int(offset).to_bytes(4, "little", signed=True))
            writer.write(len(cdf).to_bytes(2, "little"))
            writer.write(np.asarray(cdf, dtype="<u2").tobytes())
    print(f"{path}: {len(tables.cdfs)} CDFs, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
