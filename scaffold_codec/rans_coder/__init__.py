from pathlib import Path

from torch.utils.cpp_extension import load


source_dir = Path(__file__).resolve().parent
build_dir = source_dir / "build"
build_dir.mkdir(exist_ok=True)
extension = load(
    name="scaffold_codec_rans",
    sources=[
        str(source_dir / "rans_wrapper.cpp"),
        str(source_dir / "cdf_ops.cpp"),
    ],
    extra_cflags=["-O3", "-Wall", "-Wextra"],
    build_directory=str(build_dir),
    verbose=False,
)

quantize_pmfs = extension.quantize_pmfs
build_inverse_cdfs = extension.build_inverse_cdfs
RansEncoder = extension.RansEncoder
RansDecoder = extension.RansDecoder
CdfTables = extension.CdfTables
