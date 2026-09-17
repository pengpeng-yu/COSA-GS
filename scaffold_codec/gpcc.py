import subprocess
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def write_xyz_ply(xyz: torch.Tensor, path: Path):
    xyz = xyz.cpu().numpy().astype("<f4", copy=False)
    vertices = np.empty(xyz.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def gpcc_encode(input_path: Path, output_path: Path, tmc3: Path):
    subprocess.run([
        str(tmc3),
        "--mode=0",
        "--trisoupNodeSizeLog2=0",
        "--mergeDuplicatedPoints=1",
        "--neighbourAvailBoundaryLog2=8",
        "--intra_pred_max_node_size_log2=6",
        "--positionQuantizationScale=1",
        "--maxNumQtBtBeforeOt=4",
        "--minQtbtSizeLog2=0",
        "--planarEnabled=1",
        "--planarModeIdcmUse=0",
        "--disableAttributeCoding=1",
        f"--uncompressedDataPath={input_path}",
        f"--compressedStreamPath={output_path}",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)


def gpcc_decode(input_path: Path, output_path: Path, tmc3: Path):
    subprocess.run([
        str(tmc3),
        "--mode=1",
        f"--compressedStreamPath={input_path}",
        f"--reconstructedDataPath={output_path}",
        "--outputBinaryPly=1",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
