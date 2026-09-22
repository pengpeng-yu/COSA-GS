<p align="center">
  <h1 align="center">Context without Spatial Aggregation: Anchor-Wise Causal Factorization for 3DGS Compression</h1>
  <h3 align="center">
    <a href="https://arxiv.org/abs/">Paper</a> |
    <a href="https://github.com/pengpeng-yu/COSA-GS">Code</a>
  </h3>
</p>

**COSA-GS** compresses 3D Gaussian Splatting through anchor-wise causal factorization, without spatial aggregation.
A compact learnable latent is encoded channel by channel, then fused with each anchor's own coordinate-derived context to model attribute groups causally.
Using only anchor-wise linear layers and activations for context modeling, COSA-GS achieves state-of-the-art compression performance with fast decoding on Mip-NeRF360, DeepBlending, Tanks&Temples, and BungeeNeRF.

## Installation

The code is developed with Python 3.10 and PyTorch 2.9.1. Please refer to the official
[PyTorch installation guide](https://pytorch.org/get-started/locally/) if your
CUDA version or platform differs from the example below.

```bash
conda create -y -n py310torch291 python=3.10
conda activate py310torch291

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

Run the following commands from the project root:

```bash
git submodule update --init --recursive

pip install -r requirements.txt
pip install --no-build-isolation \
  ./submodules/diff-gaussian-rasterization \
  ./submodules/simple-knn

git clone https://github.com/NVIDIA/cutlass.git /your/path/to/cutlass
export CUTLASS_HOME=/your/path/to/cutlass
python scaffold_codec_int/int_nn_ops/setup.py build

# Optional: precompile extensions; otherwise they compile on first import.
python -c "from scaffold_codec import rans_coder"
python -c "from scaffold_codec import space_filling_curves"
```

## Data

Place COLMAP-format scenes under `datasets/`, or link that directory to your
dataset path. For example:

```text
datasets/
  mipnerf360/
    bicycle/
      images/
      sparse/0/
  deep_blending/
    drjohnson/
      images/
      sparse/0/
  tandt/
    train/
      images/
      sparse/0/
```

Dataset-specific training configurations are provided in `config/`.

## Training

First train a Scaffold-GS representation (15K iterations):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold.train \
  config/scaffold15k_mipnerf360.yaml \
  data.scene=bicycle \
  data.model_path=outputs/scaffold15k_mipnerf360/bicycle
```

Then perform rate-distortion optimization for compression:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.train \
  config/codec30k_mipnerf360.yaml \
  data.source_path=datasets/mipnerf360/bicycle \
  data.scaffold_model=outputs/scaffold15k_mipnerf360/bicycle/point_cloud/iteration_15000 \
  data.model_path=outputs/mipnerf360_30k_int/bicycle_lmb0.0006 \
  train.lambda_rate=0.0006
```

Change `train.lambda_rate` to select a rate-distortion trade-off. Training
automatically performs final compression, decompression, and evaluation,
saving `checkpoints/final.pt`, `scene.bin`, `results.json`, and `per_view.json`.

## Evaluation

Compress and evaluate a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.compress_and_evaluate \
  --config config/codec30k_mipnerf360.yaml \
  --source-path datasets/mipnerf360/bicycle \
  --checkpoint outputs/mipnerf360_30k_int/bicycle_lmb0.0006/checkpoints/final.pt \
  --output outputs/mipnerf360_30k_int/bicycle_lmb0.0006/eval \
  --save-images
```

To decode and evaluate an integer bitstream:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.decompress_and_evaluate \
  --config config/codec30k_mipnerf360.yaml \
  --source-path datasets/mipnerf360/bicycle \
  --bitstream outputs/mipnerf360_30k_int/bicycle_lmb0.0006/scene.bin \
  --output outputs/mipnerf360_30k_int/bicycle_lmb0.0006/decoded_eval \
  --save-images
```

Decoding runs one warmup by default (`--warmup 0` disables it). Warmup results
are saved under `warmup_1/`, etc.; the final results are saved directly to `--output`.
Decoded symbols, integer anchor coordinates, and reconstructed attributes are
saved as NumPy arrays in each result directory's `cross_platform_check/`.

## Acknowledgements

Our implementation builds on [Scaffold-GS](https://github.com/city-super/Scaffold-GS),
[HAC++](https://github.com/YihangChen-ee/HAC-plus),
and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
We thank their authors for making the code available.
