import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scaffold.scene import Scene
from scaffold_codec.compress_and_evaluate import backup_existing_output, evaluate_views
from .config import load_config
from .model import CompressedGaussianModel


def parse_args():
    parser = argparse.ArgumentParser(description="Decode an integer scene bitstream and evaluate test views.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-images", action="store_true", help="Save render/target PNG files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing result JSON files without backups.")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The Scaffold rasterizer requires CUDA.")
    cfg = load_config(args.config)
    bitstream = Path(args.bitstream).read_bytes()
    model = CompressedGaussianModel(cfg.model).cuda()
    attrs = model.decompress(bitstream)
    model.eval()

    data = SimpleNamespace(**vars(cfg.data))
    data.model_path = args.output
    scene = Scene(data, model, shuffle=False, initialize_model=False, save_metadata=False)
    cameras = scene.getTestCameras()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, per_view = evaluate_views(
        model, attrs, cameras, cfg.data.white_background, output, save_images=args.save_images)
    results = {
        **metrics,
        "actual_total_bytes": len(bitstream),
        "actual_total_MB": len(bitstream) / 1_000_000.0,
        "actual_total_MiB": len(bitstream) / (2 << 19),
        "anchors": int(model.anchor.shape[0]),
        **model.codec_times,
    }
    for filename, values in (("results.json", results), ("per_view.json", per_view)):
        path = output / filename
        if not args.overwrite:
            backup_existing_output(path)
        path.write_text(json.dumps({"decoded": values}, indent=2), encoding="utf-8")
    print(json.dumps({"decoded": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
