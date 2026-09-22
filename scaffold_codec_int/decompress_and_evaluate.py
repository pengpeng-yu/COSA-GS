import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scaffold.scene import Scene
from scaffold_codec.compress_and_evaluate import backup_existing_output, evaluate_views
from .config import load_config
from .model import CompressedGaussianModel


def parse_args():
    parser = argparse.ArgumentParser(description="Decode an integer scene bitstream and evaluate test views.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-path", help="Override data.source_path in the config.")
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup decodes saved to warmup_N/ (default: 1).")
    parser.add_argument("--save-images", action="store_true", help="Save render/target PNG files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing result JSON files without backups.")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The Scaffold rasterizer requires CUDA.")
    cfg = load_config(args.config)
    if args.source_path is not None:
        cfg.data.source_path = args.source_path
    bitstream = Path(args.bitstream).read_bytes()
    model = CompressedGaussianModel(cfg.model).cuda()
    data = SimpleNamespace(**vars(cfg.data))
    data.model_path = args.output

    for run in range(args.warmup + 1):
        output = Path(args.output)
        if run < args.warmup:
            output = output / f"warmup_{run + 1}"
        decoded = model.decompress(bitstream)
        model.eval()

        if run == 0:
            scene = Scene(data, model, shuffle=False, initialize_model=False, save_metadata=False)
            cameras = scene.getTestCameras()

        output.mkdir(parents=True, exist_ok=True)
        check_output = output / "cross_platform_check"
        check_output.mkdir(exist_ok=True)
        for name, value in decoded.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                np.save(check_output / f"{name}.npy", value, allow_pickle=False)
        metrics, per_view = evaluate_views(
            model, model.raw_attrs(), cameras, cfg.data.white_background, output, save_images=args.save_images)
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
        print(output, flush=True)
        print(json.dumps({"decoded": results}, indent=2), flush=True)
        del decoded


if __name__ == "__main__":
    main()
