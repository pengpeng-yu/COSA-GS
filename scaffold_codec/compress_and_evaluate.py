import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import lpips
import torch
from torchvision.utils import save_image

from . import CompressedGaussianModel, load_config
from .renderer import prefilter_voxel, render
from scaffold.scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress, decompress, and render a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-path", help="Override data.source_path in the config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup encode-decode evaluations saved to warmup_N/ (default: 1).")
    parser.add_argument("--save-images", action="store_true",
                        help="Save render/target PNG files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing bitstream and result JSON files without backups.")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


def backup_existing_output(path):
    if not path.exists():
        return
    created = int(subprocess.check_output(["stat", "--format=%W", str(path)], text=True).strip())
    assert created > 0, f"Creation time unavailable for {path}"
    backup = path.with_name(f"{path.stem}_{created}{path.suffix}")
    assert not backup.exists(), f"Backup already exists: {backup}"
    path.rename(backup)
    print(f"Renamed existing file: {path} -> {backup}", flush=True)


@torch.no_grad()
def evaluate_views(model, attrs, cameras, white_background, output, save_images=False):
    pipe = SimpleNamespace(debug=False)
    background_color = [1.0, 1.0, 1.0] if white_background else [0.0] * 3
    background = torch.tensor(background_color, device="cuda")
    lpips_model = lpips.LPIPS(net="vgg").cuda().eval()
    if save_images:
        (output / "render").mkdir(parents=True, exist_ok=True)
        (output / "target").mkdir(parents=True, exist_ok=True)
    l1 = []
    quantized_psnr = []
    quantized_ssim = []
    quantized_lpips = []
    per_view = {}
    for index, camera in enumerate(cameras):
        visible = prefilter_voxel(camera, model, pipe, background, attrs)
        prediction = render(camera, model, pipe, background,
                            attrs=attrs,
                            visible_mask=visible)["render"].clamp(0.0, 1.0)
        target = camera.original_image.to(prediction.device)
        quantized_prediction = (prediction * 255).round() / 255
        if save_images:
            filename = f"{index:05d}_{camera.image_name}.png"
            save_image(quantized_prediction, output / "render" / filename)
            save_image(target, output / "target" / filename)

        l1.append(l1_loss(prediction, target))
        psnr_value = psnr(quantized_prediction, target).mean()
        ssim_value = ssim(quantized_prediction, target)
        lpips_value = lpips_model(
            quantized_prediction.unsqueeze(0), target.unsqueeze(0)).mean()
        quantized_psnr.append(psnr_value)
        quantized_ssim.append(ssim_value)
        quantized_lpips.append(lpips_value)
        per_view[camera.image_name] = {
            "PSNR": psnr_value.item(),
            "SSIM": ssim_value.item(),
            "LPIPS": lpips_value.item(),
        }

    return {
        "PSNR": torch.stack(quantized_psnr).mean().item(),
        "SSIM": torch.stack(quantized_ssim).mean().item(),
        "LPIPS": torch.stack(quantized_lpips).mean().item(),
        "L1": torch.stack(l1).mean().item(),
    }, per_view


@torch.no_grad()
def main(model_class=CompressedGaussianModel, config_loader=load_config, args=None):
    if args is None:
        args = parse_args()
    output = Path(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("The Scaffold rasterizer requires CUDA.")
    cfg = config_loader(args.config)
    if args.source_path is not None:
        cfg.data.source_path = args.source_path
    model = model_class(cfg.model).cuda()
    data = SimpleNamespace(**vars(cfg.data))
    data.model_path = args.output
    scene = Scene(data, model, shuffle=False, initialize_model=False)
    model.distance_scale = scene.cameras_extent
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.restore_model_state(checkpoint)
    model.eval()
    cameras = scene.getTestCameras()
    key = f"ours_{checkpoint['iteration']}"
    train_seconds = checkpoint.get("train_seconds")
    previous_results_path = output / "results.json"
    if train_seconds is None and previous_results_path.exists():
        previous_results = json.loads(previous_results_path.read_text(encoding="utf-8"))
        train_seconds = previous_results.get(key, {}).get("train_seconds")
    del checkpoint
    coded_parameters = tuple(parameter for _, parameter in model.coded_model_parameters())
    coded_parameter_state = tuple(parameter.clone() for parameter in coded_parameters)

    for run in range(args.warmup + 1):
        output = Path(args.output)
        if run < args.warmup:
            output = output / f"warmup_{run + 1}"
        output.mkdir(parents=True, exist_ok=True)
        bitstream = model.compress()
        codec_times = dict(model.codec_times)
        stream_path = output / "scene.bin"
        if not args.overwrite:
            backup_existing_output(stream_path)
        stream_path.write_bytes(bitstream)

        decompressed_model = model_class(cfg.model).cuda()
        decompressed_model.decompress(bitstream)
        codec_times.update(decompressed_model.codec_times)
        # Floating-point compression quantizes network parameters in place.
        for parameter, value in zip(coded_parameters, coded_parameter_state):
            parameter.copy_(value)
        decompressed_model.eval()
        attrs = decompressed_model.raw_attrs()
        metrics, per_view = evaluate_views(
            decompressed_model, attrs, cameras, cfg.data.white_background, output,
            save_images=args.save_images)

        results = {
            **metrics,
            "actual_total_bytes": len(bitstream),
            "actual_total_MB": len(bitstream) / 1_000_000.0,
            "actual_total_MiB": len(bitstream) / (2 << 19),
            "anchors": int(decompressed_model.anchor.shape[0]),
            **codec_times,
        }
        if train_seconds is not None:
            results["train_seconds"] = train_seconds
        for filename, values in (("results.json", results), ("per_view.json", per_view)):
            path = output / filename
            if not args.overwrite:
                backup_existing_output(path)
            path.write_text(json.dumps({key: values}, indent=2), encoding="utf-8")
        print(output, flush=True)
        print(json.dumps({key: results}, indent=2), flush=True)
        del attrs, decompressed_model


if __name__ == "__main__":
    main()
