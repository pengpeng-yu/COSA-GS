import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import lpips
import numpy as np
import torch
from tqdm import tqdm

from .config import load_config
from .model import GaussianModel
from .renderer import prefilter_voxel, render
from .scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


@torch.no_grad()
def evaluate(cameras, model, pipe, background, lpips_model):
    model.eval()
    float_psnr = []
    quantized_psnr = []
    quantized_ssim = []
    quantized_lpips = []
    per_view = {}
    for camera in cameras:
        visible = prefilter_voxel(camera, model, pipe, background)
        image = render(camera, model, pipe, background,
                       visible_mask=visible)["render"].clamp(0.0, 1.0)
        target = camera.original_image.to(image.device)
        quantized = (image * 255).round() / 255
        float_value = psnr(image, target).mean()
        quantized_value = psnr(quantized, target).mean()
        ssim_value = ssim(quantized, target)
        lpips_value = lpips_model(quantized.unsqueeze(0), target.unsqueeze(0)).mean()
        float_psnr.append(float_value)
        quantized_psnr.append(quantized_value)
        quantized_ssim.append(ssim_value)
        quantized_lpips.append(lpips_value)
        per_view[camera.image_name] = {
            "PSNR": quantized_value.item(),
            "SSIM": ssim_value.item(),
            "LPIPS": lpips_value.item(),
        }
    model.train()
    return {
        "test_psnr": torch.stack(quantized_psnr).mean().item(),
        "test_psnr_float": torch.stack(float_psnr).mean().item(),
        "test_ssim": torch.stack(quantized_ssim).mean().item(),
        "test_lpips": torch.stack(quantized_lpips).mean().item(),
        "test_views": len(cameras),
    }, per_view


def main():
    cfg = load_config(sys.argv[1], sys.argv[2:])
    scene_name = cfg.data.scene
    data_root = Path(cfg.data.root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(cfg.data.model_path or Path(cfg.data.output_root) / f"{scene_name}_{stamp}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data = SimpleNamespace(
        source_path=str(data_root / scene_name),
        model_path=str(output),
        images=cfg.data.images,
        resolution=cfg.data.resolution,
        data_device=cfg.data.data_device,
        eval=cfg.data.eval,
        lod=cfg.data.lod,
        white_background=cfg.data.white_background,
    )
    cfg.data.root = str(data_root)
    cfg.data.model_path = str(output)
    (output / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(output / "train.log"),
                  logging.StreamHandler()],
    )
    lpips_model = lpips.LPIPS(net="vgg").cuda().eval()
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)

    model = GaussianModel(
        cfg.model.anchor_feat_channels,
        cfg.model.n_offsets,
        cfg.model.physical_voxel_size,
        cfg.model.anchor_grid_size,
        cfg.model.grow_physical_size_factors,
        cfg.model.grow_grid_size_factors,
        cfg.model.grow_threshold_factors,
        cfg.model.use_feat_bank,
        cfg.model.appearance_dim,
        cfg.model.point_ratio,
        cfg.model.add_opacity_dist,
        cfg.model.add_cov_dist,
        cfg.model.add_color_dist,
    )
    scene = Scene(data, model)
    model.training_setup(cfg.train)
    model.train()

    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if data.white_background else [0.0, 0.0, 0.0],
        device="cuda")
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    camera_stack = []
    metrics_path = output / "metrics.jsonl"
    ema_loss = 0.0
    final_metrics = None
    final_per_view = None
    progress = tqdm(range(1, cfg.train.iterations + 1), desc="Training")

    torch.cuda.synchronize(model.get_anchor.device)
    train_start = time.perf_counter()
    for iteration in progress:
        model.update_learning_rate(iteration)
        if not camera_stack:
            camera_stack = train_cameras.copy()
        camera = camera_stack.pop(random.randrange(len(camera_stack)))
        visible = prefilter_voxel(camera, model, pipe, background)
        collect_stats = cfg.train.start_stat < iteration < cfg.train.update_until
        result = render(camera, model, pipe, background,
                        visible_mask=visible, retain_grad=collect_stats)
        image = result["render"]
        target = camera.original_image.to(image.device)
        l1 = l1_loss(image, target)
        dssim = 1.0 - ssim(image, target)
        scaling_reg = result["scaling"].prod(1).mean()
        loss = ((1.0 - cfg.train.lambda_dssim) * l1
                + cfg.train.lambda_dssim * dssim
                + cfg.train.scaling_regularization * scaling_reg)
        loss.backward()
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

        if iteration in cfg.train.test_iterations or iteration == cfg.train.iterations:
            if iteration == cfg.train.iterations:
                torch.cuda.synchronize(model.get_anchor.device)
                final_test_start = time.perf_counter()
            torch.cuda.empty_cache()
            test_metrics, per_view = evaluate(
                test_cameras, model, pipe, background, lpips_model)
            torch.cuda.empty_cache()
            test_metrics.update({
                "iteration": iteration,
                "anchors": model.get_anchor.shape[0],
            })
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(test_metrics) + "\n")
            logging.info(
                "[%d] test PSNR %.4f dB, float PSNR %.4f dB, LPIPS %.6f, anchors %d",
                iteration, test_metrics["test_psnr"],
                test_metrics["test_psnr_float"], test_metrics["test_lpips"],
                test_metrics["anchors"])
            if iteration == cfg.train.iterations:
                final_metrics = test_metrics
                final_per_view = per_view
                torch.cuda.synchronize(model.get_anchor.device)
                final_test_seconds = time.perf_counter() - final_test_start

        with torch.no_grad():
            if collect_stats:
                model.training_statis(
                    result["viewspace_points"], result["neural_opacity"],
                    result["visibility_filter"], result["selection_mask"],
                    visible)
                if (iteration > cfg.train.update_from
                        and iteration % cfg.train.update_interval == 0):
                    model.adjust_anchor(
                        check_interval=cfg.train.update_interval,
                        success_threshold=cfg.train.success_threshold,
                        grad_threshold=cfg.train.densify_grad_threshold,
                        min_opacity=cfg.train.min_opacity)
            elif iteration == cfg.train.update_until:
                del model.opacity_accum
                del model.offset_gradient_accum
                del model.offset_denom
                del model.anchor_demon
                torch.cuda.empty_cache()

            if iteration < cfg.train.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

        if iteration % cfg.train.log_interval == 0:
            progress.set_postfix(loss=f"{ema_loss:.7f}",
                                 anchors=model.get_anchor.shape[0])

    torch.cuda.synchronize(model.get_anchor.device)
    train_seconds = time.perf_counter() - train_start - final_test_seconds
    logging.info("Training completed in %.3f seconds", train_seconds)

    scene.save(cfg.train.iterations, model)
    if cfg.model.physical_voxel_size == 0:
        torch.save({
            "physical_voxel_size": model.physical_voxel_size,
            "anchor_grid_size": model.anchor_grid_size,
        }, output / "point_cloud" / f"iteration_{cfg.train.iterations}" / "extra_params.pt")
    key = f"ours_{cfg.train.iterations}"
    with (output / "results.json").open("w", encoding="utf-8") as file:
        json.dump({key: {
            "SSIM": final_metrics["test_ssim"],
            "PSNR": final_metrics["test_psnr"],
            "LPIPS": final_metrics["test_lpips"],
            "anchors": final_metrics["anchors"],
            "train_seconds": train_seconds,
        }}, file, indent=2)
    with (output / "per_view.json").open("w", encoding="utf-8") as file:
        json.dump({key: final_per_view}, file, indent=2)
    logging.info("Saved Scaffold-GS scene to %s", output)


if __name__ == "__main__":
    main()
