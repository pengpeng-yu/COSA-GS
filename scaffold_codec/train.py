import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import lpips
import numpy as np
import torch
from tqdm import tqdm

from . import CompressedGaussianModel, load_config
from .renderer import prefilter_voxel, render
from scaffold.scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


BITS_PER_MB = 8_000_000.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model,
    cameras,
    pipe,
    background,
    lpips_model,
    max_views: int | None = None,
):
    was_training = model.training
    coded_parameters = tuple(parameter for _, parameter in model.coded_model_parameters())
    coded_parameter_state = tuple(parameter.clone() for parameter in coded_parameters)
    bitstream = model.compress()
    decoded_model = type(model)(model.cfg).to(model.anchor.device)
    decoded_model.decompress(bitstream)
    for parameter, value in zip(coded_parameters, coded_parameter_state):
        parameter.copy_(value)
    model.train(was_training)
    del coded_parameters, coded_parameter_state

    codec_times = {**model.codec_times, **decoded_model.codec_times}
    if not cameras:
        return codec_times, {}, decoded_model, bitstream

    attrs = decoded_model.raw_attrs()
    l1 = []
    float_psnr = []
    quantized_psnr = []
    quantized_ssim = []
    quantized_lpips = []
    image_names = []
    selected_cameras = cameras if max_views is None else cameras[:max_views]
    for camera in selected_cameras:
        visible = prefilter_voxel(
            camera, decoded_model, pipe, background, attrs)
        prediction = render(camera, decoded_model, pipe, background,
                            attrs=attrs,
                            visible_mask=visible)["render"].clamp(0.0, 1.0)
        target = camera.original_image.to(prediction.device)
        quantized_prediction = (prediction * 255).round() / 255
        l1.append(l1_loss(prediction, target))
        float_psnr.append(psnr(prediction, target).mean())
        quantized_psnr.append(psnr(quantized_prediction, target).mean())
        quantized_ssim.append(ssim(quantized_prediction, target))
        quantized_lpips.append(
            lpips_model(quantized_prediction.unsqueeze(0), target.unsqueeze(0)).mean())
        image_names.append(camera.image_name)
    l1 = torch.stack(l1)
    float_psnr = torch.stack(float_psnr)
    quantized_psnr = torch.stack(quantized_psnr)
    quantized_ssim = torch.stack(quantized_ssim)
    quantized_lpips = torch.stack(quantized_lpips)
    metrics = {
        "test_psnr": quantized_psnr.mean().item(),
        "test_psnr_float": float_psnr.mean().item(),
        "test_l1": l1.mean().item(),
        "test_ssim": quantized_ssim.mean().item(),
        "test_lpips": quantized_lpips.mean().item(),
        "evaluated_test_views": len(selected_cameras),
        "actual_total_bytes": len(bitstream),
        "actual_total_MB": len(bitstream) / 1_000_000.0,
        "actual_total_MiB": len(bitstream) / (2 << 19),
        "anchors": int(decoded_model.anchor.shape[0]),
        **codec_times,
    }
    per_view = {
        name: {"PSNR": psnr_value, "SSIM": ssim_value, "LPIPS": lpips_value}
        for name, psnr_value, ssim_value, lpips_value in zip(
            image_names,
            quantized_psnr.tolist(),
            quantized_ssim.tolist(),
            quantized_lpips.tolist(),
        )
    }
    return metrics, per_view, decoded_model, bitstream


def save_json_line(path: Path, metrics: dict):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")


def load_resume(model, checkpoint_path, train_cfg):
    checkpoint = torch.load(checkpoint_path, map_location=model.anchor.device,
                            weights_only=False)
    model.restore_model_state(checkpoint)
    model.offset_mask_logits.requires_grad_(
        checkpoint["iteration"] < train_cfg.offset_mask_prune_iterations[-1])
    model.training_setup(train_cfg)
    model.optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["iteration"])


def rate_weight(lambda_rate: float, rd_iteration: int, warmup_steps: int) -> float:
    if warmup_steps == 0:
        return lambda_rate
    return lambda_rate * min(rd_iteration / warmup_steps, 1.0)


def main(model_class=CompressedGaussianModel, config_loader=load_config):
    cfg = config_loader(sys.argv[1], sys.argv[2:])
    prune_iterations = cfg.train.offset_mask_prune_iterations
    preserve_optimizer_modes = cfg.train.preserve_optimizer_after_mask_pruning
    if len(prune_iterations) != len(preserve_optimizer_modes):
        assert len(preserve_optimizer_modes) == 1
        preserve_optimizer_modes = preserve_optimizer_modes * len(prune_iterations)
    output = Path(cfg.data.model_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")

    set_seed(cfg.train.seed)
    model = model_class(cfg.model).cuda()
    scene = Scene(
        SimpleNamespace(**vars(cfg.data)), model, shuffle=False,
        initialize_model=False)
    model.distance_scale = scene.cameras_extent
    if cfg.train.resume:
        start_iteration = load_resume(model, cfg.train.resume, cfg.train)
    else:
        model.load_scaffold_model(cfg.data.scaffold_model)
        model.training_setup(cfg.train)
        start_iteration = 0
    lpips_model = lpips.LPIPS(net="vgg").cuda().eval()
    model.train()
    set_seed(cfg.train.seed)

    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    pipe = SimpleNamespace(debug=False)
    background_color = [1.0, 1.0, 1.0] if cfg.data.white_background else [0.0] * 3
    background = torch.tensor(background_color, device="cuda")
    metrics_path = output / "metrics.jsonl"
    if not cfg.train.resume and metrics_path.exists():
        metrics_path.unlink()
    camera_stack = []
    ema_loss = torch.zeros((), device="cuda")
    model_bits = model.coded_model_bits()
    end_iteration = min(cfg.train.stop_iteration or cfg.train.iterations, cfg.train.iterations)
    progress = tqdm(range(start_iteration + 1, end_iteration + 1), desc="Training")
    prune_options = dict(zip(
        prune_iterations,
        preserve_optimizer_modes,
    ))
    final_prune_iteration = prune_iterations[-1]

    previous_train_seconds = model.train_seconds
    torch.cuda.synchronize(model.anchor.device)
    train_start = time.perf_counter()
    for iteration in progress:
        iteration_start = time.perf_counter()
        periodic_log = iteration % cfg.train.log_interval == 0
        should_log = iteration == start_iteration + 1 or periodic_log
        if iteration in prune_options:
            final_prune = iteration == final_prune_iteration
            preserve_optimizer = prune_options[iteration]
            mask_stats = model.apply_offset_mask(
                finalize=final_prune,
                preserve_optimizer=preserve_optimizer,
            )
            if preserve_optimizer == 0:
                model.training_setup(cfg.train)
            mask_stats["iteration"] = iteration
            text = json.dumps(mask_stats, indent=2)
            path = output / ("offset_mask_pruning.json" if final_prune
                             else f"offset_mask_pruning_{iteration}.json")
            path.write_text(text, encoding="utf-8")
            print(json.dumps({"offset_mask_pruning": mask_stats}, indent=2))
        rd_iteration = max(iteration - cfg.train.rd_start_iteration, 0)
        model.update_learning_rate(iteration, rd_iteration)
        if not camera_stack:
            camera_stack = train_cameras.copy()
        camera = camera_stack.pop(random.randrange(len(camera_stack)))

        anchor_latent_rate_weight = rate_weight(
            cfg.train.lambda_rate, rd_iteration,
            cfg.train.anchor_latent_rate_warmup_steps)
        geom_attr_rate_weight = rate_weight(
            cfg.train.lambda_rate, rd_iteration,
            cfg.train.geom_attr_rate_warmup_steps)
        feat_rate_weight = rate_weight(
            cfg.train.lambda_rate, rd_iteration,
            cfg.train.feat_rate_warmup_steps)
        rate_summary, attr_quantization = model(rd_iteration)

        geom_attr_bits = (
            rate_summary.offset_bits
            + rate_summary.position_scaling_bits
            + rate_summary.gaussian_scaling_bits
        )
        rate_loss = (
            anchor_latent_rate_weight * rate_summary.anchor_latent_bits
            + geom_attr_rate_weight * geom_attr_bits
            + feat_rate_weight * rate_summary.feat_bits
        ) / BITS_PER_MB
        rate_loss.backward()
        detached_rate_loss = rate_loss.detach()

        if should_log:
            with torch.no_grad():
                stacked_rate_summary = torch.stack((
                    rate_summary.geom_bits,
                    rate_summary.anchor_latent_bits,
                    rate_summary.offset_mask_bits,
                    rate_summary.offset_bits,
                    rate_summary.position_scaling_bits,
                    rate_summary.gaussian_scaling_bits,
                    rate_summary.feat_bits,
                ))
                pred_total_bits = stacked_rate_summary.sum().item() + model_bits
                (
                    pred_geom_bits,
                    pred_anchor_latent_bits,
                    pred_offset_mask_bits,
                    pred_offset_bits,
                    pred_position_scaling_bits,
                    pred_gaussian_scaling_bits,
                    pred_feat_bits,
                ) = stacked_rate_summary.tolist()
                del stacked_rate_summary
        del rate_summary, geom_attr_bits, rate_loss

        attrs = model.training_attrs(rd_iteration, attr_quantization)
        del attr_quantization
        visible_mask = prefilter_voxel(camera, model, pipe, background, attrs)
        render_result = render(
            camera, model, pipe, background,
            attrs=attrs,
            visible_mask=visible_mask,
        )
        prediction = render_result["render"]
        target = camera.original_image.to(prediction.device)
        l1 = l1_loss(prediction, target)
        dssim = 1.0 - ssim(prediction, target)
        gaussian_scaling = render_result["scaling"]
        gaussian_scaling_reg = (
            (
                gaussian_scaling[:, 0]
                * gaussian_scaling[:, 1]
                * gaussian_scaling[:, 2]
            ).mean()
            if gaussian_scaling.shape[0]
            else prediction.new_zeros(())
        )
        distortion = (
            (1.0 - cfg.train.lambda_dssim) * l1
            + cfg.train.lambda_dssim * dssim
            + cfg.train.scaling_regularization * gaussian_scaling_reg
        )
        distortion.backward()

        if should_log:
            with torch.no_grad():
                train_psnr = psnr(prediction, target).mean().item()
        del attrs, visible_mask, render_result, prediction, target, gaussian_scaling

        loss = distortion.detach() + detached_rate_loss
        ema_loss = 0.4 * loss + 0.6 * ema_loss

        coding_scale_parameters = (
            model.log_feat_coding_scale,
            model.log_offset_coding_scale,
            model.log_position_scaling_coding_scale,
            model.log_gaussian_scaling_coding_scale,
            model.log_anchor_latent_coding_scale,
        )
        max_norm = cfg.train.coding_scale_grad_clip or float("inf")
        coding_scale_grad_norm = torch.nn.utils.clip_grad_norm_(
            coding_scale_parameters, max_norm, error_if_nonfinite=True)
        codec_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.codec_parameters,
            cfg.train.codec_grad_clip,
            error_if_nonfinite=True,
        )

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        if should_log:
            rate_loss_value = detached_rate_loss.item()
            distortion_value = distortion.item()
            loss_value = loss.item()
            ema_loss_value = ema_loss.item()
            codec_grad_norm_value = codec_grad_norm.item()

            if rd_iteration == 0:
                position_scaling_mode = "raw"
            elif rd_iteration >= cfg.train.position_scaling_ste_start:
                position_scaling_mode = "ste"
            else:
                position_scaling_mode = "noise"

            offset_mask_lr = 0.0 if model.hard_offset_mask is not None else next(
                group["lr"] for group in model.optimizer.param_groups
                if group["name"] == "offset_mask")
            codec_lr = next(
                group["lr"] for group in model.optimizer.param_groups
                if group["name"] == "codec")
            anchor_latent_lr = next(
                group["lr"] for group in model.optimizer.param_groups
                if group["name"] == "anchor_latent")

            training_metrics = {
                "iteration": iteration,
                "rd_iteration": rd_iteration,
                "loss": loss_value,
                "ema_loss": ema_loss_value,
                "distortion_loss": distortion_value,
                "l1": l1.item(),
                "dssim": dssim.item(),
                "train_psnr": train_psnr,
                "rate_loss": rate_loss_value,
                "anchor_latent_rate_weight": anchor_latent_rate_weight,
                "geom_attr_rate_weight": geom_attr_rate_weight,
                "feat_rate_weight": feat_rate_weight,
                "position_scaling_mode": position_scaling_mode,
                "pred_total_MB": pred_total_bits / BITS_PER_MB,
                "model_MB": model_bits / BITS_PER_MB,
                "geom_MB": pred_geom_bits / BITS_PER_MB,
                "pred_anchor_latent_MB": pred_anchor_latent_bits / BITS_PER_MB,
                "pred_offset_mask_MB": pred_offset_mask_bits / BITS_PER_MB,
                "pred_position_scaling_MB": pred_position_scaling_bits / BITS_PER_MB,
                "pred_offset_MB": pred_offset_bits / BITS_PER_MB,
                "pred_gaussian_scaling_MB": pred_gaussian_scaling_bits / BITS_PER_MB,
                "pred_feat_MB": pred_feat_bits / BITS_PER_MB,
                "offset_coding_scale": (model.log_offset_coding_scale.exp() + 1e-8).item(),
                "position_scaling_coding_scale": (model.log_position_scaling_coding_scale.exp() + 1e-8).item(),
                "gaussian_scaling_coding_scale": (model.log_gaussian_scaling_coding_scale.exp() + 1e-8).item(),
                "feat_coding_scale": (model.log_feat_coding_scale.exp() + 1e-8).item(),
                "anchor_latent_coding_scale": (model.log_anchor_latent_coding_scale.exp() + 1e-8).item(),
                "codec_lr": codec_lr,
                "anchor_latent_lr": anchor_latent_lr,
                "offset_mask_lr": offset_mask_lr,
                "coding_scale_grad_norm": coding_scale_grad_norm.item(),
                "codec_grad_norm": codec_grad_norm_value,
                "codec_grad_clipped": codec_grad_norm_value > cfg.train.codec_grad_clip,
                "anchors": int(model.anchor.shape[0]),
                "iteration_seconds": time.perf_counter() - iteration_start,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
            save_json_line(metrics_path, training_metrics)
            progress.set_postfix({
                "loss": f"{loss_value:.5f}",
                "EMA": f"{ema_loss_value:.5f}",
                "D": f"{distortion_value:.5f}",
                "R": f"{rate_loss_value:.6f}",
                "PSNR": f"{train_psnr:.2f}",
                "MB": f"{training_metrics['pred_total_MB']:.3f}",
                "anchors": training_metrics["anchors"],
            })

        if cfg.train.eval_interval > 0 and \
            iteration % cfg.train.eval_interval == 0 and \
                iteration != end_iteration:
            torch.cuda.empty_cache()
            periodic_test_metrics, _, decoded_model, bitstream = evaluate(
                model, test_cameras, pipe, background, lpips_model,
                cfg.train.periodic_test_views)
            periodic_test_metrics["iteration"] = iteration
            del decoded_model, bitstream
            torch.cuda.empty_cache()
            save_json_line(metrics_path, periodic_test_metrics)

        save_checkpoint = (cfg.train.checkpoint_interval > 0
                           and iteration % cfg.train.checkpoint_interval == 0
                           and iteration < end_iteration)
        if save_checkpoint:
            path = output / "checkpoints" / f"iteration_{iteration}.pt"
            torch.cuda.synchronize(model.anchor.device)
            if previous_train_seconds is not None:
                model.train_seconds = previous_train_seconds + time.perf_counter() - train_start
            model.save_checkpoint(path, iteration)

    torch.cuda.synchronize(model.anchor.device)
    if previous_train_seconds is not None:
        model.train_seconds = previous_train_seconds + time.perf_counter() - train_start
    final_checkpoint = output / "checkpoints" / "final.pt"
    model.save_checkpoint(final_checkpoint, end_iteration)
    final_metrics, per_view, decoded_model, bitstream = evaluate(
        model, test_cameras, pipe, background, lpips_model)
    if model.train_seconds is not None:
        final_metrics["train_seconds"] = model.train_seconds
    stream_path = output / "scene.bin"
    stream_path.write_bytes(bitstream)
    del model, bitstream
    decoded_model.save_ply(output / "point_cloud" / "decoded.ply")
    final_eval = {
        "iteration": end_iteration,
        **final_metrics,
    }
    save_json_line(metrics_path, final_eval)
    key = f"ours_{end_iteration}"
    results = {
        "PSNR": final_eval["test_psnr"],
        "SSIM": final_eval["test_ssim"],
        "LPIPS": final_eval["test_lpips"],
        "L1": final_eval["test_l1"],
        "actual_total_bytes": final_eval["actual_total_bytes"],
        "actual_total_MB": final_eval["actual_total_MB"],
        "actual_total_MiB": final_eval["actual_total_MiB"],
        "anchors": final_eval["anchors"],
        **{name: seconds for name, seconds in final_metrics.items() if name.endswith("_seconds")},
    }
    (output / "results.json").write_text(
        json.dumps({key: results}, indent=2), encoding="utf-8")
    (output / "per_view.json").write_text(
        json.dumps({key: per_view}, indent=2), encoding="utf-8")
    print(json.dumps(final_eval, indent=2))


if __name__ == "__main__":
    main()
