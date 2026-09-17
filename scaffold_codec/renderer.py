import math

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def generate_neural_gaussians(
    viewpoint_camera,
    model,
    attrs=None,
    visible_mask: torch.Tensor | None = None,
    is_training: bool = False,
):
    if visible_mask is None:
        visible_mask = torch.ones(
            model.anchor.shape[0],
            dtype=torch.bool,
            device=model.anchor.device,
        )

    if attrs is None:
        attrs = model.raw_attrs()
    feat, offsets, log_position_scaling, log_gaussian_scaling = attrs
    feat = feat[visible_mask]
    anchor = model.anchor[visible_mask]
    offsets = offsets[visible_mask]
    position_scaling = log_position_scaling[visible_mask].exp()
    gaussian_scaling = log_gaussian_scaling[visible_mask].exp()
    offset_mask = model.offset_mask()[visible_mask].reshape(-1, 1)

    view = anchor - viewpoint_camera.camera_center
    distance = view.norm(dim=1, keepdim=True)
    view = view / distance
    distance_context = distance / model.distance_scale
    feat_view = torch.cat((feat, view), dim=1)
    uses_distance = (
        model.cfg.add_opacity_dist
        or model.cfg.add_color_dist
        or model.cfg.add_cov_dist
    )
    feat_view_distance = (
        torch.cat((feat, view, distance_context), dim=1)
        if uses_distance else None)

    opacity_input = (
        feat_view_distance if model.cfg.add_opacity_dist else feat_view)
    neural_opacity = model.mlp_opacity(opacity_input).reshape(-1, 1)
    selection_mask = neural_opacity[:, 0] > 0.0
    if not is_training:
        selection_mask &= offset_mask[:, 0].bool()
    opacity = neural_opacity[selection_mask]
    if is_training:
        selected_offset_mask = offset_mask[selection_mask]
        opacity = opacity * selected_offset_mask

    color_input = feat_view_distance if model.cfg.add_color_dist else feat_view
    color = model.mlp_color(color_input).reshape(
        anchor.shape[0] * model.cfg.n_offsets, 3
    )
    covariance_input = (
        feat_view_distance if model.cfg.add_cov_dist else feat_view)
    scale_rotation = model.mlp_cov(covariance_input).reshape(
        anchor.shape[0] * model.cfg.n_offsets, 7
    )

    repeated = torch.cat((
        position_scaling, gaussian_scaling, anchor), 1
    ).repeat_interleave(model.cfg.n_offsets, 0)
    values = (repeated, color, scale_rotation, offsets.reshape(-1, 3))
    combined = torch.cat(values, dim=1)[selection_mask]
    (
        repeated_position_scaling,
        repeated_gaussian_scaling,
        repeated_anchor,
        color,
        scale_rotation,
        offsets,
    ) = combined.split((position_scaling.shape[1], 3, 3, 3, 7, 3), dim=1)

    scaling = repeated_gaussian_scaling * torch.sigmoid(
        scale_rotation[:, :3]
    )
    if is_training:
        scaling = scaling * selected_offset_mask
    rotation = F.normalize(scale_rotation[:, 3:7], dim=-1)
    xyz = repeated_anchor + offsets * repeated_position_scaling
    if is_training:
        return (
            xyz,
            color,
            opacity,
            scaling,
            rotation,
            neural_opacity,
            selection_mask,
        )
    return xyz, color, opacity, scaling, rotation


def raster_settings(viewpoint_camera, pipe, background, scaling_modifier):
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=background,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )


def render(
    viewpoint_camera,
    model,
    pipe,
    background: torch.Tensor,
    attrs=None,
    scaling_modifier: float = 1.0,
    visible_mask: torch.Tensor | None = None,
    retain_grad: bool = False,
):
    is_training = model.training
    generated = generate_neural_gaussians(
        viewpoint_camera,
        model,
        attrs,
        visible_mask,
        is_training=is_training,
    )
    if is_training:
        (
            xyz,
            color,
            opacity,
            scaling,
            rotation,
            neural_opacity,
            selection_mask,
        ) = generated
    else:
        xyz, color, opacity, scaling, rotation = generated

    screenspace_points = torch.zeros_like(
        xyz, requires_grad=True
    )
    if retain_grad:
        screenspace_points.retain_grad()

    rasterizer = GaussianRasterizer(
        raster_settings=raster_settings(
            viewpoint_camera, pipe, background, scaling_modifier
        )
    )
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rotation,
        cov3D_precomp=None,
    )
    result = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    if is_training:
        result.update(
            {
                "selection_mask": selection_mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
            }
        )
    return result


@torch.no_grad()
def prefilter_voxel(
    viewpoint_camera,
    model,
    pipe,
    background: torch.Tensor,
    attrs=None,
    scaling_modifier: float = 1.0,
) -> torch.Tensor:
    if attrs is None:
        attrs = model.raw_attrs()
    rasterizer = GaussianRasterizer(
        raster_settings=raster_settings(
            viewpoint_camera, pipe, background, scaling_modifier
        )
    )
    radii = rasterizer.visible_filter(
        means3D=model.anchor,
        scales=model.prefilter_scaling(attrs),
        rotations=F.normalize(model.rotation, dim=-1),
        cov3D_precomp=None,
    )
    return radii > 0
