"""Hierarchical backward by replaying the Python/Warp forward queues.

No per-pixel Gaussian history is stored. CUDA's front-to-back derivative uses
the saved final image/transmittance and the same blending sequence.
"""

from .dispatch import FrameBindings, kernel_arg, launch
from .settings import GaussianRasterizationSettings
from .types import PreprocessedGaussians, TileBins, RenderOutput, RenderGradients

from functools import cache

import torch
import warp as wp

from .backward_intrinsics import global_atomic_add
from .interop import current_stream, contiguous_features
from .render_adjoint import RenderAdjoint, _make_accumulate_ray_gradient
from .render_hierarchical_warp import INF, _pixel_type, _make_evaluation


@cache
def _make_blend_backward(
    channels: int, screen_grad: bool = False, absgrad: bool = False
) -> wp.Function:
    Pixel = _pixel_type(channels)
    feature_type = wp.types.vector(channels, wp.float32)
    accumulate_ray_gradient = _make_accumulate_ray_gradient(screen_grad, absgrad)

    @wp.func
    def blend_backward(pixel: Pixel, colors: wp.array(dtype=feature_type)):
        p = pixel
        p.head_count -= 1
        if p.active:
            alpha = p.head_alphas[0]
            next_t = p.transmittance * (1.0 - alpha)
            if next_t < 0.0001:
                p.active = False
            else:
                id = p.head_ids[0]
                rgb = colors[id]
                alpha_gradient = float(0.0)
                weight = alpha * p.transmittance
                for channel in range(wp.static(channels)):
                    ch = wp.static(channel)
                    p.color[ch] = p.color[ch] + rgb[ch] * alpha * p.transmittance
                    behind = (p.final_color[ch] - p.color[ch]) / next_t
                    alpha_gradient += (rgb[ch] - behind) * p.gradient[ch]
                    global_atomic_add(
                        p.adjoint.color_gradient,
                        id * wp.static(channels) + ch,
                        weight * p.gradient[ch],
                    )
                alpha_gradient *= p.transmittance
                alpha_gradient += (
                    -p.final_transmittance / (1.0 - alpha)
                ) * p.background_gradient
                accumulate_ray_gradient(
                    p.adjoint, id, p.x, p.y, p.head_gaussians[0], alpha_gradient
                )
                p.transmittance = next_t
                for index in range(1, 4):
                    j = wp.static(index)
                    p.head_depths[j - 1] = p.head_depths[j]
                    p.head_alphas[j - 1] = p.head_alphas[j]
                    p.head_ids[j - 1] = p.head_ids[j]
                    p.head_gaussians[j - 1] = p.head_gaussians[j]
                p.head_depths[3] = INF
        return p

    return blend_backward


@cache
def _make_backward(
    cull: bool, channels: int = 3, screen_grad: bool = False, absgrad: bool = False
) -> wp.Kernel:
    feature_type = wp.types.vector(channels, wp.float32)
    evaluate = _make_evaluation(
        cull, _make_blend_backward(channels, screen_grad, absgrad), True, channels
    )

    @wp.kernel(
        module="unique",
        enable_backward=False,
        launch_bounds=256,
        module_options={"max_unroll": 0},
    )
    def replay(
        width: int,
        height: int,
        ranges: wp.array(dtype=int),
        points: wp.array(dtype=int),
        colors: wp.array(dtype=feature_type),
        final_color: wp.array(dtype=float),
        final_t: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
        transforms: wp.array(dtype=wp.mat44),
        opacity: wp.array(dtype=float),
        background: wp.array(dtype=float),
        pixel_gradient: wp.array(dtype=float),
        transform_gradient: wp.array(dtype=float),
        opacity_gradient: wp.array(dtype=float),
        color_gradient: wp.array(dtype=float),
        screen_gradient: wp.array(dtype=float),
        screen_abs_gradient: wp.array(dtype=float),
    ):
        # Build the view on-device; host arrays use forward's direct descriptors.
        adjoint = RenderAdjoint()
        adjoint.transforms = transforms
        adjoint.opacity = opacity
        adjoint.background = background
        adjoint.pixel_gradient = pixel_gradient
        adjoint.transform_gradient = transform_gradient
        adjoint.opacity_gradient = opacity_gradient
        adjoint.color_gradient = color_gradient
        adjoint.screen_gradient = screen_gradient
        adjoint.screen_abs_gradient = screen_abs_gradient
        evaluate(
            wp.tid(),
            width,
            height,
            ranges,
            points,
            adjoint.transforms,
            adjoint.opacity,
            colors,
            adjoint.background,
            final_color,
            final_t,
            contributors,
            adjoint,
        )

    return replay


@torch.no_grad()
def backward_render(
    preprocessed: PreprocessedGaussians,
    bins: TileBins,
    output: RenderOutput,
    grad_color: torch.Tensor,
    raster_settings: GaussianRasterizationSettings,
    *,
    screen_grad: bool = False,
    absgrad: bool = False,
    _bindings: FrameBindings | None = None,
) -> RenderGradients:
    """Return dFeatures, dOpacity and transposed transform gradients.

    The legacy ``rgb`` gradient key has shape (N,C).
    The matrix gradient has CUDA's backward buffer layout, i.e. transpose it
    to obtain the ordinary derivative w.r.t. the forward row-major matrix.
    ``screen_grad`` adds (N,2) rigid footprint translation derivatives in pixel
    units; ``absgrad`` adds their componentwise absolute per-pixel sums. These
    optional statistics share the replay, without changing model gradients.
    """
    wp.init()
    if (
        not raster_settings.settings.eval_3D
        or int(raster_settings.settings.sort_settings.sort_mode) != 3
        or raster_settings.render_depth
    ):
        raise ValueError("backward requires eval_3D=True, HIER and color rendering")
    sizes = raster_settings.settings.sort_settings.queue_sizes
    if (sizes.tile_4x4, sizes.tile_2x2, sizes.per_pixel) != (64, 8, 4):
        raise ValueError("backward requires HIER 64/8/4")
    n = preprocessed["radii"].numel()
    device = preprocessed["radii"].device
    channels = preprocessed["rgb"].shape[1]
    if (
        grad_color.shape
        != (channels, raster_settings.image_height, raster_settings.image_width)
        or grad_color.device != device
        or grad_color.dtype != torch.float32
    ):
        raise ValueError("grad_color must be float32 (C,H,W) on the rendering device")
    # Disjoint contiguous views share one zero-initialized allocation.
    # All atomic destinations still start at zero, with one fill submission.
    if absgrad and not screen_grad:
        raise ValueError("absgrad requires screen_grad=True")
    screen_channels = 2 * (int(screen_grad) + int(absgrad))
    storage = torch.zeros(
        n * (17 + channels + screen_channels), device=device, dtype=torch.float32
    )
    opacity_end = n * (17 + channels)
    gradients: RenderGradients = dict(
        gauss2screen=storage[: n * 16].view(n, 4, 4),
        rgb=storage[n * 16 : n * (16 + channels)].view(n, channels),
        opacity=storage[n * (16 + channels) : opacity_end],
    )
    screen = storage[opacity_end : opacity_end + n * 2] if screen_grad else storage[:0]
    screen_abs = storage[opacity_end + n * 2 :] if absgrad else storage[:0]
    if screen_grad:
        gradients["means2D"] = screen.view(n, 2)
    if absgrad:
        gradients["means2D_abs"] = screen_abs.view(n, 2)
    if n == 0:
        return gradients
    transforms = preprocessed["gauss2screen"].contiguous()
    if transforms.data_ptr() % 16:
        transforms = transforms.clone()

    def arg(tensor, dtype=wp.float32):
        return kernel_arg(tensor.contiguous(), dtype, _bindings)

    width, height = raster_settings.image_width, raster_settings.image_height
    kernel = _make_backward(
        bool(raster_settings.settings.culling_settings.hierarchical_4x4_culling),
        channels,
        screen_grad,
        absgrad,
    )
    launch(
        kernel,
        dim=256 * ((width + 15) // 16) * ((height + 15) // 16),
        inputs=[
            width,
            height,
            kernel_arg(bins["ranges"].view(-1), wp.int32, _bindings),
            kernel_arg(bins["point_list"], wp.int32, _bindings),
            arg(
                contiguous_features(preprocessed["rgb"]),
                wp.types.vector(channels, wp.float32),
            ),
            kernel_arg(output["color"].view(-1), wp.float32, _bindings),
            kernel_arg(output["final_T"].view(-1), wp.float32, _bindings),
            kernel_arg(output["contributors"].view(-1), wp.int32, _bindings),
            arg(transforms, wp.mat44),
            arg(preprocessed["opacity"]),
            arg(raster_settings.bg),
            arg(grad_color.contiguous().view(-1)),
        ],
        outputs=[
            arg(gradients["gauss2screen"].view(-1)),
            arg(gradients["opacity"]),
            arg(gradients["rgb"].view(-1)),
            arg(screen),
            arg(screen_abs),
        ],
        block_dim=256,
        stream=current_stream(device, bindings=_bindings),
        bindings=_bindings,
    )
    return gradients
