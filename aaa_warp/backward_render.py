"""Hierarchical backward by replaying the Python/Warp forward queues.

No per-pixel Gaussian history is stored. CUDA's front-to-back derivative uses
the saved final image/transmittance and the same blending sequence.
"""

import torch
import warp as wp

from .backward_intrinsics import global_atomic_add
from .dispatch import kernel_arg, launch
from .interop import current_stream
from .render_adjoint import RenderAdjoint, accumulate_ray_gradient
from .render_hierarchical_warp import Pixel, INF, _make_evaluation


@wp.func
def blend_backward(pixel: Pixel, colors: wp.array(dtype=wp.vec3)):
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
            for channel in range(3):
                ch = wp.static(channel)
                p.color[ch] = p.color[ch] + rgb[ch] * alpha * p.transmittance
                behind = (p.final_color[ch] - p.color[ch]) / next_t
                alpha_gradient += (rgb[ch] - behind) * p.gradient[ch]
                global_atomic_add(
                    p.adjoint.color_gradient, id * 3 + ch, weight * p.gradient[ch]
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


def _make_backward(cull):
    evaluate = _make_evaluation(cull, blend_backward, True)

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
        colors: wp.array(dtype=wp.vec3),
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


_backward_cull = _make_backward(True)
_backward_unculled = _make_backward(False)


@torch.no_grad()
def backward_render(
    preprocessed, bins, output, grad_color, raster_settings, *, _bindings=None
):
    """Return dRGB, dOpacity and CUDA-layout (transposed) transform gradients.

    The matrix gradient has CUDA's backward buffer layout, i.e. transpose it
    to obtain the ordinary derivative w.r.t. the forward row-major matrix.
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
    if (
        grad_color.shape
        != (3, raster_settings.image_height, raster_settings.image_width)
        or grad_color.device != device
        or grad_color.dtype != torch.float32
    ):
        raise ValueError("grad_color must be float32 (3,H,W) on the rendering device")
    # Three disjoint contiguous views share one zero-initialized allocation.
    # All atomic destinations still start at zero, with one fill submission.
    storage = torch.zeros(n * 20, device=device, dtype=torch.float32)
    gradients = dict(
        gauss2screen=storage[: n * 16].view(n, 4, 4),
        rgb=storage[n * 16 : n * 19].view(n, 3),
        opacity=storage[n * 19 :],
    )
    if n == 0:
        return gradients
    transforms = preprocessed["gauss2screen"].contiguous()
    if transforms.data_ptr() % 16:
        transforms = transforms.clone()

    def arg(tensor, dtype=wp.float32):
        return kernel_arg(tensor.contiguous(), dtype, _bindings)

    width, height = raster_settings.image_width, raster_settings.image_height
    kernel = (
        _backward_cull
        if raster_settings.settings.culling_settings.hierarchical_4x4_culling
        else _backward_unculled
    )
    launch(
        kernel,
        dim=256 * ((width + 15) // 16) * ((height + 15) // 16),
        inputs=[
            width,
            height,
            kernel_arg(bins["ranges"].view(-1), wp.int32, _bindings),
            kernel_arg(bins["point_list"], wp.int32, _bindings),
            kernel_arg(preprocessed["rgb"], wp.vec3, _bindings),
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
        ],
        block_dim=256,
        stream=current_stream(device, bindings=_bindings),
        bindings=_bindings,
    )
    return gradients
