"""gsplat-style rasterization API backed by AAA-Gaussians / NVIDIA Warp.

Parameter/layout compatibility targets gsplat 1.5.3. Rendering retains AAA's
3D sampling, filtering and StopThePop hierarchy; it is not a gsplat 2D oracle.
No gsplat package or reference CUDA extension is needed at runtime.
"""

from copy import deepcopy
import math
from numbers import Integral

import torch
import torch.nn.functional as F

from .autograd import _stage_options
from .dispatch import FrameBindings
from .preprocess import preprocess
from .rasterizer import GaussianRasterizer
from .settings import (
    CullingSettings,
    ExtendedSettings,
    GaussianRasterizationSettings,
    GlobalSortOrder,
    SortMode,
    SortSettings,
)

_rasterizer = GaussianRasterizer()


def _spherical_harmonics(coefficients, directions, degree):
    """Degree 0–3 real SH, with Torch derivatives through coefficients/direction."""
    direction = F.normalize(directions, dim=-1)
    x, y, z = direction.unbind(-1)
    basis = [torch.full_like(x, 0.28209479177387814)]
    if degree >= 1:
        basis += [
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
        ]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        basis += [
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2 * zz - xx - yy),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (xx - yy),
        ]
    if degree >= 3:
        basis += [
            -0.5900435899266435 * y * (3 * xx - yy),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4 * zz - xx - yy),
            0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
            -0.4570457994644658 * x * (4 * zz - xx - yy),
            1.445305721320277 * z * (xx - yy),
            -0.5900435899266435 * x * (xx - 3 * yy),
        ]
    weights = torch.stack(basis, dim=-1)
    return (
        (coefficients[:, : len(basis)] * weights[..., None]).sum(-2) + 0.5
    ).clamp_min(0)


def _check_tensor(name, tensor, device, shape=None):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != torch.float32
    ):
        raise ValueError(f"{name} must be float32 on {device}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )


def _view_values(name, value, batch_dims, batch_count, cameras, n, device, *, sh=False):
    _check_tensor(name, value, device)
    trailing = 3 if sh else 2
    shared_rank = len(batch_dims) + trailing
    per_view = value.ndim == shared_rank + 1
    tail = tuple(value.shape[-trailing:])
    expected = batch_dims + ((cameras,) if per_view else ()) + tail
    if (
        value.ndim not in (shared_rank, shared_rank + 1)
        or tuple(value.shape) != expected
        or tail[0] != n
        or tail[-1] <= 0
    ):
        raise ValueError(
            f"{name} must have batch dimensions {batch_dims} and shape (...,N,D) or (...,C,N,D)"
            if not sh
            else f"{name} must be (...,N,K,3) or (...,C,N,K,3)"
        )
    if sh and tail[-1] != 3:
        raise ValueError("SH colors must have three channels")
    shape = (batch_count,) + ((cameras,) if per_view else ()) + tail
    return value.reshape(shape), per_view


def _default_settings(antialiased):
    return ExtendedSettings(
        sort_settings=SortSettings(
            sort_mode=SortMode.HIER, sort_order=GlobalSortOrder.PTD_MAX
        ),
        culling_settings=CullingSettings(True, True, True, True),
        load_balancing=True,
        proper_ewa_scaling=antialiased,
        eval_3D=True,
        new_aabb=True,
    )


def rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    near_plane=0.01,
    far_plane=1e10,
    radius_clip=0.0,
    eps2d=0.3,
    sh_degree=None,
    packed=True,
    tile_size=16,
    backgrounds=None,
    render_mode="RGB",
    sparse_grad=False,
    absgrad=False,
    rasterize_mode="classic",
    channel_chunk=32,
    distributed=False,
    camera_model="pinhole",
    segmented=False,
    covars=None,
    with_ut=False,
    with_eval3d=None,
    radial_coeffs=None,
    tangential_coeffs=None,
    thin_prism_coeffs=None,
    ftheta_coeffs=None,
    rolling_shutter=None,
    viewmats_rs=None,
    *,
    extra_signals=None,
    extra_signals_sh_degree=None,
    aaa_settings=None,
    filter3D=None,
):
    """Return (render_colors, render_alphas, meta) with gsplat tensor layouts.

    Supports arbitrary Gaussian batch dimensions, C pinhole cameras, shared or
    per-camera features/SH0–3, packed/dense metadata, radius/near/far culling,
    RGB/D/ED/RGB+D/RGB+ED, and first-order Gaussian/feature/alpha/depth gradients.
    Cameras are rendered sequentially on Torch's current CUDA stream.

    Inputs are activated scales/opacities and wxyz quaternions (normalized here).
    Output layouts are (...,C,H,W,D) and (...,C,H,W,1). Depth uses Gaussian
    center view-space Z; ED divides accumulated depth by alpha (floor 1e-10).
    Alpha is blended as a constant-one feature, so it remains differentiable.

    Always uses AAA eval_3D; explicitly requesting with_eval3d=False is rejected.
    eps2d is fixed at 0.3; rasterize_mode controls AAA opacity compensation.
    channel_chunk is an accepted performance hint: all channels are fused.
    Unsupported camera/background gradients, sparse gradients,
    distortion/UT/rolling shutter, covariance inputs and distributed rendering
    raise explicit errors.
    Projected-mean gradients measure rigid pixel-space translation of the AAA
    footprint. absgrad=True also records the sum of absolute per-pixel gradients
    at meta["means2d"].absgrad, for gsplat DefaultStrategy/AbsGS densification.
    """
    unsupported = dict(
        sparse_grad=sparse_grad,
        distributed=distributed,
        segmented=segmented,
        with_ut=with_ut,
    )
    for name, enabled in unsupported.items():
        if enabled:
            raise NotImplementedError(
                f"AAA/Warp gsplat adapter does not support {name}=True"
            )
    for name, value in dict(
        covars=covars,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        viewmats_rs=viewmats_rs,
        extra_signals_sh_degree=extra_signals_sh_degree,
    ).items():
        if value is not None:
            raise NotImplementedError(
                f"AAA/Warp gsplat adapter does not support {name}"
            )
    shutter = getattr(rolling_shutter, "name", rolling_shutter)
    if shutter not in (None, 0, "GLOBAL", "global"):
        raise NotImplementedError("only global shutter is supported")
    if camera_model != "pinhole":
        raise NotImplementedError("only pinhole cameras are supported")
    if with_eval3d is False:
        raise NotImplementedError(
            "this adapter always uses AAA eval_3D; omit with_eval3d or set it True"
        )
    if eps2d != 0.3:
        raise NotImplementedError("AAA's adaptive filter currently has fixed eps2d=0.3")
    if tile_size not in (None, 16):
        raise NotImplementedError("AAA's hierarchy requires tile_size=16")
    if render_mode not in ("RGB", "D", "ED", "RGB+D", "RGB+ED"):
        raise NotImplementedError(f"unsupported render_mode: {render_mode}")
    if rasterize_mode not in ("classic", "antialiased"):
        raise ValueError("rasterize_mode must be 'classic' or 'antialiased'")
    if any(
        isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
        for v in (width, height, channel_chunk)
    ):
        raise ValueError("width, height and channel_chunk must be positive integers")
    if (
        not math.isfinite(near_plane)
        or near_plane <= 0
        or math.isnan(far_plane)
        or far_plane <= near_plane
    ):
        raise ValueError("require 0 < near_plane < far_plane")
    if not math.isfinite(radius_clip) or radius_clip < 0:
        raise ValueError("radius_clip must be finite and nonnegative")
    if sh_degree is not None and (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, Integral)
        or sh_degree not in range(4)
    ):
        raise ValueError("sh_degree must be None or an integer in [0,3]")
    if (
        not isinstance(means, torch.Tensor)
        or not means.is_cuda
        or means.ndim < 2
        or means.shape[-1] != 3
    ):
        raise ValueError("means must be a CUDA float32 tensor with shape (...,N,3)")
    device = means.device
    _check_tensor("means", means, device)
    batch_dims = tuple(means.shape[:-2])
    batch_count, n = math.prod(batch_dims), means.shape[-2]
    if batch_count <= 0:
        raise ValueError("Gaussian batch dimensions must be nonempty")
    _check_tensor("quats", quats, device, batch_dims + (n, 4))
    _check_tensor("scales", scales, device, batch_dims + (n, 3))
    _check_tensor("opacities", opacities, device, batch_dims + (n,))
    _check_tensor("viewmats", viewmats, device)
    if (
        viewmats.ndim != len(batch_dims) + 3
        or viewmats.shape[-2:] != (4, 4)
        or tuple(viewmats.shape[:-3]) != batch_dims
    ):
        raise ValueError(
            "viewmats must have shape (...,C,4,4) with the Gaussian batch dimensions"
        )
    cameras = viewmats.shape[-3]
    if cameras <= 0:
        raise ValueError("at least one camera is required")
    _check_tensor("Ks", Ks, device, batch_dims + (cameras, 3, 3))
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (viewmats, Ks, backgrounds)
    ):
        raise NotImplementedError(
            "camera/intrinsics/background gradients are not supported"
        )
    if colors is None:
        if "RGB" in render_mode:
            raise ValueError("colors is required for RGB render modes")
        if sh_degree is not None:
            raise ValueError("sh_degree requires colors")
        color_bank, colors_per_view, d = None, False, 0
    else:
        color_bank, colors_per_view = _view_values(
            "colors",
            colors,
            batch_dims,
            batch_count,
            cameras,
            n,
            device,
            sh=sh_degree is not None,
        )
        if sh_degree is not None and color_bank.shape[-2] < (sh_degree + 1) ** 2:
            raise ValueError("not enough SH coefficients for sh_degree")
        d = color_bank.shape[-1]
    background_bank = None
    if backgrounds is not None:
        _check_tensor("backgrounds", backgrounds, device, batch_dims + (cameras, d))
        background_bank = backgrounds.reshape(batch_count, cameras, d)
    extra_bank, extras_per_view, e = None, False, 0
    if extra_signals is not None:
        extra_bank, extras_per_view = _view_values(
            "extra_signals", extra_signals, batch_dims, batch_count, cameras, n, device
        )
        e = extra_bank.shape[-1]
    filters = None
    if filter3D is not None:
        _check_tensor("filter3D", filter3D, device)
        if tuple(filter3D.shape) not in (batch_dims + (n,), batch_dims + (n, 1)):
            raise ValueError("filter3D must have shape (...,N) or (...,N,1)")
        filters = filter3D.reshape(batch_count, n)
    # Read the small intrinsic array once: the core API takes scalar FOVs.
    intrinsics = Ks.detach().reshape(batch_count, cameras, 3, 3).cpu()
    if (
        not bool(torch.isfinite(intrinsics).all())
        or bool((intrinsics[..., 0, 0] <= 0).any())
        or bool((intrinsics[..., 1, 1] <= 0).any())
    ):
        raise ValueError("Ks must be finite with positive fx/fy")
    expected_bottom = torch.tensor([0.0, 0.0, 1.0])
    if (
        not torch.equal(
            intrinsics[..., 2, :], expected_bottom.expand(batch_count, cameras, 3)
        )
        or bool((intrinsics[..., 0, 1] != 0).any())
        or bool((intrinsics[..., 1, 0] != 0).any())
    ):
        raise NotImplementedError("Ks must be a zero-skew pinhole intrinsic matrix")
    means_bank = means.reshape(batch_count, n, 3)
    quats_bank = F.normalize(quats.reshape(batch_count, n, 4), dim=-1)
    scales_bank = scales.reshape(batch_count, n, 3)
    opacity_bank = opacities.reshape(batch_count, n)
    views = viewmats.detach().reshape(batch_count, cameras, 4, 4)
    inverse_views = torch.linalg.inv(views)
    projections = torch.zeros(batch_count, cameras, 4, 4, dtype=torch.float32)
    projections[..., 0, 0] = 2 * intrinsics[..., 0, 0] / width
    projections[..., 1, 1] = 2 * intrinsics[..., 1, 1] / height
    projections[..., 0, 2] = 2 * intrinsics[..., 0, 2] / width - 1
    projections[..., 1, 2] = 2 * intrinsics[..., 1, 2] / height - 1
    projections[..., 2, 2] = (
        1 if math.isinf(far_plane) else far_plane / (far_plane - near_plane)
    )
    projections[..., 2, 3] = (
        -near_plane
        if math.isinf(far_plane)
        else -far_plane * near_plane / (far_plane - near_plane)
    )
    projections[..., 3, 2] = 1
    full = (projections.to(device) @ views).transpose(-1, -2).contiguous()
    inverse_full = torch.linalg.inv(full).contiguous()
    base_settings = (
        _default_settings(rasterize_mode == "antialiased")
        if aaa_settings is None
        else deepcopy(aaa_settings)
    )
    if (
        not isinstance(base_settings, ExtendedSettings)
        or not base_settings.eval_3D
        or int(base_settings.sort_settings.sort_mode) != 3
    ):
        raise ValueError("aaa_settings must use eval_3D=True and HIER sorting")
    frames = []
    outputs, alphas, extra_outputs = [], [], []
    metadata = {
        key: []
        for key in ("radii", "means2d", "depths", "opacities", "tiles_per_gauss")
    }
    has_rgb, has_depth = "RGB" in render_mode, "D" in render_mode
    color_channels = d if has_rgb else 0
    ones = torch.ones(n, 1, device=device, dtype=torch.float32)
    for batch in range(batch_count):
        for camera in range(cameras):
            view, m = views[batch, camera], means_bank[batch]
            # View-space center depth is also a differentiable attribute for D/ED.
            with torch.set_grad_enabled(torch.is_grad_enabled() and has_depth):
                points = m @ view[:3, :3].T + view[:3, 3]
            z = points[:, 2]
            selected = (z > near_plane) & (z < far_plane)
            effective_input_opacity = torch.where(selected, opacity_bank[batch], 0)
            fx, fy, cx, cy = (
                float(intrinsics[batch, camera, row, col])
                for row, col in ((0, 0), (1, 1), (0, 2), (1, 2))
            )
            settings = deepcopy(base_settings)
            attributes = []
            if has_rgb:
                value = (
                    color_bank[batch, camera] if colors_per_view else color_bank[batch]
                )
                if sh_degree is not None:
                    value = _spherical_harmonics(
                        value, m - inverse_views[batch, camera, :3, 3], sh_degree
                    )
                attributes.append(value)
            if has_depth:
                attributes.append(z[:, None])
            if e:
                attributes.append(
                    extra_bank[batch, camera] if extras_per_view else extra_bank[batch]
                )
            attributes.append(ones)
            features = torch.cat(attributes, dim=-1)
            background = torch.zeros(
                features.shape[-1], device=device, dtype=torch.float32
            )
            if has_rgb and background_bank is not None:
                background[:d] = background_bank[batch, camera]
            config = GaussianRasterizationSettings(
                image_height=height,
                image_width=width,
                tanfovx=width / (2 * fx),
                tanfovy=height / (2 * fy),
                bg=background,
                scale_modifier=1.0,
                viewmatrix=view.T.contiguous(),
                projmatrix=full[batch, camera],
                inv_viewprojmatrix=inverse_full[batch, camera],
                sh_degree=0,
                campos=inverse_views[batch, camera, :3, 3].contiguous(),
                prefiltered=False,
                settings=settings,
                render_depth=False,
                debug=False,
            )
            # Preprocess each view once, so packed/dense projected means can be
            # constructed BEFORE rendering and participate in its autograd graph.
            inputs = dict(
                means3D=m,
                rotations=quats_bank[batch],
                scales=scales_bank[batch],
                opacities=effective_input_opacity,
                colors_precomp=features,
                filter3D=None if filters is None else filters[batch],
            )
            bindings = FrameBindings(_rasterizer._launch_cache)
            state = preprocess(
                **inputs,
                raster_settings=config,
                principal_point=(cx, cy),
                _bindings=bindings,
            )
            _stage_options(state, dict(radius_clip=radius_clip))
            frames.append((batch, camera, config, inputs, state, bindings))
            with torch.no_grad():
                valid = state["radii"] > 0
                radii = torch.where(
                    valid[:, None], state["rects2D"].ceil().to(torch.int32), 0
                )
                screen = torch.stack(
                    (
                        fx * points[:, 0] / z.clamp_min(1e-12) + cx,
                        fy * points[:, 1] / z.clamp_min(1e-12) + cy,
                    ),
                    dim=-1,
                )
                metadata["radii"].append(radii)
                metadata["means2d"].append(torch.where(valid[:, None], screen, 0))
                metadata["depths"].append(torch.where(valid, z, 0))
                metadata["opacities"].append(torch.where(valid, state["opacity"], 0))
                metadata["tiles_per_gauss"].append(state["tiles_touched"])
    dense = {
        name: torch.stack(values).reshape(
            (batch_count, cameras, n) + tuple(values[0].shape[1:])
        )
        for name, values in metadata.items()
    }
    meta = dict(
        width=width,
        height=height,
        tile_size=16,
        tile_width=(width + 15) // 16,
        tile_height=(height + 15) // 16,
        n_cameras=cameras,
        packed=packed,
        backend="aaa-warp",
        with_eval3d=True,
        conics=None,
        supports_means2d_grad=True,
        screen_gradient_mode="rigid_footprint_translation",
    )
    if packed:
        batch_ids, camera_ids, gaussian_ids = (
            (dense["radii"] > 0).all(-1).nonzero(as_tuple=True)
        )
        meta.update(
            batch_ids=batch_ids, camera_ids=camera_ids, gaussian_ids=gaussian_ids
        )
        meta.update(
            {
                name: value[batch_ids, camera_ids, gaussian_ids]
                for name, value in dense.items()
            }
        )
    else:
        meta.update(batch_ids=None, camera_ids=None, gaussian_ids=None)
        meta.update(
            {
                name: value.reshape(batch_dims + tuple(value.shape[1:]))
                for name, value in dense.items()
            }
        )
    requires_screen_grad = torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in (means, quats, scales, opacities, colors, extra_signals)
    )
    # This auxiliary input is deliberately independent of the model tensors:
    # the analytic AAA backward already returns their complete derivatives.
    # Chaining this statistic back through projection would count them twice.
    meta["means2d"] = meta["means2d"].detach().requires_grad_(requires_screen_grad)
    screen_means = meta["means2d"]
    if absgrad and requires_screen_grad:
        screen_means.absgrad = torch.zeros_like(screen_means)
    if packed:
        dense_screen = torch.zeros_like(dense["means2d"]).index_put(
            (batch_ids, camera_ids, gaussian_ids), screen_means
        )
    else:
        dense_screen = screen_means.reshape(batch_count, cameras, n, 2)
    for batch, camera, config, inputs, state, bindings in frames:
        options = dict(
            preprocessed=state, bindings=bindings, screen_grad=requires_screen_grad
        )
        if absgrad and requires_screen_grad:
            if packed:
                rows = (
                    ((batch_ids == batch) & (camera_ids == camera)).nonzero().flatten()
                )
                options["absgrad_target"] = (
                    screen_means.absgrad,
                    rows,
                    gaussian_ids[rows],
                )
            else:
                options["absgrad_target"] = (
                    screen_means.absgrad.reshape(batch_count, cameras, n, 2)[
                        batch, camera
                    ],
                    None,
                    None,
                )
        actual = _rasterizer(
            **inputs,
            means2D=dense_screen[batch, camera],
            raster_settings=config,
            return_aux=True,
            _stage_options=options,
        )
        image = actual["color"].permute(1, 2, 0)
        # gsplat defines an empty scene as background, unlike AAA's wrapper.
        if n == 0:
            image = image + config.bg
        alpha = image[..., -1:]
        rendered = image[..., : color_channels + int(has_depth)]
        if render_mode in ("ED", "RGB+ED"):
            rendered = torch.cat(
                (rendered[..., :-1], rendered[..., -1:] / alpha.clamp_min(1e-10)),
                dim=-1,
            )
        outputs.append(rendered)
        alphas.append(alpha)
        if e:
            extra_outputs.append(image[..., color_channels + int(has_depth) : -1])
    image_shape = batch_dims + (cameras, height, width)
    render_colors = torch.stack(outputs).reshape(
        image_shape + (color_channels + int(has_depth),)
    )
    render_alphas = torch.stack(alphas).reshape(image_shape + (1,))
    if e:
        meta["render_extra_signals"] = torch.stack(extra_outputs).reshape(
            image_shape + (e,)
        )
    return render_colors, render_alphas, meta


__all__ = ["rasterization"]
