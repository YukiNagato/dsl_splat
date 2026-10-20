"""gsplat-style batching and training metadata for the GEER renderer.

Camera geometry remains GEER: PBF bounds, exact ray response and radial sorting.
"""

from typing import Any

import torch
from torch import Tensor

from .api import _prepare_view, _render_view
from .cameras import RayCamera
from ..inputs import batch_item, normalize_inputs
from ..training import prepare_training_metadata


def rasterization(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: int | None = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Tensor | None = None,
    render_mode: str = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: str = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: str = "pinhole",
    *,
    raymap: Tensor | None = None,
    radial_coeffs: Tensor | None = None,
    tan_theta: Tensor | None = None,
    tan_phi: Tensor | None = None,
    tan_fov: tuple[float, float] | None = None,
    extra_signals: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return (...,C,H,W,D) features/alpha and dense or packed metadata.

    Gaussian batches and camera dimensions follow the other method adapters.
    Camera arrays may be shared or have the Gaussian batch/camera dimensions.
    GEER inverse *radial* depth is in info['inverse_depth']; it is not gsplat's
    camera-Z depth, so render_mode='D'/'ED' are deliberately unsupported.
    Means2d gradients are GEER's densification proxy, not projected-mean VJPs.
    The fixed 3D filter is used for both classic and antialiased spellings.
    """
    if sparse_grad or distributed:
        raise NotImplementedError(
            "sparse gradients and distributed rendering are unsupported"
        )
    if tile_size != 16 or radius_clip != 0.0 or eps2d != 0.3:
        raise NotImplementedError(
            "GEER uses tile_size=16, fixed 3D filtering, and no radius_clip"
        )
    if render_mode != "RGB":
        raise NotImplementedError(
            "GEER supports RGB/features; inverse radial depth is available separately in metadata"
        )
    if rasterize_mode not in ("classic", "antialiased"):
        raise ValueError("rasterize_mode must be classic or antialiased")
    if camera_model not in ("pinhole", "beap", "fisheye"):
        raise NotImplementedError("camera_model must be pinhole, beap or fisheye")
    if (
        isinstance(channel_chunk, bool)
        or not isinstance(channel_chunk, int)
        or channel_chunk <= 0
    ):
        raise ValueError("channel_chunk must be a positive integer")
    if sh_degree is not None and (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, int)
        or sh_degree not in range(4)
    ):
        raise ValueError("sh_degree must be None or an integer in [0,3]")
    layout = normalize_inputs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        backgrounds,
        extra_signals,
        None,
        sh_degree,
        normalize_quats=False,  # GEER normalizes inside its geometry kernel.
    )

    def at_camera(value: Tensor | None, b: int, c: int, rank: int):
        if value is None or value.ndim == rank:
            return value
        tail = value.shape[-rank:]
        expected = layout.batch_dims + (layout.cameras,) + tail
        if value.shape != expected:
            raise ValueError(f"camera array must be shared or have shape {expected}")
        return batch_item(
            batch_item(value.reshape(layout.batch_count, layout.cameras, *tail), b), c
        )

    frames = []
    for b in range(layout.batch_count):
        for c in range(layout.cameras):
            view = batch_item(batch_item(layout.viewmats, b), c)
            intrinsic = batch_item(batch_item(layout.Ks, b), c)
            coeffs = at_camera(radial_coeffs, b, c, 1)
            if camera_model == "fisheye" and coeffs is None:
                coeffs = means.new_zeros(4)
            camera = RayCamera(
                view,
                width,
                height,
                None if camera_model == "beap" else intrinsic,
                model=camera_model,
                tan_theta=at_camera(tan_theta, b, c, 1),
                tan_phi=at_camera(tan_phi, b, c, 1),
                raymap=at_camera(raymap, b, c, 3),
                distortion_coeffs=coeffs,
                tan_fov=tan_fov,
                near_plane=near_plane,
                far_plane=far_plane,
            )
            color = layout.colors.at(b, c)
            frames.append(
                _prepare_view(
                    layout.means[b],
                    layout.quats[b],
                    layout.scales[b],
                    layout.opacities[b],
                    camera,
                    features=color if sh_degree is None else None,
                    sh_coeffs=color if sh_degree is not None else None,
                    sh_degree=sh_degree,
                    background=(
                        None if layout.backgrounds is None else layout.backgrounds[b, c]
                    ),
                    extra_features=(
                        None if layout.extras is None else layout.extras.at(b, c)
                    ),
                )
            )
    radii = torch.stack([f.bounds.pixel_radii for f in frames]).reshape(
        layout.batch_count, layout.cameras, layout.n, 2
    )
    dense = dict(
        radii=radii,
        means2d=torch.zeros_like(radii),
        tile_radii=torch.stack([f.bounds.radii for f in frames]).reshape(
            layout.batch_count, layout.cameras, layout.n
        ),
    )
    training = prepare_training_metadata(dense, layout, packed=packed, absgrad=absgrad)
    images, alphas, inverse, extras = [], [], [], []
    for i, frame in enumerate(frames):
        b, c = divmod(i, layout.cameras)
        screen = (
            dense["means2d"][b, c] if training.screen is None else training.screen[b, c]
        )
        result = _render_view(frame, screen, training.absolute_targets[i])
        images.append(result.features[..., : frame.color_channels])
        alphas.append(result.alpha)
        inverse.append(result.inverse_depth)
        if result.extra_features is not None:
            extras.append(result.extra_features)
    prefix = layout.batch_dims + (layout.cameras, height, width)
    info = training.values
    info.update(
        width=width,
        height=height,
        n_cameras=layout.cameras,
        tile_size=16,
        packed=packed,
        backend="geer-warp",
        screen_gradient_mode="geer_canonical_proxy",
        supports_means2d_grad=True,
        inverse_depth=torch.stack(inverse).reshape(*prefix, 1),
    )
    if extras:
        info["render_extra_signals"] = torch.stack(extras).reshape(
            *prefix, extras[0].shape[-1]
        )
    return (
        torch.stack(images).reshape(*prefix, images[0].shape[-1]),
        torch.stack(alphas).reshape(*prefix, 1),
        info,
    )
