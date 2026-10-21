"""Method-local GEER API: a central ray camera, PBF association and exact evaluation."""

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from .binning import ParticleBounds, bin_and_sort, bound_gaussians
from .cameras import RayCamera, prepare_camera
from .geometry import launch_cache, prepare_gaussians
from .render import render
from ..cameras import PinholeCamera
from ..inputs import check_tensor
from ..primitives.spherical_harmonics import spherical_harmonics
from ..primitives.tiling import TileBins
from ..runtime.dispatch import FrameBindings
from ..runtime.gradients import AbsGradientTarget

# Fixed variance added by geometry.py's canonical transform.
FILTER_VARIANCE = 1.0e-7


@dataclass(slots=True)
class RasterizationResult:
    features: Tensor
    alpha: Tensor
    inverse_depth: Tensor
    meta: dict[str, Any]

    @property
    def radii(self) -> Tensor:
        """Reference GEER radius in tiles, not pixels."""
        return self.meta["radii"]

    @property
    def means2d(self) -> Tensor:
        """GEER's synthetic densification-gradient proxy; forward values are zero."""
        return self.meta["means2d"]

    @property
    def extra_features(self) -> Tensor | None:
        return self.meta.get("render_extra_signals")


@dataclass(slots=True)
class _PreparedView:
    geometry: tuple[Tensor, Tensor, Tensor, Tensor]
    features: Tensor
    rays: Tensor
    bounds: ParticleBounds
    bins: TileBins
    background: Tensor
    camera: RayCamera
    mode: int
    color_channels: int


def _prepare_view(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    camera: RayCamera | PinholeCamera,
    *,
    features: Tensor | None = None,
    sh_coeffs: Tensor | None = None,
    sh_degree: int | None = None,
    background: Tensor | None = None,
    extra_features: Tensor | None = None,
) -> _PreparedView:
    """Prepare geometry/features once, before binding dense or packed proxies."""
    if (
        not isinstance(means, Tensor)
        or not means.is_cuda
        or means.ndim != 2
        or means.shape[1] != 3
    ):
        raise ValueError("means must be CUDA float32 (N,3)")
    n, device = len(means), means.device
    for name, value, shape in (
        ("means", means, (n, 3)),
        ("quats", quats, (n, 4)),
        ("scales", scales, (n, 3)),
    ):
        check_tensor(name, value, device, shape)
    check_tensor("opacities", opacities, device)
    if opacities.shape not in ((n,), (n, 1)):
        raise ValueError("opacities must have shape (N,) or (N,1)")
    if (features is None) == (sh_coeffs is None):
        raise ValueError("provide exactly one of features or sh_coeffs")
    if sh_degree is not None and (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, int)
        or sh_degree not in range(4)
    ):
        raise ValueError("sh_degree must be None or an integer in [0,3]")
    bindings = FrameBindings(launch_cache)
    camera, mode, rays, limits = prepare_camera(camera, bindings)
    if camera.viewmat.device != device:
        raise ValueError("camera and Gaussians must share a device")
    means, quats, scales, opacities = [
        v.contiguous() for v in (means, quats, scales, opacities.reshape(n))
    ]
    p, m, o, d = prepare_gaussians(
        means, quats, scales, opacities, camera.viewmat.contiguous()
    )
    bounds = bound_gaussians(p, m, o, camera, mode, limits, bindings)
    if sh_coeffs is not None:
        check_tensor("sh_coeffs", sh_coeffs, device)
        if sh_coeffs.ndim != 3 or sh_coeffs.shape[0] != n or sh_coeffs.shape[2] != 3:
            raise ValueError("sh_coeffs must have shape (N,K,3)")
        if sh_degree is None:
            sh_degree = min(3, math.isqrt(sh_coeffs.shape[1]) - 1)
        center = -camera.viewmat[:3, :3].T @ camera.viewmat[:3, 3]
        features = spherical_harmonics(
            sh_coeffs, means - center, sh_degree, radii=bounds.radii, bindings=bindings
        )
    else:
        assert features is not None
        check_tensor("features", features, device)
        if sh_degree is not None:
            raise ValueError("sh_degree requires sh_coeffs")
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] < 1:
        raise ValueError("features must have shape (N,D), D>=1")
    color_channels = features.shape[1]
    if extra_features is not None:
        check_tensor("extra_features", extra_features, device)
        if (
            extra_features.ndim != 2
            or extra_features.shape[0] != n
            or extra_features.shape[1] < 1
        ):
            raise ValueError("extra_features must have shape (N,E), E>=1")
        features = torch.cat((features, extra_features), dim=-1)
    features = features.contiguous()
    channels = features.shape[1]
    if background is None:
        background = means.new_zeros(channels)
    else:
        check_tensor("background", background, device)
        if extra_features is not None and background.shape == (color_channels,):
            background = torch.cat(
                (background, background.new_zeros(channels - color_channels))
            )
        if background.shape != (channels,):
            raise ValueError("background must match color or total feature channels")
    bins = bin_and_sort(bounds, d, camera.width, camera.height, bindings)
    return _PreparedView(
        (p, m, o, d),
        features,
        rays,
        bounds,
        bins,
        background.contiguous(),
        camera,
        mode,
        color_channels,
    )


def _render_view(
    frame: _PreparedView,
    screen: Tensor,
    absolute_target: AbsGradientTarget | None = None,
) -> RasterizationResult:
    p, m, o, d = frame.geometry
    camera, bounds, bins = frame.camera, frame.bounds, frame.bins
    channels = frame.features.shape[1]
    image, alpha, inverse = render(
        p,
        m,
        o,
        d,
        frame.features,
        frame.rays,
        bounds,
        bins,
        frame.background,
        screen,
        camera.width,
        camera.height,
        frame.mode,
        absolute_target,
    )
    image = image.view(camera.height, camera.width, channels)
    meta = dict(
        radii=bounds.radii,
        pixel_radii=bounds.pixel_radii,
        means2d=screen,
        opacities=o,
        pbf=bounds.tangents,
        pbf_pixels=bounds.pixels,
        tiles_per_gauss=bounds.tile_counts,
        ranges=bins["ranges"],
        gaussian_ids=bins["point_list"],
        width=camera.width,
        height=camera.height,
        n_cameras=1,
        backend="geer-warp",
        opacity_filter_variance=FILTER_VARIANCE,
        screen_gradient_mode="geer_canonical_proxy",
        supports_means2d_grad=True,
    )
    if channels > frame.color_channels:
        meta["render_extra_signals"] = image[..., frame.color_channels :]
    return RasterizationResult(
        image,
        alpha.view(camera.height, camera.width, 1),
        inverse.view(camera.height, camera.width, 1),
        meta,
    )


def rasterize(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    camera: RayCamera | PinholeCamera,
    *,
    features: Tensor | None = None,
    sh_coeffs: Tensor | None = None,
    sh_degree: int | None = None,
    background: Tensor | None = None,
    extra_features: Tensor | None = None,
    absgrad: bool = False,
) -> RasterizationResult:
    """Render one view as HWC features, HW1 alpha and inverse radial depth.

    Inputs are activated CUDA float32 arrays (N,3), (N,4), (N,3), (N,) with
    wxyz quaternions normalized in Warp. Provide (N,D) features or SH0--3
    (N,K,3) coefficients. Extra features append channels after color. Gaussian,
    feature, pose, background and supplied raymap gradients are first-order.
    Pixel centers are (x+0.5,y+0.5); arbitrary pinhole principal points work.

    Training retains GEER's straight-through alpha clamp and stopped scale
    derivative of opacity compensation. Inverse depth includes the direct
    radial-distance derivative omitted upstream. Means2d gradients are GEER's
    densification proxy, not projected-mean derivatives. See geer/README.md.
    """
    frame = _prepare_view(
        means,
        quats,
        scales,
        opacities,
        camera,
        features=features,
        sh_coeffs=sh_coeffs,
        sh_degree=sh_degree,
        background=background,
        extra_features=extra_features,
    )
    needs = torch.is_grad_enabled() and any(
        v.requires_grad
        for v in (*frame.geometry, frame.features, frame.rays, frame.background)
    )
    screen = means.new_zeros((len(means), 2), requires_grad=needs)
    absolute = None
    if absgrad:
        screen.absgrad = torch.zeros_like(screen)
        absolute = AbsGradientTarget(screen.absgrad)
    return _render_view(frame, screen, absolute)
