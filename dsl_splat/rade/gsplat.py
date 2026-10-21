"""gsplat-signature adapter for RaDe-GS; no dependency on AAA."""

import math
from numbers import Integral
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from .api import rasterize
from ..inputs import RenderMode
from ..statistics import StatisticsRequest


def _validate_options(
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    sh_degree: int | None,
    tile_size: int | None,
    render_mode: RenderMode,
    rasterize_mode: str,
    channel_chunk: int,
    camera_model: str,
    with_eval3d: bool | None,
    rolling_shutter: object | None,
    unsupported: dict[str, bool],
    unsupported_values: dict[str, object],
) -> None:
    for name, enabled in unsupported.items():
        if enabled:
            raise NotImplementedError(
                f"RaDe-GS/Warp gsplat adapter does not support {name}=True"
            )
    for name, value in unsupported_values.items():
        if value is not None:
            raise NotImplementedError(
                f"RaDe-GS/Warp gsplat adapter does not support {name}"
            )
    shutter = getattr(rolling_shutter, "name", rolling_shutter)
    if shutter not in (None, 0, "GLOBAL", "global"):
        raise NotImplementedError("only global shutter is supported")
    if camera_model != "pinhole":
        raise NotImplementedError("only pinhole cameras are supported")
    if with_eval3d is True:
        raise NotImplementedError(
            "RaDe-GS uses projected 2D conics; with_eval3d=True is unsupported"
        )
    if not math.isfinite(eps2d) or eps2d < 0:
        raise ValueError("eps2d must be finite and nonnegative")
    if tile_size not in (None, 16):
        raise NotImplementedError("RaDe-GS currently requires tile_size=16")
    if render_mode not in ("RGB", "D", "ED", "RGB+D", "RGB+ED"):
        raise ValueError("unsupported render_mode")
    if rasterize_mode != "classic":
        raise NotImplementedError("RaDe-GS supports rasterize_mode='classic' only")
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


def rasterization(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 10000000000.0,
    radius_clip: float = 0.0,
    eps2d: float = 0.0,
    sh_degree: int | None = None,
    packed: bool = True,
    tile_size: int | None = 16,
    backgrounds: Tensor | None = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: str = "pinhole",
    segmented: bool = False,
    covars: Tensor | None = None,
    with_ut: bool = False,
    with_eval3d: bool | None = None,
    radial_coeffs: Tensor | None = None,
    tangential_coeffs: Tensor | None = None,
    thin_prism_coeffs: Tensor | None = None,
    ftheta_coeffs: object | None = None,
    rolling_shutter: object | None = None,
    viewmats_rs: Tensor | None = None,
    *,
    extra_signals: Tensor | None = None,
    extra_signals_sh_degree: int | None = None,
    statistics: Sequence[StatisticsRequest] = (),
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """gsplat calling convention with RaDe-GS rendering and gradients.

    eps2d is RaDe-GS's kernel_size (default 0). ED uses its expected camera-Z
    depth; D is that expectation times alpha. Geometry outputs (including median
    depth and camera-space normals) are also available in metadata. Packed only
    changes training metadata storage. This is not gsplat's depth estimator.
    """
    _validate_options(
        width,
        height,
        near_plane,
        far_plane,
        radius_clip,
        eps2d,
        sh_degree,
        tile_size,
        render_mode,
        rasterize_mode,
        channel_chunk,
        camera_model,
        with_eval3d,
        rolling_shutter,
        dict(
            sparse_grad=sparse_grad,
            distributed=distributed,
            segmented=segmented,
            with_ut=with_ut,
        ),
        dict(
            covars=covars,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            viewmats_rs=viewmats_rs,
            extra_signals_sh_degree=extra_signals_sh_degree,
        ),
    )
    result = rasterize(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        features=colors if sh_degree is None else None,
        sh_coeffs=colors if sh_degree is not None else None,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        extra_features=extra_signals,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        kernel_size=eps2d,
        packed=packed,
        absgrad=absgrad,
        statistics=statistics,
    )
    if statistics:
        result.meta["statistics"] = result.statistics
    image = result.features
    if render_mode != "RGB":
        depth = result.depth if "ED" in render_mode else result.depth * result.alpha
        if render_mode.startswith("RGB+"):
            image = torch.cat((image, depth), dim=-1)
        else:
            image = depth
    return image, result.alpha, result.meta


__all__ = ["rasterization"]
