"""Modern OpenCV-camera entry point, independent of compatibility adapters."""

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import Tensor

from .cameras import CameraBatch, prepare_cameras, validate_camera_options
from .config import default_settings, validate_settings
from .frame import FrameOptions
from .inputs import Inputs, normalize_inputs, spherical_harmonics
from .metadata import ExtendedMetadata, ViewFrame, build_metadata
from .pipeline import default_context
from .settings import ExtendedSettings
from .types import GaussianInputs


@dataclass(slots=True)
class RasterizationResult:
    """Owned image tensors plus dense or packed training statistics.

    features: (..., C, H, W, D); alpha: (..., C, H, W, 1).
    means2d is a screen-translation statistic, not a second geometry path.
    """

    features: Tensor
    alpha: Tensor
    meta: ExtendedMetadata

    @property
    def radii(self) -> Tensor:
        return self.meta["radii"]

    @property
    def means2d(self) -> Tensor:
        return self.meta["means2d"]

    @property
    def extra_features(self) -> Tensor | None:
        return self.meta.get("render_extra_signals")


def _prepare_frames(
    layout: Inputs,
    cameras: CameraBatch,
    sh_degree: int | None,
    settings: ExtendedSettings,
    radius_clip: float,
) -> list[ViewFrame]:
    frames: list[ViewFrame] = []
    colors = layout.colors
    ones = torch.ones(layout.n, 1, device=layout.device, dtype=torch.float32)
    for batch in range(layout.batch_count):
        for camera in range(layout.cameras):
            means = layout.means[batch]
            value = colors.at(batch, camera)
            if sh_degree is not None:
                value = spherical_harmonics(
                    value, means - cameras.centers[batch, camera], sh_degree
                )
            attributes: list[Tensor] = [value]
            if layout.extras is not None:
                attributes.append(layout.extras.at(batch, camera))
            attributes.append(ones)
            features = torch.cat(attributes, dim=-1)
            background = torch.zeros(
                features.shape[-1], device=layout.device, dtype=torch.float32
            )
            if layout.backgrounds is not None:
                background[: colors.channels] = layout.backgrounds[batch, camera]
            config = cameras.config(batch, camera, background, settings)
            inputs: GaussianInputs = dict(
                means3D=means,
                rotations=layout.quats[batch],
                scales=layout.scales[batch],
                opacities=layout.opacities[batch],
                colors_precomp=features,
                filter3D=None if layout.filters is None else layout.filters[batch],
            )
            prepared = default_context.prepare(
                inputs, config, FrameOptions(radius_clip=radius_clip)
            )
            frames.append(
                ViewFrame(
                    batch,
                    camera,
                    config,
                    inputs,
                    prepared,
                    cameras.intrinsics[batch][camera],
                )
            )
    return frames


def rasterize(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    *,
    features: Tensor | None = None,
    sh_coeffs: Tensor | None = None,
    sh_degree: int | None = None,
    backgrounds: Tensor | None = None,
    extra_features: Tensor | None = None,
    filter3d: Tensor | None = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    absgrad: bool = False,
    settings: ExtendedSettings | None = None,
) -> RasterizationResult:
    """Render AAA Gaussians with OpenCV world-to-camera poses and zero-skew K.

    Pass features (...,N,D) / (...,C,N,D), or SH coefficients (...,N,K,3)
    / (...,C,N,K,3). Scales and opacities are activated values; quaternions
    use wxyz and are normalized differentiably. Defaults use AAA's recommended
    antialiasing and HIER 64/8/4 configuration. Cameras execute sequentially.

    First-order gradients cover Gaussian parameters, attributes, SH and poses.
    K/background/filter gradients and second derivatives are unsupported.
    near/far retain AAA's frontend projection and contribution-culling semantics.
    packed changes metadata layout only; rendering storage remains dense.
    """
    validate_camera_options(width, height, near_plane, far_plane, radius_clip)
    if (features is None) == (sh_coeffs is None):
        raise ValueError("provide exactly one of features or sh_coeffs")
    if sh_coeffs is not None and (
        not isinstance(sh_coeffs, Tensor) or sh_coeffs.ndim < 3
    ):
        raise ValueError("sh_coeffs must have shape (...,N,K,3) or (...,C,N,K,3)")
    if sh_coeffs is not None and sh_degree is None:
        sh_degree = min(3, math.isqrt(sh_coeffs.shape[-2]) - 1)
    if features is not None and sh_degree is not None:
        raise ValueError("sh_degree requires sh_coeffs")
    if sh_degree is not None and (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, Integral)
        or sh_degree not in range(4)
    ):
        raise ValueError("sh_degree must be an integer in [0,3]")
    layout = normalize_inputs(
        means,
        quats,
        scales,
        opacities,
        features if sh_coeffs is None else sh_coeffs,
        viewmats,
        Ks,
        backgrounds,
        extra_features,
        filter3d,
        sh_degree,
    )
    if settings is not None:
        validate_settings(settings)
    config = default_settings() if settings is None else settings.copy()
    cameras = prepare_cameras(
        layout.viewmats, layout.Ks, width, height, near_plane, far_plane
    )
    frames = _prepare_frames(layout, cameras, sh_degree, config, radius_clip)
    meta, screen, options = build_metadata(
        frames, layout, width, height, packed=packed, absgrad=absgrad
    )
    channels = layout.colors.channels
    extra_channels = 0 if layout.extras is None else layout.extras.channels
    outputs: list[Tensor] = []
    alphas: list[Tensor] = []
    extras: list[Tensor] = []
    for frame, option in zip(frames, options):
        actual = default_context.render(
            frame.inputs,
            frame.config,
            means2d=None if screen is None else screen[frame.batch, frame.camera],
            options=option,
        )
        image = actual["color"].permute(1, 2, 0)
        if layout.n == 0:
            image = image + frame.config.bg
        outputs.append(image[..., :channels])
        alphas.append(image[..., -1:])
        if layout.extras is not None:
            extras.append(image[..., channels:-1])
    image_shape = layout.batch_dims + (layout.cameras, height, width)
    if extras:
        meta["render_extra_signals"] = torch.stack(extras).reshape(
            image_shape + (extra_channels,)
        )
    return RasterizationResult(
        torch.stack(outputs).reshape(image_shape + (channels,)),
        torch.stack(alphas).reshape(image_shape + (1,)),
        meta,
    )
