"""Public render-stage dispatch: native RGB baseline or specialized Warp features."""

from .render_hierarchical_native import render_hierarchical_3d as native_renderer
from .render_hierarchical_warp import render_hierarchical_3d as warp_renderer


def render_hierarchical_3d(
    preprocessed, bins, raster_settings, *, output=None, _bindings=None
):
    """Blend (N,C) features into (C,H,W), keeping the RGB baseline unchanged."""
    channels = preprocessed["rgb"].shape[1]
    if channels <= 0:
        raise ValueError("features must have at least one channel")
    if raster_settings.bg.shape != (channels,):
        raise ValueError("bg must have shape (C,) matching the features")
    renderer = native_renderer if channels == 3 else warp_renderer
    return renderer(
        preprocessed, bins, raster_settings, output=output, _bindings=_bindings
    )
