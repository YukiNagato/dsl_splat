"""Execution ownership for the complete, single-Function rendering path."""

from torch import Tensor

from .autograd import rasterize
from .config import FrameConfig
from ..runtime.dispatch import LaunchCache
from .frame import FrameOptions, PreparedFrame, prepare_frame
from .render import render_hierarchical_3d
from .types import GaussianInputs, RasterizationOutput, Renderer


class RenderContext:
    """Own host launch packets; each frame owns tensors and descriptors.

    The context imposes no fixed Gaussian count, shape or data pointer. It is
    shared across preprocessing and rendering without an nn.Module adapter.
    """

    def __init__(self, renderer: Renderer = render_hierarchical_3d) -> None:
        self.renderer = renderer
        self.launch_cache = LaunchCache()

    def prepare(
        self,
        inputs: GaussianInputs,
        config: FrameConfig,
        options: FrameOptions | None = None,
    ) -> PreparedFrame:
        return prepare_frame(inputs, config, self.launch_cache, options)

    def render(
        self,
        inputs: GaussianInputs,
        config: FrameConfig,
        *,
        means2d: Tensor | None = None,
        options: FrameOptions | None = None,
    ) -> RasterizationOutput:
        return rasterize(
            inputs, means2d, config, self.renderer, self.launch_cache, options
        )


default_context = RenderContext()
