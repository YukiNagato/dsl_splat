"""First-order Torch autograd bridge to the analytic Warp backward kernels."""

from copy import copy

import torch
from torch.autograd.function import once_differentiable

from .backward_preprocess import backward_preprocess
from .backward_render import backward_render
from .binning import bin_and_sort
from .dispatch import FrameBindings
from .preprocess import preprocess

INPUT_NAMES = (
    "means3D",
    "means2D",
    "opacities",
    "filter3D",
    "shs",
    "colors_precomp",
    "scales",
    "rotations",
    "cov3D_precomp",
)
CAMERA_NAMES = ("bg", "viewmatrix", "projmatrix", "inv_viewprojmatrix", "campos")


class Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        opacities,
        filter3D,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp,
        config,
        renderer,
        launch_cache,
    ):
        values = (
            means3D,
            means2D,
            opacities,
            filter3D,
            shs,
            colors_precomp,
            scales,
            rotations,
            cov3D_precomp,
        )
        inputs = dict(zip(INPUT_NAMES, values))
        bindings = FrameBindings(launch_cache)
        state = preprocess(
            **{key: value for key, value in inputs.items() if key != "means2D"},
            raster_settings=config,
            _bindings=bindings,
        )
        bins = bin_and_sort(state, config, diagnostics=False, _bindings=bindings)
        output = renderer(state, bins, config, _bindings=bindings)
        if means3D.shape[0] == 0:
            output["color"].zero_()
        # Keep every frame's own intermediates. Save via Torch so in-place
        # changes to inputs/camera/auxiliary outputs are detected by autograd.
        named = []
        for namespace, mapping in (
            ("inputs", inputs),
            (
                "state",
                {
                    key: state[key]
                    for key in ("radii", "gauss2screen", "opacity", "rgb", "clamped")
                    if key in state
                },
            ),
            ("bins", {key: bins[key] for key in ("ranges", "point_list")}),
            ("output", output),
        ):
            named.extend(
                (namespace, key, value)
                for key, value in mapping.items()
                if isinstance(value, torch.Tensor)
            )
        named.extend(("camera", key, getattr(config, key)) for key in CAMERA_NAMES)
        ctx.names = [(namespace, key) for namespace, key, _ in named]
        ctx.save_for_backward(*(value for _, _, value in named))
        # Snapshot the four mutable dataclass nodes without recursively copying
        # their scalar/enum fields. Each outstanding frame retains its settings.
        settings = copy(config.settings)
        settings.sort_settings = copy(settings.sort_settings)
        settings.sort_settings.queue_sizes = copy(settings.sort_settings.queue_sizes)
        settings.culling_settings = copy(settings.culling_settings)
        ctx.config = config._replace(settings=settings)
        ctx.launch_cache = launch_cache
        ctx.mark_non_differentiable(
            state["radii"], output["final_T"], output["contributors"]
        )
        ctx.set_materialize_grads(False)
        return (
            output["color"],
            state["radii"],
            output["final_T"],
            output["contributors"],
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_color, grad_radii, grad_t, grad_contributors):
        if grad_color is None:
            return (None,) * 12
        mappings = {
            name: {} for name in ("inputs", "state", "bins", "output", "camera")
        }
        for (namespace, key), value in zip(ctx.names, ctx.saved_tensors):
            mappings[namespace][key] = value
        inputs = mappings["inputs"]
        config = ctx.config._replace(**mappings["camera"])
        bindings = FrameBindings(ctx.launch_cache)
        render_gradients = backward_render(
            mappings["state"],
            mappings["bins"],
            mappings["output"],
            grad_color,
            config,
            _bindings=bindings,
        )
        gradients = backward_preprocess(
            inputs, mappings["state"], render_gradients, config, _bindings=bindings
        )
        result = []
        for name in INPUT_NAMES:
            if name not in inputs or name in ("filter3D", "cov3D_precomp"):
                result.append(None)
            elif name == "means2D":
                # In eval_3D CUDA never writes the projected-mean gradient.
                result.append(
                    gradients[name]
                    if gradients[name].shape == inputs[name].shape
                    else torch.zeros_like(inputs[name])
                )
            else:
                result.append(gradients[name].reshape_as(inputs[name]))
        return (*result, None, None, None)


def rasterize_with_grad(inputs, means2D, config, renderer, launch_cache):
    values = dict(inputs, means2D=means2D)
    color, radii, final_t, contributors = Rasterize.apply(
        *(values.get(name) for name in INPUT_NAMES), config, renderer, launch_cache
    )
    return dict(color=color, radii=radii, final_T=final_t, contributors=contributors)
