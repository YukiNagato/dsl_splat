"""Original-3DGS binning/blending boundary and its projection VJP."""

from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from . import projection, render
from .binning import bin_and_sort
from .cameras import ViewConfig
from ..runtime.dispatch import FrameBindings, LaunchCache
from ..runtime.gradients import AbsGradientTarget
from ..statistics import PreparedStatistics


class Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        features: Tensor,
        extra_features: Tensor | None,
        view: Tensor,
        full_projection: Tensor,
        intrinsics: Tensor,
        screen: Tensor | None,
        background: Tensor | None,
        state: projection.ProjectedGaussians,
        config: ViewConfig,
        bindings: FrameBindings,
        absolute: AbsGradientTarget | None,
        statistics: PreparedStatistics | None,
        grad_enabled: bool,
    ) -> Tensor:
        # Attribute packing belongs to this first-order rendering operation.
        # Avoid exposing CatBackward/PermuteBackward nodes to every training step.
        channels = features.shape[-1]
        attributes = [features]
        if extra_features is not None:
            attributes.append(extra_features)
        attributes.append(features.new_ones((len(features), 1)))
        features = torch.cat(attributes, dim=-1)
        packed_background = features.new_zeros(features.shape[-1])
        if background is not None:
            packed_background[:channels] = background
        background = packed_background
        bins = bin_and_sort(state, config.width, config.height, bindings)
        output = render.render(
            state,
            bins,
            features,
            background,
            config.width,
            config.height,
            bindings,
            statistics,
        )
        if grad_enabled and any(ctx.needs_input_grad[:10]):
            ctx.save_for_backward(
                means,
                quats,
                scales,
                opacities,
                features,
                view,
                full_projection,
                intrinsics,
                background,
                state.centers,
                state.conics,
                state.radii,
                state.tile_counts,
                state.depths,
                bins["ranges"],
                bins["point_list"],
                output.transmittance,
                output.contributors,
            )
            ctx.config, ctx.launch_cache = config, bindings.launch_cache
            ctx.absolute = absolute
            ctx.channels = channels
        ctx.set_materialize_grads(False)
        return output.features.permute(1, 2, 0)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: Tensor | None) -> tuple[Tensor | None, ...]:
        if gradient is None:
            return (None,) * 17
        (
            means,
            quats,
            scales,
            opacities,
            features,
            view,
            full_projection,
            intrinsics,
            background,
            centers,
            conics,
            radii,
            tile_counts,
            depths,
            ranges,
            ids,
            final_t,
            contributors,
        ) = ctx.saved_tensors
        state = projection.ProjectedGaussians(
            centers, conics, radii, tile_counts, depths
        )
        config = ctx.config
        bindings = FrameBindings(ctx.launch_cache)
        grad_centers, grad_conics, grad_features, absolute = render.backward(
            state,
            dict(ranges=ranges, point_list=ids),
            features,
            background,
            final_t,
            contributors,
            gradient.permute(2, 0, 1),
            config.width,
            config.height,
            bindings,
            absgrad=ctx.absolute is not None,
            constant_tail=1,
        )
        if ctx.absolute is not None:
            ctx.absolute.write(absolute)
        needs = tuple(ctx.needs_input_grad[i] for i in (0, 1, 2, 3, 6, 7))
        gradients = (
            projection.backward(
                means,
                quats,
                scales,
                opacities,
                view,
                full_projection,
                intrinsics,
                config,
                state,
                grad_centers,
                grad_conics,
                needs,
                bindings,
            )
            if any(needs)
            else [None] * 6
        )
        return (
            *gradients[:4],
            grad_features[:, : ctx.channels] if ctx.needs_input_grad[4] else None,
            grad_features[:, ctx.channels : -1] if ctx.needs_input_grad[5] else None,
            *gradients[4:],
            None,
            grad_centers if ctx.needs_input_grad[9] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


launch_cache = LaunchCache()
