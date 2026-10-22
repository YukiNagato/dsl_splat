"""Training-view sampling and no-grad EAS rendering, outside the strategy."""

from dataclasses import dataclass, fields
from typing import Any, Callable

import torch
from torch import Tensor

from dsl_splat.statistics import WeightedContributionSum
from dsl_splat.strategy.improved_gs_ops import compute_edge_map, EdgeScoreAccumulator

if __package__:
    from .colmap_dataset import Dataset, ImageFrame
    from .model import GaussianContainer
    from .training import TrainConfig
else:
    from colmap_dataset import Dataset, ImageFrame
    from model import GaussianContainer
    from training import TrainConfig


@dataclass
class ImprovedGSOptions:
    """Strategy-specific defaults, independent of each method's normal policy."""

    grow_grad2d: float = 0.0003
    prune_opa: float = 0.005
    reset_every: int = 3000
    split_distance: float = 0.45
    opacity_reduction: float = 0.6
    budget_warmup_offset: int = 500
    gradient_only_window: int = 500
    prune_stop_window: int = 100
    reset_opacity: float = 0.05
    initial_prune_iter: int | None = 300
    initial_prune_opa: float = 0.02
    rap_prune_ratio: float = 0.2
    rap_prune_offset: int = 300
    rap_rounds: int = 2
    mu_start_iter: int = 15000
    mu_second_start_iter: int = 22500
    mu_interval: int = 5
    mu_second_interval: int = 20
    score_views: int = 10
    score_full_view_every: int = 3000
    score_full_view_offset: int = 400
    score_full_view_stop: int = 9000


def improved_strategy_options(
    options: ImprovedGSOptions, cfg: TrainConfig, dataset: Dataset
) -> dict[str, Any]:
    """Share population limits/timing, while retaining explicit strategy defaults."""
    values = {
        field.name: getattr(options, field.name)
        for field in fields(ImprovedGSOptions)
        if not field.name.startswith("score_")
    }
    values.update(
        max_gs_num=1_000_000 if cfg.max_gs_num is None else cfg.max_gs_num,
        absgrad=cfg.absgrad,
        refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=cfg.refine_stop_iter,
        refine_every=cfg.refine_every,
        pause_refine_after_reset=(
            len(dataset) + 100
            if cfg.pause_refine_after_reset == "auto"
            else cfg.pause_refine_after_reset
        ),
        verbose=True,
    )
    return values


class ImprovedGSScorer:
    """Rotate through views; periodically score all views, as in the reference.

    Cached uint8 edge maps stay on CPU. Each score call returns a new dense
    [N] tensor for the current Gaussian population, even with packed metadata.
    The strategy never owns this object or invokes the renderer itself.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_views: int = 10,
        *,
        full_view_every: int = 3000,
        full_view_offset: int = 400,
        full_view_stop: int = 9000,
    ) -> None:
        if not len(dataset) or num_views == 0 or num_views < -1:
            raise ValueError("use a nonempty dataset and num_views > 0 or -1 for all")
        if full_view_every <= 0 or not 0 <= full_view_offset < full_view_every:
            raise ValueError("invalid full-view scoring interval/offset")
        self.dataset, self.num_views = dataset, num_views
        self.full_view_every, self.full_view_offset = full_view_every, full_view_offset
        self.full_view_stop = full_view_stop
        self.pool: list[int] = []
        self.edges = [
            (compute_edge_map(frame.image) * 255).round().to(torch.uint8).cpu()
            for frame in dataset.frames
        ]

    def sample_views(self, step: int) -> list[int]:
        n = len(self.dataset)
        if (
            self.num_views == -1
            or self.num_views >= n
            or (
                step < self.full_view_stop
                and step % self.full_view_every == self.full_view_offset
            )
        ):
            return list(range(n))
        sampled = []
        for _ in range(self.num_views):
            if not self.pool:
                self.pool = list(range(n))
            sampled.append(self.pool.pop())
        return sampled

    @classmethod
    def from_options(
        cls, dataset: Dataset, options: ImprovedGSOptions
    ) -> "ImprovedGSScorer":
        return cls(
            dataset,
            options.score_views,
            full_view_every=options.score_full_view_every,
            full_view_offset=options.score_full_view_offset,
            full_view_stop=options.score_full_view_stop,
        )

    @torch.no_grad()
    def collect(
        self,
        step: int,
        n: int,
        device: torch.device,
        render: Callable[[ImageFrame, WeightedContributionSum], tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Render with the method's own activation/filtering; reduce dense scores."""
        accumulator = EdgeScoreAccumulator(n, device)
        for index in self.sample_views(step):
            weights = self.edges[index].to(device=device, dtype=torch.float32) / 255
            contributions, visible = render(
                self.dataset[index], WeightedContributionSum("eas", weights)
            )
            accumulator.add(contributions, visible)
        return accumulator.result()

    @torch.no_grad()
    def __call__(
        self,
        step: int,
        gaussian: GaussianContainer,
        rasterizer: Callable,
        degree: int,
    ) -> Tensor:
        params = gaussian.gauss_params
        device = params["means"].device
        colors, sh_degree = gaussian.appearance(degree)
        scales, opacities = (
            params["scales"].exp(),
            params["opacities"].sigmoid().flatten(),
        )

        def render(frame: ImageFrame, request: WeightedContributionSum):
            _, _, info = rasterizer(
                means=params["means"],
                quats=params["quats"],
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=frame.w2c.to(device)[None],
                Ks=frame.K.to(device)[None],
                width=frame.width,
                height=frame.height,
                near_plane=0.001,
                far_plane=1e10,
                sh_degree=sh_degree,
                packed=False,
                absgrad=False,
                statistics=(request,),
            )
            return info["statistics"]["eas"][0], (info["radii"][0] > 0).all(-1)

        return self.collect(step, len(params["means"]), device, render)
