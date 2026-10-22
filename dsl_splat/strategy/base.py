# SPDX-License-Identifier: Apache-2.0
# Derived from gsplat v1.4.0, strategy/base.py (gsplat contributors).
# Modified for DSL Splat: shared parameter/optimizer type aliases.
from dataclasses import dataclass
from typing import Dict, Union, TypeAlias
from collections.abc import Mapping

import torch

Params: TypeAlias = dict[str, torch.nn.Parameter] | torch.nn.ParameterDict
Optimizers: TypeAlias = Mapping[str, torch.optim.Optimizer]


@dataclass
class Strategy:
    """Base class for the GS densification strategy.

    This class is an base class that defines the interface for the GS
    densification strategy.
    """

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers."""
        assert set(params.keys()) == set(optimizers.keys()), (
            "params and optimizers must have the same keys, "
            f"but got {params.keys()} and {optimizers.keys()}"
        )

        for optimizer in optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                "that cooresponds to each parameter, "
                f"but got {len(optimizer.param_groups)}"
            )

    def step_pre_backward(
        self,
        *args,
        **kwargs,
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    def step_post_backward(
        self,
        *args,
        **kwargs,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        pass
