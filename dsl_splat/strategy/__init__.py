"""Renderer-independent Gaussian population strategies."""

from .default import DefaultStrategy
from .mcmc import MCMCStrategy
from .improved_gs import ImprovedGSStrategy

__all__ = ["DefaultStrategy", "MCMCStrategy", "ImprovedGSStrategy"]
