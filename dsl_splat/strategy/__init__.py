"""Renderer-independent Gaussian population strategies."""

from .default import DefaultStrategy
from .mcmc import MCMCStrategy

__all__ = ["DefaultStrategy", "MCMCStrategy"]
