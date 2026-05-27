"""
gradfield — Gradient field & loss landscape visualizer for PyTorch / NumPy
"""

from .sampler import sample_grid, sample_grid_torch
from .renderer import render
from .torch_adapter import model_loss_fn
from .notebook import plot
from .optimize import optimize

__version__ = "0.5.0"
__all__ = ["sample_grid", "sample_grid_torch", "render", "model_loss_fn", "plot", "optimize"]