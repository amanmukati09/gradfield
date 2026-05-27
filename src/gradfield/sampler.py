"""
sampler.py — Sample a scalar function over a 2D grid and compute gradients.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple


def sample_grid(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    resolution: int = 60,
) -> dict:
    """
    Sample a 2D scalar function over a grid and compute its gradient field.

    Parameters
    ----------
    fn : callable
        A function f(x, y) -> scalar or array of same shape as x.
    x_range : (float, float)
        Min and max for the x axis.
    y_range : (float, float)
        Min and max for the y axis.
    resolution : int
        Number of grid points per axis.

    Returns
    -------
    dict with keys:
        X, Y     : 2D meshgrid arrays
        Z        : function values over the grid
        GX, GY   : gradient components (dZ/dx, dZ/dy)
    """
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    Z = np.array(fn(X, Y), dtype=np.float64)

    # Gradient via central differences
    GY, GX = np.gradient(Z, ys, xs)

    return {"X": X, "Y": Y, "Z": Z, "GX": GX, "GY": GY}


def sample_grid_torch(
    fn: Callable,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    resolution: int = 60,
) -> dict:
    """
    Like sample_grid but uses torch.autograd for exact gradients.

    Parameters
    ----------
    fn : callable
        A function f(x, y) -> scalar torch.Tensor.
        x and y will be torch.Tensors with requires_grad=True.

    Returns
    -------
    Same dict structure as sample_grid.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for sample_grid_torch. pip install torch")

    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X_np, Y_np = np.meshgrid(xs, ys)

    Z_np  = np.zeros_like(X_np)
    GX_np = np.zeros_like(X_np)
    GY_np = np.zeros_like(X_np)

    for i in range(resolution):
        for j in range(resolution):
            x = torch.tensor(X_np[i, j], dtype=torch.float64, requires_grad=True)
            y = torch.tensor(Y_np[i, j], dtype=torch.float64, requires_grad=True)

            z = fn(x, y)
            z.backward()

            Z_np[i, j]  = z.item()
            GX_np[i, j] = x.grad.item() if x.grad is not None else 0.0
            GY_np[i, j] = y.grad.item() if y.grad is not None else 0.0

    return {"X": X_np, "Y": Y_np, "Z": Z_np, "GX": GX_np, "GY": GY_np}