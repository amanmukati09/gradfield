"""
optimize.py — Built-in optimizer trajectory recorders for gradfield.
Implements SGD, Momentum, Adam, RMSProp from scratch using NumPy.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple, Union


def _numerical_grad(fn, x, y, eps=1e-5):
    """Central difference gradient of fn at (x, y)."""
    gx = (fn(x + eps, y) - fn(x - eps, y)) / (2 * eps)
    gy = (fn(x, y + eps) - fn(x, y - eps)) / (2 * eps)
    return float(gx), float(gy)


def _sgd(fn, x0, y0, lr, steps):
    x, y = x0, y0
    traj = [[x, y]]
    for _ in range(steps):
        gx, gy = _numerical_grad(fn, x, y)
        x -= lr * gx
        y -= lr * gy
        traj.append([x, y])
    return np.array(traj)


def _momentum(fn, x0, y0, lr, steps, beta=0.9):
    x, y = x0, y0
    vx, vy = 0.0, 0.0
    traj = [[x, y]]
    for _ in range(steps):
        gx, gy = _numerical_grad(fn, x, y)
        vx = beta * vx + (1 - beta) * gx
        vy = beta * vy + (1 - beta) * gy
        x -= lr * vx
        y -= lr * vy
        traj.append([x, y])
    return np.array(traj)


def _adam(fn, x0, y0, lr, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    x, y = x0, y0
    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0
    traj = [[x, y]]
    for t in range(1, steps + 1):
        gx, gy = _numerical_grad(fn, x, y)
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        vx = beta2 * vx + (1 - beta2) * gx**2
        vy = beta2 * vy + (1 - beta2) * gy**2
        mx_hat = mx / (1 - beta1**t)
        my_hat = my / (1 - beta1**t)
        vx_hat = vx / (1 - beta2**t)
        vy_hat = vy / (1 - beta2**t)
        x -= lr * mx_hat / (np.sqrt(vx_hat) + eps)
        y -= lr * my_hat / (np.sqrt(vy_hat) + eps)
        traj.append([x, y])
    return np.array(traj)


def _rmsprop(fn, x0, y0, lr, steps, beta=0.9, eps=1e-8):
    x, y = x0, y0
    vx, vy = 0.0, 0.0
    traj = [[x, y]]
    for _ in range(steps):
        gx, gy = _numerical_grad(fn, x, y)
        vx = beta * vx + (1 - beta) * gx**2
        vy = beta * vy + (1 - beta) * gy**2
        x -= lr * gx / (np.sqrt(vx) + eps)
        y -= lr * gy / (np.sqrt(vy) + eps)
        traj.append([x, y])
    return np.array(traj)


OPTIMIZERS = {
    "sgd":      _sgd,
    "momentum": _momentum,
    "adam":     _adam,
    "rmsprop":  _rmsprop,
}


def optimize(
    fn_or_name: Union[Callable, str],
    optimizer: str = "adam",
    start: Optional[Tuple[float, float]] = None,
    steps: int = 300,
    lr: float = 0.01,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    **kwargs,
) -> np.ndarray:
    """
    Run an optimizer on a function and return its trajectory.

    Parameters
    ----------
    fn_or_name : callable or str
        A function f(x, y) or a named function like 'rosenbrock'.
    optimizer : str
        One of 'sgd', 'momentum', 'adam', 'rmsprop'.
    start : (float, float), optional
        Starting point. Defaults to a random point in x_range/y_range.
    steps : int
        Number of optimization steps.
    lr : float
        Learning rate.
    x_range, y_range : tuple
        Used for random start point sampling if start is None.
    **kwargs
        Extra args passed to the optimizer (e.g. beta, beta1, beta2).

    Returns
    -------
    np.ndarray of shape (steps+1, 2)
        The (x, y) position at each step including the start.
    """
    # Resolve named function
    if isinstance(fn_or_name, str):
        from gradfield.cli import NAMED_FUNCTIONS, parse_expr
        name = fn_or_name.strip().lower()
        if name in NAMED_FUNCTIONS:
            fn, default_x, default_y = NAMED_FUNCTIONS[name]
            x_range = x_range if x_range != (-3.0, 3.0) else default_x
            y_range = y_range if y_range != (-3.0, 3.0) else default_y
        else:
            fn = parse_expr(fn_or_name)
    else:
        fn = fn_or_name

    # Default random start
    if start is None:
        rng = np.random.default_rng()
        x0 = rng.uniform(x_range[0] * 0.8, x_range[1] * 0.8)
        y0 = rng.uniform(y_range[0] * 0.8, y_range[1] * 0.8)
    else:
        x0, y0 = start

    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: {list(OPTIMIZERS)}")

    opt_fn = OPTIMIZERS[optimizer]
    return opt_fn(fn, x0, y0, lr, steps, **kwargs)