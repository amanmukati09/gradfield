"""
tests/test_gradfield.py
"""

import numpy as np
import pytest
from gradfield import sample_grid, sample_grid_torch


def test_sample_grid_shape():
    result = sample_grid(lambda x, y: x**2 + y**2, resolution=20)
    assert result["X"].shape == (20, 20)
    assert result["Z"].shape == (20, 20)
    assert result["GX"].shape == (20, 20)


def test_sample_grid_bowl():
    """f(x,y) = x^2 + y^2 should have minimum at (0,0)"""
    result = sample_grid(lambda x, y: x**2 + y**2, resolution=51)
    Z = result["Z"]
    min_idx = np.unravel_index(Z.argmin(), Z.shape)
    center = Z.shape[0] // 2
    assert abs(min_idx[0] - center) <= 2
    assert abs(min_idx[1] - center) <= 2


def test_sample_grid_rosenbrock():
    """Rosenbrock should not crash and produce finite values."""
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    result = sample_grid(rosenbrock, x_range=(-2, 2), y_range=(-1, 3), resolution=30)
    assert np.all(np.isfinite(result["Z"]))


def test_sample_grid_gradient_direction():
    """Gradient of x^2+y^2 at positive x should point in +x direction."""
    result = sample_grid(lambda x, y: x**2 + y**2, resolution=51)
    GX = result["GX"]
    center = GX.shape[0] // 2
    assert GX[center, center + 5] > 0


def test_sample_grid_torch_shape():
    def fn(x, y):
        return x**2 + y**2

    result = sample_grid_torch(fn, resolution=10)
    assert result["Z"].shape == (10, 10)
    assert result["GX"].shape == (10, 10)


def test_sample_grid_torch_autograd_accuracy():
    """Autograd gradient of x^2 at x=1 should be exactly 2.0"""
    def fn(x, y):
        return x**2 + y**2

    result = sample_grid_torch(fn, x_range=(1.0, 1.0), y_range=(0.0, 0.0), resolution=1)
    assert abs(result["GX"][0, 0] - 2.0) < 1e-5


def test_render_creates_file(tmp_path):
    from gradfield import render

    field = sample_grid(lambda x, y: x**2 + y**2, resolution=20)
    out = str(tmp_path / "test_output.html")
    result = render(field, output=out, open_browser=False)

    import os
    assert os.path.exists(result)
    assert os.path.getsize(result) > 1000


def test_render_with_trajectory(tmp_path):
    from gradfield import render

    field = sample_grid(lambda x, y: x**2 + y**2, resolution=20)
    traj = np.array([[2.0, 2.0], [1.5, 1.5], [1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    out = str(tmp_path / "traj_output.html")
    result = render(field, trajectories=[traj], output=out, open_browser=False)

    import os
    assert os.path.exists(result)