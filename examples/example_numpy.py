"""
examples/example_numpy.py
Visualize the Rosenbrock "banana" function — a classic optimization challenge.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gradfield import sample_grid, render


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


print("Sampling gradient field...")
field = sample_grid(
    rosenbrock,
    x_range=(-2.0, 2.0),
    y_range=(-1.0, 3.0),
    resolution=80,
)


def rosenbrock_grad(x, y):
    gx = -2 * (1 - x) - 400 * x * (y - x**2)
    gy = 200 * (y - x**2)
    return gx, gy


traj = []
x, y = -1.5, 2.5
lr = 0.001
traj.append([x, y])
for _ in range(300):
    gx, gy = rosenbrock_grad(x, y)
    x -= lr * gx
    y -= lr * gy
    traj.append([x, y])

traj = np.array(traj)

print("Rendering...")
render(
    field,
    trajectories=[traj],
    title="Rosenbrock Banana Function — Gradient Descent",
    output="rosenbrock.html",
    colorscale="Plasma",
)