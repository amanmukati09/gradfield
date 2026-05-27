# gradfield 🔥

**Gradient field & loss landscape visualizer for PyTorch and NumPy**

> Visualize any function's loss surface, gradient vector field, and optimization trajectories — in a beautiful interactive 3D HTML chart. No server, no config, one function call.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyPI](https://img.shields.io/pypi/v/gradfield)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI Downloads](https://img.shields.io/pypi/dm/gradfield)

---

## Why gradfield?

Every ML researcher and student has stared at a loss curve and wondered *what does this landscape actually look like?* Existing tools either require a full server setup (MLflow), are buried inside massive libraries (Open3D, matplotlib internals), or don't support PyTorch models at all.

`gradfield` fixes this with one `pip install` and one function call.

---

## Install

```bash
pip install gradfield
```

Optional — for PyTorch model landscape visualization:
```bash
pip install gradfield torch
```

---

## Quick Start

### Any Python / NumPy function

```python
from gradfield import plot

# Named built-in functions
plot("rosenbrock")
plot("himmelblau")
plot("rastrigin")

# Or any math expression
plot("sin(x) * cos(y)")
plot("x**2 + y**2")
```

### With custom ranges and colorscale

```python
from gradfield import sample_grid, render

field = sample_grid(
    lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2,
    x_range=(-2, 2),
    y_range=(-1, 3),
    resolution=80,
)

render(field, title="Rosenbrock Banana", colorscale="Plasma")
```

### PyTorch model loss landscape

```python
import torch.nn as nn
from gradfield import model_loss_fn, sample_grid, render

landscape = model_loss_fn(
    model=my_model,
    loss_fn=nn.CrossEntropyLoss(),
    inputs=X_batch,
    targets=y_batch,
    scale=0.5,
)

field = sample_grid(landscape, resolution=50)
render(field, title="My Model Loss Landscape")
```

### Built-in optimizer trajectories

```python
from gradfield import optimize, plot
import numpy as np

# Run multiple optimizers and compare their paths
sgd_traj  = optimize("rosenbrock", optimizer="sgd",      start=(-1.5, 2.5), steps=500, lr=0.001)
adam_traj = optimize("rosenbrock", optimizer="adam",     start=(-1.5, 2.5), steps=500, lr=0.01)
rms_traj  = optimize("rosenbrock", optimizer="rmsprop",  start=(-1.5, 2.5), steps=500, lr=0.01)
mom_traj  = optimize("rosenbrock", optimizer="momentum", start=(-1.5, 2.5), steps=500, lr=0.001)

plot(
    "rosenbrock",
    trajectories=[sgd_traj, adam_traj, rms_traj, mom_traj],
    title="SGD vs Adam vs RMSProp vs Momentum on Rosenbrock",
    colorscale="Plasma",
)
```

### Jupyter / Kaggle / Colab

Works out of the box — renders inline in notebook cells:

```python
from gradfield import plot

plot("himmelblau", resolution=80, colorscale="Viridis")
```

### CLI — no code needed

```bash
# List all built-in functions
gradfield list

# Plot a named function
gradfield plot rosenbrock
gradfield plot himmelblau --colorscale Plasma
gradfield plot rastrigin --resolution 80

# Plot any math expression
gradfield plot "sin(x) * cos(y)" --resolution 100
gradfield plot "x**2 + y**2" --xrange -3 3 --yrange -3 3

# Save without opening browser
gradfield plot rosenbrock --output my_plot.html --no-browser
```

---

## Built-in Functions

| Name | Description | Global Minimum |
|------|-------------|----------------|
| `rosenbrock` | Banana-shaped valley, hard to optimize | (1, 1) |
| `himmelblau` | Four equal minima | (3,2), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848) |
| `rastrigin` | Highly multimodal, many local minima | (0, 0) |
| `sphere` | Simple bowl, convex | (0, 0) |
| `beale` | Flat with steep ridges | (3, 0.5) |
| `booth` | Elongated valley | (1, 3) |
| `matyas` | Nearly flat with gentle slope | (0, 0) |

---

## Built-in Optimizers

| Name | Description |
|------|-------------|
| `sgd` | Vanilla stochastic gradient descent |
| `momentum` | SGD with momentum (beta=0.9) |
| `adam` | Adaptive moment estimation |
| `rmsprop` | RMSProp with moving average |

All implemented from scratch in NumPy — no PyTorch needed for trajectory recording.

---

## API Reference

### `plot(fn_or_name, ...)`
One-shot sample + render. Auto-detects Jupyter and renders inline.

```python
plot(
    fn_or_name,           # callable or str name or math expression
    x_range=(-3, 3),
    y_range=(-3, 3),
    resolution=60,
    trajectories=None,    # list of (N, 2) numpy arrays
    title=None,
    colorscale="Viridis",
    show_gradients=True,
    output="gradfield_output.html",
    use_torch=False,      # use torch.autograd for exact gradients
)
```

### `sample_grid(fn, x_range, y_range, resolution)`
Sample a vectorized function over a 2D grid. Returns dict with `X, Y, Z, GX, GY`.

### `sample_grid_torch(fn, x_range, y_range, resolution)`
Same as above but uses `torch.autograd` for exact gradients — useful for functions that aren't differentiable via NumPy.

### `model_loss_fn(model, loss_fn, inputs, targets, scale)`
Wraps a PyTorch model into a 2D landscape function using random filter-normalized directions (Li et al. 2018). Plug the result into `sample_grid` or `plot`.

### `optimize(fn_or_name, optimizer, start, steps, lr, x_range, y_range)`
Run a built-in optimizer and return its `(steps+1, 2)` trajectory array. Pass the result to `plot(trajectories=[...])`.

### `render(field, trajectories, title, output, colorscale, show_gradients, open_browser)`
Render a sampled field as an interactive Plotly HTML with:
- Interactive 3D surface plot
- 2D contour heatmap
- Gradient vector field overlay
- Optimization trajectory paths with start/end markers

---

## Examples

```bash
cd examples
python example_numpy.py        # Rosenbrock + gradient descent trajectory
python example_torch_model.py  # MLP loss landscape on XOR data
```

---

## How It Works

**Loss landscape visualization** projects the high-dimensional parameter space of a model (or any function) onto a 2D plane defined by two random directions, then evaluates the loss at each point on a grid.

For PyTorch models, `gradfield` implements the **filter normalization** technique from:
> Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018).
> *Visualizing the Loss Landscape of Neural Nets.* NeurIPS 2018.

This normalization ensures the directions are scale-invariant across layers, making the landscape comparable across different model sizes.

---

## Roadmap

- [ ] 1D loss curve along a single direction
- [ ] Side-by-side multi-model landscape comparison
- [ ] C++ accelerated grid sampler via pybind11 (for non-vectorizable functions)
- [ ] Trajectory recorder for live PyTorch training loops
- [ ] Weights & Biases / MLflow integration

---

## Contributing

PRs welcome. Please run `pytest tests/ -v` before submitting.

```bash
git clone https://github.com/yourusername/gradfield
cd gradfield
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT — built with ❤️ by Aman