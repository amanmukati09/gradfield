"""
notebook.py — Jupyter / IPython inline rendering support.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List


def is_notebook() -> bool:
    """Detect if we are running inside a Jupyter / IPython environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return True
    except ImportError:
        return False


def plot(
    fn_or_name,
    x_range=(-3.0, 3.0),
    y_range=(-3.0, 3.0),
    resolution: int = 60,
    trajectories: Optional[List[np.ndarray]] = None,
    title: Optional[str] = None,
    colorscale: str = "Viridis",
    show_gradients: bool = True,
    output: str = "gradfield_output.html",
    use_torch: bool = False,
) -> None:
    """
    One-shot function: sample + render.
    Auto-detects Jupyter and renders inline, otherwise opens browser.

    Parameters
    ----------
    fn_or_name : callable or str
        A function f(x, y) or a named function string like 'rosenbrock'.
    x_range, y_range : tuple
        Axis ranges.
    resolution : int
        Grid resolution.
    trajectories : list of np.ndarray, optional
        Optimization paths to overlay.
    title : str, optional
        Plot title. Defaults to function name or expression.
    colorscale : str
        Plotly colorscale.
    show_gradients : bool
        Whether to show gradient arrows.
    output : str
        Output HTML path (used when not in notebook).
    use_torch : bool
        Use torch.autograd for exact gradients instead of finite differences.
    """
    from gradfield.sampler import sample_grid, sample_grid_torch
    from gradfield.renderer import render

    # Resolve named functions
    if isinstance(fn_or_name, str):
        from gradfield.cli import NAMED_FUNCTIONS, parse_expr
        name = fn_or_name.strip().lower()
        if name in NAMED_FUNCTIONS:
            fn, default_x, default_y = NAMED_FUNCTIONS[name]
            x_range = x_range if x_range != (-3.0, 3.0) else default_x
            y_range = y_range if y_range != (-3.0, 3.0) else default_y
            title = title or name.capitalize()
        else:
            fn = parse_expr(fn_or_name)
            title = title or fn_or_name
    else:
        fn = fn_or_name
        title = title or "Loss Landscape"

    # Sample
    if use_torch:
        field = sample_grid_torch(fn, x_range=x_range, y_range=y_range, resolution=resolution)
    else:
        field = sample_grid(fn, x_range=x_range, y_range=y_range, resolution=resolution)

    # Render
    if is_notebook():
        _render_inline(
            field,
            trajectories=trajectories,
            title=title,
            colorscale=colorscale,
            show_gradients=show_gradients,
        )
    else:
        render(
            field,
            trajectories=trajectories,
            title=title,
            output=output,
            colorscale=colorscale,
            show_gradients=show_gradients,
            open_browser=True,
        )


def _render_inline(
    field: dict,
    trajectories=None,
    title: str = "Loss Landscape",
    colorscale: str = "Viridis",
    show_gradients: bool = True,
) -> None:
    """Render plotly figure inline inside a Jupyter cell."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Plotly is required. pip install plotly")

    from gradfield.renderer import _interpolate_z
    from IPython.display import display, HTML
    import plotly.io as pio

    X  = field["X"]
    Y  = field["Y"]
    Z  = field["Z"]
    GX = field["GX"]
    GY = field["GY"]

    xs = X[0, :]
    ys = Y[:, 0]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss Surface (3D)", "Contour + Gradient Field"),
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        horizontal_spacing=0.05,
    )

    fig.add_trace(
        go.Surface(
            x=xs, y=ys, z=Z,
            colorscale=colorscale,
            showscale=False,
            opacity=0.92,
            name="Loss Surface",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Contour(
            x=xs, y=ys, z=Z,
            colorscale=colorscale,
            showscale=True,
            contours=dict(coloring="heatmap", showlabels=True,
                          labelfont=dict(size=9, color="white")),
            line=dict(width=0.5),
            name="Contour",
        ),
        row=1, col=2,
    )

    if show_gradients:
        gradient_density = 15
        step = max(1, len(xs) // gradient_density)
        arrow_x, arrow_y = [], []
        for i in range(0, len(ys), step):
            for j in range(0, len(xs), step):
                x0, y0 = X[i, j], Y[i, j]
                gx, gy = GX[i, j], GY[i, j]
                mag = np.sqrt(gx**2 + gy**2) + 1e-10
                scale = (xs[-1] - xs[0]) / (gradient_density * 2)
                arrow_x += [x0, x0 - gx / mag * scale, None]
                arrow_y += [y0, y0 - gy / mag * scale, None]

        fig.add_trace(
            go.Scatter(
                x=arrow_x, y=arrow_y,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.55)", width=1),
                name="Gradient",
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

    if trajectories:
        colors = ["#FF4136", "#2ECC40", "#FFDC00", "#FF69B4", "#7FDBFF"]
        for idx, traj in enumerate(trajectories):
            traj  = np.array(traj)
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0], y=traj[:, 1],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5, color=color),
                    name=f"Trajectory {idx + 1}",
                ),
                row=1, col=2,
            )
            traj_z = _interpolate_z(traj, X, Y, Z)
            fig.add_trace(
                go.Scatter3d(
                    x=traj[:, 0], y=traj[:, 1], z=traj_z + 0.05,
                    mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(size=3, color=color),
                    showlegend=False,
                ),
                row=1, col=1,
            )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=20), x=0.5),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="#e0e0e0"),
        height=600,
        margin=dict(l=20, r=20, t=80, b=20),
        scene=dict(
            bgcolor="#0f0f1a",
            xaxis=dict(backgroundcolor="#0f0f1a", gridcolor="#222"),
            yaxis=dict(backgroundcolor="#0f0f1a", gridcolor="#222"),
            zaxis=dict(backgroundcolor="#0f0f1a", gridcolor="#222"),
        ),
    )

    fig.show()