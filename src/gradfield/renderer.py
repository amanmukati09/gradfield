"""
renderer.py — Render sampled gradient fields and loss landscapes as interactive HTML.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List
import os


def render(
    field: dict,
    trajectories: Optional[List[np.ndarray]] = None,
    title: str = "Loss Landscape",
    output: str = "gradfield_output.html",
    colorscale: str = "Viridis",
    show_gradients: bool = True,
    gradient_density: int = 15,
    open_browser: bool = True,
) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Plotly is required. pip install plotly")

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

    # 3D Surface
    fig.add_trace(
        go.Surface(
            x=xs, y=ys, z=Z,
            colorscale=colorscale,
            showscale=False,
            opacity=0.92,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
            ),
            name="Loss Surface",
        ),
        row=1, col=1,
    )

    # Contour map
    fig.add_trace(
        go.Contour(
            x=xs, y=ys, z=Z,
            colorscale=colorscale,
            showscale=True,
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=9, color="white")
            ),
            line=dict(width=0.5),
            name="Contour",
        ),
        row=1, col=2,
    )

    # Gradient arrows
    if show_gradients:
        step = max(1, len(xs) // gradient_density)
        xi_idx = range(0, len(xs), step)
        yi_idx = range(0, len(ys), step)

        arrow_x, arrow_y = [], []
        for i in yi_idx:
            for j in xi_idx:
                x0 = X[i, j]
                y0 = Y[i, j]
                gx = GX[i, j]
                gy = GY[i, j]
                mag = np.sqrt(gx**2 + gy**2) + 1e-10
                norm_gx = gx / mag
                norm_gy = gy / mag
                scale = (xs[-1] - xs[0]) / (gradient_density * 2)
                arrow_x += [x0, x0 - norm_gx * scale, None]
                arrow_y += [y0, y0 - norm_gy * scale, None]

        fig.add_trace(
            go.Scatter(
                x=arrow_x, y=arrow_y,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.55)", width=1),
                name="Gradient (descent)",
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

    # Trajectories
    if trajectories:
        colors = ["#FF4136", "#2ECC40", "#FFDC00", "#FF69B4", "#7FDBFF"]
        for idx, traj in enumerate(trajectories):
            traj  = np.array(traj)
            color = colors[idx % len(colors)]
            label = f"Trajectory {idx + 1}"

            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0], y=traj[:, 1],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5, color=color,
                                line=dict(width=1, color="white")),
                    name=label,
                ),
                row=1, col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=[traj[0, 0]], y=[traj[0, 1]],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=12,
                                color=color, line=dict(width=2.5)),
                    name=f"{label} start",
                    showlegend=False,
                ),
                row=1, col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=[traj[-1, 0]], y=[traj[-1, 1]],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color=color),
                    name=f"{label} end",
                    showlegend=False,
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
                    name=f"{label} (3D)",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # Styling
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=20), x=0.5),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="#e0e0e0", family="JetBrains Mono, monospace"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444", borderwidth=1),
        height=620,
        margin=dict(l=20, r=20, t=80, b=20),
        scene=dict(
            bgcolor="#0f0f1a",
            xaxis=dict(backgroundcolor="#0f0f1a", gridcolor="#222", title="x"),
            yaxis=dict(backgroundcolor="#0f0f1a", gridcolor="#222", title="y"),
            zaxis=dict(backgroundcolor="#0f0f1a", gridcolor="#222", title="loss"),
        ),
    )
    fig.update_xaxes(gridcolor="#222", zerolinecolor="#444", row=1, col=2)
    fig.update_yaxes(gridcolor="#222", zerolinecolor="#444", row=1, col=2)

    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")

    inject = """
    <style>
      body { background: #0f0f1a; margin: 0; padding: 12px; font-family: 'JetBrains Mono', monospace; }
      .gradfield-badge {
        position: fixed; bottom: 12px; right: 16px;
        font-size: 11px; color: #555; font-family: monospace;
      }
    </style>
    <div class="gradfield-badge">made with gradfield 🔥</div>
    """
    html_str = html_str.replace("</body>", inject + "</body>")

    with open(output, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[gradfield] ✓ Saved → {os.path.abspath(output)}")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file:///{os.path.abspath(output)}")

    return output


def _interpolate_z(traj, X, Y, Z):
    xs = X[0, :]
    ys = Y[:, 0]
    traj_z = []
    for pt in traj:
        xi = np.clip(np.searchsorted(xs, pt[0]), 0, len(xs) - 1)
        yi = np.clip(np.searchsorted(ys, pt[1]), 0, len(ys) - 1)
        traj_z.append(Z[yi, xi])
    return np.array(traj_z)