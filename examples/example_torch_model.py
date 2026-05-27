"""
examples/example_torch_model.py
Visualize the loss landscape of a small PyTorch MLP on XOR data.
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gradfield import sample_grid, render, model_loss_fn


X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


model = MLP()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Training MLP on XOR...")
for _ in range(500):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss_fn(model(X), y).item():.4f}")

print("Building landscape function...")
landscape = model_loss_fn(model, loss_fn, X, y, scale=0.5)

print("Sampling grid (this may take ~30s)...")
field = sample_grid(landscape, x_range=(-1, 1), y_range=(-1, 1), resolution=40)

print("Rendering...")
render(
    field,
    title="MLP Loss Landscape on XOR — Two Random Directions",
    output="mlp_xor_landscape.html",
    colorscale="Viridis",
    show_gradients=True,
)