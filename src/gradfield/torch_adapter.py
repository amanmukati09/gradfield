"""
torch_adapter.py — Wraps a PyTorch model so its loss landscape can be explored
along two random directions in parameter space (as in Li et al. 2018).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional


def model_loss_fn(
    model,
    loss_fn: Callable,
    inputs,
    targets,
    center=None,
    scale: float = 1.0,
) -> Callable:
    """
    Create a 2D loss landscape function by perturbing a PyTorch model's
    parameters along two random orthogonal directions.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. pip install torch")

    import torch
    import copy

    original_model = copy.deepcopy(model)
    original_params = [p.data.clone() for p in original_model.parameters()]

    direction1 = []
    direction2 = []

    for p in original_params:
        d1 = torch.randn_like(p)
        d2 = torch.randn_like(p)

        if p.dim() > 1:
            for f in range(p.shape[0]):
                p_norm = p[f].norm()
                d1[f] = d1[f] / (d1[f].norm() + 1e-10) * p_norm
                d2[f] = d2[f] / (d2[f].norm() + 1e-10) * p_norm
        else:
            d1 = d1 / (d1.norm() + 1e-10) * p.norm()
            d2 = d2 / (d2.norm() + 1e-10) * p.norm()

        direction1.append(d1)
        direction2.append(d2)

    eval_model = copy.deepcopy(original_model)

    def landscape_fn(alpha, beta):
        alpha = np.asarray(alpha)
        beta  = np.asarray(beta)
        scalar_input = alpha.ndim == 0

        if scalar_input:
            alpha = alpha.reshape(1)
            beta  = beta.reshape(1)

        results = np.zeros(alpha.shape)
        it = np.nditer([alpha, beta], flags=["multi_index"])

        while not it.finished:
            a_val = float(it[0])
            b_val = float(it[1])
            idx   = it.multi_index

            with torch.no_grad():
                for param, p0, d1, d2 in zip(
                    eval_model.parameters(), original_params, direction1, direction2
                ):
                    param.data = p0 + scale * a_val * d1 + scale * b_val * d2

            eval_model.eval()
            with torch.no_grad():
                outputs = eval_model(inputs)
                loss    = loss_fn(outputs, targets)

            results[idx] = loss.item()
            it.iternext()

        with torch.no_grad():
            for param, p0 in zip(eval_model.parameters(), original_params):
                param.data = p0.clone()

        return results.squeeze() if scalar_input else results

    return landscape_fn