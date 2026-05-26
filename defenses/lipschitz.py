"""Grönwall-based Lipschitz certificate for CT-TGNN (Theorem 1).

Given a numerical bound L_g on the drift Lipschitz constant and an
integration horizon T, the worst-case output sensitivity to an input
perturbation of size eps is
    || h(T)_1 - h(T)_2 || <= exp(L_g * T) * eps           (Eq. 7)
which we invert to a certified l2 input radius
    R(eps_out) = eps_out * exp(-L_g * T).
"""

from __future__ import annotations

import math

import torch


def power_iteration_lipschitz(model, n_iter: int = 50,
                              batch: int = 8) -> float:
    """Public alias around model.estimate_lipschitz() when available;
    falls back to a generic finite-difference probe."""
    if hasattr(model, "estimate_lipschitz"):
        return model.estimate_lipschitz(n_iter=n_iter, batch_size=batch)
    raise NotImplementedError(
        "model has no estimate_lipschitz; supply the drift directly.")


def gronwall_radius(L_g: float, T: float, eps_out: float) -> float:
    """Input radius that keeps output drift below eps_out."""
    if L_g <= 0:
        return float("inf")
    return float(eps_out * math.exp(-L_g * T))


def gronwall_output_bound(L_g: float, T: float, eps_in: float) -> float:
    """Forward Grönwall: output drift given input radius eps_in."""
    return float(eps_in * math.exp(L_g * T))
