"""CT-TGNN (M1) on the satellite-receiver graph for GNSS spoof detection.

Each window of TEXBAT IQ becomes a graph G_t = (V, E, X_t, A_t):
    V       = visible satellites (n_sats) plus an implicit receiver root
    A_t     = visibility adjacency (here fully-connected over the visible set)
    X_t     = per-satellite CAF features

Node hidden states evolve under a learned vector field
    d h_v / dt = f_theta(h_v, A, X, t)
solved with torchdiffeq if installed, else with an explicit RK4
fallback. The output of the integrator is pooled and fed to a binary
spoof / clean head. The Lipschitz constant of f_theta is the constant
that drives the Grönwall radius computed in defenses/lipschitz.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from torchdiffeq import odeint
    _HAS_TDE = True
except Exception:
    _HAS_TDE = False


@dataclass
class GNSSGraphSpec:
    n_sats: int = 8
    feat_dim: int = 8
    hidden: int = 32


class GNSSGraph(nn.Module):
    """Per-step graph drift function f_theta(h, A, X, t).

    Uses a Lipschitz-constrained linear mixer (spectral norm) and a
    bounded non-linearity (tanh) so the Grönwall bound is tight.
    """

    def __init__(self, spec: GNSSGraphSpec):
        super().__init__()
        self.spec = spec
        self.x_enc = nn.Linear(spec.feat_dim, spec.hidden)
        self.mix = nn.utils.parametrizations.spectral_norm(
            nn.Linear(spec.hidden, spec.hidden, bias=False)
        )
        self.upd = nn.utils.parametrizations.spectral_norm(
            nn.Linear(2 * spec.hidden, spec.hidden)
        )

    def forward(self, t, h, *, adj, x):
        """h: (B, N, H);  adj: (B, N, N);  x: (B, N, F)."""
        msg = torch.bmm(adj, self.mix(h))
        deg = adj.sum(-1, keepdim=True).clamp_min(1.0)
        msg = msg / deg
        z = torch.cat([h, msg + self.x_enc(x)], dim=-1)
        return torch.tanh(self.upd(z))


class _ODEFunc(nn.Module):
    """Adapter so torchdiffeq calls f(t, h) with adj/x captured."""

    def __init__(self, drift: GNSSGraph):
        super().__init__()
        self.drift = drift
        self.adj = None
        self.x = None

    def forward(self, t, h):
        return self.drift(t, h, adj=self.adj, x=self.x)


class CTTGNN(nn.Module):
    """Continuous-time temporal GNN classifier.

    Args:
        spec: graph spec.
        t_span: (t0, t1) integration interval; smaller intervals lower
                the certified Grönwall radius but speed training.
        method: torchdiffeq method, or "rk4" for the no-dep fallback.
    """

    def __init__(self, spec: GNSSGraphSpec | None = None,
                 t_span: tuple[float, float] = (0.0, 1.0),
                 method: str = "rk4", n_classes: int = 2):
        super().__init__()
        self.spec = spec or GNSSGraphSpec()
        self.drift = GNSSGraph(self.spec)
        self.func = _ODEFunc(self.drift)
        self.t_span = t_span
        self.method = method
        self.head = nn.Sequential(
            nn.Linear(self.spec.hidden, self.spec.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.spec.hidden, n_classes),
        )

    # --- integrator ------------------------------------------------------
    def _rk4(self, h0: torch.Tensor, adj, x, steps: int = 4) -> torch.Tensor:
        t0, t1 = self.t_span
        dt = (t1 - t0) / steps
        h = h0
        t = torch.tensor(t0, device=h.device)
        for _ in range(steps):
            k1 = self.drift(t,            h,            adj=adj, x=x)
            k2 = self.drift(t + dt / 2,   h + dt * k1 / 2, adj=adj, x=x)
            k3 = self.drift(t + dt / 2,   h + dt * k2 / 2, adj=adj, x=x)
            k4 = self.drift(t + dt,       h + dt * k3,    adj=adj, x=x)
            h = h + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t = t + dt
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F); adj: (B, N, N)."""
        h0 = torch.zeros(x.size(0), x.size(1), self.spec.hidden,
                         device=x.device, dtype=x.dtype)
        if _HAS_TDE and self.method != "rk4":
            self.func.adj, self.func.x = adj, x
            ts = torch.tensor(self.t_span, device=x.device, dtype=x.dtype)
            h = odeint(self.func, h0, ts, method=self.method)[-1]
        else:
            h = self._rk4(h0, adj, x)
        pooled = h.mean(dim=1)
        return self.head(pooled)

    # --- Lipschitz handle ------------------------------------------------
    @torch.no_grad()
    def estimate_lipschitz(self, n_iter: int = 50, batch_size: int = 8,
                           device: torch.device | None = None) -> float:
        """Power-iteration estimate of L_g on the drift field.

        Returns a numerical upper bound on ||f_theta(h_1) - f_theta(h_2)|| /
        ||h_1 - h_2|| over a random batch — feeds gronwall_radius().
        """
        device = device or next(self.parameters()).device
        spec = self.spec
        adj = (torch.ones(batch_size, spec.n_sats, spec.n_sats, device=device)
               - torch.eye(spec.n_sats, device=device))
        x = torch.randn(batch_size, spec.n_sats, spec.feat_dim, device=device)
        h = torch.randn(batch_size, spec.n_sats, spec.hidden, device=device)
        v = torch.randn_like(h)
        v = v / v.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1)
        L = 0.0
        eps = 1e-3
        for _ in range(n_iter):
            d_plus = self.drift(0.0, h + eps * v, adj=adj, x=x)
            d_zero = self.drift(0.0, h,           adj=adj, x=x)
            jv = (d_plus - d_zero) / eps
            n = jv.flatten(1).norm(dim=1).max().item()
            L = max(L, n)
            v = jv / (jv.flatten(1).norm(dim=1, keepdim=True)
                      .view(-1, 1, 1) + 1e-8)
        return float(L)
