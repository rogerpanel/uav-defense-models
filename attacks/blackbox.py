"""Decision-based black-box attacks: HopSkipJump (Chen 2020) and
BoundaryAttack (Brendel 2018). Both call only model(x, adj).argmax."""

from __future__ import annotations

import torch


@torch.no_grad()
def _decide(model, x, adj):
    return model(x, adj).argmax(-1)


def hopskipjump(model, x: torch.Tensor, adj: torch.Tensor, y: torch.Tensor,
                max_queries: int = 1000, B: int = 100,
                init_norm: float = 0.5) -> torch.Tensor:
    """Single-batch HSJ: Monte-Carlo gradient estimator on the boundary."""
    device = x.device
    dim = x[0].numel()
    delta = torch.randn_like(x) * init_norm
    out = x + delta
    queries = 0
    while queries < max_queries:
        u = torch.randn(B, *x.shape, device=device)
        u = u / u.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
        sigma = 1.0 / (queries + 1) ** 0.5
        probes = (out.unsqueeze(0) + sigma * u).reshape(-1, *x.shape[1:])
        adj_rep = adj.repeat(B, 1, 1)
        decisions = _decide(model, probes, adj_rep).view(B, -1)
        y_b = y.unsqueeze(0).expand_as(decisions)
        phi = (decisions != y_b).float() * 2 - 1
        grad = (phi.unsqueeze(-1).unsqueeze(-1) * u).mean(0)
        step = 0.1 * grad.flatten(1).norm(dim=1).clamp_min(1e-6)
        out = out + step.view(-1, 1, 1) * grad / (grad.flatten(1)
              .norm(dim=1).clamp_min(1e-8).view(-1, 1, 1))
        queries += B
    return out.detach()


def boundary(model, x: torch.Tensor, adj: torch.Tensor, y: torch.Tensor,
             steps: int = 1000, source_step: float = 1e-2,
             spherical_step: float = 1e-2) -> torch.Tensor:
    """Boundary attack (random walk along the decision boundary)."""
    out = x + torch.randn_like(x) * 0.5
    for _ in range(steps):
        eta = torch.randn_like(out)
        sph = out + spherical_step * eta
        proposal = sph + source_step * (x - sph)
        with torch.no_grad():
            cur = _decide(model, proposal, adj)
        adv = (cur != y).float().view(-1, 1, 1)
        out = adv * proposal + (1 - adv) * out
    return out.detach()
