"""White-box adversarial attacks against (model, x, y).

Equations match the proposal's Section 3.2: (2) FGSM, (3) PGD, (4) CW.
The CW box-constraint change of variables is used so we never need an
explicit projection.

The model is assumed to expose `model(x, adj) -> logits`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fgsm(model, x: torch.Tensor, adj: torch.Tensor, y: torch.Tensor,
         eps: float = 4 / 255) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x, adj), y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps * grad.sign()).detach()


def pgd(model, x: torch.Tensor, adj: torch.Tensor, y: torch.Tensor,
        eps: float = 4 / 255, alpha: float | None = None,
        steps: int = 20, restarts: int = 1) -> torch.Tensor:
    alpha = alpha or eps / 4
    best = x.detach().clone()
    best_loss = torch.full((x.size(0),), -float("inf"), device=x.device)
    for _ in range(restarts):
        delta = torch.empty_like(x).uniform_(-eps, eps).requires_grad_(True)
        for _ in range(steps):
            logits = model(x + delta, adj)
            loss = F.cross_entropy(logits, y, reduction="none").sum()
            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps).detach()
            delta.requires_grad_(True)
        with torch.no_grad():
            logits = model(x + delta, adj)
            per = F.cross_entropy(logits, y, reduction="none")
            better = per > best_loss
            best[better] = (x + delta).detach()[better]
            best_loss[better] = per[better]
    return best


def cw(model, x: torch.Tensor, adj: torch.Tensor, y: torch.Tensor,
       c: float = 1.0, kappa: float = 0.0, steps: int = 200,
       lr: float = 1e-2) -> torch.Tensor:
    """Carlini-Wagner L2 with the tanh box-mapping (Eq. 4 in the paper).

    We pick c as a single search-step value to keep Phase-A runs cheap;
    the full 9-step binary search lives in evaluate.py.
    """
    x = x.detach()
    w = torch.atanh((x.clamp(-0.999, 0.999)) * 0.999).requires_grad_(True)
    optim = torch.optim.Adam([w], lr=lr)
    n_classes = model(x, adj).size(-1)
    for _ in range(steps):
        adv = 0.5 * (torch.tanh(w) + 1.0) * 2.0 - 1.0
        logits = model(adv, adj)
        one_hot = F.one_hot(y, n_classes).float()
        real = (one_hot * logits).sum(-1)
        other = ((1 - one_hot) * logits - 1e4 * one_hot).max(-1).values
        f6 = torch.clamp(real - other + kappa, min=0.0)
        l2 = (adv - x).pow(2).flatten(1).sum(-1)
        loss = (l2 + c * f6).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    return (0.5 * (torch.tanh(w) + 1.0) * 2.0 - 1.0).detach()
