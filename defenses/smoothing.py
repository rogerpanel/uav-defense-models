"""Randomised smoothing (Cohen, Rosenfeld, Kolter, ICML 2019).

Implements Eq. (6) of the proposal: certified l2 radius
    R = (sigma / 2) * (Phi^{-1}(p_A) - Phi^{-1}(p_B))
at confidence (1 - alpha) over n Monte-Carlo samples. Abstains when the
lower bound on p_A is below 0.5, matching the reference implementation.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
from scipy.stats import norm, beta


ABSTAIN = -1


@torch.no_grad()
def _sample_counts(model, x, adj, sigma: float, n: int,
                   batch: int = 64) -> Counter:
    cnt = Counter()
    for i in range(0, n, batch):
        b = min(batch, n - i)
        noisy = x.unsqueeze(0).expand(b, *x.shape) + sigma * torch.randn(
            b, *x.shape, device=x.device)
        adj_rep = adj.unsqueeze(0).expand(b, *adj.shape)
        pred = model(noisy, adj_rep).argmax(-1).tolist()
        for p in pred:
            cnt[p] += 1
    return cnt


def _lower_conf_bound(k: int, n: int, alpha: float) -> float:
    """Clopper-Pearson lower bound on a binomial proportion."""
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def certify_l2(model, x: torch.Tensor, adj: torch.Tensor,
               sigma: float = 0.25, n0: int = 100, n: int = 1000,
               alpha: float = 1e-3) -> tuple[int, float]:
    """Returns (predicted_class, certified_radius). predicted_class is
    ABSTAIN when the confidence interval covers <= 1/2.
    """
    counts0 = _sample_counts(model, x, adj, sigma, n0)
    c_a = counts0.most_common(1)[0][0]
    counts = _sample_counts(model, x, adj, sigma, n)
    k_a = counts[c_a]
    p_a_low = _lower_conf_bound(k_a, n, alpha)
    if p_a_low <= 0.5:
        return ABSTAIN, 0.0
    radius = sigma * norm.ppf(p_a_low)
    return c_a, float(radius)


@torch.no_grad()
def smooth_predict(model, x: torch.Tensor, adj: torch.Tensor,
                   sigma: float = 0.25, n: int = 100) -> torch.Tensor:
    """Hard-label smoothed classifier (no abstention)."""
    counts = _sample_counts(model, x, adj, sigma, n)
    return torch.tensor(counts.most_common(1)[0][0])
