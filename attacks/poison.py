"""Clean-label poisoning (Shafahi et al., NeurIPS 2018, "Poison Frogs!").

Given a target sample (x_t, y_t) and a base class (x_b, y_b), find a
poison x_p that visually resembles x_b but whose feature representation
matches x_t. Insert into the training set; the resulting model
mis-classifies x_t at test time.
"""

from __future__ import annotations

import torch


def clean_label_poison(feature_extractor, x_target: torch.Tensor,
                       x_base: torch.Tensor, adj: torch.Tensor,
                       beta: float = 0.25, lr: float = 1e-2,
                       steps: int = 500) -> torch.Tensor:
    """Returns x_poison ~ x_base but with feature(x_poison) ~ feature(x_target)."""
    with torch.no_grad():
        f_t = feature_extractor(x_target, adj)
    x = x_base.detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        f_x = feature_extractor(x, adj)
        feat_loss = (f_x - f_t).pow(2).flatten(1).sum(-1).mean()
        img_loss = (x - x_base).pow(2).flatten(1).sum(-1).mean()
        loss = feat_loss + beta * img_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
    return x.detach()
