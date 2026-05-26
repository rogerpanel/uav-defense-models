"""M4 MambaShield — selective state-space block.

A compact PyTorch reference implementation of the SSM block used by
MambaShield (Anaedevha 2026): input-dependent gating + diagonal state
transition + FFT-based convolution, giving O(L log L) sequence
complexity. This is intentionally small; the full version lives in
mambashield-ssmodel_v3.ipynb and is imported by RobustIDPS.ai.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaShieldBlock(nn.Module):
    def __init__(self, d_model: int = 64, d_state: int = 16,
                 expand: int = 2, kernel: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        A = -torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1).log()
        self.A_log = nn.Parameter(A)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel,
                              groups=self.d_inner, padding=kernel - 1)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.kernel = kernel

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, D) -> (B, L, D)."""
        B, L, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :L]
        x = F.silu(x).transpose(1, 2)
        dt = F.softplus(self.dt_proj(x))
        BC = self.x_proj(x).chunk(2, dim=-1)
        Bm, Cm = BC
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x, dt, A, Bm, Cm)
        y = y + self.D.view(1, 1, -1) * x
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, Bm, Cm):
        """Recurrent scan; O(L) per step, O(L log L) total via grouping.

        We use the simple recurrent form here for fidelity; the FFT path
        is in the full notebook. Sequence lengths in the Phase-A regime
        are short (TEXBAT windows ~1 ms) so this is fine.
        """
        B, L, D = x.shape
        N = A.size(-1)
        h = x.new_zeros(B, D, N)
        outs = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, t, :].unsqueeze(-1) * Bm[:, t, :].unsqueeze(1)
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            outs.append((h * Cm[:, t, :].unsqueeze(1)).sum(-1))
        return torch.stack(outs, dim=1)
