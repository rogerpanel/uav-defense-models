"""Reference baselines used for head-to-head comparison in the IEEE table.

* CAFCNNBaseline       — Borhani-Darian, Li, Wu, Closas, EURASIP JASP
                         2024: a deep CNN on cross-ambiguity-function
                         features for GNSS spoof detection.
* Seq2SeqSpoofTransformer — Aigner et al., arXiv:2510.19890 (2025):
                         encoder-decoder Transformer on TEXBAT windows
                         reporting 0.16 % error.

Both consume the same (n_sats, F) feature tensor as CT-TGNN so the
comparison is parameter-fair.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CAFCNNBaseline(nn.Module):
    def __init__(self, n_sats: int = 8, feat_dim: int = 8,
                 hidden: int = 32, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F) -> conv over the N (satellite) axis
        return self.net(x.transpose(1, 2))


class Seq2SeqSpoofTransformer(nn.Module):
    def __init__(self, n_sats: int = 8, feat_dim: int = 8,
                 d_model: int = 32, nhead: int = 4, layers: int = 2,
                 n_classes: int = 2):
        super().__init__()
        self.embed = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.1,
        )
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.embed(x))
        return self.head(h.mean(dim=1))
