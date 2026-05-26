"""Cheap import + tensor-shape sanity tests."""

import torch

from uav_defense.attacks import fgsm, pgd, cw
from uav_defense.datasets.texbat import make_synth
from uav_defense.defenses import gronwall_radius
from uav_defense.models import CTTGNN, GNSSGraphSpec, CAFCNNBaseline


def test_forward_shapes():
    spec = GNSSGraphSpec(n_sats=8, feat_dim=8, hidden=16)
    m = CTTGNN(spec)
    x = torch.randn(4, 8, 8)
    adj = torch.ones(4, 8, 8) - torch.eye(8).unsqueeze(0)
    out = m(x, adj)
    assert out.shape == (4, 2)


def test_attacks_run():
    spec = GNSSGraphSpec(n_sats=8, feat_dim=8, hidden=16)
    m = CTTGNN(spec)
    x = torch.randn(2, 8, 8)
    adj = torch.ones(2, 8, 8) - torch.eye(8).unsqueeze(0)
    y = torch.tensor([0, 1])
    x_fgsm = fgsm(m, x, adj, y, eps=4 / 255)
    x_pgd = pgd(m, x, adj, y, eps=4 / 255, steps=3)
    x_cw = cw(m, x, adj, y, steps=10)
    assert x_fgsm.shape == x.shape == x_pgd.shape == x_cw.shape


def test_synth_corpus():
    ds = make_synth(n_clean=8, n_spoof=8)
    assert len(ds) == 16
    f, a, y = ds[0]
    assert f.shape == (8, 8)
    assert a.shape == (8, 8)
    assert y.shape == ()


def test_gronwall_radius():
    assert gronwall_radius(2.0, 1.0, 1.0) < 1.0 / 2.71  # exp(-2) ~ 0.135


def test_baseline_shapes():
    b = CAFCNNBaseline()
    x = torch.randn(2, 8, 8)
    adj = torch.zeros(2, 8, 8)
    out = b(x, adj)
    assert out.shape == (2, 2)
