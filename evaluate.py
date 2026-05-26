"""Phase-A robustness evaluation.

Loads a checkpoint produced by train.py, runs the six attacks of the
proposal's section 4.4.1.1 at the hyperparameter grid of section 7.2,
and writes a single metrics.json that maps directly onto the IEEE
submission's quantitative-target table (Table 6 of the proposal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from .attacks import fgsm, pgd, cw, hopskipjump, boundary
from .datasets.texbat import TEXBAT, TEXBATConfig, make_synth
from .defenses import certify_l2, gronwall_radius
from .models import CTTGNN, GNSSGraphSpec


def _rebuild(cfg):
    mcfg = cfg["model"]
    spec = GNSSGraphSpec(n_sats=cfg["data"]["n_sats"],
                         feat_dim=8, hidden=mcfg["hidden"])
    return CTTGNN(spec, t_span=tuple(mcfg["t_span"]), method=mcfg["method"])


def _val_loader(cfg):
    dcfg = cfg["data"]
    if dcfg["source"] == "synth":
        ds = make_synth(n_clean=128, n_spoof=128)
    else:
        ds = TEXBAT(
            root=Path(dcfg["root"]),
            scenarios=dcfg["scenarios"],
            cfg=TEXBATConfig(n_sats=dcfg["n_sats"]),
            max_windows_per_scenario=dcfg["max_windows_per_scenario"],
        )
        if len(ds) == 0:
            ds = make_synth(n_clean=128, n_spoof=128)
    n_val = max(1, int(0.2 * len(ds)))
    n_tr = len(ds) - n_val
    _, va = random_split(ds, [n_tr, n_val],
                         generator=torch.Generator().manual_seed(cfg["seed"]))
    return DataLoader(va, batch_size=cfg["train"]["batch_size"])


def _acc(model, loader, device, perturb=None):
    correct, total = 0, 0
    for x, adj, y in loader:
        x, adj, y = x.to(device), adj.to(device), y.to(device)
        if perturb is None:
            with torch.no_grad():
                pred = model(x, adj).argmax(-1)
        else:
            x_in = perturb(x, adj, y)
            with torch.no_grad():
                pred = model(x_in, adj).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    blob = torch.load(args.ckpt, map_location="cpu")
    cfg = blob["cfg"]
    device = torch.device(args.device)
    model = _rebuild(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    loader = _val_loader(cfg)
    results: dict = {}
    results["clean_acc"] = _acc(model, loader, device)

    ecfg = cfg["eval"]
    # White-box attacks
    for eps in ecfg["attacks"]["fgsm"]["eps"]:
        f = lambda x, adj, y, e=eps: fgsm(model, x, adj, y, eps=e)
        results[f"fgsm_acc_eps_{eps}"] = _acc(model, loader, device, f)
    for eps in ecfg["attacks"]["pgd"]["eps"]:
        f = lambda x, adj, y, e=eps: pgd(
            model, x, adj, y, eps=e,
            steps=ecfg["attacks"]["pgd"]["steps"],
            restarts=ecfg["attacks"]["pgd"]["restarts"])
        results[f"pgd_acc_eps_{eps}"] = _acc(model, loader, device, f)
    for k in ecfg["attacks"]["cw"]["kappa"]:
        f = lambda x, adj, y, kk=k: cw(model, x, adj, y, kappa=kk,
                                       steps=ecfg["attacks"]["cw"]["steps"])
        results[f"cw_acc_kappa_{k}"] = _acc(model, loader, device, f)

    # Black-box (use a small subset for query budget reasons)
    sub = next(iter(loader))
    x, adj, y = (sub[0].to(device), sub[1].to(device), sub[2].to(device))
    x_hsj = hopskipjump(model, x, adj, y,
                        max_queries=ecfg["attacks"]["hsj"]["max_queries"],
                        B=ecfg["attacks"]["hsj"]["B"])
    results["hsj_acc"] = (model(x_hsj, adj).argmax(-1) == y).float().mean().item()
    x_bnd = boundary(model, x, adj, y,
                     steps=ecfg["attacks"]["boundary"]["steps"])
    results["boundary_acc"] = (model(x_bnd, adj).argmax(-1) == y).float().mean().item()

    # Certified radii
    scfg = ecfg["smoothing"]
    radii = []
    abstains = 0
    for x, adj, y in loader:
        x, adj, y = x.to(device), adj.to(device), y.to(device)
        for i in range(x.size(0)):
            pred, r = certify_l2(model, x[i], adj[i],
                                 sigma=scfg["sigma"], n0=scfg["n0"],
                                 n=scfg["n"], alpha=scfg["alpha"])
            if pred == -1:
                abstains += 1
            else:
                radii.append(r)
        break  # one batch is enough for Phase-A
    results["smoothed_mean_radius"] = (sum(radii) / max(1, len(radii)))
    results["smoothed_abstain_rate"] = abstains / max(1, abstains + len(radii))

    L_g = blob.get("L_g", float("nan"))
    lcfg = ecfg["lipschitz"]
    results["L_g"] = L_g
    results["gronwall_input_radius"] = gronwall_radius(
        L_g, lcfg["horizon_T"], lcfg["eps_out"])

    out = args.out or args.ckpt.parent / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
