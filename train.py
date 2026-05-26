"""Phase-A training entry point.

Runs Algorithm 1 of the proposal (single-client variant):
    1. FGSM warm-start                       - eq. (2)
    2. PGD inner-max for K steps             - eq. (3)
    3. Forward through CT-TGNN               - M1
    4. Cross-entropy + (optional) ELBO term
    5. Adam step

Logs per-epoch clean / robust accuracy and saves the final checkpoint
plus a Lipschitz estimate that evaluate.py consumes for the Grönwall
radius computation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from .attacks import fgsm, pgd
from .datasets.texbat import TEXBAT, TEXBATConfig, make_synth
from .models import CTTGNN, GNSSGraphSpec, CAFCNNBaseline, Seq2SeqSpoofTransformer


def _build_data(cfg: dict):
    dcfg = cfg["data"]
    if dcfg["source"] == "synth":
        ds = make_synth(n_clean=512, n_spoof=512,
                        cfg=TEXBATConfig(n_sats=dcfg["n_sats"],
                                         win_ms=dcfg["win_ms"],
                                         hop_ms=dcfg["hop_ms"]))
    else:
        ds = TEXBAT(
            root=Path(dcfg["root"]),
            scenarios=dcfg["scenarios"],
            cfg=TEXBATConfig(n_sats=dcfg["n_sats"],
                             win_ms=dcfg["win_ms"],
                             hop_ms=dcfg["hop_ms"]),
            max_windows_per_scenario=dcfg["max_windows_per_scenario"],
        )
        if len(ds) == 0:
            print("[warn] TEXBAT root empty; falling back to synth corpus.")
            ds = make_synth(n_clean=512, n_spoof=512,
                            cfg=TEXBATConfig(n_sats=dcfg["n_sats"]))
    n_val = max(1, int(0.2 * len(ds)))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val],
                          generator=torch.Generator().manual_seed(cfg["seed"]))
    return tr, va


def _build_model(cfg: dict):
    mcfg = cfg["model"]
    spec = GNSSGraphSpec(n_sats=cfg["data"]["n_sats"],
                         feat_dim=8, hidden=mcfg["hidden"])
    kind = mcfg.get("kind", "ct_tgnn")
    if kind == "ct_tgnn":
        return CTTGNN(spec, t_span=tuple(mcfg["t_span"]),
                      method=mcfg["method"])
    if kind == "caf_cnn":
        return CAFCNNBaseline(spec.n_sats, spec.feat_dim, spec.hidden)
    if kind == "s2s_transformer":
        return Seq2SeqSpoofTransformer(spec.n_sats, spec.feat_dim,
                                       d_model=spec.hidden)
    raise ValueError(kind)


def _accuracy(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()


def run_round(model, loader, opt, device, cfg, train: bool):
    model.train(train)
    acc_clean, acc_adv, n_batches = 0.0, 0.0, 0
    adv_cfg = cfg["train"]["adv_train"]
    for x, adj, y in loader:
        x, adj, y = x.to(device), adj.to(device), y.to(device)
        x_adv = x
        if adv_cfg["enable"]:
            x_adv = fgsm(model, x, adj, y, eps=adv_cfg["alpha"])
            x_adv = pgd(model, x_adv, adj, y,
                        eps=adv_cfg["eps"], alpha=adv_cfg["alpha"],
                        steps=adv_cfg["steps"])
        logits_c = model(x, adj)
        logits_a = model(x_adv, adj)
        loss = 0.5 * (F.cross_entropy(logits_c, y)
                      + F.cross_entropy(logits_a, y))
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc_clean += _accuracy(logits_c, y)
        acc_adv += _accuracy(logits_a, y)
        n_batches += 1
    return acc_clean / n_batches, acc_adv / n_batches


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=Path("uav_defense/configs/texbat_ctgnn.yaml"))
    p.add_argument("--out", type=Path, default=Path("uav_defense/runs"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg["seed"])

    tr, va = _build_data(cfg)
    bs = cfg["train"]["batch_size"]
    tr_loader = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=bs, num_workers=0)

    device = torch.device(args.device)
    model = _build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    run_dir = args.out / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()
    for ep in range(cfg["train"]["epochs"]):
        ac, aa = run_round(model, tr_loader, opt, device, cfg, train=True)
        vc, va_ = run_round(model, va_loader, opt, device, cfg, train=False)
        rec = dict(epoch=ep, train_clean=ac, train_adv=aa,
                   val_clean=vc, val_adv=va_)
        print(f"[ep {ep}] {rec}")
        history.append(rec)

    L_g = (model.estimate_lipschitz()
           if hasattr(model, "estimate_lipschitz") else float("nan"))
    print(f"[done] L_g = {L_g:.3f}  in {time.time() - t0:.1f}s")
    torch.save(dict(model=model.state_dict(),
                    cfg=cfg, L_g=L_g, history=history),
               run_dir / "ckpt.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
