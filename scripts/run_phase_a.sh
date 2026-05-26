#!/usr/bin/env bash
# Phase-A end-to-end smoke test: synthetic TEXBAT corpus -> CT-TGNN ->
# FGSM/PGD/CW attacks -> Lipschitz + randomised-smoothing certificates.
# No external data required.
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root

CONFIG=uav_defense/configs/texbat_ctgnn_synth.yaml
cat > "$CONFIG" <<'YAML'
run_name: phase_a_smoke
seed: 0
data:
  source: synth
  root: data/texbat
  scenarios: [1, 2, 3, 7]
  max_windows_per_scenario: 256
  win_ms: 1.0
  hop_ms: 1.0
  n_sats: 8
model:
  kind: ct_tgnn
  hidden: 32
  t_span: [0.0, 1.0]
  method: rk4
train:
  epochs: 5
  batch_size: 32
  lr: 1.0e-3
  weight_decay: 1.0e-5
  adv_train:
    enable: true
    method: pgd
    eps: 0.0313
    alpha: 0.0078
    steps: 7
eval:
  attacks:
    fgsm:    {eps: [0.0078, 0.0157, 0.0313]}
    pgd:     {eps: [0.0078, 0.0157, 0.0313], steps: 10, restarts: 1}
    cw:      {kappa: [0, 5], steps: 50}
    hsj:     {max_queries: 200, B: 20}
    boundary: {steps: 100}
  smoothing:
    sigma: 0.25
    n0: 50
    n: 200
    alpha: 1.0e-3
  lipschitz:
    horizon_T: 1.0
    eps_out: 0.5
YAML

python -m uav_defense.train --config "$CONFIG" --out uav_defense/runs
python -m uav_defense.evaluate \
    --ckpt uav_defense/runs/phase_a_smoke/ckpt.pt

echo
echo "=== Phase-A smoke test complete ==="
echo "metrics.json -> uav_defense/runs/phase_a_smoke/metrics.json"
