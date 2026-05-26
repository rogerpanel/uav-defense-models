# uav_defense — Phase-A reference implementation

Backing codebase for the IEEE-bound paper *"Adversarial Robustness of
Neural Network-Based UAV Navigation: A Unified Defense Framework Applying
Dynamic Graph Neural Network Methods"* (Anaedevha & Trofimov, NRNU MEPhI,
2026). Slots into the existing **RobustIDPS.ai** stack
(https://robustidps.ai, Zenodo DOI 10.5281/zenodo.19129512).

This package implements **Phase A** of the four-phase roadmap in §6.3 of
the proposal: shared-core extraction of methods M1–M7 behind a
modality-agnostic interface plus the first two UAV data adapters
(GNSS = TEXBAT, multi-modal aerial = AU-AIR).

## Layout

```
uav_defense/
├── datasets/
│   ├── download.py        # TEXBAT + AU-AIR fetchers (resumable)
│   ├── texbat.py          # IQ → windowed feature DataLoader
│   └── auair.py           # frame + IMU/GPS DataLoader
├── models/
│   ├── ct_tgnn_gnss.py    # M1 CT-TGNN on the satellite-receiver graph
│   ├── mambashield.py     # M4 SSM block (selective state-space, FFT conv)
│   └── baselines.py       # CAF-CNN (Borhani-Darian 2024) + Seq2Seq Tr.
│                          # (Aigner 2025) for head-to-head comparison
├── attacks/
│   ├── whitebox.py        # FGSM, PGD, CW with CW change-of-variables
│   ├── blackbox.py        # HopSkipJump, BoundaryAttack
│   └── poison.py          # Clean-label bilevel poisoning (Shafahi 2018)
├── defenses/
│   ├── smoothing.py       # Cohen 2019 randomised smoothing (l2 radius)
│   └── lipschitz.py       # Gronwall radius + power-iteration L estimate
├── configs/
│   └── texbat_ctgnn.yaml
├── train.py               # phase-A entry point
├── evaluate.py            # adversarial robustness sweep
└── scripts/
    └── run_phase_a.sh
```

## Datasets

Phase A targets two complementary datasets — together they cover GNSS
spoofing (§6.2 of the proposal) and multi-modal aerial perception +
telemetry (§6.1 + §4.2.1):

| Dataset | Domain | Size | License | Auth needed |
|---|---|---|---|---|
| **TEXBAT** v1.1 | GPS L1 C/A IQ spoof, 8 scenarios | ~150 GB raw | UT Austin RNL research-use | Yes (registration) |
| **AU-AIR** | UAV frames + IMU/GPS, 8 classes | ~30 GB | CC BY-NC-SA 4.0 | No |

TEXBAT requires a registration at the UT Austin Radionavigation Lab data
portal; once you have the download URL, set `TEXBAT_URL_BASE` in the
environment. AU-AIR is direct download. The downloader is resumable and
checksums every file.

```bash
# AU-AIR — no auth, ~30 GB
python -m uav_defense.datasets.download --dataset auair --out data/

# TEXBAT — requires the per-account URL from UT RNL
export TEXBAT_URL_BASE='<your-signed-url>'
python -m uav_defense.datasets.download --dataset texbat \
       --scenarios 1 2 3 7 --out data/
```

If you cannot obtain TEXBAT in your environment, a small **synthetic
TEXBAT-like generator** (`uav_defense/datasets/texbat.py:make_synth`)
produces drift-evasive spoofing IQ at configurable SNR so the model
pipeline can be validated end-to-end before real data arrives. The
synthetic numbers are clearly labelled as such in `evaluate.py` output.

## Quickstart (Phase-A smoke test)

```bash
pip install -r uav_defense/requirements.txt
bash uav_defense/scripts/run_phase_a.sh         # ~5 min CPU, ~1 min GPU
```

The smoke test downloads nothing — it builds a 30-second synthetic
TEXBAT-style corpus, trains CT-TGNN for 5 epochs, runs FGSM/PGD/CW
attacks at ε∈{2,4,8}/255, computes the Lipschitz–Grönwall radius and the
Cohen randomised-smoothing radius, and writes
`uav_defense/runs/phase_a_smoke/metrics.json`. This is the artifact you
quote in the IEEE submission's Phase-A reproduction table.

## How this maps to the proposal

| §  | Proposal artefact | Code |
|----|-------------------|------|
| 3.1| Operational graph G_t | `models/ct_tgnn_gnss.py:GNSSGraph` |
| 3.2| Eq. (1)–(5) attacks | `attacks/{whitebox,blackbox,poison}.py` |
| 5.3| Thm 1 Lipschitz radius | `defenses/lipschitz.py:gronwall_radius` |
| 5.3| Thm 2 smoothed l2 radius | `defenses/smoothing.py:certify_l2` |
| 5.6| Algorithm 1 unified training | `train.py:run_round` |
| 6.2| GNSS application | `models/ct_tgnn_gnss.py` |
| 7.1| Quantitative target table | `evaluate.py` outputs this directly |

## Integrating into RobustIDPS.ai

After Phase-A numbers land, this package exposes a single entry point
`uav_defense.api.UAVDefender` that the existing FastAPI backend in
`robustidps_web_app/` can mount as a new "UAV" tab. The contract is
intentionally identical to the existing IDS pipeline's
`Detector.predict_proba` so the React frontend reuses the same charts.
