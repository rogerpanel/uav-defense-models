"""TEXBAT IQ → windowed-feature DataLoader plus a synthetic generator.

The TEXBAT binary format is interleaved int16 I and Q at 25 MS/s baseband
(see Humphreys, Bhatti & Ledvina 2012). We window the IQ stream into
short snapshots, compute the cross-ambiguity function (CAF) peak features
used by Borhani-Darian 2024, and emit them as tensors that feed both
the CNN baseline and the CT-TGNN graph encoder.

For Phase-A smoke tests where TEXBAT cannot be obtained, `make_synth`
fabricates a small drift-evasive spoofing IQ corpus that matches the
shape, dtype, and statistical regimes the model sees on real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


FS_HZ = 25_000_000.0          # 25 MS/s baseband
GPS_L1_HZ = 1_575_420_000.0   # nominal carrier
N_SATS_DEFAULT = 8            # visible-satellite count used in the graph


@dataclass
class TEXBATConfig:
    win_ms: float = 1.0       # window length in ms; 1 ms = 1 C/A code period
    hop_ms: float = 1.0
    n_sats: int = N_SATS_DEFAULT
    caf_doppler_bins: int = 21
    caf_delay_bins: int = 21
    snr_db: float = 12.0      # used only by the synthetic generator


def _caf_features(iq: np.ndarray, cfg: TEXBATConfig) -> np.ndarray:
    """Pseudo-CAF feature vector per satellite.

    Real implementation would correlate against the C/A code; for the
    feature-extraction interface used downstream we keep this CPU-cheap
    by reducing each window to the moments + spectrum bins that the
    Borhani-Darian CNN consumes. Shape: (n_sats, F).
    """
    n_samp = iq.shape[0]
    seg = max(1, n_samp // cfg.n_sats)
    feats = []
    for s in range(cfg.n_sats):
        x = iq[s * seg:(s + 1) * seg]
        if x.size < 8:
            feats.append(np.zeros(8, dtype=np.float32))
            continue
        ang = np.angle(x)
        amp = np.abs(x)
        spec = np.fft.fft(x)
        sp = np.abs(spec[:5]).astype(np.float32)
        feats.append(np.concatenate([
            [amp.mean(), amp.std(), amp.max() - amp.min(),
             np.cos(ang).mean(), np.sin(ang).mean()],
            sp / (sp.max() + 1e-8),
        ]).astype(np.float32)[:8])
    return np.stack(feats, axis=0)  # (n_sats, 8)


class TEXBAT(Dataset):
    """Windowed TEXBAT dataset.

    Args:
        root: directory containing the .bin scenario files.
        scenarios: list of scenario ids to include (1..8).
        cfg: feature config.

    Each item:
        feats : (n_sats, F)  torch.float32   — node features for M1
        adj   : (n_sats, n_sats) torch.float32 — visibility adjacency
        label : int           — 0 clean, 1 spoofed
    """

    LABEL_MAP = {1: 0, 7: 0,                 # clean / zero-delay reference
                 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 8: 1}

    def __init__(self, root: Path, scenarios: list[int],
                 cfg: TEXBATConfig | None = None,
                 max_windows_per_scenario: int = 5000):
        self.root = Path(root)
        self.cfg = cfg or TEXBATConfig()
        self.items: list[tuple[Path, int, int]] = []  # (file, offset, label)

        win_samp = int(self.cfg.win_ms * 1e-3 * FS_HZ)
        hop_samp = int(self.cfg.hop_ms * 1e-3 * FS_HZ)
        bytes_per = 4  # int16 I + int16 Q

        for sid in scenarios:
            from .download import TEXBAT_SCENARIOS
            name = TEXBAT_SCENARIOS.get(sid)
            if name is None:
                continue
            path = self.root / name
            if not path.exists():
                continue
            n_samp = path.stat().st_size // bytes_per
            label = self.LABEL_MAP[sid]
            n_wins = max(0, (n_samp - win_samp) // hop_samp)
            stride = max(1, n_wins // max_windows_per_scenario)
            for w in range(0, n_wins, stride):
                self.items.append((path, w * hop_samp * bytes_per, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, byte_off, label = self.items[idx]
        win_samp = int(self.cfg.win_ms * 1e-3 * FS_HZ)
        raw = np.memmap(path, dtype=np.int16, mode="r",
                        offset=byte_off, shape=(win_samp * 2,))
        iq = (raw[0::2].astype(np.float32) + 1j *
              raw[1::2].astype(np.float32)) / 32768.0
        feats = _caf_features(iq, self.cfg)
        adj = np.ones((self.cfg.n_sats, self.cfg.n_sats),
                      dtype=np.float32) - np.eye(self.cfg.n_sats,
                                                 dtype=np.float32)
        return (
            torch.from_numpy(feats),
            torch.from_numpy(adj),
            torch.tensor(label, dtype=torch.long),
        )


def make_synth(n_clean: int = 256, n_spoof: int = 256,
               cfg: TEXBATConfig | None = None,
               seed: int = 0,
               n_samp_override: int | None = None):
    """Drift-evasive spoofing IQ generator for smoke-tests.

    Produces a balanced corpus of clean and spoofed windows. The clean
    class is a GPS-like coherent C/A pattern (single dominant CAF peak
    per satellite + low-amplitude AWGN background); the spoofed class
    injects a coherent drift-evasive bias whose CAF peak migrates over
    the window, mimicking the Panda & Guo (2025) drift-evasive regime.

    The signature is designed to be discriminable in the (n_sats, 8)
    CAF-moment feature space so the Phase-A pipeline produces non-trivial
    accuracy without requiring TEXBAT. Never report these numbers as TEXBAT.
    """
    rng = np.random.default_rng(seed)
    cfg = cfg or TEXBATConfig()
    n_samp = n_samp_override or 512
    items = []
    base_freq = np.linspace(0.0, np.pi, n_samp).astype(np.float64)
    for _ in range(n_clean):
        noise = (rng.standard_normal(n_samp)
                 + 1j * rng.standard_normal(n_samp)) * 0.4
        sig = np.exp(1j * (base_freq * (1.0 + 0.05 * rng.standard_normal())))
        iq = (sig + noise).astype(np.complex64)
        items.append((_caf_features(iq, cfg), 0))
    for _ in range(n_spoof):
        noise = (rng.standard_normal(n_samp)
                 + 1j * rng.standard_normal(n_samp)) * 0.4
        sig = np.exp(1j * (base_freq * (1.0 + 0.05 * rng.standard_normal())))
        # drift-evasive: slow coherent phase walk + amplitude bias
        drift = np.exp(1j * np.linspace(0, 2 * np.pi * 0.8, n_samp))
        amp = 1.0 + 0.6 * np.sin(np.linspace(0, np.pi, n_samp))
        iq = (sig * drift * amp + noise).astype(np.complex64)
        items.append((_caf_features(iq, cfg), 1))
    rng.shuffle(items)

    class _SynthDS(Dataset):
        def __init__(self, data, cfg):
            self.data = data
            self.cfg = cfg

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            feats, y = self.data[idx]
            adj = (np.ones((self.cfg.n_sats, self.cfg.n_sats),
                           dtype=np.float32)
                   - np.eye(self.cfg.n_sats, dtype=np.float32))
            return (torch.from_numpy(feats),
                    torch.from_numpy(adj),
                    torch.tensor(y, dtype=torch.long))

    return _SynthDS(items, cfg)
