"""AU-AIR multi-modal aerial DataLoader.

AU-AIR (Bozcan & Kayacan, ICRA 2020) pairs each video frame with IMU
(roll/pitch/yaw, angular velocities, linear acceleration), GPS
(lat/lon/alt) and bounding-box labels over 8 object categories. This
loader emits frames + the IMU/GPS feature vector that the graph encoder
attaches to the egonode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


CATEGORIES = (
    "Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer",
)


def _imu_vec(meta: dict) -> np.ndarray:
    """Pull the 12-d IMU/GPS state used as ego-node features."""
    keys = [
        ("roll", 0.0), ("pitch", 0.0), ("yaw", 0.0),
        ("ang_vel_x", 0.0), ("ang_vel_y", 0.0), ("ang_vel_z", 0.0),
        ("lin_acc_x", 0.0), ("lin_acc_y", 0.0), ("lin_acc_z", 0.0),
        ("latitude", 0.0), ("longitude", 0.0), ("altitude", 0.0),
    ]
    return np.array([float(meta.get(k, d)) for k, d in keys],
                    dtype=np.float32)


class AUAIR(Dataset):
    """AU-AIR frames + IMU/GPS.

    Frames are returned as float tensors in [0, 1], CHW; if the underlying
    images.zip has not been extracted, frames are returned as zero
    placeholders so the IMU branch can still be exercised. The boolean
    `has_frames` attribute reports whether real images are available.
    """

    def __init__(self, root: Path,
                 split: str = "train",
                 categories: Sequence[str] = CATEGORIES,
                 img_size: int = 224):
        self.root = Path(root)
        self.cats = list(categories)
        self.img_size = img_size
        ann_path = self.root / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"annotations.json missing under {self.root}; "
                "run `python -m uav_defense.datasets.download "
                "--dataset auair` first."
            )
        with open(ann_path) as f:
            ann = json.load(f)
        self.entries = ann["annotations"]

        # train/val split: stable 90/10 by index
        idx = np.arange(len(self.entries))
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        cut = int(0.9 * len(idx))
        self.indices = idx[:cut] if split == "train" else idx[cut:]

        img_dir = self.root / "images"
        self.has_frames = img_dir.exists() and any(img_dir.iterdir())
        self.img_dir = img_dir

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        rec = self.entries[int(self.indices[idx])]
        imu = _imu_vec(rec)
        # Per-frame multi-hot class vector across the 8 AU-AIR classes.
        y = np.zeros(len(self.cats), dtype=np.float32)
        for box in rec.get("bbox", []):
            c = int(box.get("class", -1))
            if 0 <= c < len(self.cats):
                y[c] = 1.0

        if self.has_frames:
            from PIL import Image
            img_path = self.img_dir / rec["image_name"]
            img = (Image.open(img_path).convert("RGB")
                   .resize((self.img_size, self.img_size)))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            frame = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            frame = torch.zeros(3, self.img_size, self.img_size)

        return frame, torch.from_numpy(imu), torch.from_numpy(y)
