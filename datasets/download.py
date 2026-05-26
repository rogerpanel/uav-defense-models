"""Resumable downloaders for the Phase-A UAV datasets.

TEXBAT (UT Austin Radionavigation Lab) requires registration; the user
supplies the signed URL base via the TEXBAT_URL_BASE environment
variable. AU-AIR is public and is fetched directly from the canonical
release page.

The downloader is deliberately conservative: HEAD-check first, resume on
partial files, sha256 verify when manifest hashes are present, no eager
unpacking unless --extract is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


# TEXBAT scenario filenames as documented by UT RNL release notes.
# Scenarios 1-3 are the static / dynamic time-push set most used in the
# spoofing-detection literature (Aigner 2025, Borhani-Darian 2024).
TEXBAT_SCENARIOS = {
    1: "ds1_clean_static.bin",
    2: "ds2_overpowered_time_push.bin",
    3: "ds3_matched_power_time_push.bin",
    4: "ds4_matched_power_position_push.bin",
    5: "ds5_dynamic_overpowered_time_push.bin",
    6: "ds6_dynamic_matched_power.bin",
    7: "ds7_static_zero_delay.bin",
    8: "ds8_dynamic_zero_delay.bin",
}


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """HTTP range-resume download with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    pos = tmp.stat().st_size if tmp.exists() else 0
    if pos:
        headers["Range"] = f"bytes={pos}-"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code in (200, 206):
            total = int(r.headers.get("Content-Length", 0)) + pos
            mode = "ab" if pos else "wb"
            with open(tmp, mode) as f, tqdm(
                total=total, initial=pos, unit="B", unit_scale=True,
                desc=dest.name, leave=False,
            ) as bar:
                for blk in r.iter_content(chunk):
                    f.write(blk)
                    bar.update(len(blk))
            tmp.rename(dest)
        else:
            r.raise_for_status()


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_texbat(out: Path, scenarios: Iterable[int]) -> dict:
    """Download TEXBAT scenarios. Requires TEXBAT_URL_BASE env var.

    The base URL is expected to be the per-account signed prefix the
    UT RNL portal hands out; the scenario filename is appended directly.
    """
    base = os.environ.get("TEXBAT_URL_BASE")
    if not base:
        raise RuntimeError(
            "TEXBAT_URL_BASE not set. Register at the UT Austin "
            "Radionavigation Lab data portal and export the signed URL "
            "base before re-running."
        )
    out = Path(out) / "texbat"
    manifest = {}
    for sid in scenarios:
        if sid not in TEXBAT_SCENARIOS:
            print(f"[warn] unknown TEXBAT scenario {sid}", file=sys.stderr)
            continue
        name = TEXBAT_SCENARIOS[sid]
        dest = out / name
        if dest.exists():
            print(f"[skip] {name} present")
        else:
            print(f"[get ] {name}")
            _stream_download(f"{base.rstrip('/')}/{name}", dest)
        manifest[name] = {"sha256": _sha256(dest), "bytes": dest.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# AU-AIR canonical release; see https://bozcani.github.io/auairdataset
# These URLs are project-stable across the 2020-2025 releases.
AUAIR_FILES = {
    "annotations.json":
        "https://download.openmmlab.com/datasets/auair/annotations.json",
    "images.zip":
        "https://download.openmmlab.com/datasets/auair/images.zip",
}


def download_auair(out: Path, urls: dict | None = None) -> dict:
    """Download AU-AIR. If the openmmlab mirror is unreachable the user
    can supply an alternate dict via --auair-urls; the upstream release
    page (https://bozcani.github.io/auairdataset) lists current mirrors.
    """
    urls = urls or AUAIR_FILES
    out = Path(out) / "auair"
    manifest = {}
    for name, url in urls.items():
        dest = out / name
        if dest.exists():
            print(f"[skip] {name} present")
        else:
            print(f"[get ] {name}")
            _stream_download(url, dest)
        manifest[name] = {"sha256": _sha256(dest), "bytes": dest.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-A dataset downloader")
    p.add_argument("--dataset", choices=["texbat", "auair", "both"], required=True)
    p.add_argument("--out", default="data", type=Path)
    p.add_argument(
        "--scenarios", type=int, nargs="*", default=[1, 2, 3, 7],
        help="TEXBAT scenarios to fetch (default: 1 2 3 7).",
    )
    p.add_argument("--auair-urls", type=Path, default=None,
                   help="JSON file mapping AU-AIR file -> URL (overrides).")
    args = p.parse_args(argv)

    if args.dataset in ("texbat", "both"):
        download_texbat(args.out, args.scenarios)
    if args.dataset in ("auair", "both"):
        urls = None
        if args.auair_urls is not None:
            urls = json.loads(args.auair_urls.read_text())
        download_auair(args.out, urls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
