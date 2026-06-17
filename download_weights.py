"""
Download trained model weights for DeepFake Sentinel.

Trained weights (.pt files) are not committed to this repository — they're
attached to a GitHub Release instead, since vit_base.pt alone (~340MB)
exceeds GitHub's 100MB per-file limit for a normal git push, and committing
large binaries directly bloats clone time for everyone who visits the repo.

Usage:
    python download_weights.py

This downloads efficientnet_b4.pt, vit_base.pt, and cnn_lstm.pt (if available)
from the release tag configured below into backend/weights/.

If you haven't published a release yet, see the "Publishing a release"
section in README.md, or train your own weights using the notebooks in
training/ and skip this script entirely.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

# --- Configuration ---------------------------------------------------------
# Update these once you've published a GitHub Release with the .pt files
# attached as release assets.
GITHUB_REPO = "yourusername/deepfake-detection-platform"
RELEASE_TAG = "v1.0-weights"

WEIGHT_FILES = ["efficientnet_b4.pt", "vit_base.pt", "cnn_lstm.pt"]
# -----------------------------------------------------------------------------

WEIGHTS_DIR = Path(__file__).parent / "backend" / "weights"


def download(filename: str) -> bool:
    """Download a single weight file from the configured GitHub Release."""
    dest = WEIGHTS_DIR / filename
    if dest.is_file():
        print(f"  [skip] {filename} already exists.")
        return True

    url = (
        f"https://github.com/{GITHUB_REPO}/releases/download/"
        f"{RELEASE_TAG}/{filename}"
    )
    print(f"  Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Done ({size_mb:.1f} MB).")
        return True
    except urllib.error.HTTPError as e:
        print(f"  [failed] {filename}: HTTP {e.code} — release asset not found.")
        dest.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"  [failed] {filename}: {e}")
        dest.unlink(missing_ok=True)
        return False


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading weights from {GITHUB_REPO} @ {RELEASE_TAG}\n")

    results = {f: download(f) for f in WEIGHT_FILES}

    print("\nSummary:")
    for filename, ok in results.items():
        status = "OK" if ok else "MISSING"
        print(f"  {filename}: {status}")

    if not results.get("efficientnet_b4.pt") or not results.get("vit_base.pt"):
        print(
            "\nEfficientNet-B4 and/or ViT-B/16 weights are missing. The "
            "platform will still run in 'simulation' mode without them — "
            "see README.md for how to either publish a release with these "
            "assets, or train your own using training/phase1_train_efficientnet_vit.ipynb."
        )
        sys.exit(1)

    if not results.get("cnn_lstm.pt"):
        print(
            "\nNote: cnn_lstm.pt is optional. DetectionEngine will run with "
            "just EfficientNet + ViT and approximate the temporal score."
        )

    print("\nDone. Run `python run.py` to start the server.")


if __name__ == "__main__":
    main()
