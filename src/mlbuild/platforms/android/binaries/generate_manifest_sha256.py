#!/usr/bin/env python3
"""
generate_manifest_sha256.py

Automatically computes SHA256 for all binaries in this folder
and updates manifest.json. Ensures manifest is always consistent
with deployed binaries.
"""

import json
import hashlib
from pathlib import Path

# Paths
BINARIES_DIR = Path(__file__).parent
MANIFEST_PATH = BINARIES_DIR / "manifest.json"
BINARY_NAME = "benchmark_model"  # must match deploy.py

def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    updated = False
    for abi, entry in manifest.items():
        binary_path = BINARIES_DIR / abi / BINARY_NAME
        if not binary_path.exists():
            print(f"[WARN] Binary not found for ABI '{abi}': {binary_path}")
            continue

        sha256_hash = compute_sha256(binary_path)
        if entry.get("sha256") != sha256_hash:
            entry["sha256"] = sha256_hash
            updated = True
            print(f"[INFO] Updated SHA256 for {abi}: {sha256_hash}")

    if updated:
        with MANIFEST_PATH.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("[INFO] Manifest.json updated successfully")
    else:
        print("[INFO] Manifest.json already up-to-date")

if __name__ == "__main__":
    main()