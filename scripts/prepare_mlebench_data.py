"""
Prepare MLE-bench competition data after raw downloads.
Unzips raw data and runs prepare() to create public/private splits.

Usage: python prepare_mlebench_data.py <competition_id>
  e.g. python prepare_mlebench_data.py vesuvius-challenge-ink-detection
"""
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "/home/jarnav/MLScientist/mle-bench")

DATA_DIR = Path("/scratch/jarnav/mlebench_data")
MLGYM_DATA = Path("/home/jarnav/MLScientist/MLGym/data")

# Map competition IDs to MLGym task data dir names
COMP_TO_MLGYM = {
    "vesuvius-challenge-ink-detection": "mlebenchVesuvius",
    "bms-molecular-translation": "mlebenchBMS",
    "3d-object-detection-for-autonomous-vehicles": "mlebench3DDetection",
}


def prepare_competition(comp_id: str):
    comp_dir = DATA_DIR / comp_id
    raw_dir = comp_dir / "raw"
    public_dir = comp_dir / "prepared" / "public"
    private_dir = comp_dir / "prepared" / "private"

    # Step 1: Unzip
    zip_path = raw_dir / f"{comp_id}.zip"
    if zip_path.exists():
        print(f"Unzipping {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        print(f"Unzipped to {raw_dir}")
    else:
        print(f"No zip found at {zip_path}, checking for existing raw data...")

    # List what we have
    print(f"Raw contents: {sorted(os.listdir(raw_dir))[:20]}")

    # Step 2: Run prepare()
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running prepare({raw_dir}, {public_dir}, {private_dir})...")
    from mlebench.utils import import_fn
    prepare_fn = import_fn(f"mlebench.competitions.{comp_id}.prepare:prepare")
    prepare_fn(raw_dir, public_dir, private_dir)
    print("Prepare done!")

    # Step 3: Create symlink from MLGym data dir
    mlgym_name = COMP_TO_MLGYM.get(comp_id)
    if mlgym_name:
        link_path = MLGYM_DATA / mlgym_name / "data"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(public_dir)
        print(f"Symlinked {link_path} -> {public_dir}")

    # Verify
    print(f"Public: {sorted(os.listdir(public_dir))[:10]}")
    print(f"Private: {sorted(os.listdir(private_dir))[:10]}")
    print(f"DONE: {comp_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_mlebench_data.py <competition_id>")
        print("Available:", list(COMP_TO_MLGYM.keys()))
        sys.exit(1)

    comp_id = sys.argv[1]
    prepare_competition(comp_id)
