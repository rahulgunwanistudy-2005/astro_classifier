"""
prepare_data.py — Convert Galaxy10 DECals HDF5 into train/val/test folder structure.

Galaxy10 DECals has 10 classes. We keep all 10 for maximum impressiveness.

Classes:
    0: Disturbed Galaxies
    1: Merging Galaxies
    2: Round Smooth Galaxies
    3: In-between Round Smooth Galaxies
    4: Cigar Shaped Smooth Galaxies
    5: Barred Spiral Galaxies
    6: Unbarred Tight Spiral Galaxies
    7: Unbarred Loose Spiral Galaxies
    8: Edge-on Galaxies without Bulge
    9: Edge-on Galaxies with Bulge

Output structure:
    data/
        train/disturbed/, train/merging/, ...
        val/disturbed/,   val/merging/,   ...
        test/disturbed/,  test/merging/,  ...

Usage:
    python scripts/prepare_data.py
"""

import random
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Config 
OUTPUT_DIR = Path("data")
SPLITS     = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED       = 42

CLASS_NAMES = {
    0: "disturbed",
    1: "merging",
    2: "round_smooth",
    3: "in_between",
    4: "cigar_shaped",
    5: "barred_spiral",
    6: "unbarred_tight_spiral",
    7: "unbarred_loose_spiral",
    8: "edge_on_no_bulge",
    9: "edge_on_with_bulge",
}

# ── Main 
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Reading: {HDF5_PATH}")
    assert HDF5_PATH.exists(), f"File not found: {HDF5_PATH}"

    with h5py.File(HDF5_PATH, "r") as f:
        images = f["images"][:]   # shape: (N, 256, 256, 3)  uint8
        labels = f["ans"][:]      # shape: (N,)               int

    print(f"Total images: {len(images):,}")
    print(f"Image shape:  {images.shape[1:]}")
    print(f"Label dtype:  {labels.dtype}")
    print()

    # ── Class distribution 
    print("Class distribution:")
    print("-" * 45)
    unique, counts = np.unique(labels, return_counts=True)
    for cls_id, count in zip(unique, counts):
        name = CLASS_NAMES[int(cls_id)]
        bar  = "█" * int(count / max(counts) * 30)
        print(f"  {int(cls_id):2d}  {name:<25} {count:>5,}  {bar}")
    print("-" * 45)
    print(f"  Total: {len(labels):,}")
    print(f"  Imbalance ratio: {max(counts)/min(counts):.1f}:1")
    print()

    # ── Create output directories
    for split in SPLITS:
        for name in CLASS_NAMES.values():
            (OUTPUT_DIR / split / name).mkdir(parents=True, exist_ok=True)

    # ── Split and save 
    split_counts = {s: {n: 0 for n in CLASS_NAMES.values()} for s in SPLITS}

    # Group indices by class for stratified splitting
    class_indices = {cls_id: [] for cls_id in CLASS_NAMES}
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)

    all_assignments = []  # (idx, split, class_name)

    for cls_id, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])

        assignments = (
            [(i, "train") for i in indices[:n_train]] +
            [(i, "val")   for i in indices[n_train:n_train + n_val]] +
            [(i, "test")  for i in indices[n_train + n_val:]]
        )
        all_assignments.extend([(i, split, CLASS_NAMES[cls_id]) for i, split in assignments])

    print(f"Saving images to {OUTPUT_DIR}/...")
    for idx, split, class_name in tqdm(all_assignments, ncols=80):
        img_array = images[idx]  # (256, 256, 3) uint8
        img = Image.fromarray(img_array, mode="RGB")

        out_path = OUTPUT_DIR / split / class_name / f"img_{idx:06d}.png"
        img.save(out_path, optimize=True)
        split_counts[split][class_name] += 1

    # ── Summary 
    print()
    print("=" * 60)
    print("DONE — Final split summary:")
    print("=" * 60)
    print(f"{'CLASS':<26} {'TRAIN':>7} {'VAL':>7} {'TEST':>7} {'TOTAL':>7}")
    print("-" * 60)
    for name in CLASS_NAMES.values():
        tr = split_counts["train"][name]
        va = split_counts["val"][name]
        te = split_counts["test"][name]
        print(f"  {name:<24} {tr:>7,} {va:>7,} {te:>7,} {tr+va+te:>7,}")
    print("=" * 60)
    print()
    print("Next step:")
    print("  Update configs/base_config.yaml:")
    print("    model.num_classes: 10")
    print("    data.classes: [disturbed, merging, round_smooth, in_between,")
    print("                   cigar_shaped, barred_spiral, unbarred_tight_spiral,")
    print("                   unbarred_loose_spiral, edge_on_no_bulge, edge_on_with_bulge]")


if __name__ == "__main__":
    main()