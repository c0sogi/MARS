import os
import glob
import random
import shutil
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# Constants
INPUT_DIR = Path("./input")
META_DIR = Path("./metadata")
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


def get_fragment_info(fragment_dir, split_name):
    """
    Parses a fragment directory to extract file paths and metadata.
    """
    fragment_id = fragment_dir.name

    # Define expected paths
    mask_path = fragment_dir / "mask.png"
    surface_vol_path = fragment_dir / "surface_volume"

    # Train/Val specific paths
    inklabels_path = fragment_dir / "inklabels.png"
    ir_path = fragment_dir / "ir.png"

    info = {
        "fragment_id": fragment_id,
        "split": split_name,
        "surface_volume_path": str(surface_vol_path.relative_to(INPUT_DIR)),
        "mask_path": (
            str(mask_path.relative_to(INPUT_DIR)) if mask_path.exists() else None
        ),
    }

    # Get dimensions from mask
    if mask_path.exists():
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            info["height"] = h
            info["width"] = w
        else:
            info["height"] = 0
            info["width"] = 0
    else:
        info["height"] = 0
        info["width"] = 0

    # Add labels and IR for train/val
    if split_name in ["train", "val"]:
        info["inklabels_path"] = (
            str(inklabels_path.relative_to(INPUT_DIR))
            if inklabels_path.exists()
            else None
        )
        info["ir_path"] = (
            str(ir_path.relative_to(INPUT_DIR)) if ir_path.exists() else None
        )
    else:
        info["inklabels_path"] = None
        info["ir_path"] = None

    return info


def generate_metadata():
    # Ensure metadata directory exists and is empty of previous runs
    if META_DIR.exists():
        shutil.rmtree(META_DIR)
    META_DIR.mkdir(parents=True)

    # 1. Parse Test Data
    test_fragments = []
    test_dir = INPUT_DIR / "test"
    if test_dir.exists():
        for f_dir in sorted(test_dir.iterdir()):
            if f_dir.is_dir():
                test_fragments.append(get_fragment_info(f_dir, "test"))

    # 2. Parse Train Data
    train_fragments_all = []
    train_dir = INPUT_DIR / "train"
    if train_dir.exists():
        for f_dir in sorted(train_dir.iterdir()):
            if f_dir.is_dir():
                train_fragments_all.append(
                    get_fragment_info(f_dir, "train")
                )  # Temp label

    # 3. Split Train/Val
    # Shuffle with fixed seed
    random.seed(RANDOM_STATE)
    random.shuffle(train_fragments_all)

    n_total = len(train_fragments_all)
    n_train = int(n_total * TRAIN_RATIO)

    # Ensure at least one validation sample if possible
    if n_total > 1 and n_train == n_total:
        n_train = n_total - 1
    elif n_total > 1 and n_train == 0:
        n_train = 1

    train_subset = train_fragments_all[:n_train]
    val_subset = train_fragments_all[n_train:]

    # Update split label in dicts
    for item in val_subset:
        item["split"] = "val"

    # Create DataFrames
    df_train = pd.DataFrame(train_subset)
    df_val = pd.DataFrame(val_subset)
    df_test = pd.DataFrame(test_fragments)

    # Save to CSV
    df_train.to_csv(META_DIR / "train.csv", index=False)
    if not df_val.empty:
        df_val.to_csv(META_DIR / "val.csv", index=False)
    if not df_test.empty:
        df_test.to_csv(META_DIR / "test.csv", index=False)

    print(f"Metadata generated in {META_DIR}")


def validate_metadata():
    print("\n--- Validating Metadata ---")

    datasets = {}
    for name in ["train", "val", "test"]:
        p = META_DIR / f"{name}.csv"
        if p.exists():
            datasets[name] = pd.read_csv(p)
        else:
            datasets[name] = pd.DataFrame()

    # 1. Summary Statistics
    for name, df in datasets.items():
        if df.empty:
            print(f"Dataset {name} is empty.")
            continue

        print(f"\nDataset: {name}")
        print(f"  Count: {len(df)}")
        if "fragment_id" in df.columns:
            print(f"  Fragment IDs: {df['fragment_id'].tolist()}")
        if "width" in df.columns and "height" in df.columns:
            print(
                f"  Avg Shape (HxW): {df['height'].mean():.1f} x {df['width'].mean():.1f}"
            )

    # 2. Verify Split Logic (Stratification/Grouping)
    train_ids = (
        set(datasets["train"]["fragment_id"]) if not datasets["train"].empty else set()
    )
    val_ids = (
        set(datasets["val"]["fragment_id"]) if not datasets["val"].empty else set()
    )

    # Intersection check
    intersection = train_ids.intersection(val_ids)
    if intersection:
        raise AssertionError(
            f"Train and Validation sets overlap! Overlapping IDs: {intersection}"
        )

    # Check if validation set exists (if we had enough data)
    # We know from the generation step we tried to create it.
    # If original train had >= 2 items, val should not be empty.
    # We can't strictly assert val is not empty if input only had 1 item, but assuming standard dataset:
    if len(train_ids) + len(val_ids) >= 2 and len(val_ids) == 0:
        raise AssertionError(
            "Validation set is empty despite having multiple training samples."
        )

    print("\nSplit verification passed: Train and Val are disjoint.")

    # 3. File Existence Check (Sampling 1000 paths)
    all_paths_to_check = []

    for name, df in datasets.items():
        if df.empty:
            continue

        # Collect columns that contain paths
        path_cols = [c for c in df.columns if "path" in c]

        for _, row in df.iterrows():
            # Add direct file paths
            for col in path_cols:
                if pd.notna(row[col]):
                    # If it's the volume directory, we want to check files INSIDE it to reach 1000 samples
                    if col == "surface_volume_path":
                        vol_dir = INPUT_DIR / row[col]
                        if vol_dir.exists():
                            # List all tifs
                            tifs = list(vol_dir.glob("*.tif"))
                            # Add relative paths of tifs
                            all_paths_to_check.extend(
                                [str(p.relative_to(INPUT_DIR)) for p in tifs]
                            )
                        else:
                            # Add the dir itself so it fails later
                            all_paths_to_check.append(row[col])
                    else:
                        all_paths_to_check.append(row[col])

    # Randomly sample 1000 paths if we have more than that
    if len(all_paths_to_check) > 1000:
        sampled_paths = random.sample(all_paths_to_check, 1000)
    else:
        sampled_paths = all_paths_to_check

    print(f"\nChecking existence of {len(sampled_paths)} sampled file paths...")

    missing_count = 0
    missing_samples = []

    for rel_path in sampled_paths:
        full_path = INPUT_DIR / rel_path
        if not full_path.exists():
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / len(sampled_paths) if len(sampled_paths) > 0 else 0
    print(f"Missing File Ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing files:")
        for m in missing_samples:
            print(f"  - {m}")
        raise FileNotFoundError(
            f"High missing file ratio detected ({missing_ratio:.2%}). Check path generation."
        )

    print("File existence check passed.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
