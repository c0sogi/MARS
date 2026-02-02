import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TILE_SIZE = 512  # Size of the crops
STRIDE = 512  # Non-overlapping for dataset generation, or overlapping if preferred.
# Using non-overlapping to keep sample count reasonable and distinct.
VALID_PIXEL_THRESHOLD = (
    0.05  # Fraction of pixels in mask that must be valid to keep the tile
)


def generate_metadata():
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    train_fragments = sorted(glob.glob(os.path.join(INPUT_DIR, "train", "*")))
    patch_data = []

    print(f"Found {len(train_fragments)} training fragments.")

    for frag_path in train_fragments:
        if not os.path.isdir(frag_path):
            continue

        frag_id = os.path.basename(frag_path)
        mask_path = os.path.join(frag_path, "mask.png")
        label_path = os.path.join(frag_path, "inklabels.png")
        volume_dir = os.path.join(frag_path, "surface_volume")

        # Relative paths for metadata
        rel_mask_path = os.path.relpath(mask_path, INPUT_DIR)
        rel_label_path = os.path.relpath(label_path, INPUT_DIR)
        rel_volume_dir = os.path.relpath(volume_dir, INPUT_DIR)

        if not os.path.exists(mask_path) or not os.path.exists(label_path):
            print(f"Skipping fragment {frag_id}: mask or labels missing.")
            continue

        # Read images
        # cv2.imread returns None if file doesn't exist, but we checked existence.
        # Use IMREAD_GRAYSCALE
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if mask is None or label is None:
            print(f"Error reading images for fragment {frag_id}")
            continue

        h, w = mask.shape

        # Generate patches
        for y in range(0, h, STRIDE):
            for x in range(0, w, STRIDE):
                # Handle edge cases by cropping if necessary, or just skip if too small
                # Here we take fixed crops. If it goes out of bounds, we clip.
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)

                # If the remaining chunk is too small, skip or pad.
                # For simplicity in this script, we skip chunks smaller than TILE_SIZE
                # to ensure consistent tensor shapes later, unless we are at the edge.
                # Let's strictly enforce size for this metadata generation to avoid complexity.
                if (y_end - y) < TILE_SIZE or (x_end - x) < TILE_SIZE:
                    continue

                mask_patch = mask[y:y_end, x:x_end]

                # Check validity
                valid_pixels = np.count_nonzero(mask_patch)
                total_pixels = mask_patch.size

                if (valid_pixels / total_pixels) >= VALID_PIXEL_THRESHOLD:
                    label_patch = label[y:y_end, x:x_end]
                    has_ink = 1 if np.count_nonzero(label_patch) > 0 else 0

                    patch_data.append(
                        {
                            "fragment_id": frag_id,
                            "x": x,
                            "y": y,
                            "width": TILE_SIZE,
                            "height": TILE_SIZE,
                            "has_ink": has_ink,
                            "mask_path": rel_mask_path,
                            "label_path": rel_label_path,
                            "volume_path": rel_volume_dir,
                        }
                    )

    df_patches = pd.DataFrame(patch_data)

    if len(df_patches) == 0:
        raise ValueError("No valid patches generated from training data.")

    print(f"Generated {len(df_patches)} patches from training data.")

    # Split Train/Val
    # Stratify by 'has_ink' to ensure ink representation in both sets
    train_df, val_df = train_test_split(
        df_patches,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_patches["has_ink"],
    )

    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "validation.csv"), index=False)

    # --- Process Test Data ---
    # For test, we usually predict on the whole image, so we provide fragment-level metadata.
    test_fragments = sorted(glob.glob(os.path.join(INPUT_DIR, "test", "*")))
    test_data = []

    for frag_path in test_fragments:
        if not os.path.isdir(frag_path):
            continue

        frag_id = os.path.basename(frag_path)
        mask_path = os.path.join(frag_path, "mask.png")
        volume_dir = os.path.join(frag_path, "surface_volume")

        test_data.append(
            {
                "fragment_id": frag_id,
                "mask_path": os.path.relpath(mask_path, INPUT_DIR),
                "volume_path": os.path.relpath(volume_dir, INPUT_DIR),
            }
        )

    test_df = pd.DataFrame(test_data)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)
    print(f"Generated metadata for {len(test_df)} test fragments.")


def validate_metadata():
    print("\n--- Validating Metadata ---")

    try:
        train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(METADATA_DIR, "validation.csv"))
        test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Metadata file missing: {e}")

    # 1. Summary Statistics
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    train_ink_ratio = train_df["has_ink"].mean()
    val_ink_ratio = val_df["has_ink"].mean()

    print(f"Train Ink Ratio: {train_ink_ratio:.4f}")
    print(f"Val Ink Ratio:   {val_ink_ratio:.4f}")

    # 2. Check Split Ratio
    total_train_val = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(f"Validation split ratio {val_ratio} is not close to 0.2")

    # 3. Check Stratification
    # Ratios should be roughly similar
    if abs(train_ink_ratio - val_ink_ratio) > 0.05:
        raise AssertionError(
            "Stratification failed: Ink ratios differ significantly between train and val."
        )

    # 4. Check File Paths
    # Collect paths from all dataframes
    # Columns containing paths: mask_path, label_path, volume_path
    all_paths = []

    if "mask_path" in train_df.columns:
        all_paths.extend(train_df["mask_path"].tolist())
    if "label_path" in train_df.columns:
        all_paths.extend(train_df["label_path"].tolist())
    if "volume_path" in train_df.columns:
        all_paths.extend(train_df["volume_path"].tolist())

    if "mask_path" in val_df.columns:
        all_paths.extend(val_df["mask_path"].tolist())
    if "label_path" in val_df.columns:
        all_paths.extend(val_df["label_path"].tolist())
    if "volume_path" in val_df.columns:
        all_paths.extend(val_df["volume_path"].tolist())

    if "mask_path" in test_df.columns:
        all_paths.extend(test_df["mask_path"].tolist())
    if "volume_path" in test_df.columns:
        all_paths.extend(test_df["volume_path"].tolist())

    # Remove duplicates
    all_paths = list(set(all_paths))

    # Select 1000 random paths (or all if less than 1000)
    check_count = min(len(all_paths), 1000)
    paths_to_check = random.sample(all_paths, check_count)

    missing_count = 0
    missing_samples = []

    for p in paths_to_check:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / check_count
    print(
        f"Checked {check_count} paths. Missing: {missing_count}. Ratio: {missing_ratio:.4f}"
    )

    if missing_ratio > 0.5:
        print("Sample missing paths:")
        for mp in missing_samples:
            print(f"  {mp}")
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio} exceeds threshold 0.5"
        )

    print("Validation successful.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
