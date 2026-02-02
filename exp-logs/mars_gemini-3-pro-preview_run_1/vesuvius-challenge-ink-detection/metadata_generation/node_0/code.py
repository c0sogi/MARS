import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
PATCH_SIZE = 512
STRIDE = 512
VAL_SIZE = 0.2

# Handle large images
Image.MAX_IMAGE_PIXELS = None


def setup_directories():
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)


def get_relative_path(full_path):
    return os.path.relpath(full_path, INPUT_DIR)


def generate_patches(fragment_dir, fragment_id, mode="train"):
    """
    Generates patch metadata for a given fragment.
    """
    patches = []

    # Define paths
    mask_path = os.path.join(fragment_dir, "mask.png")
    surface_vol_path = os.path.join(fragment_dir, "surface_volume")

    # Check if files exist
    if not os.path.exists(mask_path):
        return []

    # Load mask to determine valid areas and dimensions
    mask_img = Image.open(mask_path)
    width, height = mask_img.size
    mask_arr = np.array(mask_img)

    # For training, load labels
    ink_arr = None
    inklabels_rel_path = None
    ir_rel_path = None

    if mode == "train":
        ink_path = os.path.join(fragment_dir, "inklabels.png")
        if os.path.exists(ink_path):
            ink_arr = np.array(Image.open(ink_path))
            inklabels_rel_path = get_relative_path(ink_path)

        ir_path = os.path.join(fragment_dir, "ir.png")
        if os.path.exists(ir_path):
            ir_rel_path = get_relative_path(ir_path)

    mask_rel_path = get_relative_path(mask_path)
    surface_vol_rel_path = get_relative_path(surface_vol_path)

    # Generate grid
    for y in range(0, height, STRIDE):
        for x in range(0, width, STRIDE):
            # Adjust for boundaries
            w = min(PATCH_SIZE, width - x)
            h = min(PATCH_SIZE, height - y)

            # Extract mask patch to check validity
            mask_patch = mask_arr[y : y + h, x : x + w]

            # Only keep patch if it contains valid pixels
            if np.any(mask_patch):
                patch_info = {
                    "sample_id": f"{fragment_id}_{y}_{x}",
                    "fragment_id": fragment_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "mask_path": mask_rel_path,
                    "surface_volume_path": surface_vol_rel_path,
                }

                if mode == "train":
                    patch_info["inklabels_path"] = inklabels_rel_path
                    patch_info["ir_path"] = ir_rel_path

                    # Determine label (has ink or not)
                    if ink_arr is not None:
                        ink_patch = ink_arr[y : y + h, x : x + w]
                        patch_info["has_ink"] = 1 if np.any(ink_patch) else 0
                    else:
                        patch_info["has_ink"] = 0

                patches.append(patch_info)

    return patches


def main():
    setup_directories()

    # --- Process Training Data ---
    train_dir = os.path.join(INPUT_DIR, "train")
    train_patches = []

    if os.path.exists(train_dir):
        fragment_ids = sorted(
            [
                d
                for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
        )
        print(f"Found training fragments: {fragment_ids}")

        for fid in fragment_ids:
            f_path = os.path.join(train_dir, fid)
            patches = generate_patches(f_path, fid, mode="train")
            train_patches.extend(patches)
            print(f"  Fragment {fid}: Generated {len(patches)} patches.")

    df_full_train = pd.DataFrame(train_patches)

    # --- Split Train/Val ---
    if not df_full_train.empty:
        # Stratified Shuffle Split based on 'has_ink'
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )

        # We split based on the 'has_ink' column to ensure ink is represented in validation
        split_idx = list(splitter.split(df_full_train, df_full_train["has_ink"]))
        train_idx, val_idx = split_idx[0]

        df_train = df_full_train.iloc[train_idx].copy()
        df_val = df_full_train.iloc[val_idx].copy()

        # Save
        df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
        df_val.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
        print(
            f"Saved train.csv ({len(df_train)} samples) and val.csv ({len(df_val)} samples)."
        )
    else:
        # Handle case with no training data (e.g. empty input dir in some test environments)
        pd.DataFrame(columns=["sample_id"]).to_csv(
            os.path.join(METADATA_DIR, "train.csv"), index=False
        )
        pd.DataFrame(columns=["sample_id"]).to_csv(
            os.path.join(METADATA_DIR, "val.csv"), index=False
        )
        print("Warning: No training data found.")

    # --- Process Test Data ---
    test_dir = os.path.join(INPUT_DIR, "test")
    test_patches = []

    if os.path.exists(test_dir):
        fragment_ids = sorted(
            [
                d
                for d in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, d))
            ]
        )
        print(f"Found test fragments: {fragment_ids}")

        for fid in fragment_ids:
            f_path = os.path.join(test_dir, fid)
            patches = generate_patches(f_path, fid, mode="test")
            test_patches.extend(patches)
            print(f"  Fragment {fid}: Generated {len(patches)} patches.")

    df_test = pd.DataFrame(test_patches)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)
    print(f"Saved test.csv ({len(df_test)} samples).")

    # --- Verification ---
    print("\n--- Starting Verification ---")

    # 1. Load Datasets
    try:
        df_train_v = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        df_val_v = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        df_test_v = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    except Exception as e:
        raise AssertionError(f"Failed to load generated metadata files: {e}")

    # 2. Summary Statistics
    print(f"Train samples: {len(df_train_v)}")
    if "has_ink" in df_train_v.columns:
        print(
            f"Train Ink distribution:\n{df_train_v['has_ink'].value_counts(normalize=True)}"
        )

    print(f"Val samples: {len(df_val_v)}")
    if "has_ink" in df_val_v.columns:
        print(
            f"Val Ink distribution:\n{df_val_v['has_ink'].value_counts(normalize=True)}"
        )

    print(f"Test samples: {len(df_test_v)}")

    # 3. Path Verification
    # Collect all path-like columns
    path_cols = ["mask_path", "inklabels_path", "ir_path", "surface_volume_path"]
    paths_to_check = []

    for df in [df_train_v, df_val_v, df_test_v]:
        if df.empty:
            continue
        # Sample up to 340 rows per df to get approx 1000 total if all exist
        sample_n = min(len(df), 340)
        sample_df = df.sample(n=sample_n, random_state=RANDOM_STATE)

        for col in path_cols:
            if col in sample_df.columns:
                # Filter out NaNs
                valid_paths = sample_df[col].dropna().tolist()
                paths_to_check.extend(valid_paths)

    # Shuffle and pick 1000
    if len(paths_to_check) > 1000:
        random.seed(RANDOM_STATE)
        paths_to_check = random.sample(paths_to_check, 1000)

    print(f"Checking {len(paths_to_check)} file paths...")

    missing_count = 0
    missing_samples = []

    for p in paths_to_check:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / len(paths_to_check) if paths_to_check else 0
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing paths:", missing_samples)
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio} exceeds threshold 0.5"
        )

    # 4. Split Verification
    if not df_train_v.empty and not df_val_v.empty:
        total_train_val = len(df_train_v) + len(df_val_v)
        actual_val_ratio = len(df_val_v) / total_train_val
        print(f"Actual validation ratio: {actual_val_ratio:.4f}")

        # Allow small deviation due to discrete number of patches
        if abs(actual_val_ratio - VAL_SIZE) > 0.05:
            raise AssertionError(
                f"Validation split ratio {actual_val_ratio} deviates significantly from {VAL_SIZE}"
            )

        # Verify Stratification
        train_ink_rate = df_train_v["has_ink"].mean()
        val_ink_rate = df_val_v["has_ink"].mean()
        print(f"Train Ink Rate: {train_ink_rate:.4f}, Val Ink Rate: {val_ink_rate:.4f}")

        if abs(train_ink_rate - val_ink_rate) > 0.1:
            raise AssertionError(
                "Stratification failed: significant difference in ink distribution between train and val."
            )

    print("Verification passed successfully.")


if __name__ == "__main__":
    main()
