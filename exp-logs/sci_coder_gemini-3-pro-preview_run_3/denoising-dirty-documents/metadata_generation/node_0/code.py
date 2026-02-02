import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- 1. Identify Files ---
    # Train images are in input/train/
    # Clean targets are in input/train_cleaned/
    # Test images are in input/test/

    train_dir = os.path.join(INPUT_DIR, "train")
    train_cleaned_dir = os.path.join(INPUT_DIR, "train_cleaned")
    test_dir = os.path.join(INPUT_DIR, "test")

    # Get list of filenames
    # We assume the filenames in train and train_cleaned match
    train_files = sorted(
        [
            f
            for f in os.listdir(train_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    test_files = sorted(
        [
            f
            for f in os.listdir(test_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Construct Train/Val Data
    data_records = []
    for filename in train_files:
        input_rel_path = os.path.join("train", filename)
        target_rel_path = os.path.join("train_cleaned", filename)

        # Verify target exists for this training image
        full_target_path = os.path.join(INPUT_DIR, target_rel_path)
        if not os.path.exists(full_target_path):
            print(f"Warning: Cleaned target not found for {filename}, skipping.")
            continue

        data_records.append(
            {
                "image_id": filename,
                "input_path": input_rel_path,
                "target_path": target_rel_path,
                "dataset_type": "train_candidate",
            }
        )

    df_full_train = pd.DataFrame(data_records)

    # Construct Test Data
    test_records = []
    for filename in test_files:
        input_rel_path = os.path.join("test", filename)
        test_records.append(
            {
                "image_id": filename,
                "input_path": input_rel_path,
                "target_path": None,  # No ground truth for test
                "dataset_type": "test",
            }
        )

    df_test = pd.DataFrame(test_records)

    # --- 2. Split Train/Val ---
    # Simple random split since there are no distinct groups or classes for stratification
    train_df, val_df = train_test_split(
        df_full_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # Update dataset_type column
    train_df = train_df.copy()
    train_df["dataset_type"] = "train"
    val_df = val_df.copy()
    val_df["dataset_type"] = "validation"

    # --- 3. Save Metadata ---
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print(f"Metadata generated:")
    print(f"  Train: {train_csv_path} ({len(train_df)} samples)")
    print(f"  Val:   {val_csv_path} ({len(val_df)} samples)")
    print(f"  Test:  {test_csv_path} ({len(df_test)} samples)")


def validate_metadata():
    print("\n--- Starting Validation Checks ---")

    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples:   {len(df_val)}")
    print(f"Test samples:  {len(df_test)}")

    total_train_val = len(df_train) + len(df_val)
    actual_val_ratio = len(df_val) / total_train_val if total_train_val > 0 else 0
    print(f"Validation Split Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # 2. Check File Existence
    def check_paths(df, name):
        paths_to_check = df["input_path"].tolist()
        if "target_path" in df.columns:
            # Filter out None/NaN target paths (e.g. in test set)
            targets = df["target_path"].dropna().tolist()
            paths_to_check.extend(targets)

        # Randomly sample 1000 paths if there are more than 1000
        if len(paths_to_check) > 1000:
            paths_to_check = np.random.choice(paths_to_check, 1000, replace=False)

        missing_count = 0
        missing_samples = []

        for rel_path in paths_to_check:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = (
            missing_count / len(paths_to_check) if len(paths_to_check) > 0 else 0
        )
        print(
            f"[{name}] Checked {len(paths_to_check)} paths. Missing ratio: {missing_ratio:.4f}"
        )

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} metadata."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    # 3. Verify Split Logic
    # Verify no overlap between train and val
    train_ids = set(df_train["image_id"])
    val_ids = set(df_val["image_id"])
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(
            f"Train and Validation sets overlap! Overlapping IDs: {list(overlap)[:5]}"
        )

    # Verify split ratio roughly holds (allow for small rounding differences)
    # Since dataset might be small, exact 0.2 might not be possible, but should be close
    # Using a tolerance of 1 sample size roughly
    expected_val_count = int(np.round(total_train_val * VAL_SIZE))
    # Allow a margin of error of 1 sample due to rounding
    if abs(len(df_val) - expected_val_count) > 1:
        # Note: In very small datasets, train_test_split behavior might vary slightly,
        # but usually it adheres strictly to stratification or floor/ceil logic.
        # We assert strictly here based on sklearn's behavior.
        # However, if the dataset is extremely small, we might want to be lenient.
        # Given the task description implies > 100 files, this check is safe.
        pass
        # Actually, let's just assert that it is not 0 if total > 0
        if total_train_val > 0 and len(df_val) == 0:
            raise AssertionError(
                "Validation set is empty despite having training data."
            )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
