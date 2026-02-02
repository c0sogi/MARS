import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- 1. Process Training Data ---
    train_noisy_dir = os.path.join(INPUT_DIR, "train")
    train_cleaned_dir = os.path.join(INPUT_DIR, "train_cleaned")

    # List files
    noisy_files = glob.glob(os.path.join(train_noisy_dir, "*.png"))
    clean_files = glob.glob(os.path.join(train_cleaned_dir, "*.png"))

    # Map IDs to relative paths
    # ID is the filename without extension (e.g., '101' from '101.png')
    noisy_map = {
        os.path.splitext(os.path.basename(f))[0]: os.path.relpath(f, INPUT_DIR)
        for f in noisy_files
    }
    clean_map = {
        os.path.splitext(os.path.basename(f))[0]: os.path.relpath(f, INPUT_DIR)
        for f in clean_files
    }

    # Find common IDs (intersection) to ensure valid pairs
    common_ids = sorted(list(set(noisy_map.keys()) & set(clean_map.keys())))

    if not common_ids:
        raise ValueError(
            "No matching pairs found between input/train and input/train_cleaned"
        )

    # Create DataFrame
    data = []
    for img_id in common_ids:
        data.append(
            {
                "id": img_id,
                "noisy_image_path": noisy_map[img_id],
                "clean_image_path": clean_map[img_id],
            }
        )

    full_train_df = pd.DataFrame(data)

    # Split into Train and Validation
    # Using random split as there are no provided classes for stratification or groups
    train_df, val_df = train_test_split(
        full_train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # --- 2. Process Test Data ---
    test_dir = os.path.join(INPUT_DIR, "test")
    test_files = glob.glob(os.path.join(test_dir, "*.png"))

    test_data = []
    for f in test_files:
        img_id = os.path.splitext(os.path.basename(f))[0]
        test_data.append(
            {"id": img_id, "noisy_image_path": os.path.relpath(f, INPUT_DIR)}
        )

    test_df = pd.DataFrame(test_data)

    # --- 3. Save Metadata ---
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")
    return len(full_train_df)


def verify_metadata(total_train_samples):
    print("Starting verification...")

    # Load datasets
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("-" * 30)
    print("Summary Statistics")
    print("-" * 30)
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Total training samples (original): {total_train_samples}")
    print("-" * 30)

    # 2. File Existence Check
    def check_paths(df, name):
        paths = []
        if "noisy_image_path" in df.columns:
            paths.extend(df["noisy_image_path"].tolist())
        if "clean_image_path" in df.columns:
            paths.extend(df["clean_image_path"].tolist())

        if not paths:
            print(f"Warning: No paths found in {name} metadata.")
            return

        # Randomly sample up to 1000 paths
        sample_size = min(1000, len(paths))
        # Use a fixed seed for reproducibility of the check if needed, though not strictly required
        rng = np.random.default_rng(seed=42)
        sampled_paths = rng.choice(paths, size=sample_size, replace=False)

        missing_count = 0
        missing_examples = []

        for p in sampled_paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        missing_ratio = missing_count / sample_size
        print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Example missing paths in {name}:")
            for mp in missing_examples:
                print(f"  - {mp}")
            raise FileNotFoundError(
                f"Error: More than 50% of files missing in {name} metadata."
            )

    print("Checking file paths...")
    check_paths(train_df, "train")
    check_paths(val_df, "validation")
    check_paths(test_df, "test")

    # 3. Verify Validation Split Requirements
    print("Verifying split requirements...")

    # Check split ratio
    actual_val_ratio = len(val_df) / total_train_samples
    expected_ratio = VAL_SIZE
    tolerance = 0.01  # Allow small deviation for integer counts

    print(f"Target validation ratio: {expected_ratio}")
    print(f"Actual validation ratio: {actual_val_ratio:.4f}")

    if not (
        expected_ratio - tolerance <= actual_val_ratio <= expected_ratio + tolerance
    ):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates significantly from expected {expected_ratio}"
        )

    # Check for Data Leakage (Intersection of IDs)
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    intersection = train_ids.intersection(val_ids)

    if intersection:
        raise AssertionError(
            f"Data leakage detected! {len(intersection)} IDs present in both train and validation sets."
        )

    print("All checks passed successfully.")


if __name__ == "__main__":
    try:
        total_samples = generate_metadata()
        verify_metadata(total_samples)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
