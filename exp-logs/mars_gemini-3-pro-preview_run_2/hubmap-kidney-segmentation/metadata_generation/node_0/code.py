import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def generate_metadata():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading source files...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    info_csv_path = os.path.join(INPUT_DIR, "HuBMAP-20-dataset_information.csv")

    train_df = pd.read_csv(train_csv_path)
    info_df = pd.read_csv(info_csv_path)

    # 2. Preprocess Info DataFrame
    # Extract ID from image_file (remove .tiff extension) to match train.csv format
    info_df["id"] = info_df["image_file"].apply(lambda x: os.path.splitext(x)[0])

    # 3. Construct Training Metadata
    # Merge train labels with dataset info
    # Note: train.csv contains 'id' and 'encoding'. info_df contains metadata.
    full_train_df = pd.merge(train_df, info_df, on="id", how="left")

    # Construct relative paths for training data
    # Images are in input/train/{id}.tiff
    full_train_df["image_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("train", f"{x}.tiff")
    )
    full_train_df["json_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("train", f"{x}.json")
    )
    full_train_df["anatomical_json_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("train", f"{x}-anatomical-structure.json")
    )

    # 4. Create Train/Val Split
    # Use GroupShuffleSplit to respect patient_number
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

    # We split based on patient_number.
    # If patient_number is missing (NaN), we treat each as a unique group (though dataset desc implies no NaNs in patient_number)
    groups = full_train_df["patient_number"]

    train_idx, val_idx = next(gss.split(full_train_df, groups=groups))

    train_split_df = full_train_df.iloc[train_idx].copy()
    val_split_df = full_train_df.iloc[val_idx].copy()

    # 5. Construct Test Metadata
    print("Scanning test directory...")
    test_files = glob.glob(os.path.join(INPUT_DIR, "test", "*.tiff"))
    test_ids = [os.path.splitext(os.path.basename(f))[0] for f in test_files]

    test_df = pd.DataFrame({"id": test_ids})

    # Merge with info_df if available (dataset info covers both train and test usually)
    test_df = pd.merge(test_df, info_df, on="id", how="left")

    # Construct relative paths for test data
    test_df["image_path"] = test_df["id"].apply(
        lambda x: os.path.join("test", f"{x}.tiff")
    )
    test_df["json_path"] = test_df["id"].apply(
        lambda x: os.path.join("test", f"{x}.json")
    )
    test_df["anatomical_json_path"] = test_df["id"].apply(
        lambda x: os.path.join("test", f"{x}-anatomical-structure.json")
    )

    # 6. Save Metadata
    print("Saving metadata...")
    train_split_df.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_split_df.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    return train_split_df, val_split_df, test_df


def check_paths(df, name, input_dir):
    """Checks if files exist for a random sample of paths."""
    print(f"Checking file paths for {name}...")

    # Columns that contain paths
    path_cols = ["image_path", "json_path", "anatomical_json_path"]

    # Gather all paths
    all_paths = []
    for col in path_cols:
        if col in df.columns:
            all_paths.extend(df[col].dropna().tolist())

    if not all_paths:
        return

    # Sample 1000 or all if less
    sample_size = min(1000, len(all_paths))
    sampled_paths = np.random.choice(all_paths, sample_size, replace=False)

    missing_count = 0
    missing_samples = []

    for rel_path in sampled_paths:
        full_path = os.path.join(input_dir, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / sample_size
    print(f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})")

    if missing_ratio > 0.5:
        print("  Sample missing files:", missing_samples)
        raise FileNotFoundError(
            f"More than 50% of files are missing in {name} dataset."
        )


def verify_split(train_df, val_df):
    """Verifies that the split is valid and respects groups."""
    print("Verifying split integrity...")

    # Check 1: No ID overlap
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    overlap_ids = train_ids.intersection(val_ids)
    assert (
        len(overlap_ids) == 0
    ), f"Found overlapping IDs between train and val: {overlap_ids}"

    # Check 2: No Patient overlap (Group Leakage)
    train_patients = set(train_df["patient_number"])
    val_patients = set(val_df["patient_number"])
    overlap_patients = train_patients.intersection(val_patients)

    if overlap_patients:
        print(f"Train patients: {train_patients}")
        print(f"Val patients: {val_patients}")
        raise AssertionError(
            f"Data leakage detected! Patients found in both train and val: {overlap_patients}"
        )

    print("  Split integrity check passed (No patient leakage).")


def print_stats(name, df):
    print(f"\n--- {name} Dataset Statistics ---")
    print(f"Total samples: {len(df)}")
    if "patient_number" in df.columns:
        print(f"Unique patients: {df['patient_number'].nunique()}")
    if "sex" in df.columns:
        print(f"Sex distribution:\n{df['sex'].value_counts()}")
    if "race" in df.columns:
        print(f"Race distribution:\n{df['race'].value_counts()}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    try:
        # Generate metadata
        train_df, val_df, test_df = generate_metadata()

        # Load and Check
        INPUT_DIR = "./input"

        # Print Stats
        print_stats("Training", train_df)
        print_stats("Validation", val_df)
        print_stats("Test", test_df)

        # Verify Split
        verify_split(train_df, val_df)

        # Check Paths
        check_paths(train_df, "Training", INPUT_DIR)
        check_paths(val_df, "Validation", INPUT_DIR)
        check_paths(test_df, "Test", INPUT_DIR)

        print("\nMetadata generation and verification completed successfully.")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise e
