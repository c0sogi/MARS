import os
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

    # Load raw data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Generate file paths relative to ./input
    # Format: train/{id}/geometry.xyz or test/{id}/geometry.xyz
    df_train_full["file_path"] = df_train_full["id"].apply(
        lambda x: os.path.join("train", str(x), "geometry.xyz")
    )
    df_test["file_path"] = df_test["id"].apply(
        lambda x: os.path.join("test", str(x), "geometry.xyz")
    )

    # Stratified Split for Validation
    # Since this is a regression task, we bin the target 'formation_energy_ev_natom' to stratify
    # We use qcut to create bins. If there are not enough unique values, we reduce bins.
    num_bins = 10
    try:
        df_train_full["stratify_bin"] = pd.qcut(
            df_train_full["formation_energy_ev_natom"],
            q=num_bins,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        # Fallback if qcut fails (e.g. too few unique values), just use random shuffle without stratification
        print("Warning: Could not bin target for stratification. Using random split.")
        df_train_full["stratify_bin"] = 0

    # Split the data
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df_train_full["stratify_bin"],
    )

    # Remove the temporary stratification column
    train_df = train_df.drop(columns=["stratify_bin"])
    val_df = val_df.drop(columns=["stratify_bin"])
    # Note: df_train_full still has it, but we are saving the split dfs.

    # Save metadata files
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(
        f"Metadata generated:\n {train_meta_path}\n {val_meta_path}\n {test_meta_path}"
    )


def check_file_paths(df, dataset_name):
    """
    Checks if file paths in the dataframe exist in the input directory.
    Raises error if missing ratio > 0.5.
    """
    if "file_path" not in df.columns:
        return

    # Sample up to 1000 paths
    sample_size = min(1000, len(df))
    sample_paths = df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / sample_size
    print(
        f"[{dataset_name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
    )

    if missing_ratio > 0.5:
        print(f"Sample missing paths from {dataset_name}:")
        for p in missing_samples:
            print(f"  {p}")
        raise FileNotFoundError(
            f"More than 50% of file paths are missing for {dataset_name} dataset."
        )


def validate_metadata():
    print("\n--- Validating Metadata ---")

    # Load generated metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Summary Statistics
    print("\nTraining Set Summary:")
    print(f"Shape: {train_df.shape}")
    if "formation_energy_ev_natom" in train_df.columns:
        print(train_df[["formation_energy_ev_natom", "bandgap_energy_ev"]].describe())

    print("\nValidation Set Summary:")
    print(f"Shape: {val_df.shape}")
    if "formation_energy_ev_natom" in val_df.columns:
        print(val_df[["formation_energy_ev_natom", "bandgap_energy_ev"]].describe())

    print("\nTest Set Summary:")
    print(f"Shape: {test_df.shape}")

    # 2. Check File Paths
    check_file_paths(train_df, "Train")
    check_file_paths(val_df, "Validation")
    check_file_paths(test_df, "Test")

    # 3. Verify Validation Split
    total_train_val = len(train_df) + len(val_df)
    actual_val_ratio = len(val_df) / total_train_val
    print(f"\nActual Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow a small tolerance for split ratio due to integer division
    if not (abs(actual_val_ratio - VAL_SIZE) < 0.01):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates significantly from expected {VAL_SIZE}"
        )

    # Verify labels exist in train/val
    required_labels = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    for label in required_labels:
        if label not in train_df.columns:
            raise AssertionError(f"Missing label {label} in training metadata")
        if label not in val_df.columns:
            raise AssertionError(f"Missing label {label} in validation metadata")

    print("\nValidation successful.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
