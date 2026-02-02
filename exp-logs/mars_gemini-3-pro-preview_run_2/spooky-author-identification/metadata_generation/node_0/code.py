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
    """
    Reads raw data, performs stratified split, and saves metadata files.
    """
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # 1. Load Raw Data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Loaded raw training data: {df_train_full.shape}")
    print(f"Loaded raw test data: {df_test.shape}")

    # 2. Split Training Data (Stratified)
    # The task is classification with 'author' as the target.
    # We use stratified sampling to maintain class distribution.
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        stratify=df_train_full["author"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # 3. Add source file information (relative path to input)
    # This acts as the 'file path' metadata pointing to origin
    train_df["source_path"] = "train.csv"
    val_df["source_path"] = "train.csv"
    df_test["source_path"] = "test.csv"

    # 4. Save Metadata
    # We save the full content as metadata for efficient loading later.
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")
    return train_meta_path, val_meta_path, test_meta_path


def validate_metadata(train_path, val_path, test_path):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\n--- Starting Validation ---")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print(f"\nTraining Set: {df_train.shape}")
    print(df_train["author"].value_counts())
    print(f"\nValidation Set: {df_val.shape}")
    print(df_val["author"].value_counts())
    print(f"\nTest Set: {df_test.shape}")

    # 2. Check File Paths
    # We check if the 'source_path' column resolves to an existing file in ./input
    # We combine all dfs to check a random sample of paths
    all_dfs = [df_train, df_val, df_test]

    for i, df in enumerate(all_dfs):
        name = ["Train", "Val", "Test"][i]
        if "source_path" in df.columns:
            paths = df["source_path"].values
            # Select up to 1000 random paths
            n_samples = min(1000, len(paths))
            indices = np.random.choice(len(paths), n_samples, replace=False)
            sampled_paths = paths[indices]

            missing_count = 0
            missing_samples = []

            for p in sampled_paths:
                full_path = os.path.join(INPUT_DIR, p)
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_samples) < 5:
                        missing_samples.append(p)

            missing_ratio = missing_count / n_samples
            print(f"[{name}] Missing file ratio: {missing_ratio:.4f}")

            if missing_ratio > 0.5:
                print(f"Sample missing paths: {missing_samples}")
                raise FileNotFoundError(f"Too many missing files in {name} metadata.")
        else:
            print(f"[{name}] No 'source_path' column found to validate.")

    # 3. Verify Split Requirements
    # Check Split Ratio
    total_train_val = len(df_train) + len(df_val)
    actual_train_ratio = len(df_train) / total_train_val
    print(f"\nActual Train Ratio: {actual_train_ratio:.4f} (Target: {1.0 - VAL_SIZE})")

    # Allow small deviation due to rounding
    if not (0.79 < actual_train_ratio < 0.81):
        raise AssertionError(
            f"Split ratio validation failed. Expected ~0.8, got {actual_train_ratio}"
        )

    # Check Stratification
    # We compare the distribution of classes in Train vs Val
    train_dist = df_train["author"].value_counts(normalize=True).sort_index()
    val_dist = df_val["author"].value_counts(normalize=True).sort_index()

    print("\nClass Distribution (Train):")
    print(train_dist)
    print("\nClass Distribution (Val):")
    print(val_dist)

    # Calculate maximum difference in class proportions
    diff = (train_dist - val_dist).abs().max()
    print(f"Max class distribution difference: {diff:.6f}")

    # Tolerance for stratification mismatch
    if diff > 0.015:
        raise AssertionError(
            f"Stratification validation failed. Max difference {diff} exceeds tolerance."
        )

    print("\nValidation Successful!")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        validate_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
