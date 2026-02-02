import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Read raw data
    print("Reading raw data from ./input...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found.")

    # Read data
    # We load the full dataframe to perform stratified splitting based on the target
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Split training data into Train and Validation
    print("Performing stratified split (80:20)...")

    # Stratified split ensures class distribution is preserved
    train_subset, val_subset = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["target"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # 3. Generate Metadata Files
    print("Generating metadata files...")

    # Helper function to create metadata DataFrame
    def create_meta_df(df, filename, include_target=True):
        # We store ID, Target (if available), and the relative source path
        meta_data = {"id": df["id"], "source_path": filename}  # Relative to ./input
        if include_target:
            meta_data["target"] = df["target"]
        return pd.DataFrame(meta_data)

    # Create DataFrames
    train_meta = create_meta_df(train_subset, "train.csv", include_target=True)
    val_meta = create_meta_df(val_subset, "train.csv", include_target=True)
    test_meta = create_meta_df(test_df, "test.csv", include_target=False)

    # Define output paths
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Save to disk
    train_meta.to_csv(train_meta_path, index=False)
    val_meta.to_csv(val_meta_path, index=False)
    test_meta.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    # 4. Verification Steps
    print("\n--- Starting Verification ---")

    # Load datasets from metadata
    train_meta_loaded = pd.read_csv(train_meta_path)
    val_meta_loaded = pd.read_csv(val_meta_path)
    test_meta_loaded = pd.read_csv(test_meta_path)

    # A. Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train set samples: {len(train_meta_loaded)}")
    print(f"Val set samples:   {len(val_meta_loaded)}")
    print(f"Test set samples:  {len(test_meta_loaded)}")

    print(f"Train target mean: {train_meta_loaded['target'].mean():.4f}")
    print(f"Val target mean:   {val_meta_loaded['target'].mean():.4f}")

    # B. File Path Check
    print("\nChecking file paths...")

    def verify_paths(df, name):
        # Randomly check 1000 paths
        n_samples = min(1000, len(df))
        sample = df.sample(n=n_samples, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            rel_path = row["source_path"]
            # Resolve relative to ./input
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        ratio = missing_count / n_samples
        print(f"{name}: Missing path ratio = {ratio:.4f}")

        if ratio > 0.5:
            print(f"Sample missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Missing file ratio for {name} is {ratio}, which exceeds 0.5"
            )

    verify_paths(train_meta_loaded, "Train Metadata")
    verify_paths(val_meta_loaded, "Val Metadata")
    verify_paths(test_meta_loaded, "Test Metadata")

    # C. Validation Split Requirements Check
    print("\nVerifying Split Requirements...")

    # Check 1: Ratio 80:20
    total_train = len(train_meta_loaded)
    total_val = len(val_meta_loaded)
    total = total_train + total_val
    val_ratio = total_val / total

    print(f"Validation Split Ratio: {val_ratio:.4f}")
    # Allow very minor floating point deviation, though exact integers usually align perfectly
    if not (0.199 <= val_ratio <= 0.201):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is not approximately 0.2 (20%)"
        )

    # Check 2: Stratification
    # We compare the mean of the target variable (proportion of class 1)
    train_mean = train_meta_loaded["target"].mean()
    val_mean = val_meta_loaded["target"].mean()

    diff = abs(train_mean - val_mean)
    print(f"Class distribution difference (mean target): {diff:.6f}")

    # Stratified split should result in very similar means
    if diff > 0.01:
        raise AssertionError(
            f"Stratification failed. Class distributions differ by {diff:.6f}"
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
