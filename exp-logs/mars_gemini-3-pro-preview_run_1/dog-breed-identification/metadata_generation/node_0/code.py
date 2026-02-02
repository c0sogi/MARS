import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def main():
    # 1. Setup Directory
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Starting metadata generation...")

    # 2. Process Training and Validation Data
    labels_path = os.path.join(INPUT_DIR, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"{labels_path} not found.")

    # Load labels
    df = pd.read_csv(labels_path)

    # Construct relative file paths: train/<id>.jpg
    df["file_path"] = df["id"].apply(lambda x: os.path.join("train", f"{x}.jpg"))

    # Perform Stratified Split (80% Train, 20% Val)
    # Stratify by 'breed' to ensure class distribution is preserved
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["breed"], random_state=RANDOM_STATE
    )

    # Save Train and Val metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print(f"Generated {train_csv_path} with {len(train_df)} samples.")
    print(f"Generated {val_csv_path} with {len(val_df)} samples.")

    # 3. Process Test Data
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"{sample_sub_path} not found.")

    # Load sample submission to get test IDs
    test_df_raw = pd.read_csv(sample_sub_path)

    # Create test metadata DataFrame
    test_df = pd.DataFrame()
    test_df["id"] = test_df_raw["id"]
    # Construct relative file paths: test/<id>.jpg
    test_df["file_path"] = test_df["id"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    # Save Test metadata
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Generated {test_csv_path} with {len(test_df)} samples.")

    # 4. Verification Step
    print("\n--- Verifying Generated Metadata ---")

    # Reload datasets to verify integrity
    train_meta = pd.read_csv(train_csv_path)
    val_meta = pd.read_csv(val_csv_path)
    test_meta = pd.read_csv(test_csv_path)

    # 4.1 Print Summary Statistics
    print(
        f"Train Shape: {train_meta.shape}, Unique Breeds: {train_meta['breed'].nunique()}"
    )
    print(
        f"Val Shape:   {val_meta.shape}, Unique Breeds: {val_meta['breed'].nunique()}"
    )
    print(f"Test Shape:  {test_meta.shape}")

    # 4.2 Check File Path Existence
    def verify_paths(df, name):
        # Check random 1000 paths
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Paths in metadata are relative to ./input
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        ratio = missing_count / sample_size
        print(f"[{name}] Missing file ratio: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Examples of missing files in {name}: {missing_examples}")
            raise FileNotFoundError(
                f"Verification Failed: More than 50% of files missing in {name} metadata."
            )

    verify_paths(train_meta, "Train")
    verify_paths(val_meta, "Validation")
    verify_paths(test_meta, "Test")

    # 4.3 Verify Validation Split Requirements
    # Check 1: Split Ratio (approx 20%)
    total_samples = len(train_meta) + len(val_meta)
    val_ratio = len(val_meta) / total_samples
    print(f"Validation Split Ratio: {val_ratio:.4f}")

    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio} deviates significantly from 0.2"
        )

    # Check 2: Stratification Success
    # Compare normalized value counts (probabilities) of classes
    train_dist = train_meta["breed"].value_counts(normalize=True).sort_index()
    val_dist = val_meta["breed"].value_counts(normalize=True).sort_index()

    # Reindex to ensure both have same keys for comparison (fill missing with 0)
    all_breeds = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_breeds, fill_value=0)
    val_dist = val_dist.reindex(all_breeds, fill_value=0)

    # Calculate Mean Absolute Error between class distributions
    mae = (train_dist - val_dist).abs().mean()
    print(f"Class Distribution MAE between Train and Val: {mae:.6f}")

    # Assert that distributions are similar.
    # With 120 classes, uniform prob is ~0.008. A high MAE would indicate bad stratification.
    if mae > 0.01:
        raise AssertionError(
            "Stratification check failed: Class distributions in Train and Validation differ significantly."
        )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
