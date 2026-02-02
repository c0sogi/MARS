import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading source CSVs...")
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    sample_submission_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(train_labels_path) or not os.path.exists(
        sample_submission_path
    ):
        raise FileNotFoundError(
            "Could not find train_labels.csv or sample_submission.csv in ./input"
        )

    df_train_full = pd.read_csv(train_labels_path)
    df_test = pd.read_csv(sample_submission_path)

    # Construct relative file paths
    # The dataset structure is train/{first_char}/{id}.npy
    print("Constructing file paths...")

    def get_train_path(row):
        return os.path.join("train", str(row["id"])[0], f"{row['id']}.npy")

    def get_test_path(row):
        return os.path.join("test", str(row["id"])[0], f"{row['id']}.npy")

    df_train_full["file_path"] = df_train_full.apply(get_train_path, axis=1)
    df_test["file_path"] = df_test.apply(get_test_path, axis=1)

    # Split training data into train and validation
    print(f"Splitting data (Train/Val ratio: {1-VAL_SIZE}/{VAL_SIZE})...")
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        stratify=df_train_full["target"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save metadata
    print("Saving metadata files...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    # --- Verification Step ---
    print("\n=== Verification & Statistics ===")

    # 1. Load data back
    df_train_check = pd.read_csv(train_save_path)
    df_val_check = pd.read_csv(val_save_path)
    df_test_check = pd.read_csv(test_save_path)

    datasets = {
        "Train": df_train_check,
        "Validation": df_val_check,
        "Test": df_test_check,
    }

    # 2. Print Summary Statistics
    for name, df in datasets.items():
        print(f"\nDataset: {name}")
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        if "target" in df.columns:
            target_counts = df["target"].value_counts()
            target_ratio = df["target"].mean()
            print(f"  Target distribution:\n{target_counts}")
            print(f"  Positive class ratio: {target_ratio:.4f}")

    # 3. Check File Paths
    print("\nChecking file path resolution...")
    for name, df in datasets.items():
        # Sample up to 1000 paths
        n_sample = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=n_sample, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_examples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(
            f"  {name}: Missing file ratio = {missing_ratio:.4f} ({missing_count}/{n_sample})"
        )

        if missing_ratio > 0.5:
            print(f"  Sample missing paths from {name}:")
            for p in missing_examples:
                print(f"    {p}")
            raise FileNotFoundError(
                f"More than 50% of sampled file paths in {name} set do not exist."
            )

    # 4. Verify Validation Split Requirements
    print("\nVerifying validation split logic...")

    # Check for ID overlap
    train_ids = set(df_train_check["id"])
    val_ids = set(df_val_check["id"])
    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise AssertionError(
            f"Found {len(overlap)} overlapping IDs between Train and Validation sets."
        )
    else:
        print("  Overlap check passed: No shared IDs between Train and Validation.")

    # Check Stratification
    train_pos_ratio = df_train_check["target"].mean()
    val_pos_ratio = df_val_check["target"].mean()
    diff = abs(train_pos_ratio - val_pos_ratio)

    print(f"  Train positive ratio: {train_pos_ratio:.4f}")
    print(f"  Val positive ratio:   {val_pos_ratio:.4f}")

    # Allow a small margin of error for stratification
    if diff > 0.01:
        raise AssertionError(
            f"Stratification failed. Target ratios differ significantly: {diff:.4f}"
        )
    else:
        print("  Stratification check passed.")

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
