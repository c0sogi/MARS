import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Read raw data
    print("Reading raw data...")
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # 2. Add file paths
    # Structure: {train|test}/{id}/geometry.xyz relative to input dir
    train_df["file_path"] = train_df["id"].apply(lambda x: f"train/{x}/geometry.xyz")
    test_df["file_path"] = test_df["id"].apply(lambda x: f"test/{x}/geometry.xyz")

    # 3. Split training data
    # Using random split as per requirements for regression task.
    # We use random_state=42 and shuffle=True.
    print("Splitting data...")
    train_meta, val_meta = train_test_split(
        train_df, test_size=0.2, random_state=42, shuffle=True
    )

    # 4. Save metadata
    print("Saving metadata...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # 5. Validation and Checks
    print("Performing validation checks...")

    # Load back the data to ensure it was saved correctly
    train_loaded = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_loaded = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_loaded = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train samples: {len(train_loaded)}")
    print(f"Val samples: {len(val_loaded)}")
    print(f"Test samples: {len(test_loaded)}")

    # Check for overlap
    train_ids = set(train_loaded["id"])
    val_ids = set(val_loaded["id"])
    overlap = train_ids.intersection(val_ids)
    if len(overlap) > 0:
        raise AssertionError(
            f"Found {len(overlap)} overlapping IDs between train and val sets."
        )

    # Verify split ratio
    total_train_val = len(train_loaded) + len(val_loaded)
    val_ratio = len(val_loaded) / total_train_val
    print(f"Validation split ratio: {val_ratio:.4f}")

    # Check if ratio is approximately 0.2 (allow small variance due to integer rounding)
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(f"Validation split ratio {val_ratio} is not close to 0.2")

    # Check file paths
    def check_file_existence(df, name):
        # Select 1000 random samples or all if less than 1000
        n_samples = min(1000, len(df))
        samples = df.sample(n=n_samples, random_state=42)

        missing_paths = []
        for _, row in samples.iterrows():
            rel_path = row["file_path"]
            # Path relative to ./input
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_paths.append(rel_path)

        missing_ratio = len(missing_paths) / n_samples
        print(f"{name} set missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths from {name}:")
            for p in missing_paths[:5]:
                print(p)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not resolve."
            )

    check_file_existence(train_loaded, "Train")
    check_file_existence(val_loaded, "Validation")
    check_file_existence(test_loaded, "Test")

    print("\nMetadata generation and verification successful.")


if __name__ == "__main__":
    main()
