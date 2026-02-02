import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_path = os.path.join(input_dir, "train.json")
    test_path = os.path.join(input_dir, "test.json")

    os.makedirs(metadata_dir, exist_ok=True)

    print("Loading raw data...")
    # Load train data
    with open(train_path, "r") as f:
        train_data = json.load(f)

    # Load test data
    with open(test_path, "r") as f:
        test_data = json.load(f)

    # Create DataFrames
    # We only keep essential metadata: ID, index, source path, and label

    # Process Train Data
    train_records = []
    for idx, entry in enumerate(train_data):
        train_records.append(
            {
                "request_id": entry["request_id"],
                "requester_received_pizza": int(
                    entry["requester_received_pizza"]
                ),  # Convert bool to int
                "sample_index": idx,
                "source_file": os.path.join("input", "train.json"),
            }
        )
    df_full_train = pd.DataFrame(train_records)

    # Process Test Data
    test_records = []
    for idx, entry in enumerate(test_data):
        test_records.append(
            {
                "request_id": entry["request_id"],
                # Label is not available in test set
                "sample_index": idx,
                "source_file": os.path.join("input", "test.json"),
            }
        )
    df_test = pd.DataFrame(test_records)

    print(f"Full training set shape: {df_full_train.shape}")
    print(f"Test set shape: {df_test.shape}")

    # Stratified Split
    print("Splitting training data into train and validation sets...")
    X = df_full_train.drop(columns=["requester_received_pizza"])
    y = df_full_train["requester_received_pizza"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Reconstruct DataFrames
    df_train = X_train.copy()
    df_train["requester_received_pizza"] = y_train

    df_val = X_val.copy()
    df_val["requester_received_pizza"] = y_val

    # Save Metadata
    print("Saving metadata...")
    train_meta_path = os.path.join(metadata_dir, "train.csv")
    val_meta_path = os.path.join(metadata_dir, "val.csv")
    test_meta_path = os.path.join(metadata_dir, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Step
    # ==========================================
    print("\nStarting verification...")

    # 1. Load datasets
    df_train_v = pd.read_csv(train_meta_path)
    df_val_v = pd.read_csv(val_meta_path)
    df_test_v = pd.read_csv(test_meta_path)

    # 2. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {len(df_train_v)} samples")
    print(
        f"Train Class Distribution:\n{df_train_v['requester_received_pizza'].value_counts(normalize=True)}"
    )

    print(f"\nValidation Set: {len(df_val_v)} samples")
    print(
        f"Validation Class Distribution:\n{df_val_v['requester_received_pizza'].value_counts(normalize=True)}"
    )

    print(f"\nTest Set: {len(df_test_v)} samples")

    # 3. Check File Paths
    print("\nChecking file paths...")

    def check_paths(df, name):
        if "source_file" not in df.columns:
            return

        # Sample 1000 paths (or all if less than 1000)
        n_sample = min(1000, len(df))
        sample_paths = df["source_file"].sample(n=n_sample, random_state=42)

        missing_count = 0
        missing_samples = []

        for path in sample_paths:
            # Paths are relative to current working directory (which contains input/)
            # The metadata stores 'input/train.json', so we check if that exists relative to CWD.
            if not os.path.exists(path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(path)

        missing_ratio = missing_count / n_sample
        print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    check_paths(df_train_v, "Train Metadata")
    check_paths(df_val_v, "Validation Metadata")
    check_paths(df_test_v, "Test Metadata")

    # 4. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check Split Ratio
    total_train_val = len(df_train_v) + len(df_val_v)
    val_ratio = len(df_val_v) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Allow small floating point tolerance
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio is {val_ratio}, expected approx 0.20"
        )

    # Check Stratification
    train_pos_rate = df_train_v["requester_received_pizza"].mean()
    val_pos_rate = df_val_v["requester_received_pizza"].mean()

    print(f"Train Positive Rate: {train_pos_rate:.4f}")
    print(f"Val Positive Rate: {val_pos_rate:.4f}")

    # Stratification should keep rates very close
    if abs(train_pos_rate - val_pos_rate) > 0.01:
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly between train and val."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
