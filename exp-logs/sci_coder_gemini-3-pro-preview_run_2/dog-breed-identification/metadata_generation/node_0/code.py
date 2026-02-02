import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def generate_metadata():
    """
    Generates metadata CSVs for train, val, and test sets.
    """
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    labels_file = os.path.join(INPUT_DIR, "labels.csv")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"{labels_file} not found.")

    df = pd.read_csv(labels_file)

    # Construct file paths for training data: train/<id>.jpg
    # Note: Paths are relative to ./input
    df["file_path"] = df["id"].apply(lambda x: os.path.join("train", f"{x}.jpg"))

    # Split into train and validation (80:20, stratified)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["breed"], random_state=RANDOM_STATE, shuffle=True
    )

    # --- Process Test Data ---
    # Using sample_submission.csv to get the list of test IDs
    sample_sub_file = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_file):
        raise FileNotFoundError(f"{sample_sub_file} not found.")

    test_df = pd.read_csv(sample_sub_file)
    # Keep only the ID and create file paths
    test_df = test_df[["id"]].copy()
    test_df["file_path"] = test_df["id"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    # --- Save Metadata ---
    train_csv = os.path.join(METADATA_DIR, "train.csv")
    val_csv = os.path.join(METADATA_DIR, "val.csv")
    test_csv = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_csv, val_csv, test_csv


def check_file_existence(df, name):
    """
    Checks if files exist in the input directory.
    Raises error if missing ratio > 0.5.
    """
    print(f"Checking file existence for {name} dataset...")

    # Select random sample of 1000 paths (or all if less than 1000)
    n_samples = min(1000, len(df))
    if n_samples == 0:
        print(f"  No samples in {name} dataset.")
        return

    sample_paths = (
        df["file_path"].sample(n=n_samples, random_state=RANDOM_STATE).tolist()
    )

    missing_count = 0
    missing_examples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(rel_path)

    ratio = missing_count / n_samples
    print(f"  Missing file ratio: {ratio:.4f} ({missing_count}/{n_samples})")

    if ratio > 0.5:
        print("  Sample missing paths:")
        for p in missing_examples:
            print(f"    {p}")
        raise FileNotFoundError(
            f"Missing file ratio for {name} is {ratio}, which exceeds 0.5"
        )


def verify_stratification(train_df, val_df):
    """
    Verifies that the train/val split respects the 80:20 ratio and stratification.
    """
    print("Verifying stratification...")

    # Check split ratio
    n_train = len(train_df)
    n_val = len(val_df)
    total = n_train + n_val

    if total == 0:
        raise ValueError("No training data found.")

    val_ratio = n_val / total

    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    print(f"  Validation ratio: {val_ratio:.4f}")

    # Assert ratio is approximately 0.2
    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(f"Validation ratio {val_ratio:.4f} is not close to 0.2")

    # Check class distribution consistency
    train_dist = train_df["breed"].value_counts(normalize=True).sort_index()
    val_dist = val_df["breed"].value_counts(normalize=True).sort_index()

    # Align indices to ensure we compare same breeds
    all_breeds = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_breeds, fill_value=0)
    val_dist = val_dist.reindex(all_breeds, fill_value=0)

    # Calculate Mean Absolute Error between class probabilities
    mae = (train_dist - val_dist).abs().mean()
    print(f"  Class distribution MAE: {mae:.6f}")

    # Threshold for stratification failure
    # With 120 classes, perfect stratification isn't always possible for small counts,
    # but the average difference should be very small.
    if mae > 0.01:
        raise AssertionError(
            f"Stratification failed. Class distribution MAE: {mae:.6f}"
        )


def main():
    print("Starting metadata generation...")
    train_path, val_path, test_path = generate_metadata()
    print("Metadata generated successfully.")

    # Load back the generated metadata for verification
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("\n--- Summary Statistics ---")
    print(
        f"Train Set: {train_df.shape[0]} samples, {train_df['breed'].nunique()} breeds"
    )
    print(f"Val Set:   {val_df.shape[0]} samples, {val_df['breed'].nunique()} breeds")
    print(f"Test Set:  {test_df.shape[0]} samples")

    # Programmatic Checks
    check_file_existence(train_df, "Train")
    check_file_existence(val_df, "Validation")
    check_file_existence(test_df, "Test")

    verify_stratification(train_df, val_df)

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
