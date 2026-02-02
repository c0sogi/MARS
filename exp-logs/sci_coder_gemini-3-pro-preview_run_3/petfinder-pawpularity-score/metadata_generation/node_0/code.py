import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


# ==========================================
# 1. Setup and Data Loading
# ==========================================
def main():
    print("Starting metadata generation...")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw datasets
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    print(
        f"Loaded raw data: Train shape {df_train_full.shape}, Test shape {df_test.shape}"
    )

    # ==========================================
    # 2. Add File Paths
    # ==========================================
    # Paths must be relative to ./input
    # Structure: input/train/{id}.jpg and input/test/{id}.jpg

    df_train_full["file_path"] = df_train_full["Id"].apply(
        lambda x: os.path.join("train", f"{x}.jpg")
    )
    df_test["file_path"] = df_test["Id"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    # ==========================================
    # 3. Stratified Split (Train/Val)
    # ==========================================
    # Pawpularity is continuous (1-100). To stratify, we bin the target.
    # We use Sturges' rule to determine a reasonable number of bins for stratification.
    num_bins = int(np.floor(1 + np.log2(len(df_train_full))))

    # Create a temporary column for stratification
    df_train_full["stratify_bin"] = pd.cut(
        df_train_full["Pawpularity"], bins=num_bins, labels=False
    )

    # Perform split
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["stratify_bin"],
        shuffle=True,
    )

    # Clean up temporary column
    train_df = train_df.drop(columns=["stratify_bin"])
    val_df = val_df.drop(columns=["stratify_bin"])

    print(f"Split completed. Train: {len(train_df)}, Val: {len(val_df)}")

    # ==========================================
    # 4. Save Metadata
    # ==========================================
    train_meta_path = os.path.join(METADATA_DIR, "train_meta.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_meta.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_meta.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    # ==========================================
    # 5. Verification
    # ==========================================
    verify_metadata(train_meta_path, val_meta_path, test_meta_path)


def verify_metadata(train_path, val_path, test_path):
    print("\nStarting verification...")

    # Load generated metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 5.1 Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train Set: {len(df_train)} samples")
    print(f"Val Set:   {len(df_val)} samples")
    print(f"Test Set:  {len(df_test)} samples")

    print("\nTrain Target (Pawpularity) Stats:")
    print(df_train["Pawpularity"].describe())
    print("\nVal Target (Pawpularity) Stats:")
    print(df_val["Pawpularity"].describe())

    # 5.2 File Path Verification
    print("\n=== Checking File Paths ===")
    check_file_paths(df_train, "Train")
    check_file_paths(df_val, "Val")
    check_file_paths(df_test, "Test")

    # 5.3 Verify Split Requirements
    print("\n=== Verifying Split Requirements ===")

    # Check Ratio
    total_train_val = len(df_train) + len(df_val)
    actual_val_ratio = len(df_val) / total_train_val
    print(f"Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    if not (0.19 < actual_val_ratio < 0.21):
        raise AssertionError(
            f"Split ratio mismatch. Expected ~0.2, got {actual_val_ratio}"
        )

    # Check Stratification
    # We check if the distribution of the target variable is similar in train and val
    # using Kolmogorov-Smirnov test or simply comparing mean/std relative difference.
    # Here we check the difference in means is small relative to the range.

    train_mean = df_train["Pawpularity"].mean()
    val_mean = df_val["Pawpularity"].mean()
    train_std = df_train["Pawpularity"].std()
    val_std = df_val["Pawpularity"].std()

    print(f"Train Mean: {train_mean:.2f}, Val Mean: {val_mean:.2f}")
    print(f"Train Std:  {train_std:.2f}, Val Std:  {val_std:.2f}")

    # Allow a small margin of error for stratification on continuous variable
    if (
        abs(train_mean - val_mean) > 2.0
    ):  # 2.0 is roughly 2% of the 100 scale, lenient but safe
        raise AssertionError(
            "Stratification failed: Means of Train and Val are too different."
        )

    # Also verify that IDs do not overlap
    train_ids = set(df_train["Id"])
    val_ids = set(df_val["Id"])
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(
            f"Data Leakage: {len(overlap)} IDs found in both Train and Val sets."
        )

    print("Verification passed successfully.")


def check_file_paths(df, name):
    # Select 1000 random paths or all if less than 1000
    n_samples = min(1000, len(df))
    sample_paths = df["file_path"].sample(n=n_samples, random_state=42).tolist()

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        # Resolve path relative to input directory
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / n_samples
    print(f"[{name}] Checked {n_samples} paths. Missing ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing paths:", missing_samples)
        raise FileNotFoundError(f"[{name}] More than 50% of image files are missing.")


if __name__ == "__main__":
    main()
