import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw training data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    train_df = pd.read_csv(train_path)

    # Create binary target for stratification (Target >= 0.5 is considered toxic)
    # This ensures the validation set has the same distribution of toxic comments
    train_df["stratify_label"] = (train_df["target"] >= 0.5).astype(int)

    print("Performing stratified split (80/20)...")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_STATE
    )

    # Get indices for split
    train_idx, val_idx = next(splitter.split(train_df, train_df["stratify_label"]))

    # Create Train and Validation DataFrames
    train_meta = train_df.iloc[train_idx].copy()
    val_meta = train_df.iloc[val_idx].copy()

    # Add source_file column to act as a relative file path pointer
    train_meta["source_file"] = "train.csv"
    val_meta["source_file"] = "train.csv"

    # Drop the temporary stratification label
    train_meta.drop(columns=["stratify_label"], inplace=True)
    val_meta.drop(columns=["stratify_label"], inplace=True)

    # Remove 'comment_text' to create lightweight metadata files
    # Downstream scripts will load text from 'source_file' using 'id' or by loading the full file
    if "comment_text" in train_meta.columns:
        train_meta.drop(columns=["comment_text"], inplace=True)
    if "comment_text" in val_meta.columns:
        val_meta.drop(columns=["comment_text"], inplace=True)

    print("Saving training and validation metadata...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "validation.csv"), index=False)

    # Process Test Data
    print("Processing test data...")
    test_path = os.path.join(INPUT_DIR, "test.csv")
    test_df = pd.read_csv(test_path)

    test_meta = test_df.copy()
    test_meta["source_file"] = "test.csv"

    if "comment_text" in test_meta.columns:
        test_meta.drop(columns=["comment_text"], inplace=True)

    print("Saving test metadata...")
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # -------------------------------------------------------------------------
    # Verification Steps
    # -------------------------------------------------------------------------
    print("\n=== Verifying Generated Metadata ===")

    # Load metadata back
    m_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    m_val = pd.read_csv(os.path.join(METADATA_DIR, "validation.csv"))
    m_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print(f"Train samples:      {len(m_train)}")
    print(f"Validation samples: {len(m_val)}")
    print(f"Test samples:       {len(m_test)}")

    # Calculate class distribution (using threshold 0.5)
    train_pos_ratio = (m_train["target"] >= 0.5).mean()
    val_pos_ratio = (m_val["target"] >= 0.5).mean()

    print(f"Train Toxicity Ratio: {train_pos_ratio:.5f}")
    print(f"Val Toxicity Ratio:   {val_pos_ratio:.5f}")

    # 2. Check File Paths
    # We verify that 'source_file' points to an existing file in ./input
    print("\nChecking file path resolution for 1000 random samples per file...")
    datasets = {"Train": m_train, "Validation": m_val, "Test": m_test}

    for name, df in datasets.items():
        # Sample 1000 items (or all if less than 1000)
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            rel_path = row["source_file"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(f"[{name}] Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} metadata are invalid."
            )

    # 3. Verify Validation Split Requirements
    print("\nVerifying split constraints...")

    # Assert Split Ratio (approx 20%)
    total_train_val = len(m_train) + len(m_val)
    val_split_ratio = len(m_val) / total_train_val
    print(f"Actual Validation Split Ratio: {val_split_ratio:.4f}")

    if not (0.19 < val_split_ratio < 0.21):
        raise AssertionError(
            f"Split ratio validation failed. Expected ~0.20, got {val_split_ratio:.4f}"
        )

    # Assert Stratification
    # The difference in toxicity ratio should be minimal
    diff = abs(train_pos_ratio - val_pos_ratio)
    print(f"Class distribution difference: {diff:.6f}")

    if diff > 0.01:
        raise AssertionError(
            f"Stratification validation failed. Difference {diff:.6f} exceeds tolerance."
        )

    print("\nAll verification checks passed successfully.")


if __name__ == "__main__":
    main()
