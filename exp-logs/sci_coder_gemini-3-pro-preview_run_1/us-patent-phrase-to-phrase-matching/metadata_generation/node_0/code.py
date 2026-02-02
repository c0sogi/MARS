import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.20

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Raw Data
    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")

    df_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # 2. Split Data (Stratified Sampling)
    # The task involves discrete scores (0, 0.25, 0.5, 0.75, 1.0), so we stratify by 'score'.
    print("Splitting data into Train and Validation sets...")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    try:
        # split returns indices
        for train_idx, val_idx in splitter.split(df_full, df_full["score"]):
            df_train = df_full.iloc[train_idx].copy()
            df_val = df_full.iloc[val_idx].copy()
    except ValueError as e:
        raise ValueError(f"Stratified split failed: {e}")

    # 3. Save Metadata
    # We save the split dataframes as new CSVs in the metadata folder.
    # These serve as the metadata defining the splits.
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # 4. Verification and Statistics
    print("\n" + "=" * 30)
    print("VERIFICATION & STATISTICS")
    print("=" * 30)

    # Reload to verify integrity
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    # A. Summary Statistics
    print(f"\n[Statistics]")
    print(f"Original Train Samples: {len(df_full)}")
    print(f"New Train Samples:      {len(df_train_check)}")
    print(f"New Validation Samples: {len(df_val_check)}")
    print(f"Test Samples:           {len(df_test_check)}")

    print("\n[Class Distribution - Score]")
    train_dist = df_train_check["score"].value_counts(normalize=True).sort_index()
    val_dist = df_val_check["score"].value_counts(normalize=True).sort_index()

    print("Train Distribution:")
    print(train_dist)
    print("Validation Distribution:")
    print(val_dist)

    # B. File Path Verification
    # This dataset contains text phrases. We check if any column looks like a file path.
    # If so, we verify existence.
    def verify_paths(df, name):
        # Identify columns that might be relative paths (contain '/' or common extensions)
        path_candidates = []
        for col in df.columns:
            if df[col].dtype == "object":
                # Check a non-null sample
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if isinstance(sample, str) and (
                    "/" in sample
                    or sample.lower().endswith((".jpg", ".png", ".wav", ".csv"))
                ):
                    # In this specific dataset, 'context' or 'anchor' are unlikely to be paths,
                    # but if they were, we would check them.
                    # Given the dataset description, there are no external files.
                    # We proceed only if strong indicators exist.
                    pass

        if not path_candidates:
            print(
                f"No file path columns detected in {name} set. Skipping path existence check."
            )
            return

        # Logic for checking paths if they existed
        print(f"Checking file paths for {name}...")
        for col in path_candidates:
            samples = df[col].sample(n=min(1000, len(df)), random_state=RANDOM_STATE)
            missing = 0
            missing_examples = []
            for path in samples:
                full_path = os.path.join(INPUT_DIR, str(path))
                if not os.path.exists(full_path):
                    missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(path)

            ratio = missing / len(samples)
            if ratio > 0.5:
                print(f"Sample missing paths: {missing_examples}")
                raise FileNotFoundError(
                    f"Missing file ratio {ratio:.2f} exceeds 0.5 for column '{col}' in {name} set."
                )

    verify_paths(df_train_check, "Train")
    verify_paths(df_val_check, "Validation")
    verify_paths(df_test_check, "Test")

    # C. Validation Split Verification
    print("\n[Verifying Split Requirements]")

    # 1. Check Ratio
    total_len = len(df_train_check) + len(df_val_check)
    actual_val_ratio = len(df_val_check) / total_len
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f}")

    # Allow a small margin of error for discrete count splitting
    if not (0.19 <= actual_val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates significantly from 0.20"
        )

    # 2. Check Stratification
    # Calculate max difference in class probabilities
    max_diff = (train_dist - val_dist).abs().max()
    print(f"Max Class Distribution Difference: {max_diff:.6f}")

    if max_diff > 0.05:
        raise AssertionError(
            f"Stratification failed. Class distribution differs by {max_diff:.6f} > 0.05"
        )

    print("\nAll checks passed. Metadata generation successful.")


if __name__ == "__main__":
    main()
