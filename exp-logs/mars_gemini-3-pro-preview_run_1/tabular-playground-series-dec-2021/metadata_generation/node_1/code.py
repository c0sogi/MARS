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

    # --- 1. Process Training Data ---
    print("Reading training data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found.")

    # Read CSV
    df_train_full = pd.read_csv(train_path)

    target_col = "Cover_Type"
    if target_col not in df_train_full.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    print(f"Full training data shape: {df_train_full.shape}")

    # Remove classes with fewer than 2 samples to allow stratified split
    class_counts = df_train_full[target_col].value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if not rare_classes.empty:
        print(
            f"Removing {len(rare_classes)} rare classes (count < 2) to enable stratification."
        )
        df_train_full = df_train_full[~df_train_full[target_col].isin(rare_classes)]
        print(f"Filtered training data shape: {df_train_full.shape}")

    # Stratified Split
    print("Performing stratified split (80/20)...")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_STATE
    )

    try:
        train_idx, val_idx = next(
            splitter.split(df_train_full, df_train_full[target_col])
        )
    except ValueError as e:
        raise ValueError(
            f"Stratification failed (likely due to insufficient class samples): {e}"
        )

    df_train = df_train_full.iloc[train_idx]
    df_val = df_train_full.iloc[val_idx]

    # Save to metadata
    print("Saving train and val metadata...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    # Clean up memory
    del df_train_full, df_train, df_val, train_idx, val_idx

    # --- 2. Process Test Data ---
    print("Reading and processing test data...")
    test_path = os.path.join(INPUT_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found.")

    df_test = pd.read_csv(test_path)
    print(f"Test data shape: {df_test.shape}")

    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)
    del df_test

    # --- 3. Validation & Checks ---
    print("\n--- Running Validation Checks ---")

    # Load generated metadata
    meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    meta_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    meta_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 3.1 Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train Set: {len(meta_train)} samples")
    print(f"Val Set:   {len(meta_val)} samples")
    print(f"Test Set:  {len(meta_test)} samples")

    # 3.2 Verify Split Ratio
    n_train = len(meta_train)
    n_val = len(meta_val)
    ratio = n_val / (n_train + n_val)
    print(f"Validation Ratio: {ratio:.5f}")

    if not np.isclose(ratio, 0.2, atol=1e-3):
        raise AssertionError(
            f"Validation split ratio {ratio:.5f} is not 0.2 (+/- 0.001)"
        )

    # 3.3 Verify Stratification
    print("Verifying Stratification...")
    train_dist = meta_train[target_col].value_counts(normalize=True).sort_index()
    val_dist = meta_val[target_col].value_counts(normalize=True).sort_index()

    print("Train Class Dist (head):")
    print(train_dist.head())
    print("Val Class Dist (head):")
    print(val_dist.head())

    # Check max difference in proportions
    # Align indices to ensure we compare same classes (fill 0 for missing classes if any)
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_probs = train_dist.reindex(all_classes, fill_value=0)
    val_probs = val_dist.reindex(all_classes, fill_value=0)

    max_diff = (train_probs - val_probs).abs().max()
    print(f"Max class probability difference: {max_diff:.6f}")

    if max_diff > 0.01:  # 1% tolerance
        raise AssertionError(
            f"Stratification failed. Max difference {max_diff:.6f} > 0.01"
        )

    # 3.4 Check File Paths
    # We scan for columns that look like file paths
    def check_file_paths(df, name):
        # Heuristic: object columns containing '/' or '.'
        path_candidates = []
        for col in df.select_dtypes(include=["object"]):
            # Check a non-null sample
            sample = df[col].dropna()
            if len(sample) > 0:
                s = str(sample.iloc[0])
                if ("/" in s or "\\" in s) and ("." in s):
                    path_candidates.append(col)

        if not path_candidates:
            print(f"No file path columns detected in {name}.")
            return

        for col in path_candidates:
            print(f"Checking potential path column '{col}' in {name}...")
            # Sample 1000
            sample_paths = df[col].sample(
                n=min(1000, len(df)), random_state=RANDOM_STATE
            )
            missing = 0
            bad_examples = []

            for p in sample_paths:
                # Paths relative to INPUT_DIR
                full_path = os.path.join(INPUT_DIR, str(p))
                if not os.path.exists(full_path):
                    missing += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(p)

            ratio = missing / len(sample_paths)
            print(f"  Missing ratio: {ratio:.4f}")

            if ratio > 0.5:
                print(f"  Examples of missing paths: {bad_examples}")
                raise AssertionError(
                    f"File path check failed for column '{col}' in {name}. Missing ratio {ratio:.4f} > 0.5"
                )

    check_file_paths(meta_train, "Train")
    check_file_paths(meta_val, "Val")
    check_file_paths(meta_test, "Test")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
