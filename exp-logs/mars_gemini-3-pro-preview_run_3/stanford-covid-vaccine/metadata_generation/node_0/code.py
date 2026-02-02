import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # File paths
    train_path = os.path.join(INPUT_DIR, "train.json")
    test_path = os.path.join(INPUT_DIR, "test.json")

    print("Loading data...")

    # Load train data
    # The file format is likely JSON Lines based on the description snippet
    try:
        train_df = pd.read_json(train_path, lines=True)
    except ValueError:
        # Fallback if it's a standard JSON array
        train_df = pd.read_json(train_path)

    # Load test data
    try:
        test_df = pd.read_json(test_path, lines=True)
    except ValueError:
        test_df = pd.read_json(test_path)

    print(f"Original Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Determine stratification column
    # The description mentions 'SN_filter' (S/N filter).
    # We check if it exists in the dataframe columns.
    stratify_col = None
    if "SN_filter" in train_df.columns:
        stratify_col = "SN_filter"
        print(f"Stratifying by 'SN_filter'.")
    else:
        print("Column 'SN_filter' not found. Using random split.")

    # Split training data into Train and Validation (80:20)
    RANDOM_STATE = 42

    if stratify_col:
        train_split, val_split = train_test_split(
            train_df,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=train_df[stratify_col],
        )
    else:
        train_split, val_split = train_test_split(
            train_df, test_size=0.2, random_state=RANDOM_STATE
        )

    # Reset indices
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save to metadata as Parquet to preserve list/array structures efficiently
    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")

    train_split.to_parquet(train_meta_path, index=False)
    val_split.to_parquet(val_meta_path, index=False)
    test_df.to_parquet(test_meta_path, index=False)

    print("Metadata files saved.")

    # ==========================================
    # Verification and Validation
    # ==========================================
    print("\nVerifying datasets...")

    # Reload to verify
    df_train_loaded = pd.read_parquet(train_meta_path)
    df_val_loaded = pd.read_parquet(val_meta_path)
    df_test_loaded = pd.read_parquet(test_meta_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train samples: {len(df_train_loaded)}")
    print(f"Val samples:   {len(df_val_loaded)}")
    print(f"Test samples:  {len(df_test_loaded)}")

    # Check columns
    print(f"Train columns: {list(df_train_loaded.columns)}")

    # 2. Check Split Ratio
    total_train_val = len(df_train_loaded) + len(df_val_loaded)
    val_ratio = len(df_val_loaded) / total_train_val
    print(f"Validation Ratio: {val_ratio:.4f} (Target: 0.20)")

    # Assert split ratio is approximately correct
    assert (
        0.19 < val_ratio < 0.21
    ), f"Validation split ratio {val_ratio} deviates from 0.2"

    # 3. Verify Stratification (if applied)
    if stratify_col:
        train_dist = df_train_loaded[stratify_col].value_counts(normalize=True)
        val_dist = df_val_loaded[stratify_col].value_counts(normalize=True)
        print("\nStratification Check (Distribution of SN_filter):")
        print("Train Distribution:\n", train_dist)
        print("Val Distribution:\n", val_dist)

        # Check if distributions are similar (within 5% tolerance)
        for key in train_dist.index:
            diff = abs(train_dist[key] - val_dist.get(key, 0))
            if diff > 0.05:
                raise AssertionError(
                    f"Stratification failed for class {key}. Diff: {diff}"
                )
        print("Stratification verification passed.")

    # 4. Check for Data Integrity (Sample check)
    # Ensure list columns are actually lists/arrays and not strings
    # We check 'reactivity' in train if it exists
    if "reactivity" in df_train_loaded.columns:
        sample_val = df_train_loaded["reactivity"].iloc[0]
        if isinstance(sample_val, str):
            print(
                "Warning: 'reactivity' column loaded as string. Parquet should preserve arrays."
            )
        elif isinstance(sample_val, (np.ndarray, list)):
            print("'reactivity' column preserved as array/list.")

    # 5. File Path Check
    # Since we did not generate file paths (data is embedded), we skip the missing file ratio check.
    # However, we verify the source files existed by proxy of successful load.

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
