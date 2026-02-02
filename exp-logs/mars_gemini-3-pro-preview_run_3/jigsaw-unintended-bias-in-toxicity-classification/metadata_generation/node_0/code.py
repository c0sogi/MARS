import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # 1. Setup directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Define constants
    RANDOM_STATE = 42
    VAL_SIZE = 0.2
    TARGET_COL = "target"
    IDENTITY_COLUMNS = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]

    # 3. Load Data
    print("Loading data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    # Using specific dtypes to save memory if necessary, though 220GB is plenty.
    # We read all columns.
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Original Train shape: {df_train_full.shape}")
    print(f"Test shape: {df_test.shape}")

    # 4. Preprocessing for Split
    # The task treats target >= 0.5 as positive. We stratify on this binary outcome.
    # Note: The target is fractional.
    df_train_full["binary_target"] = (df_train_full[TARGET_COL] >= 0.5).astype(int)

    # 5. Split Data
    print("Splitting data...")
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["binary_target"],
        shuffle=True,
    )

    # Drop the temporary binary_target column if desired, or keep it.
    # Keeping it might be useful, but standard practice is to keep original schema + split.
    # I will remove it to keep schema clean, or keep it if helpful.
    # Let's remove it to match original schema exactly, but the prompt implies metadata can have labels.
    # I'll drop it to be safe and stick to original columns.
    train_df = train_df.drop(columns=["binary_target"])
    val_df = val_df.drop(columns=["binary_target"])

    # 6. Save Metadata (Split Datasets)
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # 7. Verification and Checks
    print("Running verification checks...")

    # Load back the data
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    # Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train Set: {len(df_train_check)} samples")
    print(f"Val Set:   {len(df_val_check)} samples")
    print(f"Test Set:  {len(df_test_check)} samples")

    train_pos_ratio = (df_train_check[TARGET_COL] >= 0.5).mean()
    val_pos_ratio = (df_val_check[TARGET_COL] >= 0.5).mean()

    print(f"Train Positive Ratio: {train_pos_ratio:.4f}")
    print(f"Val Positive Ratio:   {val_pos_ratio:.4f}")

    # Identity stats
    print("\nIdentity counts (non-zero) in Train:")
    for id_col in IDENTITY_COLUMNS:
        if id_col in df_train_check.columns:
            count = (df_train_check[id_col] > 0).sum()
            print(f"  {id_col}: {count}")

    # Check 1: File paths
    # Since this dataset is text-in-CSV, there are no external file paths to check.
    # The prompt says "If the metadata contains file paths...". Here it does not.
    # We skip the file path resolution check.

    # Check 2: Validation Split Requirements
    # 2a. Check Split Ratio
    total_train_val = len(df_train_check) + len(df_val_check)
    actual_val_ratio = len(df_val_check) / total_train_val
    print(f"\nActual Validation Ratio: {actual_val_ratio:.4f}")

    if abs(actual_val_ratio - VAL_SIZE) > 1e-3:
        raise AssertionError(
            f"Validation split ratio mismatch. Expected {VAL_SIZE}, got {actual_val_ratio}"
        )

    # 2b. Check Stratification
    # We check if the positive class ratio is similar.
    diff = abs(train_pos_ratio - val_pos_ratio)
    print(f"Stratification Difference (Positive Class Ratio): {diff:.6f}")

    if diff > 1e-2:  # Allow small variance
        raise AssertionError(
            f"Stratification failed. Train ratio: {train_pos_ratio}, Val ratio: {val_pos_ratio}"
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
