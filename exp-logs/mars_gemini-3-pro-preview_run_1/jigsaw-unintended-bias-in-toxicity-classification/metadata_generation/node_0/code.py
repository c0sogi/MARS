import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    # Load data
    # Specifying dtypes can help with memory for large datasets,
    # but pandas usually handles this fine. train.csv is ~3.8M rows.
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Original Train shape: {df_train_full.shape}")
    print(f"Original Test shape: {df_test.shape}")

    # Create binary target for stratification
    # Task description: "test set examples with target >= 0.5 will be considered to be in the positive class"
    df_train_full["binary_target"] = (df_train_full["target"] >= 0.5).astype(int)

    # Split into Train and Validation (80:20)
    print("Splitting data...")
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=42,
        stratify=df_train_full["binary_target"],
    )

    # Save metadata
    # We save the full dataframes as metadata because the text is contained within them.
    # In a scenario with external images, we would save paths. Here, the CSV is the dataset.
    print("Saving metadata to ./metadata...")

    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Step
    # ==========================================
    print("\nStarting verification...")

    # 1. Load generated metadata
    v_train = pd.read_csv(train_meta_path)
    v_val = pd.read_csv(val_meta_path)
    v_test = pd.read_csv(test_meta_path)

    # 2. Print Summary Statistics
    print("-" * 30)
    print("Summary Statistics")
    print("-" * 30)

    print(f"Train Set: {len(v_train)} samples")
    print(f"Val Set:   {len(v_val)} samples")
    print(f"Test Set:  {len(v_test)} samples")

    # Target distribution
    train_pos_rate = v_train["binary_target"].mean()
    val_pos_rate = v_val["binary_target"].mean()

    print(f"Train Positive Rate: {train_pos_rate:.4f}")
    print(f"Val Positive Rate:   {val_pos_rate:.4f}")

    # Identity Subgroup Stats (checking a few key ones if they exist)
    identities = ["male", "female", "black", "white", "muslim"]
    print("\nIdentity Subgroup Counts (Non-zero entries):")
    for ident in identities:
        if ident in v_train.columns:
            # In this dataset, identity columns are fractional.
            # Usually we consider a mention if value > 0 or >= 0.5.
            # Let's just count non-zeros for summary.
            train_count = (v_train[ident] > 0).sum()
            val_count = (v_val[ident] > 0).sum()
            print(f"  {ident}: Train={train_count}, Val={val_count}")

    # 3. Check file paths
    # The dataset provides text directly in the CSV, not as external files.
    # Therefore, we verify the existence of the source CSVs in ./input as the "paths".
    # If the metadata contained relative paths to images, we would check those.
    # Here we simulate the check by verifying we can read the text column.
    print("\nChecking data integrity...")

    for name, df in [("Train", v_train), ("Val", v_val), ("Test", v_test)]:
        if "comment_text" not in df.columns:
            raise ValueError(f"{name} metadata missing 'comment_text' column.")

        # Check for empty text (optional, just logging)
        empty_text = df["comment_text"].isna().sum()
        if empty_text > 0:
            print(f"  Warning: {name} has {empty_text} rows with missing text.")

        # Randomly check 1000 "paths" (here, just verifying the row exists/is valid)
        # Since there are no external files, this check passes if the dataframe is loaded.
        # We will strictly follow the prompt's logic: "If the metadata contains file paths..."
        # It does not, so we skip the external file check.

    # 4. Verify Validation Set Requirements
    print("\nVerifying split requirements...")

    # Assert split ratio (80:20)
    total_train_val = len(v_train) + len(v_val)
    val_ratio = len(v_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Allow a tiny margin of error for rounding
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio is {val_ratio:.4f}, expected ~0.20"
        )

    # Assert Stratification
    # The difference in positive rates should be very small
    diff = abs(train_pos_rate - val_pos_rate)
    print(f"Difference in positive rates: {diff:.6f}")

    if diff > 0.01:  # 1% tolerance
        raise AssertionError(
            "Stratification failed: Class distribution differs significantly between Train and Val."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
