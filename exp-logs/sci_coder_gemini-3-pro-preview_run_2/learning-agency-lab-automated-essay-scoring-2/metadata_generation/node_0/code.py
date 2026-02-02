import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Raw train shape: {df_train_full.shape}")
    print(f"Raw test shape: {df_test.shape}")

    # Stratified Split
    print("Performing stratified split...")
    X = df_train_full.index
    y = df_train_full["score"]

    train_idx, val_idx = train_test_split(
        X, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE, shuffle=True
    )

    df_train = df_train_full.loc[train_idx].reset_index(drop=True)
    df_val = df_train_full.loc[val_idx].reset_index(drop=True)

    # Save metadata
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # --- Verification Step ---
    print("\n--- Verifying Metadata ---")

    # Load back
    v_train = pd.read_csv(train_meta_path)
    v_val = pd.read_csv(val_meta_path)
    v_test = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print(f"Train Set: {v_train.shape[0]} samples")
    print(f"Val Set:   {v_val.shape[0]} samples")
    print(f"Test Set:  {v_test.shape[0]} samples")

    print("\nTrain Score Distribution:")
    print(v_train["score"].value_counts().sort_index())

    print("\nVal Score Distribution:")
    print(v_val["score"].value_counts().sort_index())

    # 2. Check Stratification
    train_dist = v_train["score"].value_counts(normalize=True).sort_index()
    val_dist = v_val["score"].value_counts(normalize=True).sort_index()

    # Calculate max absolute difference in proportions
    # Align indices in case some classes are missing (unlikely with stratified split but safe)
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    diffs = []
    for c in all_classes:
        t_p = train_dist.get(c, 0)
        v_p = val_dist.get(c, 0)
        diffs.append(abs(t_p - v_p))

    max_diff = max(diffs)
    print(
        f"\nMax difference in class proportions between Train and Val: {max_diff:.6f}"
    )

    # Assert stratification was successful (tolerance 1%)
    if max_diff > 0.01:
        raise AssertionError(
            f"Stratification failed! Class distribution differs by {max_diff:.6f} > 0.01"
        )

    # 3. File Path Check
    # In this task, the data is text inside the CSV, not external files.
    # However, if we had columns referencing files, we would check them here.
    # We will verify that the 'full_text' column is present and not empty for a sample.

    print("\nChecking data integrity...")
    for name, df in [("Train", v_train), ("Val", v_val), ("Test", v_test)]:
        if "full_text" not in df.columns:
            raise AssertionError(f"'full_text' column missing in {name} metadata")

        # Check a random sample for content
        sample = df.sample(min(100, len(df)), random_state=42)
        empty_text = sample["full_text"].isna() | (sample["full_text"] == "")
        if empty_text.any():
            print(sample[empty_text])
            raise AssertionError(f"Found empty text in {name} metadata")

    print("\nVerification passed successfully.")


if __name__ == "__main__":
    generate_metadata()
