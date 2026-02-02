import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2
    LABEL_COLS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Step 1: Loading Data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found.")

    # Load datasets
    # Using engine='c' for speed, handling potential mixed types by letting pandas infer
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded Train: {train_df.shape}")
    print(f"Loaded Test: {test_df.shape}")

    # Step 2: Stratified Split
    print("Step 2: Performing Stratified Split...")

    # Create stratification key
    # We combine all label columns into a single string to represent the label combination
    # This allows us to stratify based on the exact pattern of toxicity
    train_df["stratify_key"] = train_df[LABEL_COLS].astype(str).agg("".join, axis=1)

    # Filter out singletons (groups with only 1 sample) which break stratification
    key_counts = train_df["stratify_key"].value_counts()
    valid_keys = key_counts[key_counts >= 2].index

    stratifiable = train_df[train_df["stratify_key"].isin(valid_keys)]
    singletons = train_df[~train_df["stratify_key"].isin(valid_keys)]

    print(f"Stratifiable samples: {len(stratifiable)}")
    print(f"Singleton samples (forced to train): {len(singletons)}")

    # Split the stratifiable portion
    train_split, val_split = train_test_split(
        stratifiable,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratifiable["stratify_key"],
    )

    # Combine singletons back into train
    train_split = pd.concat([train_split, singletons])

    # Shuffle train to mix singletons in
    train_split = train_split.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    val_split = val_split.reset_index(drop=True)

    # Drop the temporary key
    train_split = train_split.drop(columns=["stratify_key"])
    val_split = val_split.drop(columns=["stratify_key"])

    # Step 3: Save Metadata
    print("Step 3: Saving Metadata...")
    meta_train_path = os.path.join(METADATA_DIR, "train.csv")
    meta_val_path = os.path.join(METADATA_DIR, "val.csv")
    meta_test_path = os.path.join(METADATA_DIR, "test.csv")

    train_split.to_csv(meta_train_path, index=False)
    val_split.to_csv(meta_val_path, index=False)
    test_df.to_csv(meta_test_path, index=False)

    print(f"Saved {meta_train_path}")
    print(f"Saved {meta_val_path}")
    print(f"Saved {meta_test_path}")

    # Step 4: Verification
    print("Step 4: Verifying Metadata...")

    # Load back to verify integrity
    v_train = pd.read_csv(meta_train_path)
    v_val = pd.read_csv(meta_val_path)
    v_test = pd.read_csv(meta_test_path)

    # 4a. Summary Stats
    print("\n=== Summary Statistics ===")
    print(f"Train Shape: {v_train.shape}")
    print(f"Val Shape:   {v_val.shape}")
    print(f"Test Shape:  {v_test.shape}")

    # 4b. Check Split Ratio
    n_train = len(v_train)
    n_val = len(v_val)
    total = n_train + n_val
    actual_val_ratio = n_val / total
    print(f"Validation Ratio: {actual_val_ratio:.5f} (Target: {VAL_SIZE})")

    # Assert ratio is within reasonable bounds (e.g. +/- 0.01)
    if not (0.19 <= actual_val_ratio <= 0.21):
        raise AssertionError(
            f"Validation ratio {actual_val_ratio} is out of bounds (0.19-0.21)."
        )

    # 4c. Check Stratification
    print("\n=== Label Distribution Check ===")
    train_means = v_train[LABEL_COLS].mean()
    val_means = v_val[LABEL_COLS].mean()

    print(f"{'Label':<20} {'Train Mean':<15} {'Val Mean':<15} {'Diff':<15}")
    max_diff = 0.0
    for label in LABEL_COLS:
        t_m = train_means[label]
        v_m = val_means[label]
        diff = abs(t_m - v_m)
        max_diff = max(max_diff, diff)
        print(f"{label:<20} {t_m:.5f}         {v_m:.5f}         {diff:.5f}")

    print(f"\nMax difference in label means: {max_diff:.5f}")

    # Assert stratification quality
    # We allow a small difference. 0.015 is reasonable for multi-label with potential rare classes.
    if max_diff > 0.015:
        raise AssertionError(
            f"Stratification check failed! Max difference {max_diff} > 0.015"
        )

    # 4d. File Path Check
    # The dataset contains text, not file paths. We skip the file resolution check.
    print("\nNote: Dataset contains inline text. No external file paths to verify.")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
