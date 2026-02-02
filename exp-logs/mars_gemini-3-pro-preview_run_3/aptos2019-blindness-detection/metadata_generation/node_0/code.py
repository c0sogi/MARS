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

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load raw csvs
    train_df_orig = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df_orig = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # Add relative file paths
    # Note: The requirement is paths relative to ./input
    train_df_orig["file_path"] = train_df_orig["id_code"].apply(
        lambda x: os.path.join("train_images", f"{x}.png")
    )
    test_df_orig["file_path"] = test_df_orig["id_code"].apply(
        lambda x: os.path.join("test_images", f"{x}.png")
    )

    # Perform Stratified Split
    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(
        train_df_orig,
        test_size=VAL_SIZE,
        stratify=train_df_orig["diagnosis"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save metadata
    print("Saving metadata...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df_orig.to_csv(test_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Steps
    # ==========================================
    print("\nStarting verification...")

    # Reload datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    print("\nTrain Class Distribution:")
    print(df_train["diagnosis"].value_counts(normalize=True).sort_index())

    print("\nValidation Class Distribution:")
    print(df_val["diagnosis"].value_counts(normalize=True).sort_index())

    # 2. Check File Paths
    def check_paths(df, name):
        print(f"\nChecking file paths for {name}...")
        # Sample up to 1000 paths
        n_sample = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=n_sample, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            # Construct full path based on requirement that metadata paths are relative to ./input
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(
            f"Missing file ratio for {name}: {missing_ratio:.4f} ({missing_count}/{n_sample})"
        )

        if missing_ratio > 0.5:
            print("Sample of missing paths:")
            for p in missing_samples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} are missing."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    # 3. Verify Stratification
    print("\nVerifying stratification...")
    train_dist = df_train["diagnosis"].value_counts(normalize=True).sort_index()
    val_dist = df_val["diagnosis"].value_counts(normalize=True).sort_index()

    # Calculate maximum absolute difference in class proportions
    # Since classes 0-4 should exist, we align them
    all_classes = sorted(train_df_orig["diagnosis"].unique())

    for c in all_classes:
        t_prop = train_dist.get(c, 0)
        v_prop = val_dist.get(c, 0)
        diff = abs(t_prop - v_prop)

        # We allow a small tolerance. With small datasets, exact stratification isn't always possible,
        # but with ~3000 samples, it should be close.
        # Tolerance of 0.05 (5%) is generous but sufficient to catch non-stratified random splits on imbalanced data.
        if diff > 0.05:
            raise AssertionError(
                f"Stratification failed for class {c}. Train prop: {t_prop:.4f}, Val prop: {v_prop:.4f}, Diff: {diff:.4f}"
            )

    print("Stratification verification passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
