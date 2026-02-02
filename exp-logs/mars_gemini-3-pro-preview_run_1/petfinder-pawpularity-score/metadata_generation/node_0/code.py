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

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Step 1: Loading raw data...")
    try:
        train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    except FileNotFoundError as e:
        print(f"Critical Error: Could not find input files. {e}")
        raise

    print("Step 2: Generating file paths...")
    # Generate relative file paths for images
    # Training images: input/train/{id}.jpg -> stored as 'train/{id}.jpg'
    # Test images: input/test/{id}.jpg -> stored as 'test/{id}.jpg'
    train_raw["file_path"] = train_raw["Id"].apply(
        lambda x: os.path.join("train", f"{x}.jpg")
    )
    test_raw["file_path"] = test_raw["Id"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    print("Step 3: Creating stratified validation split...")
    # Pawpularity is continuous. To ensure the validation set represents the training set
    # (especially given the peaks in distribution), we bin the target and stratify on bins.
    # Using ~14 bins based on Sturges' rule for N~10000
    num_bins = 14
    train_raw["stratify_bin"] = pd.cut(
        train_raw["Pawpularity"], bins=num_bins, labels=False
    )

    # Split 80:20
    train_df, val_df = train_test_split(
        train_raw,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_raw["stratify_bin"],
        shuffle=True,
    )

    # Clean up temporary column
    train_df = train_df.drop(columns=["stratify_bin"])
    val_df = val_df.drop(columns=["stratify_bin"])

    print("Step 4: Saving metadata...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_raw.to_csv(test_save_path, index=False)

    print("Metadata generation complete.")

    # ==================================================================================
    # Verification Checks
    # ==================================================================================
    print("\nStep 5: Running verification checks...")

    # Reload data to verify persistence
    df_train_check = pd.read_csv(train_save_path)
    df_val_check = pd.read_csv(val_save_path)
    df_test_check = pd.read_csv(test_save_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train samples: {len(df_train_check)}")
    print(f"Val samples:   {len(df_val_check)}")
    print(f"Test samples:  {len(df_test_check)}")

    print("\nTrain Target Stats:")
    print(df_train_check["Pawpularity"].describe().to_string())
    print("\nVal Target Stats:")
    print(df_val_check["Pawpularity"].describe().to_string())

    # 2. File Path Existence Check
    def check_paths(df, dataset_name):
        # Check up to 1000 random paths
        n_check = min(1000, len(df))
        sample = df.sample(n=n_check, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            # Construct full path: ./input/ + relative_path
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(row["file_path"])

        ratio = missing_count / n_check
        print(
            f"\nChecking {dataset_name} paths: {missing_count}/{n_check} missing (Ratio: {ratio:.4f})"
        )

        if ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"Missing file ratio for {dataset_name} is {ratio}, which exceeds the 0.5 limit."
            )

    check_paths(df_train_check, "Train")
    check_paths(df_val_check, "Validation")
    check_paths(df_test_check, "Test")

    # 3. Validation Split Verification
    # Check Ratio
    total_len = len(df_train_check) + len(df_val_check)
    actual_ratio = len(df_val_check) / total_len
    print(f"\nActual Validation Ratio: {actual_ratio:.4f}")
    if not (0.19 <= actual_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_ratio} is not approximately 0.2 (Target: 0.2)"
        )

    # Check Stratification (Distribution Similarity)
    train_mean = df_train_check["Pawpularity"].mean()
    val_mean = df_val_check["Pawpularity"].mean()
    diff = abs(train_mean - val_mean)

    print(f"Train Mean Pawpularity: {train_mean:.4f}")
    print(f"Val Mean Pawpularity:   {val_mean:.4f}")
    print(f"Difference:             {diff:.4f}")

    # With stratification, means should be very close.
    # We use a loose threshold of 2.0 to account for random variations, but expect < 0.5 usually.
    if diff > 2.0:
        raise AssertionError(
            "Validation set distribution differs significantly from training set. Stratification may have failed."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
