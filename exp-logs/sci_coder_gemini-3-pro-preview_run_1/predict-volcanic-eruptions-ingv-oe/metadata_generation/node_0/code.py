import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split


def main():
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Generating metadata...")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"{train_csv_path} not found.")

    df_train_full = pd.read_csv(train_csv_path)

    # Construct relative file paths
    # Format: train/{segment_id}.csv
    df_train_full["file_path"] = df_train_full["segment_id"].apply(
        lambda x: os.path.join("train", f"{x}.csv")
    )

    # Stratified Split for Regression
    # We bin the target variable to simulate classes for stratification
    num_bins = min(10, len(df_train_full) // 20)  # Ensure enough samples per bin
    if num_bins < 2:
        # Fallback to random split if not enough data for bins
        stratify_labels = None
    else:
        try:
            stratify_labels = pd.qcut(
                df_train_full["time_to_eruption"],
                q=num_bins,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            # Fallback if qcut fails (e.g., too many duplicate values)
            stratify_labels = None

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
        shuffle=True,
    )

    # Save to metadata
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)

    print(f"Saved train metadata to {train_save_path} ({len(train_df)} rows)")
    print(f"Saved val metadata to {val_save_path} ({len(val_df)} rows)")

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    test_files_pattern = os.path.join(INPUT_DIR, "test", "*.csv")
    test_file_paths = glob.glob(test_files_pattern)

    test_data = []
    for path in test_file_paths:
        filename = os.path.basename(path)
        segment_id = os.path.splitext(filename)[0]
        # Relative path from input dir
        rel_path = os.path.join("test", filename)
        test_data.append(
            {
                "segment_id": int(segment_id),
                "file_path": rel_path,
                "time_to_eruption": 0,
            }
        )

    test_df = pd.DataFrame(test_data)
    test_save_path = os.path.join(METADATA_DIR, "test.csv")
    test_df.to_csv(test_save_path, index=False)

    print(f"Saved test metadata to {test_save_path} ({len(test_df)} rows)")

    # ---------------------------------------------------------
    # 3. Verification
    # ---------------------------------------------------------
    print("\nVerifying generated metadata...")

    # Reload datasets
    df_train_check = pd.read_csv(train_save_path)
    df_val_check = pd.read_csv(val_save_path)
    df_test_check = pd.read_csv(test_save_path)

    # 3a. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {len(df_train_check)} samples")
    print(f"Train Target Mean: {df_train_check['time_to_eruption'].mean():.2f}")
    print(f"Train Target Std:  {df_train_check['time_to_eruption'].std():.2f}")

    print(f"Val Set:   {len(df_val_check)} samples")
    print(f"Val Target Mean:   {df_val_check['time_to_eruption'].mean():.2f}")
    print(f"Val Target Std:    {df_val_check['time_to_eruption'].std():.2f}")

    print(f"Test Set:  {len(df_test_check)} samples")

    # 3b. File Path Existence Check
    def check_paths(df, name):
        if len(df) == 0:
            return

        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Path in metadata is relative to ./input
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        missing_ratio = missing_count / sample_size
        print(
            f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    check_paths(df_train_check, "Train")
    check_paths(df_val_check, "Val")
    check_paths(df_test_check, "Test")

    # 3c. Validation Split Requirements Check
    total_train_val = len(df_train_check) + len(df_val_check)
    actual_val_ratio = len(df_val_check) / total_train_val

    print(f"\nValidation Split Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Assert split ratio is within reasonable bounds (allow small rounding diffs)
    if not (0.19 <= actual_val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates significantly from expected {VAL_SIZE}"
        )

    # Assert distribution similarity (simple mean check)
    # Since we stratified, means should be relatively close
    train_mean = df_train_check["time_to_eruption"].mean()
    val_mean = df_val_check["time_to_eruption"].mean()
    mean_diff_pct = abs(train_mean - val_mean) / train_mean

    print(f"Difference in target means (Train vs Val): {mean_diff_pct:.2%}")

    # We expect the difference to be small (< 10% is generous for stratified, but safe)
    if mean_diff_pct > 0.15:
        # Note: If dataset is very small or high variance, this might trigger.
        # But with 4000 samples and stratification, it should be very close.
        raise AssertionError(
            "Validation set distribution deviates significantly from training set."
        )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
