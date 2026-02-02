import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(metadata_dir, exist_ok=True)

    # --- 1. Process Training Data ---
    train_dir = os.path.join(input_dir, "train")
    cleaned_dir = os.path.join(input_dir, "train_cleaned")

    # Get all PNG files
    train_files = glob.glob(os.path.join(train_dir, "*.png"))
    cleaned_files = glob.glob(os.path.join(cleaned_dir, "*.png"))

    # Extract IDs (filenames without extension)
    # Assuming filenames are like "101.png"
    train_ids = {os.path.splitext(os.path.basename(f))[0] for f in train_files}
    cleaned_ids = {os.path.splitext(os.path.basename(f))[0] for f in cleaned_files}

    # Find intersection to ensure we have pairs
    valid_ids = list(train_ids.intersection(cleaned_ids))
    valid_ids.sort()  # Ensure deterministic order before shuffle

    print(f"Found {len(train_ids)} noisy images and {len(cleaned_ids)} cleaned images.")
    print(f"Total paired training samples: {len(valid_ids)}")

    data = []
    for img_id in valid_ids:
        data.append(
            {
                "id": img_id,
                "feature_path": os.path.join("train", f"{img_id}.png"),
                "label_path": os.path.join("train_cleaned", f"{img_id}.png"),
            }
        )

    df_full_train = pd.DataFrame(data)

    # Split 80/20
    # Since this is a denoising task (regression/image-to-image), we don't have discrete classes
    # for stratification. We use a simple random split.
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.2, random_state=42, shuffle=True
    )

    # Save Train and Val
    df_train.to_csv(os.path.join(metadata_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(metadata_dir, "val.csv"), index=False)

    # --- 2. Process Test Data ---
    test_dir = os.path.join(input_dir, "test")
    test_files = glob.glob(os.path.join(test_dir, "*.png"))

    test_data = []
    for f in test_files:
        img_id = os.path.splitext(os.path.basename(f))[0]
        test_data.append(
            {"id": img_id, "feature_path": os.path.join("test", f"{img_id}.png")}
        )

    df_test = pd.DataFrame(test_data)
    df_test.sort_values(by="id", inplace=True)  # Sort for consistency

    # Save Test
    df_test.to_csv(os.path.join(metadata_dir, "test.csv"), index=False)

    print("Metadata generation complete.")
    return len(df_full_train)


def validate_metadata(total_train_samples):
    print("\n--- Validating Metadata ---")
    metadata_dir = "./metadata"
    input_dir = "./input"

    # Load datasets
    df_train = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # 1. Summary Statistics
    print(f"Train set size: {len(df_train)}")
    print(f"Val set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    # 2. Verify Split Ratio
    total_split = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_split
    print(f"Validation ratio: {val_ratio:.4f}")

    # Assert split ratio is close to 0.2 (allowing for small rounding diffs due to dataset size)
    assert (
        0.19 <= val_ratio <= 0.21
    ), f"Validation split ratio {val_ratio} is not approximately 0.2"

    # Assert no data leakage (ids should be unique across train/val)
    train_ids = set(df_train["id"])
    val_ids = set(df_val["id"])
    intersection = train_ids.intersection(val_ids)
    assert (
        len(intersection) == 0
    ), f"Data leakage detected! IDs in both train and val: {intersection}"

    # 3. Check File Paths
    datasets = {"train": df_train, "val": df_val, "test": df_test}

    for name, df in datasets.items():
        print(f"Checking file paths for {name} dataset...")

        # Collect all path columns
        path_cols = [col for col in df.columns if "path" in col]

        # Sample up to 1000 paths
        # We sample rows, then check all path columns in those rows
        n_samples = min(1000, len(df))
        sample_df = df.sample(n=n_samples, random_state=42)

        missing_count = 0
        missing_samples = []

        total_checks = 0

        for _, row in sample_df.iterrows():
            for col in path_cols:
                rel_path = row[col]
                full_path = os.path.join(input_dir, rel_path)
                total_checks += 1

                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_samples) < 5:
                        missing_samples.append(rel_path)

        missing_ratio = missing_count / total_checks if total_checks > 0 else 0

        if missing_ratio > 0.5:
            print("Sample of missing files:")
            for m in missing_samples:
                print(f"  {m}")
            raise FileNotFoundError(
                f"Missing file ratio for {name} is {missing_ratio:.2f} (> 0.5)."
            )

        if missing_count > 0:
            print(
                f"Warning: {missing_count} files missing in {name} sample check (Ratio: {missing_ratio:.4f})"
            )
        else:
            print(f"All checked paths in {name} resolve correctly.")

    print("Validation successful.")


if __name__ == "__main__":
    total_train = generate_metadata()
    validate_metadata(total_train)
