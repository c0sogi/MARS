import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def get_salt_coverage(rle):
    """Calculates the ratio of salt pixels to total pixels from RLE string."""
    if pd.isna(rle) or str(rle).strip() == "":
        return 0.0

    # RLE format: start length start length ...
    # We only need the lengths (every second element)
    s = str(rle).split()
    lengths = [int(x) for x in s[1::2]]
    total_salt = sum(lengths)
    return total_salt / (101 * 101)


def check_files_exist(df, name, n_check=1000):
    """Checks if a random sample of file paths in the dataframe exist."""
    # Collect all relevant paths
    paths = df["image_path"].tolist()
    if "mask_path" in df.columns:
        paths.extend(df["mask_path"].tolist())

    # Sample paths
    if len(paths) > n_check:
        paths = np.random.choice(paths, n_check, replace=False)

    missing_count = 0
    missing_samples = []

    for p in paths:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    ratio = missing_count / len(paths) if len(paths) > 0 else 0
    print(f"[{name}] Missing file ratio: {ratio:.4f} ({missing_count}/{len(paths)})")

    if ratio > 0.5:
        print(f"Sample missing files: {missing_samples}")
        raise FileNotFoundError(
            f"Missing file ratio {ratio:.4f} exceeds limit of 0.5 for {name} dataset."
        )


def main():
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Raw Data
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    depths_df = pd.read_csv(os.path.join(INPUT_DIR, "depths.csv"))
    sample_sub_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    # 2. Merge Depth Information
    # train.csv has 'id', depths.csv has 'id', 'z'
    train_df = train_df.merge(depths_df, on="id", how="left")

    # Prepare test dataframe (using sample_submission ids)
    test_df = sample_sub_df[["id"]].merge(depths_df, on="id", how="left")

    # 3. Construct File Paths
    # Paths must be relative to ./input
    train_df["image_path"] = train_df["id"].apply(lambda x: f"train/images/{x}.png")
    train_df["mask_path"] = train_df["id"].apply(lambda x: f"train/masks/{x}.png")
    test_df["image_path"] = test_df["id"].apply(lambda x: f"test/images/{x}.png")

    # 4. Stratification Logic
    print("Calculating salt coverage for stratification...")
    train_df["coverage"] = train_df["rle_mask"].apply(get_salt_coverage)

    # Bin coverage into 10 classes for stratification
    # Class 0 is usually empty masks, others are quantiles or fixed bins
    # We use fixed bins to ensure consistent grouping based on salt amount
    train_df["coverage_class"] = pd.cut(train_df["coverage"], bins=10, labels=False)

    print("Splitting training data into Train/Val...")
    # Perform Stratified Split
    try:
        train_split, val_split = train_test_split(
            train_df,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_df["coverage_class"],
        )
    except ValueError as e:
        print(f"Stratification with 10 bins failed ({e}). Reducing bins to 5.")
        train_df["coverage_class"] = pd.cut(train_df["coverage"], bins=5, labels=False)
        train_split, val_split = train_test_split(
            train_df,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_df["coverage_class"],
        )

    # 5. Save Metadata
    print("Saving metadata files...")
    train_split.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    # 6. Verification
    print("\n--- Verification Step ---")

    # Reload data to ensure integrity
    v_train = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    v_val = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    v_test = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # Summary Statistics
    print(f"Train samples: {len(v_train)}")
    print(f"Val samples:   {len(v_val)}")
    print(f"Test samples:  {len(v_test)}")

    print("\nTrain Coverage Class Distribution:")
    print(v_train["coverage_class"].value_counts(normalize=True).sort_index())

    # Check File Paths
    np.random.seed(RANDOM_STATE)  # Ensure reproducibility of checks
    check_files_exist(v_train, "Train")
    check_files_exist(v_val, "Validation")
    check_files_exist(v_test, "Test")

    # Verify Stratification
    print("\nVerifying Stratification...")
    train_dist = v_train["coverage_class"].value_counts(normalize=True).sort_index()
    val_dist = v_val["coverage_class"].value_counts(normalize=True).sort_index()

    # Align indices to ensure we compare same classes
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))

    max_diff = 0
    for c in all_classes:
        t_prop = train_dist.get(c, 0)
        v_prop = val_dist.get(c, 0)
        diff = abs(t_prop - v_prop)
        max_diff = max(max_diff, diff)

    print(f"Max difference in class proportions: {max_diff:.4f}")

    # Assert stratification success (tolerance of 5%)
    if max_diff > 0.05:
        raise AssertionError(
            f"Stratification failed! Class distribution differs by {max_diff:.4f} > 0.05"
        )

    print("Metadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
