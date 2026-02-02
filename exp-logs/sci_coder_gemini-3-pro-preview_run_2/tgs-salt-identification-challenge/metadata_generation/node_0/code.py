import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
IMG_SIZE = 101


def rle_to_coverage(rle_string):
    """Calculates the number of pixels in the mask from RLE string."""
    if pd.isna(rle_string) or rle_string == "":
        return 0

    # RLE format: start length start length ...
    rle_list = [int(x) for x in rle_string.split()]
    # Sum the lengths (every second element)
    return sum(rle_list[1::2])


def check_files_exist(df, path_cols, base_dir, sample_size=1000):
    """Checks if files exist for a random sample of paths."""
    if len(df) == 0:
        return

    sample = df.sample(n=min(len(df), sample_size), random_state=RANDOM_STATE)

    for col in path_cols:
        missing_count = 0
        missing_samples = []

        for rel_path in sample[col]:
            full_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / len(sample)
        if missing_ratio > 0.5:
            print(f"Sample of missing files for column '{col}':")
            for p in missing_samples:
                print(f" - {p}")
            raise FileNotFoundError(
                f"Missing file ratio {missing_ratio:.2f} for column {col} exceeds threshold 0.5"
            )
        else:
            print(
                f"File existence check passed for {col} (missing ratio: {missing_ratio:.2f})"
            )


def main():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load csvs
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    depths_df = pd.read_csv(os.path.join(INPUT_DIR, "depths.csv"))

    # Merge depths
    train_df = pd.merge(train_df, depths_df, on="id", how="left")

    # Construct file paths (relative to input dir)
    train_df["image_path"] = "train/images/" + train_df["id"] + ".png"
    train_df["mask_path"] = "train/masks/" + train_df["id"] + ".png"

    # Calculate coverage for stratification
    print("Calculating salt coverage for stratification...")
    train_df["salt_pixels"] = train_df["rle_mask"].apply(rle_to_coverage)
    train_df["salt_coverage"] = train_df["salt_pixels"] / (IMG_SIZE * IMG_SIZE)

    # Create coverage classes for stratification
    # We bin coverage into 10 quantiles.
    # Note: Many images have 0 salt, so we treat 0 as a separate class or ensure bins handle it.
    # A simple way for this specific dataset is to bin the non-zero coverage and keep 0 separate,
    # or just use pd.cut with enough bins.
    # Here we define a simplified class: 0 coverage vs >0 quantized.

    def get_coverage_class(coverage):
        if coverage == 0:
            return 0
        # Quantize non-zero coverage into 10 bins (1-10)
        return int(np.ceil(coverage * 10))

    train_df["coverage_class"] = train_df["salt_coverage"].apply(get_coverage_class)

    print("Splitting train/validation sets...")
    # Stratified split
    train_meta, val_meta = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["coverage_class"],
    )

    # Process Test Data
    print("Processing test data...")
    test_images = glob.glob(os.path.join(INPUT_DIR, "test/images/*.png"))
    test_ids = [os.path.basename(f).replace(".png", "") for f in test_images]
    test_df = pd.DataFrame({"id": test_ids})

    # Merge depths for test
    test_df = pd.merge(test_df, depths_df, on="id", how="left")
    test_df["image_path"] = "test/images/" + test_df["id"] + ".png"

    # Save Metadata
    print("Saving metadata...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Steps
    # ==========================================
    print("\n--- Verification ---")

    # Reload datasets
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print("\nClass distribution (Coverage Class) in Train:")
    print(df_train["coverage_class"].value_counts(normalize=True).sort_index())
    print("\nClass distribution (Coverage Class) in Val:")
    print(df_val["coverage_class"].value_counts(normalize=True).sort_index())

    # 2. Check File Paths
    print("\nChecking file paths...")
    check_files_exist(df_train, ["image_path", "mask_path"], INPUT_DIR)
    check_files_exist(df_val, ["image_path", "mask_path"], INPUT_DIR)
    check_files_exist(df_test, ["image_path"], INPUT_DIR)

    # 3. Verify Split Logic
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"\nActual Validation Ratio: {val_ratio:.4f}")

    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio} is not approximately 0.2"
        )

    # Verify Stratification
    # We check if the distribution of coverage classes is similar using Kolmogorov-Smirnov or just simple variance check.
    # For this task, we'll check if the difference in proportions for any class is within a tolerance.
    train_dist = df_train["coverage_class"].value_counts(normalize=True).sort_index()
    val_dist = df_val["coverage_class"].value_counts(normalize=True).sort_index()

    # Align indices (in case some bin is missing in one set, though unlikely with stratified split)
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    for c in all_classes:
        t_prop = train_dist.get(c, 0)
        v_prop = val_dist.get(c, 0)
        diff = abs(t_prop - v_prop)
        # Tolerance: 0.05 is generous, but allows for small sample variations in bins
        if diff > 0.05:
            raise AssertionError(
                f"Stratification failed for class {c}: Train prop {t_prop:.3f}, Val prop {v_prop:.3f}"
            )

    print("All verification checks passed successfully.")


if __name__ == "__main__":
    main()
