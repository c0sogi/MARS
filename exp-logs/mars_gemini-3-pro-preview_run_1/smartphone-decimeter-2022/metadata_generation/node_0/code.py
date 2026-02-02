import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Generates metadata CSV files for train, validation, and test sets.
    """
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # --- 1. Process Training Data ---
    print("Processing training data...")
    train_data = []
    train_base_path = os.path.join(INPUT_DIR, "train")

    if os.path.exists(train_base_path):
        # List all drives (trips)
        drives = [
            d
            for d in os.listdir(train_base_path)
            if os.path.isdir(os.path.join(train_base_path, d))
        ]

        for drive_id in drives:
            drive_path = os.path.join(train_base_path, drive_id)
            # List all phones in the drive
            phones = [
                p
                for p in os.listdir(drive_path)
                if os.path.isdir(os.path.join(drive_path, p))
            ]

            for phone_name in phones:
                phone_path = os.path.join(drive_path, phone_name)
                gt_path = os.path.join(phone_path, "ground_truth.csv")

                # Check if ground truth exists
                if os.path.exists(gt_path):
                    # Read ground truth to get labels and timestamps
                    df_gt = pd.read_csv(gt_path)

                    # Construct relative paths for sensor data
                    # Paths are relative to ./input
                    rel_path_prefix = os.path.join("train", drive_id, phone_name)
                    gnss_path = os.path.join(rel_path_prefix, "device_gnss.csv")
                    imu_path = os.path.join(rel_path_prefix, "device_imu.csv")

                    # Add metadata columns
                    df_gt["drive_id"] = drive_id
                    df_gt["phone_name"] = phone_name
                    df_gt["gnss_path"] = gnss_path
                    df_gt["imu_path"] = imu_path

                    # Keep necessary columns
                    # ground_truth.csv contains: LatitudeDegrees, LongitudeDegrees, UnixTimeMillis, etc.
                    cols = [
                        "drive_id",
                        "phone_name",
                        "UnixTimeMillis",
                        "LatitudeDegrees",
                        "LongitudeDegrees",
                        "gnss_path",
                        "imu_path",
                    ]

                    # Filter to ensure we only select existing columns
                    cols = [c for c in cols if c in df_gt.columns]

                    train_data.append(df_gt[cols])

    if train_data:
        full_train_df = pd.concat(train_data, ignore_index=True)
    else:
        # Fallback for empty input
        full_train_df = pd.DataFrame(
            columns=[
                "drive_id",
                "phone_name",
                "UnixTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "gnss_path",
                "imu_path",
            ]
        )

    # --- 2. Split Train/Validation ---
    print("Splitting training data into train/val sets...")
    if not full_train_df.empty:
        # Group Sampling by drive_id to avoid leakage
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )
        groups = full_train_df["drive_id"]

        train_idx, val_idx = next(splitter.split(full_train_df, groups=groups))

        train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_train_df.iloc[val_idx].reset_index(drop=True)
    else:
        train_df = full_train_df.copy()
        val_df = pd.DataFrame(columns=full_train_df.columns)

    # Save Train/Val Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)

    # --- 3. Process Test Data ---
    print("Processing test data...")
    # Use sample_submission.csv to determine the target test samples
    sample_submission_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if os.path.exists(sample_submission_path):
        test_df = pd.read_csv(sample_submission_path)

        # Parse tripId to get drive_id and phone_name
        # Format: drive_id-phone_name (e.g., 2020-05-15-US-MTV-1-Pixel4)
        # Phone names don't have hyphens, drive_ids do. Split from right.
        parsed = test_df["tripId"].str.rsplit("-", n=1)
        test_df["drive_id"] = parsed.str[0]
        test_df["phone_name"] = parsed.str[1]

        # Construct paths
        test_df["gnss_path"] = test_df.apply(
            lambda x: os.path.join(
                "test", x["drive_id"], x["phone_name"], "device_gnss.csv"
            ),
            axis=1,
        )
        test_df["imu_path"] = test_df.apply(
            lambda x: os.path.join(
                "test", x["drive_id"], x["phone_name"], "device_imu.csv"
            ),
            axis=1,
        )

        # Select relevant columns
        cols = [
            "tripId",
            "drive_id",
            "phone_name",
            "UnixTimeMillis",
            "gnss_path",
            "imu_path",
        ]
        test_df = test_df[cols]
    else:
        test_df = pd.DataFrame(
            columns=[
                "tripId",
                "drive_id",
                "phone_name",
                "UnixTimeMillis",
                "gnss_path",
                "imu_path",
            ]
        )

    # Save Test Metadata
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    test_df.to_csv(test_meta_path, index=False)

    return train_meta_path, val_meta_path, test_meta_path


def verify_metadata(train_path, val_path, test_path):
    """
    Loads generated metadata and performs verification checks.
    """
    print("\n--- Verifying Metadata ---")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print(f"Train Samples: {len(df_train)}")
    print(f"Val Samples:   {len(df_val)}")
    print(f"Test Samples:  {len(df_test)}")

    print(
        f"Train Drives:  {df_train['drive_id'].nunique() if not df_train.empty else 0}"
    )
    print(f"Val Drives:    {df_val['drive_id'].nunique() if not df_val.empty else 0}")
    print(f"Test Trips:    {df_test['tripId'].nunique() if not df_test.empty else 0}")

    # 2. Check File Path Existence
    def check_files(df, name):
        if df.empty:
            return

        # Randomly sample 1000 paths (or all if less than 1000)
        paths = df["gnss_path"].unique()
        if len(paths) > 1000:
            paths = np.random.choice(paths, 1000, replace=False)

        missing_count = 0
        missing_examples = []

        for p in paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        ratio = missing_count / len(paths)
        print(f"[{name}] Missing GNSS file ratio: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Sample missing paths in {name}:")
            for mp in missing_examples:
                print(f"  - {mp}")
            raise FileNotFoundError(
                f"Too many missing files in {name} metadata (Ratio: {ratio:.2f})"
            )

    check_files(df_train, "Train")
    check_files(df_val, "Validation")
    check_files(df_test, "Test")

    # 3. Verify Split Integrity
    if not df_train.empty and not df_val.empty:
        train_drives = set(df_train["drive_id"].unique())
        val_drives = set(df_val["drive_id"].unique())

        # Check for intersection
        intersection = train_drives.intersection(val_drives)
        if intersection:
            raise AssertionError(
                f"Data Leakage Detected! Drives found in both train and val: {intersection}"
            )

        print("Split Verification: No drive leakage between train and validation sets.")


if __name__ == "__main__":
    train_p, val_p, test_p = generate_metadata()
    verify_metadata(train_p, val_p, test_p)
