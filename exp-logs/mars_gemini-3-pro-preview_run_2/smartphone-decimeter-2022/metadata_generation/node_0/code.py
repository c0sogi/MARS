import os
import pandas as pd
import numpy as np
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_VAL_SPLIT_RATIO = 0.8


def generate_metadata():
    # Create metadata directory if it doesn't exist
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # ---------------------------------------------------------
    # 1. Generate Train and Validation Metadata
    # ---------------------------------------------------------
    print("Generating Train/Val Metadata...")
    train_trips = []

    # Traverse train directory
    train_base_path = os.path.join(INPUT_DIR, "train")
    if os.path.exists(train_base_path):
        train_drive_ids = os.listdir(train_base_path)
        for drive_id in train_drive_ids:
            drive_path = os.path.join(train_base_path, drive_id)
            if not os.path.isdir(drive_path):
                continue

            phone_names = os.listdir(drive_path)
            for phone_name in phone_names:
                phone_path = os.path.join(drive_path, phone_name)
                if not os.path.isdir(phone_path):
                    continue

                # Paths to check
                gt_path = os.path.join(phone_path, "ground_truth.csv")

                if os.path.exists(gt_path):
                    try:
                        # Read ground truth to get timestamps and labels
                        df_gt = pd.read_csv(gt_path)

                        # Add metadata columns
                        df_gt["drive_id"] = drive_id
                        df_gt["phone_name"] = phone_name
                        # Construct tripId as drive_id + '-' + phone_name
                        df_gt["tripId"] = f"{drive_id}-{phone_name}"

                        # Construct relative file paths
                        # Note: We assume the files exist if the folder structure is correct,
                        # but we will verify a sample later.
                        # Paths are relative to ./input
                        df_gt["gnss_path"] = os.path.join(
                            "train", drive_id, phone_name, "device_gnss.csv"
                        )
                        df_gt["imu_path"] = os.path.join(
                            "train", drive_id, phone_name, "device_imu.csv"
                        )

                        train_trips.append(df_gt)
                    except Exception as e:
                        print(f"Error reading {gt_path}: {e}")
    else:
        print(f"Warning: {train_base_path} does not exist.")

    if not train_trips:
        # If no training data, we create empty dataframes to avoid crash,
        # but raise error if strictly required. Given the task, we expect data.
        raise ValueError("No training data found in ./input/train")

    full_train_df = pd.concat(train_trips, ignore_index=True)

    # Split into Train and Validation using Group Sampling on drive_id
    unique_drives = full_train_df["drive_id"].unique()

    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_drives)

    n_train = int(len(unique_drives) * TRAIN_VAL_SPLIT_RATIO)
    train_drives = unique_drives[:n_train]
    val_drives = unique_drives[n_train:]

    train_df = full_train_df[full_train_df["drive_id"].isin(train_drives)].copy()
    val_df = full_train_df[full_train_df["drive_id"].isin(val_drives)].copy()

    # Save to CSV
    train_metadata_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_metadata_path = os.path.join(METADATA_DIR, "validation_metadata.csv")

    train_df.to_csv(train_metadata_path, index=False)
    val_df.to_csv(val_metadata_path, index=False)

    print(f"Train metadata saved to {train_metadata_path}")
    print(f"Validation metadata saved to {val_metadata_path}")

    # ---------------------------------------------------------
    # 2. Generate Test Metadata
    # ---------------------------------------------------------
    print("Generating Test Metadata...")

    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Sample submission not found at {sample_sub_path}")

    test_df = pd.read_csv(sample_sub_path)

    # Scan test directory to build a lookup of valid trips to ensure correct path mapping
    valid_test_trips = {}  # tripId -> (drive_id, phone_name)
    test_base_path = os.path.join(INPUT_DIR, "test")

    if os.path.exists(test_base_path):
        test_drive_ids = os.listdir(test_base_path)
        for drive_id in test_drive_ids:
            drive_path = os.path.join(test_base_path, drive_id)
            if not os.path.isdir(drive_path):
                continue
            phone_names = os.listdir(drive_path)
            for phone_name in phone_names:
                # Construct the ID matching the format in sample_submission
                trip_id = f"{drive_id}-{phone_name}"
                valid_test_trips[trip_id] = (drive_id, phone_name)

    # Function to apply to test_df to get paths
    def get_path_info(row):
        trip_id = row["tripId"]
        drive_id = None
        phone_name = None

        if trip_id in valid_test_trips:
            drive_id, phone_name = valid_test_trips[trip_id]
        else:
            # Fallback: try to split by last hyphen if not found in scan
            parts = trip_id.rsplit("-", 1)
            if len(parts) == 2:
                drive_id, phone_name = parts

        if drive_id and phone_name:
            gnss_path = os.path.join("test", drive_id, phone_name, "device_gnss.csv")
            imu_path = os.path.join("test", drive_id, phone_name, "device_imu.csv")
            return pd.Series([drive_id, phone_name, gnss_path, imu_path])
        else:
            return pd.Series([None, None, None, None])

    path_info = test_df.apply(get_path_info, axis=1)
    path_info.columns = ["drive_id", "phone_name", "gnss_path", "imu_path"]

    # Combine sample submission with path info
    # We drop LatitudeDegrees and LongitudeDegrees from sample_submission as they are placeholders
    test_metadata = pd.concat(
        [test_df[["tripId", "UnixTimeMillis"]], path_info], axis=1
    )

    test_metadata_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    test_metadata.to_csv(test_metadata_path, index=False)
    print(f"Test metadata saved to {test_metadata_path}")

    # ---------------------------------------------------------
    # 3. Verification
    # ---------------------------------------------------------
    print("Verifying datasets...")

    # Load back the metadata
    df_train_loaded = pd.read_csv(train_metadata_path)
    df_val_loaded = pd.read_csv(val_metadata_path)
    df_test_loaded = pd.read_csv(test_metadata_path)

    # 3.1 Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train samples: {len(df_train_loaded)}")
    print(f"Val samples: {len(df_val_loaded)}")
    print(f"Test samples: {len(df_test_loaded)}")
    print(f"Train unique drives: {df_train_loaded['drive_id'].nunique()}")
    print(f"Val unique drives: {df_val_loaded['drive_id'].nunique()}")
    print(f"Test unique trips: {df_test_loaded['tripId'].nunique()}")

    # 3.2 Check File Paths
    def check_paths(df, name):
        print(f"Checking paths for {name} dataset...")
        paths_to_check = []
        if "gnss_path" in df.columns:
            paths_to_check.extend(df["gnss_path"].dropna().unique().tolist())
        if "imu_path" in df.columns:
            paths_to_check.extend(df["imu_path"].dropna().unique().tolist())

        if not paths_to_check:
            print(f"No paths to check for {name}.")
            return

        # Random sample 1000 paths
        if len(paths_to_check) > 1000:
            paths_to_check = random.sample(paths_to_check, 1000)

        missing_count = 0
        missing_samples = []
        for p in paths_to_check:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        ratio = missing_count / len(paths_to_check)
        print(f"Missing file ratio for {name}: {ratio:.4f}")

        if ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise AssertionError(
                f"Too many missing files in {name} metadata! Ratio: {ratio}"
            )

    check_paths(df_train_loaded, "Train")
    check_paths(df_val_loaded, "Validation")
    check_paths(df_test_loaded, "Test")

    # 3.3 Verify Validation Split
    print("Verifying split integrity...")
    train_drives_set = set(df_train_loaded["drive_id"].unique())
    val_drives_set = set(df_val_loaded["drive_id"].unique())

    intersection = train_drives_set.intersection(val_drives_set)
    if intersection:
        raise AssertionError(
            f"Train and Validation sets share drive_ids: {intersection}"
        )

    # Verify ratio roughly
    total_drives = len(train_drives_set) + len(val_drives_set)
    if total_drives > 0:
        actual_train_ratio = len(train_drives_set) / total_drives
        print(f"Actual Train Split Ratio (by drive): {actual_train_ratio:.2f}")

    print("Metadata generation and verification complete.")


if __name__ == "__main__":
    generate_metadata()
