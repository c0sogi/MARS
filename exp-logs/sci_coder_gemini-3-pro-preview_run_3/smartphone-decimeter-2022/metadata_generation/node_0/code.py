import os
import glob
import pandas as pd
import numpy as np
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
RANDOM_STATE = 42


def get_relative_path(abs_path):
    return os.path.relpath(abs_path, start=INPUT_DIR)


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Scanning training data...")
    # Find all ground_truth.csv files
    # Structure: input/train/[drive_id]/[phone_name]/ground_truth.csv
    gt_files = glob.glob(os.path.join(TRAIN_DIR, "*", "*", "ground_truth.csv"))

    train_rows = []

    for gt_path in gt_files:
        # Extract drive_id and phone_name
        # path parts: .../train/drive_id/phone_name/ground_truth.csv
        parts = gt_path.split(os.sep)
        phone_name = parts[-2]
        drive_id = parts[-3]

        # Construct paths for sensor data
        # Note: We store paths relative to ./input to keep metadata portable/clean
        # The task requires paths relative to ./input

        # device_gnss.csv
        gnss_path_abs = os.path.join(TRAIN_DIR, drive_id, phone_name, "device_gnss.csv")
        gnss_path_rel = os.path.relpath(gnss_path_abs, INPUT_DIR)

        # device_imu.csv
        imu_path_abs = os.path.join(TRAIN_DIR, drive_id, phone_name, "device_imu.csv")
        imu_path_rel = os.path.relpath(imu_path_abs, INPUT_DIR)

        # ground_truth.csv (relative)
        gt_path_rel = os.path.relpath(gt_path, INPUT_DIR)

        # Read Ground Truth
        df_gt = pd.read_csv(gt_path)

        # We only need specific columns
        # tripId is typically drive_id + '-' + phone_name in the submission,
        # but let's construct it consistently.
        # Looking at sample submission: 2020-06-04-US-MTV-1-GooglePixel4
        trip_id = f"{drive_id}-{phone_name}"

        df_gt["tripId"] = trip_id
        df_gt["drive_id"] = drive_id
        df_gt["phone_name"] = phone_name
        df_gt["gnss_path"] = gnss_path_rel
        df_gt["imu_path"] = imu_path_rel
        df_gt["gt_path"] = gt_path_rel

        # Keep relevant columns
        cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
            "drive_id",
            "phone_name",
            "gnss_path",
            "imu_path",
            "gt_path",
        ]

        # Filter columns that exist (GT usually has these)
        df_subset = df_gt[cols]
        train_rows.append(df_subset)

    if not train_rows:
        raise ValueError("No training data found!")

    full_train_df = pd.concat(train_rows, ignore_index=True)

    print(f"Total training samples found: {len(full_train_df)}")

    # Split Train/Val using Group Sampling on drive_id
    unique_drives = full_train_df["drive_id"].unique()
    print(f"Unique drives in training set: {len(unique_drives)}")

    # Shuffle drives
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(unique_drives)

    # 80/20 Split
    n_train = int(len(unique_drives) * 0.8)
    train_drives = unique_drives[:n_train]
    val_drives = unique_drives[n_train:]

    print(f"Drives in Train: {len(train_drives)}, Drives in Val: {len(val_drives)}")

    train_df = full_train_df[full_train_df["drive_id"].isin(train_drives)].copy()
    val_df = full_train_df[full_train_df["drive_id"].isin(val_drives)].copy()

    # Save Train and Val metadata
    train_metadata_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_metadata_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    train_df.to_csv(train_metadata_path, index=False)
    val_df.to_csv(val_metadata_path, index=False)

    print("Processing test data...")
    # Load sample submission to get target tripIds and timestamps
    if not os.path.exists(SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission not found at {SAMPLE_SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # We need to map tripId to drive_id and phone_name to build paths.
    # We can scan the test directory to find valid drive/phone combinations.
    # Structure: input/test/[drive_id]/[phone_name]

    test_trips_map = {}  # tripId -> (drive_id, phone_name)

    # Scan test directory
    test_drive_dirs = glob.glob(os.path.join(TEST_DIR, "*"))
    for d_path in test_drive_dirs:
        drive_id = os.path.basename(d_path)
        phone_dirs = glob.glob(os.path.join(d_path, "*"))
        for p_path in phone_dirs:
            phone_name = os.path.basename(p_path)
            # Construct potential tripId.
            # Based on sample: 2020-06-04-US-MTV-1-GooglePixel4
            # drive: 2020-06-04-US-MTV-1, phone: GooglePixel4
            trip_id = f"{drive_id}-{phone_name}"
            test_trips_map[trip_id] = (drive_id, phone_name)

    # Add metadata columns to submission df
    # We use a list to collect data then merge to avoid modifying dataframe row by row

    meta_data = []

    # Get unique trips from submission to optimize
    unique_sub_trips = sub_df["tripId"].unique()

    for trip_id in unique_sub_trips:
        if trip_id in test_trips_map:
            drive_id, phone_name = test_trips_map[trip_id]

            gnss_path_abs = os.path.join(
                TEST_DIR, drive_id, phone_name, "device_gnss.csv"
            )
            gnss_path_rel = os.path.relpath(gnss_path_abs, INPUT_DIR)

            imu_path_abs = os.path.join(
                TEST_DIR, drive_id, phone_name, "device_imu.csv"
            )
            imu_path_rel = os.path.relpath(imu_path_abs, INPUT_DIR)

            meta_data.append(
                {
                    "tripId": trip_id,
                    "drive_id": drive_id,
                    "phone_name": phone_name,
                    "gnss_path": gnss_path_rel,
                    "imu_path": imu_path_rel,
                }
            )
        else:
            # Fallback or error if tripId in submission doesn't match folder structure
            # This shouldn't happen based on dataset description
            print(f"Warning: TripID {trip_id} not found in test directory scan.")

    meta_df = pd.DataFrame(meta_data)

    # Merge metadata into submission df
    test_df = sub_df.merge(meta_df, on="tripId", how="left")

    # Save Test metadata
    test_metadata_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    test_df.to_csv(test_metadata_path, index=False)

    print("Metadata generation complete.")

    # --- Verification Step ---
    print("\n--- Verifying Metadata ---")

    # 1. Load datasets
    df_train_check = pd.read_csv(train_metadata_path)
    df_val_check = pd.read_csv(val_metadata_path)
    df_test_check = pd.read_csv(test_metadata_path)

    # 2. Print Summary Statistics
    print(f"Train Shape: {df_train_check.shape}")
    print(f"Val Shape: {df_val_check.shape}")
    print(f"Test Shape: {df_test_check.shape}")

    print(f"Train Unique Drives: {df_train_check['drive_id'].nunique()}")
    print(f"Val Unique Drives: {df_val_check['drive_id'].nunique()}")

    # 3. Verify Split (Disjoint drives)
    train_drives_set = set(df_train_check["drive_id"].unique())
    val_drives_set = set(df_val_check["drive_id"].unique())

    intersection = train_drives_set.intersection(val_drives_set)
    if intersection:
        raise AssertionError(
            f"Data Leakage detected! Drives {intersection} present in both Train and Val."
        )
    print("Verification Passed: Train and Val sets are disjoint by drive_id.")

    # 4. Check File Paths
    def check_paths(df, name):
        print(f"Checking file paths for {name}...")
        # Columns that contain paths
        path_cols = [c for c in df.columns if "path" in c]
        if not path_cols:
            return

        # Sample 1000 rows
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        total_checks = 0
        missing_samples = []

        for _, row in sample.iterrows():
            for col in path_cols:
                rel_path = row[col]
                # Paths are relative to ./input, so we join with INPUT_DIR's parent or just use as is relative to current CWD?
                # The requirement says "All file paths stored within the metadata must be relative to the ./input directory."
                # So if path is "train/...", full path is "./input/train/..."
                full_path = os.path.join(INPUT_DIR, rel_path)

                total_checks += 1
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_samples) < 5:
                        missing_samples.append(full_path)

        if total_checks > 0:
            ratio = missing_count / total_checks
            print(f"  Missing File Ratio: {ratio:.4f} ({missing_count}/{total_checks})")

            if ratio > 0.5:
                print("  Sample missing paths:")
                for p in missing_samples:
                    print(f"    {p}")
                raise FileNotFoundError(
                    f"High ratio of missing files in {name} metadata: {ratio:.2f}"
                )
        else:
            print("  No path columns found to check.")

    check_paths(df_train_check, "Train")
    check_paths(df_val_check, "Val")
    check_paths(df_test_check, "Test")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
