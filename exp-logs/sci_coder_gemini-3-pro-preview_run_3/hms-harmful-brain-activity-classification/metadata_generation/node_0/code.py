import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# Configuration & Constants
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_SPLIT_RATIO = 0.8
SAMPLE_CHECK_SIZE = 1000
MISSING_FILE_THRESHOLD = 0.5


# ==========================================
# Metadata Generation
# ==========================================
def generate_metadata():
    """
    Generates metadata files for train, validation, and test sets.
    Splits train into train/val grouped by patient_id.
    Adds relative file paths and normalized probability targets.
    """
    print("Generating metadata...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- 1. Process Training Data ---
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")

    df_train = pd.read_csv(train_csv_path)

    # Construct relative file paths
    # Paths are relative to ./input, so we store 'train_eegs/123.parquet'
    df_train["eeg_path"] = df_train["eeg_id"].apply(
        lambda x: os.path.join("train_eegs", f"{x}.parquet")
    )
    df_train["spec_path"] = df_train["spectrogram_id"].apply(
        lambda x: os.path.join("train_spectrograms", f"{x}.parquet")
    )

    # Normalize target votes to probabilities
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # Calculate total votes per row
    df_train["total_votes"] = df_train[vote_cols].sum(axis=1)

    # Create probability columns (e.g., seizure_prob)
    for col in vote_cols:
        prob_col = col.replace("_vote", "_prob")
        # Avoid division by zero if total_votes is 0 (though unlikely in this dataset)
        df_train[prob_col] = df_train.apply(
            lambda row: (
                row[col] / row["total_votes"] if row["total_votes"] > 0 else 0.0
            ),
            axis=1,
        )

    # Split into Train and Validation
    # We must use GroupShuffleSplit to ensure patients do not leak between sets
    gss = GroupShuffleSplit(
        n_splits=1, train_size=TRAIN_SPLIT_RATIO, random_state=RANDOM_STATE
    )

    # The split method yields indices
    train_idx, val_idx = next(gss.split(df_train, groups=df_train["patient_id"]))

    train_meta = df_train.iloc[train_idx].copy()
    val_meta = df_train.iloc[val_idx].copy()

    # Save to metadata directory
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    print(f"Saved train.csv with {len(train_meta)} rows.")
    print(f"Saved val.csv with {len(val_meta)} rows.")

    # --- 2. Process Test Data ---
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")
    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)

        # Construct relative file paths for test
        df_test["eeg_path"] = df_test["eeg_id"].apply(
            lambda x: os.path.join("test_eegs", f"{x}.parquet")
        )
        df_test["spec_path"] = df_test["spectrogram_id"].apply(
            lambda x: os.path.join("test_spectrograms", f"{x}.parquet")
        )

        # Save to metadata directory
        df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)
        print(f"Saved test.csv with {len(df_test)} rows.")
    else:
        print("Warning: test.csv not found in input.")


# ==========================================
# Validation & Checks
# ==========================================
def check_paths_exist(df, base_dir, path_cols):
    """
    Checks a random sample of paths in the dataframe to ensure files exist.
    Returns the failure ratio and a list of missing examples.
    """
    missing_files = []
    total_checked = 0

    for col in path_cols:
        # Get unique paths to avoid checking duplicates repeatedly
        unique_paths = df[col].dropna().unique()

        # Sample if too many
        if len(unique_paths) > SAMPLE_CHECK_SIZE:
            # Use numpy for reproducible sampling
            rng = np.random.default_rng(seed=RANDOM_STATE)
            paths_to_check = rng.choice(unique_paths, SAMPLE_CHECK_SIZE, replace=False)
        else:
            paths_to_check = unique_paths

        for rel_path in paths_to_check:
            full_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(full_path):
                missing_files.append(rel_path)
            total_checked += 1

    if total_checked == 0:
        return 0.0, []

    ratio = len(missing_files) / total_checked
    return ratio, missing_files[:5]  # Return top 5 missing for debug


def perform_checks():
    print("\nPerforming validation checks...")

    # Load created metadata
    try:
        train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Metadata files not found for validation: {e}")

    # 1. Summary Statistics
    print("-" * 30)
    print("Dataset Statistics")
    print("-" * 30)
    print(f"Train Rows: {len(train_df)}")
    print(f"Val Rows:   {len(val_df)}")
    print(f"Test Rows:  {len(test_df)}")

    print(f"Train Unique Patients: {train_df['patient_id'].nunique()}")
    print(f"Val Unique Patients:   {val_df['patient_id'].nunique()}")

    # Class balance check (sum of probabilities)
    prob_cols = [c for c in train_df.columns if c.endswith("_prob")]
    if prob_cols:
        print("\nClass Distribution (Sum of Probabilities - Train):")
        print(train_df[prob_cols].sum())

    # 2. File Path Verification
    print("\nChecking file path resolution...")
    datasets = [("Train", train_df), ("Val", val_df), ("Test", test_df)]

    for name, df in datasets:
        ratio, missing = check_paths_exist(df, INPUT_DIR, ["eeg_path", "spec_path"])
        print(f"[{name}] Missing File Ratio: {ratio:.4f}")

        if ratio > MISSING_FILE_THRESHOLD:
            print(f"Sample missing files: {missing}")
            raise FileNotFoundError(
                f"[{name}] Missing file ratio {ratio:.2f} exceeds threshold {MISSING_FILE_THRESHOLD}"
            )

    # 3. Validation Split Verification
    print("\nVerifying split integrity...")

    # Check for patient leakage
    train_patients = set(train_df["patient_id"].unique())
    val_patients = set(val_df["patient_id"].unique())
    intersection = train_patients.intersection(val_patients)

    if intersection:
        raise AssertionError(
            f"Data Leakage! {len(intersection)} patients found in both train and validation sets."
        )
    else:
        print("Success: No patient overlap between train and validation.")

    # Check split ratio roughly (by patient count or row count)
    total_rows = len(train_df) + len(val_df)
    actual_train_ratio = len(train_df) / total_rows
    print(f"Actual Train Split Ratio (by rows): {actual_train_ratio:.4f}")

    # Assert that we actually have data
    if len(train_df) == 0 or len(val_df) == 0:
        raise AssertionError("Split resulted in empty train or validation set.")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    perform_checks()
