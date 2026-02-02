import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42

# Column definitions
VOTE_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
PROB_COLS = [
    "seizure_prob",
    "lpd_prob",
    "gpd_prob",
    "lrda_prob",
    "grda_prob",
    "other_prob",
]


def generate_metadata():
    """
    Generates metadata files for train, validation, and test sets.
    """
    print("Initializing metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    train_source_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_source_path):
        raise FileNotFoundError(f"Source file not found: {train_source_path}")

    df_train = pd.read_csv(train_source_path)

    # Generate file paths relative to ./input
    # e.g., train_eegs/12345.parquet
    df_train["eeg_path"] = df_train["eeg_id"].apply(lambda x: f"train_eegs/{x}.parquet")
    df_train["spectrogram_path"] = df_train["spectrogram_id"].apply(
        lambda x: f"train_spectrograms/{x}.parquet"
    )

    # Normalize votes to probabilities
    # Fill NaNs with 0 to be safe
    votes = df_train[VOTE_COLS].fillna(0).values
    vote_sums = votes.sum(axis=1, keepdims=True)

    # Handle potential division by zero (though unlikely in this dataset)
    vote_sums[vote_sums == 0] = 1.0

    probs = votes / vote_sums
    df_train[PROB_COLS] = probs

    # Split Data: Group Sampling by patient_id
    unique_patients = df_train["patient_id"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_patients)

    n_patients = len(unique_patients)
    n_train = int(n_patients * 0.8)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:])

    df_train_split = df_train[df_train["patient_id"].isin(train_patients)].copy()
    df_val_split = df_train[df_train["patient_id"].isin(val_patients)].copy()

    # Save training and validation metadata
    df_train_split.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val_split.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    print(f"Generated train.csv: {len(df_train_split)} samples")
    print(f"Generated val.csv: {len(df_val_split)} samples")

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    test_source_path = os.path.join(INPUT_DIR, "test.csv")
    if os.path.exists(test_source_path):
        df_test = pd.read_csv(test_source_path)

        # Generate file paths relative to ./input
        df_test["eeg_path"] = df_test["eeg_id"].apply(
            lambda x: f"test_eegs/{x}.parquet"
        )
        df_test["spectrogram_path"] = df_test["spectrogram_id"].apply(
            lambda x: f"test_spectrograms/{x}.parquet"
        )

        # Save test metadata
        df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)
        print(f"Generated test.csv: {len(df_test)} samples")
    else:
        print("Warning: test.csv not found, skipping test metadata generation.")


def validate_metadata():
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nStarting validation checks...")

    try:
        df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Validation failed: Could not load generated metadata. {e}"
        )

    # ---------------------------------------------------------
    # 1. Summary Statistics
    # ---------------------------------------------------------
    datasets = {"Train": df_train, "Validation": df_val, "Test": df_test}

    for name, df in datasets.items():
        print(f"\n--- {name} Set Summary ---")
        print(f"Shape: {df.shape}")
        print(f"Unique Patients: {df['patient_id'].nunique()}")

        # Check class distribution for labeled sets
        if name in ["Train", "Validation"]:
            print("Class Distribution (Mean Probabilities):")
            print(df[PROB_COLS].mean())

    # ---------------------------------------------------------
    # 2. File Path Verification
    # ---------------------------------------------------------
    def check_paths(df, col_name, dataset_name):
        if col_name not in df.columns:
            return

        # Sample up to 1000 paths
        sample_size = min(1000, len(df))
        sample_paths = (
            df[col_name].sample(n=sample_size, random_state=RANDOM_STATE).values
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            # Construct full path: ./input + relative_path
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"\nChecking {dataset_name} - {col_name}: Missing Ratio = {missing_ratio:.4f}"
        )

        if missing_ratio > 0.5:
            print("Examples of missing paths:", missing_samples)
            raise FileNotFoundError(
                f"Critical Error: More than 50% of files are missing for {dataset_name} ({col_name})."
            )

    check_paths(df_train, "eeg_path", "Train")
    check_paths(df_train, "spectrogram_path", "Train")
    check_paths(df_test, "eeg_path", "Test")
    check_paths(df_test, "spectrogram_path", "Test")

    # ---------------------------------------------------------
    # 3. Split Verification
    # ---------------------------------------------------------
    train_patients = set(df_train["patient_id"])
    val_patients = set(df_val["patient_id"])

    overlap = train_patients.intersection(val_patients)

    if len(overlap) > 0:
        raise AssertionError(
            f"Data Leakage Error: {len(overlap)} patients found in both Train and Validation sets."
        )

    print(
        "\nSplit Verification Passed: No patient overlap between Train and Validation."
    )
    print("All checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
