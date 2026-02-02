import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load input CSV files: {e}")

    # -------------------------------------------------------------------------
    # 2. Preprocess Training Data
    # -------------------------------------------------------------------------
    print("Preprocessing training data...")

    # Generate relative file paths for EEGs and Spectrograms
    # Paths are stored relative to the ./input directory
    train_df["eeg_path"] = train_df["eeg_id"].apply(lambda x: f"train_eegs/{x}.parquet")
    train_df["spectrogram_path"] = train_df["spectrogram_id"].apply(
        lambda x: f"train_spectrograms/{x}.parquet"
    )

    # Normalize votes to probabilities
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    target_cols = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # Calculate total votes per row
    train_df["total_votes"] = train_df[vote_cols].sum(axis=1)

    # Filter out rows with 0 votes (sanity check, though unlikely)
    train_df = train_df[train_df["total_votes"] > 0].copy()

    # Calculate probabilities
    for v_col, t_col in zip(vote_cols, target_cols):
        train_df[t_col] = train_df[v_col] / train_df["total_votes"]

    # -------------------------------------------------------------------------
    # 3. Create Train/Validation Split
    # -------------------------------------------------------------------------
    print("Splitting data (Grouped by patient_id)...")

    # Use GroupShuffleSplit to ensure no patient leakage between train and val
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    # We split based on patient_id
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["patient_id"]))

    train_split_df = train_df.iloc[train_idx].copy()
    val_split_df = train_df.iloc[val_idx].copy()

    # -------------------------------------------------------------------------
    # 4. Preprocess Test Data
    # -------------------------------------------------------------------------
    print("Preprocessing test data...")
    test_df["eeg_path"] = test_df["eeg_id"].apply(lambda x: f"test_eegs/{x}.parquet")
    test_df["spectrogram_path"] = test_df["spectrogram_id"].apply(
        lambda x: f"test_spectrograms/{x}.parquet"
    )

    # -------------------------------------------------------------------------
    # 5. Save Metadata
    # -------------------------------------------------------------------------
    print("Saving metadata to ./metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_split_df.to_csv(train_meta_path, index=False)
    val_split_df.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    # -------------------------------------------------------------------------
    # 6. Verification Checks
    # -------------------------------------------------------------------------
    print("\n=== Performing Verification Checks ===")

    # Reload data to verify integrity
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    datasets = {
        "Train": df_train_check,
        "Validation": df_val_check,
        "Test": df_test_check,
    }

    # A. Summary Statistics
    for name, df in datasets.items():
        print(f"\n--- {name} Set ---")
        print(f"Total Samples: {len(df)}")
        if "patient_id" in df.columns:
            print(f"Unique Patients: {df['patient_id'].nunique()}")

        # Print class distribution for labeled sets
        if name in ["Train", "Validation"]:
            print("Class Distribution (Mean Probabilities):")
            print(df[target_cols].mean().to_string())

    # B. File Path Verification
    print("\n--- Checking File Path Resolution ---")

    def verify_paths(df, dataset_name, sample_size=1000):
        missing_count = 0
        total_checked = 0
        missing_examples = []

        # Check both EEG and Spectrogram paths
        path_cols = ["eeg_path", "spectrogram_path"]

        for col in path_cols:
            if col not in df.columns:
                continue

            # Select random sample
            paths_to_check = df[col].dropna()
            if len(paths_to_check) > sample_size:
                paths_to_check = paths_to_check.sample(
                    n=sample_size, random_state=RANDOM_STATE
                )

            for rel_path in paths_to_check:
                full_path = os.path.join(INPUT_DIR, rel_path)
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(rel_path)
                total_checked += 1

        if total_checked == 0:
            return

        missing_ratio = missing_count / total_checked
        print(
            f"{dataset_name}: Checked {total_checked} paths. Missing Ratio: {missing_ratio:.4f}"
        )

        if missing_ratio > 0.5:
            print(f"Examples of missing files in {dataset_name}:")
            for m in missing_examples:
                print(f"  - {m}")
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {dataset_name} metadata are invalid."
            )

    for name, df in datasets.items():
        verify_paths(df, name)

    # C. Validation Split Verification
    print("\n--- Verifying Split Strategy ---")
    train_patients = set(df_train_check["patient_id"].unique())
    val_patients = set(df_val_check["patient_id"].unique())

    overlap = train_patients.intersection(val_patients)
    overlap_count = len(overlap)

    print(f"Train Patients: {len(train_patients)}")
    print(f"Validation Patients: {len(val_patients)}")
    print(f"Patient Overlap: {overlap_count}")

    if overlap_count > 0:
        raise AssertionError(
            f"Split Validation Failed: Found {overlap_count} patients present in both Train and Validation sets. Group split failed."
        )

    print("\nSuccess: All metadata generated and verified.")


if __name__ == "__main__":
    main()
