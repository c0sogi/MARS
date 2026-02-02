import os
import sys
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split


def main():
    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    ESSENTIAL_DIR = os.path.join(INPUT_DIR, "essential_data")
    SUPPLEMENTAL_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    RANDOM_STATE = 42
    VAL_SIZE = 0.2
    NUM_SPECIES = 19

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # --- 1. Load Data ---

    # Load rec_id2filename.txt
    # This file maps recording IDs to filenames
    rec_filename_path = os.path.join(ESSENTIAL_DIR, "rec_id2filename.txt")
    df_filenames = pd.read_csv(rec_filename_path)
    df_filenames.columns = [c.strip().lower() for c in df_filenames.columns]

    # Ensure columns are identified correctly
    if "rec_id" not in df_filenames.columns:
        # Fallback if headers are missing or named differently
        df_filenames.rename(
            columns={
                df_filenames.columns[0]: "rec_id",
                df_filenames.columns[1]: "filename",
            },
            inplace=True,
        )

    # Load CVfolds_2.txt
    # This file contains the official train/test split (0=Train, 1=Test)
    cvfolds_path = os.path.join(ESSENTIAL_DIR, "CVfolds_2.txt")
    df_folds = pd.read_csv(cvfolds_path)
    df_folds.columns = [c.strip().lower() for c in df_folds.columns]

    # Load rec_labels_test_hidden.txt
    # This file contains labels. Format: rec_id,label1,label2... or rec_id,?
    labels_path = os.path.join(ESSENTIAL_DIR, "rec_labels_test_hidden.txt")
    label_rows = []
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")

            # Skip header if present
            if not parts[0].isdigit():
                continue

            rec_id = int(parts[0])
            labels = parts[1:]

            # Check for test set indicator '?'
            if "?" in labels:
                # Test set samples have no ground truth labels provided
                current_labels = []
            else:
                # Parse species IDs
                current_labels = [int(l) for l in labels if l.strip().isdigit()]

            label_rows.append((rec_id, current_labels))

    df_labels = pd.DataFrame(label_rows, columns=["rec_id", "label_list"])

    # Merge all dataframes on rec_id
    df = df_filenames.merge(df_folds, on="rec_id").merge(df_labels, on="rec_id")

    # --- 2. Feature Engineering (Paths & Labels) ---

    # Construct relative paths
    def get_wav_rel_path(fname):
        return os.path.join("essential_data", "src_wavs", fname)

    def get_spec_rel_path(fname):
        # Spectrograms are BMP files with the same basename
        bmp_name = os.path.splitext(fname)[0] + ".bmp"
        return os.path.join("supplemental_data", "spectrograms", bmp_name)

    df["file_path_wav"] = df["filename"].apply(get_wav_rel_path)
    df["file_path_spec"] = df["filename"].apply(get_spec_rel_path)

    # Create Multi-Hot Label Columns
    label_cols = [f"species_{i}" for i in range(NUM_SPECIES)]
    label_matrix = np.zeros((len(df), NUM_SPECIES), dtype=int)

    for idx, row in df.iterrows():
        for lbl in row["label_list"]:
            if 0 <= lbl < NUM_SPECIES:
                label_matrix[idx, lbl] = 1

    df_labels_expanded = pd.DataFrame(label_matrix, columns=label_cols)
    df = pd.concat([df, df_labels_expanded], axis=1)

    # --- 3. Split Data ---

    # Separate Official Test Set (Fold 1)
    test_df = df[df["fold"] == 1].copy().reset_index(drop=True)

    # Separate Development Set (Fold 0)
    dev_df = df[df["fold"] == 0].copy().reset_index(drop=True)

    # Shuffle the development set before splitting
    dev_df = dev_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Prepare data for Iterative Stratification
    # X is just the rec_id (reshaped), y is the multi-label matrix
    X_dev = dev_df["rec_id"].values.reshape(-1, 1)
    y_dev = dev_df[label_cols].values

    # Perform stratified split for multi-label data
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_dev, y_dev, test_size=VAL_SIZE
    )

    # Recover DataFrames based on rec_ids returned by the split
    train_rec_ids = X_train.flatten()
    val_rec_ids = X_val.flatten()

    train_df = (
        dev_df[dev_df["rec_id"].isin(train_rec_ids)].copy().reset_index(drop=True)
    )
    val_df = dev_df[dev_df["rec_id"].isin(val_rec_ids)].copy().reset_index(drop=True)

    # --- 4. Save Metadata ---

    output_cols = ["rec_id", "file_path_wav", "file_path_spec"] + label_cols

    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    train_df[output_cols].to_csv(train_csv_path, index=False)
    val_df[output_cols].to_csv(val_csv_path, index=False)
    test_df[output_cols].to_csv(test_csv_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")

    # --- 5. Validation & Statistics ---

    print("\n=== Dataset Statistics ===")
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set:   {len(val_df)} samples")
    print(f"Test set:  {len(test_df)} samples")

    # Check split ratio
    total_dev = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_dev
    print(f"Validation ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    # Assert split ratio is within reasonable bounds (e.g., +/- 5%)
    assert (
        abs(val_ratio - VAL_SIZE) < 0.05
    ), f"Validation split ratio {val_ratio} deviates too much from {VAL_SIZE}"

    # Check file existence
    print("\n=== Checking File Paths ===")
    all_paths = []
    for d in [train_df, val_df, test_df]:
        all_paths.extend(d["file_path_wav"].tolist())
        all_paths.extend(d["file_path_spec"].tolist())

    missing_count = 0
    checked_count = 0
    sample_missing = []

    # Check all paths (since dataset is small, < 1000 files total)
    for rel_path in all_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        checked_count += 1
        if not os.path.exists(full_path):
            missing_count += 1
            if len(sample_missing) < 5:
                sample_missing.append(rel_path)

    missing_ratio = missing_count / checked_count if checked_count > 0 else 0
    print(f"Checked {checked_count} file paths.")
    print(f"Missing files: {missing_count} (Ratio: {missing_ratio:.4f})")

    if len(sample_missing) > 0:
        print("Example missing files:")
        for p in sample_missing:
            print(f"  {p}")

    if missing_ratio > 0.5:
        raise FileNotFoundError(
            f"Critical Error: More than 50% of files are missing. Ratio: {missing_ratio}"
        )

    # Stratification Check
    print("\n=== Label Distribution Check ===")
    train_prev = train_df[label_cols].mean()
    val_prev = val_df[label_cols].mean()

    dist_df = pd.DataFrame({"Train_Freq": train_prev, "Val_Freq": val_prev})
    dist_df["Diff"] = abs(dist_df["Train_Freq"] - dist_df["Val_Freq"])
    print(dist_df.round(4))

    # Ensure datasets are not empty
    assert len(train_df) > 0, "Training set is empty"
    assert len(val_df) > 0, "Validation set is empty"

    print("\nMetadata generation and validation successful.")


if __name__ == "__main__":
    main()
