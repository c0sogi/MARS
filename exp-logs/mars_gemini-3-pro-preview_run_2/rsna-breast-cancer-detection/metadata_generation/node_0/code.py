import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """Reads raw data, generates file paths, splits data, and saves metadata."""
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Loading raw data...")
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # Generate relative file paths
    # Train format: train_images/[patient_id]/[image_id].dcm
    train_df["file_path"] = train_df.apply(
        lambda x: f"train_images/{x['patient_id']}/{x['image_id']}.dcm", axis=1
    )

    # Test format: test_images/[patient_id]/[image_id].dcm
    test_df["file_path"] = test_df.apply(
        lambda x: f"test_images/{x['patient_id']}/{x['image_id']}.dcm", axis=1
    )

    print("Splitting training data into Train and Validation sets...")
    # We must split by patient_id to avoid data leakage (group split)
    # We also want to stratify by the target (cancer)

    # Get unique patients and their max cancer label (1 if patient has any cancer, 0 otherwise)
    patient_groups = train_df.groupby("patient_id")
    unique_patients = np.array(list(patient_groups.groups.keys()))
    patient_labels = patient_groups["cancer"].max()

    # Perform stratified split on patients
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=patient_labels,
    )

    # Filter original dataframe based on patient split
    train_split = train_df[train_df["patient_id"].isin(train_patients)].copy()
    val_split = train_df[train_df["patient_id"].isin(val_patients)].copy()

    print(f"Saving metadata to {METADATA_DIR}...")
    train_split.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    return train_split, val_split, test_df


def validate_metadata(train_df, val_df, test_df):
    """Performs validation checks on the generated metadata."""
    print("\n=== Validation Checks ===")

    # 1. Summary Statistics
    datasets = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
    for name, df in datasets:
        print(f"\n--- {name} Set ---")
        print(f"Total Samples: {len(df)}")
        print(f"Unique Patients: {df['patient_id'].nunique()}")
        if "cancer" in df.columns:
            print(
                f"Class Distribution (Cancer=1): {df['cancer'].sum()} ({df['cancer'].mean():.2%})"
            )

    # 2. Check File Paths
    print("\nChecking file path existence...")
    for name, df in datasets:
        check_file_paths(df, name)

    # 3. Verify Split Requirements
    print("\nVerifying split requirements...")

    # Check for Patient Overlap
    train_pats = set(train_df["patient_id"])
    val_pats = set(val_df["patient_id"])
    overlap = train_pats.intersection(val_pats)
    if overlap:
        raise AssertionError(
            f"Split failed: Found {len(overlap)} patients overlapping between Train and Validation sets."
        )

    # Check Stratification (Patient Level)
    train_pat_prev = train_df.groupby("patient_id")["cancer"].max().mean()
    val_pat_prev = val_df.groupby("patient_id")["cancer"].max().mean()
    print(f"Train Patient Prevalence: {train_pat_prev:.2%}")
    print(f"Val Patient Prevalence:   {val_pat_prev:.2%}")

    # Assert prevalence is reasonably close (within 5% absolute difference)
    if abs(train_pat_prev - val_pat_prev) > 0.05:
        raise AssertionError(
            "Split failed: Stratification resulted in significantly different class distributions."
        )

    print("\nAll validation checks passed successfully.")


def check_file_paths(df, dataset_name):
    """Checks a random sample of file paths for existence."""
    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    ratio = missing_count / sample_size
    print(
        f"[{dataset_name}] Missing File Ratio: {ratio:.4f} ({missing_count}/{sample_size})"
    )

    if ratio > 0.5:
        print(f"Sample missing paths: {missing_samples}")
        raise FileNotFoundError(
            f"Error: More than 50% of file paths in {dataset_name} metadata do not exist."
        )


if __name__ == "__main__":
    # Generate
    t_df, v_df, te_df = generate_metadata()

    # Validate (loading from disk to ensure saved files are correct)
    t_loaded = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    v_loaded = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    te_loaded = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    validate_metadata(t_loaded, v_loaded, te_loaded)
