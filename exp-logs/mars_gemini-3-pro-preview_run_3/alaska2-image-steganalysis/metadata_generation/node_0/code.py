import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Generates metadata CSV files for Train, Validation, and Test sets.
    """
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- 1. Process Training Data ---
    # We scan the Cover directory to get the list of unique image IDs.
    # We assume the dataset structure is parallel (Cover, JMiPOD, JUNIWARD, UERD have matching filenames).
    cover_dir = os.path.join(INPUT_DIR, "Cover")
    if not os.path.exists(cover_dir):
        raise FileNotFoundError(f"Cover directory not found at {cover_dir}")

    # List all jpg files in Cover
    image_ids = [f for f in os.listdir(cover_dir) if f.lower().endswith(".jpg")]
    image_ids.sort()  # Ensure deterministic order

    print(f"Found {len(image_ids)} unique image IDs in Cover directory.")

    # Define the sources and their labels
    # Cover is 0, all Stego algorithms are 1
    sources = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 1, "UERD": 1}

    data_records = []

    # Construct the dataset in memory
    # We iterate through IDs and generate paths for all 4 variations
    for img_id in image_ids:
        for source_folder, label in sources.items():
            # Path relative to ./input
            rel_path = os.path.join(source_folder, img_id)
            data_records.append(
                {
                    "image_id": img_id,
                    "image_path": rel_path,
                    "label": label,
                    "source": source_folder,
                }
            )

    full_df = pd.DataFrame(data_records)

    # --- 2. Create Train/Val Split ---
    # We must split by 'image_id' to prevent data leakage (same content in train and val)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # The groups are the image_ids
    train_idx, val_idx = next(
        splitter.split(full_df, y=full_df["label"], groups=full_df["image_id"])
    )

    train_df = full_df.iloc[train_idx].copy()
    val_df = full_df.iloc[val_idx].copy()

    # --- 3. Process Test Data ---
    test_dir = os.path.join(INPUT_DIR, "Test")
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
        test_files.sort()

        test_records = []
        for f in test_files:
            test_records.append(
                {
                    "image_id": f,
                    "image_path": os.path.join("Test", f),
                    "label": -1,  # Placeholder for test
                }
            )
        test_df = pd.DataFrame(test_records)
    else:
        print("Warning: Test directory not found. Creating empty test metadata.")
        test_df = pd.DataFrame(columns=["image_id", "image_path", "label"])

    # --- 4. Save Metadata ---
    print("Saving metadata files...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")
    return train_df, val_df, test_df


def check_file_existence(df, name):
    """
    Checks if a random sample of file paths in the dataframe exist.
    """
    if len(df) == 0:
        return

    # Check up to 1000 random files
    n_samples = min(1000, len(df))
    sample = df.sample(n=n_samples, random_state=RANDOM_STATE)

    missing_paths = []
    for _, row in sample.iterrows():
        full_path = os.path.join(INPUT_DIR, row["image_path"])
        if not os.path.exists(full_path):
            missing_paths.append(row["image_path"])

    missing_ratio = len(missing_paths) / n_samples
    print(
        f"[{name}] Missing file ratio: {missing_ratio:.4f} ({len(missing_paths)}/{n_samples})"
    )

    if missing_ratio > 0.5:
        print(f"Sample missing paths in {name}:")
        for p in missing_paths[:5]:
            print(f" - {p}")
        raise FileNotFoundError(
            f"Error: More than 50% of files are missing in {name} dataset metadata."
        )


def validate_metadata():
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nRunning validation checks...")

    # Load datasets
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # --- Check 1: Summary Statistics ---
    print("\n--- Dataset Summary ---")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")

    print("\nTrain Label Distribution:")
    print(train_df["label"].value_counts(normalize=True))

    print("\nVal Label Distribution:")
    print(val_df["label"].value_counts(normalize=True))

    # --- Check 2: File Existence ---
    print("\n--- Checking File Paths ---")
    check_file_existence(train_df, "Train")
    check_file_existence(val_df, "Validation")
    check_file_existence(test_df, "Test")

    # --- Check 3: Verify Split Strategy (Group Sampling) ---
    print("\n--- Verifying Split Integrity ---")

    train_ids = set(train_df["image_id"].unique())
    val_ids = set(val_df["image_id"].unique())

    # Check for overlap
    overlap = train_ids.intersection(val_ids)
    if len(overlap) > 0:
        raise AssertionError(
            f"Data Leakage Detected! {len(overlap)} image IDs are present in both Train and Validation sets."
        )

    print("Success: No image ID overlap between Train and Validation sets.")

    # Check split ratio based on groups (IDs)
    n_train_groups = len(train_ids)
    n_val_groups = len(val_ids)
    total_groups = n_train_groups + n_val_groups

    actual_val_ratio = n_val_groups / total_groups
    print(f"Group-wise Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Assert ratio is approximately correct (allow small variance due to discrete nature)
    if not (0.19 < actual_val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} is significantly different from expected {VAL_SIZE}"
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
