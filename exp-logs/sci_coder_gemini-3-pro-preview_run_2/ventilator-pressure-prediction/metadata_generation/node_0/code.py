import pandas as pd
import numpy as np
import os

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def generate_metadata():
    """Generates metadata files for train, val, and test sets."""
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training and Validation Data ---
    train_csv_path = "train.csv"
    full_train_path = os.path.join(INPUT_DIR, train_csv_path)

    if not os.path.exists(full_train_path):
        raise FileNotFoundError(f"Input file not found: {full_train_path}")

    print(f"Loading {full_train_path}...")
    df_train_full = pd.read_csv(full_train_path)

    # Get unique breath IDs for Group Sampling
    unique_breaths = df_train_full["breath_id"].unique()

    # Shuffle and Split
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_breaths)

    split_index = int(len(unique_breaths) * 0.8)
    train_breaths = set(unique_breaths[:split_index])
    val_breaths = set(unique_breaths[split_index:])

    print(f"Total breaths: {len(unique_breaths)}")
    print(f"Train breaths: {len(train_breaths)}")
    print(f"Val breaths: {len(val_breaths)}")

    # Create Train Metadata
    # Filter rows belonging to train breaths
    train_meta = df_train_full[df_train_full["breath_id"].isin(train_breaths)].copy()
    train_meta["source_file"] = train_csv_path
    # Select specific columns for metadata: ID, Group ID, Label, Source
    train_meta = train_meta[["id", "breath_id", "pressure", "source_file"]]

    # Create Validation Metadata
    val_meta = df_train_full[df_train_full["breath_id"].isin(val_breaths)].copy()
    val_meta["source_file"] = train_csv_path
    val_meta = val_meta[["id", "breath_id", "pressure", "source_file"]]

    # Save to CSV
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    print(f"Saving train metadata to {train_meta_path}...")
    train_meta.to_csv(train_meta_path, index=False)

    print(f"Saving validation metadata to {val_meta_path}...")
    val_meta.to_csv(val_meta_path, index=False)

    # --- Process Test Data ---
    test_csv_path = "test.csv"
    full_test_path = os.path.join(INPUT_DIR, test_csv_path)

    if os.path.exists(full_test_path):
        print(f"Loading {full_test_path}...")
        df_test = pd.read_csv(full_test_path)

        test_meta = df_test.copy()
        test_meta["source_file"] = test_csv_path
        # Test set does not have 'pressure' column
        test_meta = test_meta[["id", "breath_id", "source_file"]]

        test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
        print(f"Saving test metadata to {test_meta_path}...")
        test_meta.to_csv(test_meta_path, index=False)
    else:
        print(f"Warning: {full_test_path} not found. Skipping test metadata.")

    print("Metadata generation complete.")


def validate_metadata():
    """Validates the generated metadata files."""
    print("\nStarting metadata validation...")

    files_to_check = ["train_metadata.csv", "val_metadata.csv", "test_metadata.csv"]

    train_meta = None
    val_meta = None

    for filename in files_to_check:
        filepath = os.path.join(METADATA_DIR, filename)
        if not os.path.exists(filepath):
            if filename == "test_metadata.csv":
                continue
            raise FileNotFoundError(f"Metadata file missing: {filepath}")

        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)

        # 1. Print Summary Statistics
        print(f"--- Stats for {filename} ---")
        print(f"Shape: {df.shape}")
        print(f"Unique breaths: {df['breath_id'].nunique()}")
        if "pressure" in df.columns:
            print(f"Pressure Mean: {df['pressure'].mean():.4f}")
            print(f"Pressure Std: {df['pressure'].std():.4f}")

        # Store for split validation
        if filename == "train_metadata.csv":
            train_meta = df
        elif filename == "val_metadata.csv":
            val_meta = df

        # 2. Check File Paths
        if "source_file" in df.columns:
            sample_size = min(1000, len(df))
            sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

            missing_count = 0
            missing_examples = []

            for _, row in sample.iterrows():
                rel_path = row["source_file"]
                abs_path = os.path.join(INPUT_DIR, rel_path)
                if not os.path.exists(abs_path):
                    missing_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(rel_path)

            missing_ratio = missing_count / sample_size
            print(f"Missing file ratio for {filename}: {missing_ratio:.4f}")

            if missing_ratio > 0.5:
                print(f"Sample missing paths: {missing_examples}")
                raise FileNotFoundError(f"High missing file ratio in {filename}")

    # 3. Verify Split Integrity
    if train_meta is not None and val_meta is not None:
        print("Verifying split integrity (stratification/grouping)...")

        train_breaths = set(train_meta["breath_id"].unique())
        val_breaths = set(val_meta["breath_id"].unique())

        # Check for overlap
        intersection = train_breaths.intersection(val_breaths)
        if len(intersection) > 0:
            raise AssertionError(
                f"Data Leakage detected! {len(intersection)} breath_ids found in both train and val."
            )

        # Check split ratio
        total_breaths = len(train_breaths) + len(val_breaths)
        train_ratio = len(train_breaths) / total_breaths
        print(f"Train split ratio (by breath_id): {train_ratio:.4f}")

        # Allow small floating point tolerance, but it should be exactly 0.8 based on the logic
        if not (0.79 <= train_ratio <= 0.81):
            raise AssertionError(
                f"Split ratio {train_ratio:.4f} deviates significantly from 0.80"
            )

        print("Split integrity verified successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
