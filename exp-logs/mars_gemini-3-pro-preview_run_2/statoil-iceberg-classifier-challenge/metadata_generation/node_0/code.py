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
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data from JSON files...")
    train_path = os.path.join(INPUT_DIR, "train.json")
    test_path = os.path.join(INPUT_DIR, "test.json")

    # Load JSON files
    # Note: We load the full JSON to extract metadata columns.
    # The 'band_1' and 'band_2' columns are large lists of floats.
    df_train_raw = pd.read_json(train_path)
    df_test_raw = pd.read_json(test_path)

    print(f"Loaded train.json with shape: {df_train_raw.shape}")
    print(f"Loaded test.json with shape: {df_test_raw.shape}")

    # --- Process Training Data ---
    # Extract only metadata columns, dropping image bands to avoid duplication
    train_meta = df_train_raw[["id", "inc_angle", "is_iceberg"]].copy()
    train_meta["filepath"] = "train.json"

    # Handle incidence angle: convert to numeric, coercing 'na' to NaN
    train_meta["inc_angle"] = pd.to_numeric(train_meta["inc_angle"], errors="coerce")

    # --- Process Test Data ---
    test_meta = df_test_raw[["id", "inc_angle"]].copy()
    test_meta["filepath"] = "test.json"
    test_meta["inc_angle"] = pd.to_numeric(test_meta["inc_angle"], errors="coerce")

    # --- Create Validation Split ---
    print("Creating stratified validation split...")
    train_split, val_split = train_test_split(
        train_meta,
        test_size=VAL_SIZE,
        stratify=train_meta["is_iceberg"],
        random_state=RANDOM_STATE,
    )

    # --- Save Metadata ---
    print("Saving metadata files to ./metadata/ ...")
    train_split.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    return train_split, val_split, test_meta


def verify_data(train_df, val_df, test_df):
    print("\n" + "=" * 30)
    print("VERIFICATION CHECKS")
    print("=" * 30)

    # 1. Summary Statistics
    datasets = {"Train": train_df, "Validation": val_df, "Test": test_df}
    for name, df in datasets.items():
        print(f"\n[{name} Dataset]")
        print(f"Shape: {df.shape}")
        print(f"Unique IDs: {df['id'].nunique()}")

        if "is_iceberg" in df.columns:
            dist = df["is_iceberg"].value_counts(normalize=True)
            print("Class Distribution:")
            print(dist)

        print("Incidence Angle Statistics:")
        print(df["inc_angle"].describe())
        print(f"Missing 'inc_angle' count: {df['inc_angle'].isna().sum()}")

    # 2. File Path Check
    print("\n[File Path Check]")
    # Combine all dataframes to sample paths
    all_data = pd.concat([train_df, val_df, test_df])

    # Sample up to 1000 paths
    n_samples = min(1000, len(all_data))
    sample_paths = all_data["filepath"].sample(n=n_samples, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / n_samples
    print(f"Checked {n_samples} file paths. Missing ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing paths:", missing_samples)
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio} exceeds allowed threshold of 0.5"
        )

    # 3. Validation Split Verification
    print("\n[Validation Split Verification]")

    # Check for ID overlap
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise AssertionError(
            f"Found {len(overlap)} overlapping IDs between Train and Validation sets!"
        )
    else:
        print("No overlap between Train and Validation sets.")

    # Check Stratification
    train_pos_ratio = train_df["is_iceberg"].mean()
    val_pos_ratio = val_df["is_iceberg"].mean()

    print(f"Train Positive Ratio: {train_pos_ratio:.4f}")
    print(f"Val Positive Ratio:   {val_pos_ratio:.4f}")

    # Assert ratios are close (within 5%)
    if abs(train_pos_ratio - val_pos_ratio) > 0.05:
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly."
        )
    else:
        print("Stratification successful.")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    train_df, val_df, test_df = generate_metadata()
    verify_data(train_df, val_df, test_df)
