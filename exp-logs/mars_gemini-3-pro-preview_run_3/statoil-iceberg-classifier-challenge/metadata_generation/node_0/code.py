import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load train.json
    train_path = os.path.join(INPUT_DIR, "train.json")
    # Reading json can be memory intensive, but dataset size fits in 220GB RAM
    df_train_raw = pd.read_json(train_path)

    # Prepare metadata for training data
    # We keep id, inc_angle, is_iceberg. We drop band_1 and band_2 to save space in metadata.
    # We add source_file and original_index to allow retrieving the bands later.
    df_train_meta = df_train_raw.drop(columns=["band_1", "band_2"]).copy()
    df_train_meta["source_file"] = "train.json"
    df_train_meta["original_index"] = df_train_meta.index

    # Handle inc_angle: convert to numeric, coerce errors (na) to NaN
    # The dataset description mentions "na" values in inc_angle
    df_train_meta["inc_angle"] = pd.to_numeric(
        df_train_meta["inc_angle"], errors="coerce"
    )

    # Split into train and validation (80/20, stratified)
    print("Splitting training data...")
    train_df, val_df = train_test_split(
        df_train_meta,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_train_meta["is_iceberg"],
    )

    # Load test.json
    test_path = os.path.join(INPUT_DIR, "test.json")
    df_test_raw = pd.read_json(test_path)

    # Prepare metadata for test data
    df_test_meta = df_test_raw.drop(columns=["band_1", "band_2"]).copy()
    df_test_meta["source_file"] = "test.json"
    df_test_meta["original_index"] = df_test_meta.index
    df_test_meta["inc_angle"] = pd.to_numeric(
        df_test_meta["inc_angle"], errors="coerce"
    )

    # Save metadata
    print("Saving metadata...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    df_test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")
    return train_df, val_df, df_test_meta


def validate_metadata():
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    print("\nValidating generated metadata...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set size: {len(train_df)}")
    print(f"Val set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    print("\nClass Distribution (is_iceberg):")
    print(f"Train:\n{train_df['is_iceberg'].value_counts(normalize=True)}")
    print(f"Val:\n{val_df['is_iceberg'].value_counts(normalize=True)}")

    print("\nMissing Incidence Angles:")
    print(f"Train: {train_df['inc_angle'].isna().sum()}")
    print(f"Val: {val_df['inc_angle'].isna().sum()}")
    print(f"Test: {test_df['inc_angle'].isna().sum()}")

    # 2. File Path Check
    print("\n--- Checking File Paths ---")

    def check_paths(df, name):
        # Select random sample of up to 1000 paths
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=42)

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            # Path is relative to ./input
            rel_path = row["source_file"]
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"{name}: Checked {sample_size} paths. Missing ratio: {missing_ratio:.4f}"
        )

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    check_paths(train_df, "Train")
    check_paths(val_df, "Val")
    check_paths(test_df, "Test")

    # 3. Validation Split Verification
    print("\n--- Verifying Split Requirements ---")

    # Check split ratio
    total_train_val = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_train_val
    print(f"Validation ratio: {val_ratio:.4f}")

    # Allow small floating point deviation
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio} is not approximately 0.2"
        )

    # Check stratification
    train_mean = train_df["is_iceberg"].mean()
    val_mean = val_df["is_iceberg"].mean()
    print(f"Train target mean: {train_mean:.4f}")
    print(f"Val target mean: {val_mean:.4f}")

    # Stratification check (allow small variance due to discrete counts)
    if abs(train_mean - val_mean) > 0.05:
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
