import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import glob


def generate_metadata():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load source CSVs
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    info_csv_path = os.path.join(INPUT_DIR, "HuBMAP-20-dataset_information.csv")

    df_train_rle = pd.read_csv(train_csv_path)
    df_sample_sub = pd.read_csv(sample_sub_path)
    df_info = pd.read_csv(info_csv_path)

    # Helper to extract ID from filename in info csv
    # image_file column is like 'afa5e8098.tiff'
    df_info["id"] = df_info["image_file"].apply(lambda x: os.path.splitext(x)[0])

    # --- Process Training Data ---
    # Merge RLE data with patient info
    # Note: df_train_rle has 'id', df_info has 'id'
    # We use a left join on df_train_rle to ensure we keep all training samples
    df_train_merged = pd.merge(df_train_rle, df_info, on="id", how="left")

    # Construct file paths for training data
    # Images and JSONs are in input/train/
    def get_train_paths(row):
        base_path = os.path.join("train", row["id"])
        return pd.Series(
            {
                "image_path": os.path.join(INPUT_DIR, base_path + ".tiff"),
                "json_path": os.path.join(INPUT_DIR, base_path + ".json"),
                "anatomical_json_path": os.path.join(
                    INPUT_DIR, base_path + "-anatomical-structure.json"
                ),
            }
        )

    path_cols = df_train_merged.apply(get_train_paths, axis=1)
    df_train_full = pd.concat([df_train_merged, path_cols], axis=1)

    # Handle missing patient_number if any (fill with ID to treat as unique group)
    if "patient_number" not in df_train_full.columns:
        df_train_full["patient_number"] = df_train_full["id"]
    df_train_full["patient_number"] = df_train_full["patient_number"].fillna(
        df_train_full["id"]
    )

    # Split into Train and Validation
    # We must group by patient_number to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(
        gss.split(df_train_full, groups=df_train_full["patient_number"])
    )

    train_set = df_train_full.iloc[train_idx].copy()
    val_set = df_train_full.iloc[val_idx].copy()

    # --- Process Test Data ---
    # Construct file paths for test data
    # Images and JSONs are in input/test/
    # We join with info if available, but primarily rely on sample_submission IDs
    df_test_merged = pd.merge(df_sample_sub[["id"]], df_info, on="id", how="left")

    def get_test_paths(row):
        base_path = os.path.join("test", row["id"])
        return pd.Series(
            {
                "image_path": os.path.join(INPUT_DIR, base_path + ".tiff"),
                "json_path": os.path.join(INPUT_DIR, base_path + ".json"),
                "anatomical_json_path": os.path.join(
                    INPUT_DIR, base_path + "-anatomical-structure.json"
                ),
            }
        )

    test_path_cols = df_test_merged.apply(get_test_paths, axis=1)
    test_set = pd.concat([df_test_merged, test_path_cols], axis=1)

    # --- Save Metadata ---
    train_set.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_set.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_set.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    print("Metadata generation complete.")
    return train_set, val_set, test_set


def validate_metadata(train_df, val_df, test_df):
    print("\n--- Validating Metadata ---")

    # 1. Summary Statistics
    print(f"Train set shape: {train_df.shape}")
    print(f"Val set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    print(f"Train unique patients: {train_df['patient_number'].nunique()}")
    print(f"Val unique patients: {val_df['patient_number'].nunique()}")

    # 2. Check File Paths
    def check_paths(df, name):
        if df.empty:
            return

        # Select columns that look like paths
        path_cols = [c for c in df.columns if "path" in c]

        # Sample up to 1000 paths
        paths_to_check = []
        for col in path_cols:
            paths = df[col].dropna().tolist()
            if paths:
                # Random sample
                sample_size = min(len(paths), 350)  # distribute 1000 across ~3 cols
                paths_to_check.extend(
                    np.random.choice(paths, sample_size, replace=False)
                )

        if not paths_to_check:
            return

        missing_count = 0
        missing_samples = []

        for p in paths_to_check:
            # Paths in metadata are relative to current working dir (./input/...)
            # We check existence directly
            if not os.path.exists(p):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / len(paths_to_check)
        print(
            f"[{name}] Checked {len(paths_to_check)} paths. Missing ratio: {missing_ratio:.4f}"
        )

        if missing_ratio > 0.5:
            print(f"Sample missing files: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    check_paths(train_df, "Train")
    check_paths(val_df, "Val")
    check_paths(test_df, "Test")

    # 3. Verify Split Requirements
    # Assert stratification/grouping
    train_patients = set(train_df["patient_number"].unique())
    val_patients = set(val_df["patient_number"].unique())

    intersection = train_patients.intersection(val_patients)
    if intersection:
        raise AssertionError(
            f"Patient leakage detected! Patients {intersection} are in both train and val."
        )

    print("Split verification passed: No patient leakage.")

    # Check split ratio roughly (might vary due to group sizes)
    total_train_val = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_train_val
    print(f"Validation ratio: {val_ratio:.2f} (Target: 0.2)")

    # We don't assert exact 0.2 because group sizes vary, but it should be reasonable
    # If we have very few groups, it might be skewed.

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    try:
        train_df, val_df, test_df = generate_metadata()

        # Reload to simulate fresh start and ensure files are correct
        train_df_loaded = pd.read_csv("./metadata/train_metadata.csv")
        val_df_loaded = pd.read_csv("./metadata/val_metadata.csv")
        test_df_loaded = pd.read_csv("./metadata/test_metadata.csv")

        validate_metadata(train_df_loaded, val_df_loaded, test_df_loaded)

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
