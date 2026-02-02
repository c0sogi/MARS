import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def generate_metadata():
    # Define input and output directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw CSVs
    train_rle_path = os.path.join(INPUT_DIR, "train.csv")
    info_path = os.path.join(INPUT_DIR, "HuBMAP-20-dataset_information.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Read data
    # train.csv contains 'id' and 'encoding' (RLE)
    train_rle_df = pd.read_csv(train_rle_path)

    # dataset_info contains 'image_file', 'patient_number', etc.
    info_df = pd.read_csv(info_path)

    # sample_submission contains 'id' for test set
    test_df = pd.read_csv(sample_sub_path)

    # Preprocess info_df to get 'id' from 'image_file' (remove extension)
    info_df["id"] = info_df["image_file"].apply(lambda x: os.path.splitext(x)[0])

    # --- Process Training Data ---
    # Merge RLE data with info to get patient numbers and other metadata
    # We use an inner join to ensure we have metadata for all training samples
    full_train_df = pd.merge(train_rle_df, info_df, on="id", how="inner")

    # Construct relative file paths for training data
    # Based on file structure, training images are in input/train/
    full_train_df["image_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("input", "train", f"{x}.tiff")
    )
    full_train_df["json_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("input", "train", f"{x}.json")
    )
    full_train_df["anatomical_json_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("input", "train", f"{x}-anatomical-structure.json")
    )

    # --- Split Train/Val ---
    # We use GroupShuffleSplit to split by patient_number to avoid data leakage
    # 80:20 split, random_state=42
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

    groups = full_train_df["patient_number"]
    train_idx, val_idx = next(gss.split(full_train_df, groups=groups))

    train_metadata = full_train_df.iloc[train_idx].copy()
    val_metadata = full_train_df.iloc[val_idx].copy()

    # Add split label for clarity (optional, but good for merged debugging)
    train_metadata["split"] = "train"
    val_metadata["split"] = "validation"

    # --- Process Test Data ---
    # Merge test IDs with info if available, otherwise just use IDs
    # Note: dataset_info might contain test images too.
    test_metadata = pd.merge(test_df[["id"]], info_df, on="id", how="left")

    # Construct relative file paths for test data
    # Based on file structure, test images are in input/test/
    test_metadata["image_path"] = test_metadata["id"].apply(
        lambda x: os.path.join("input", "test", f"{x}.tiff")
    )
    # Test set usually doesn't have the ground truth glomerulus json provided in the same way or it's hidden,
    # but the anatomical structure is often provided.
    test_metadata["anatomical_json_path"] = test_metadata["id"].apply(
        lambda x: os.path.join("input", "test", f"{x}-anatomical-structure.json")
    )
    # The main json might exist in the folder (as per file listing), so we add it, but it might not be used for prediction.
    test_metadata["json_path"] = test_metadata["id"].apply(
        lambda x: os.path.join("input", "test", f"{x}.json")
    )

    test_metadata["split"] = "test"

    # --- Save Metadata ---
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_metadata.to_csv(train_save_path, index=False)
    val_metadata.to_csv(val_save_path, index=False)
    test_metadata.to_csv(test_save_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    return train_save_path, val_save_path, test_save_path


def verify_metadata(train_path, val_path, test_path):
    print("\n--- Verifying Metadata ---")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")

    print("\nTrain Patient IDs:", df_train["patient_number"].unique())
    print("Val Patient IDs:", df_val["patient_number"].unique())

    # 2. Check File Paths
    # Combine all dfs to check paths
    all_dfs = [df_train, df_val, df_test]
    path_cols = [
        "image_path",
        "anatomical_json_path",
    ]  # json_path might not exist for all test or could be optional

    paths_to_check = []
    for df in all_dfs:
        for col in path_cols:
            if col in df.columns:
                paths_to_check.extend(df[col].dropna().tolist())

    # Also check json_path for train/val specifically as it must exist there
    for df in [df_train, df_val]:
        if "json_path" in df.columns:
            paths_to_check.extend(df["json_path"].dropna().tolist())

    # Randomly select up to 1000 paths
    if len(paths_to_check) > 1000:
        check_sample = np.random.choice(paths_to_check, 1000, replace=False)
    else:
        check_sample = paths_to_check

    missing_count = 0
    missing_samples = []

    for p in check_sample:
        # Paths in metadata are relative to input root (e.g., input/train/...)
        # The script runs from root, so we check existence directly relative to CWD
        # However, the prompt says "All file paths stored within the metadata must be relative to the ./input directory."
        # If the path is "input/train/x.tiff", and we are in root, os.path.exists("input/train/x.tiff") should work.

        # Note: If the stored path is "train/x.tiff" (relative to input), we need to prepend "./input".
        # My implementation stored "input/train/x.tiff".

        # Let's verify what "relative to ./input" means. Usually it means if I am in ./input, the path is valid.
        # But for loading, it's easier if it includes the input dir.
        # The prompt says: "All file paths stored within the metadata must be relative to the ./input directory."
        # This is slightly ambiguous. It could mean "train/image.tiff" or "./train/image.tiff".
        # Given the requirement "The script must read raw data from the ./input directory",
        # I have constructed paths starting with 'input/'.
        # If the requirement implies the path *string* should not contain 'input/', I would adjust.
        # However, usually "relative to X" means path P such that os.path.join(X, P) is the absolute path.
        # If I store 'train/img.tiff', then os.path.join('./input', 'train/img.tiff') works.
        # If I store 'input/train/img.tiff', then it's relative to current working directory (root).
        # Let's stick to paths relative to the root (starting with input/) for immediate usability,
        # or strictly relative to input (starting with train/).
        # Re-reading: "relative to the ./input directory". This strictly means `train/file.tiff`.
        # BUT, standard practice in these tasks often prefers the full relative path from execution root.
        # Let's adjust to be safe: I will store paths starting with `train/...` or `test/...`
        # AND when checking existence, I will join with `./input`.
        pass

    # ADJUSTMENT: Storing paths relative to ./input (e.g., "train/file.tiff")
    # Reloading to fix paths in memory before saving again? No, let's fix the generation logic above conceptually.
    # Actually, to avoid confusion and ensure "load data efficiently", paths relative to the project root (starting with input/) are best.
    # But if the prompt insists on "relative to ./input", I will strip "input/".
    # Let's assume "relative to ./input" means `train/filename.tiff`.

    # Refactoring the path generation in the main function to be strictly relative to ./input
    # Then in verification, we prepend ./input.

    # ... (Logic updated in the code block below to reflect this strict interpretation) ...

    # 3. Verify Validation Split
    # Check for patient overlap
    train_patients = set(df_train["patient_number"].unique())
    val_patients = set(df_val["patient_number"].unique())

    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise AssertionError(
            f"Data Leakage detected! Patients in both train and val: {overlap}"
        )

    print("Validation split verification passed: No patient overlap.")


if __name__ == "__main__":
    # 1. Generate
    # Re-defining generation to ensure paths are relative to ./input
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    train_rle_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    info_df = pd.read_csv(os.path.join(INPUT_DIR, "HuBMAP-20-dataset_information.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    info_df["id"] = info_df["image_file"].apply(lambda x: os.path.splitext(x)[0])

    # Train merge
    full_train_df = pd.merge(train_rle_df, info_df, on="id", how="inner")

    # Paths (relative to ./input)
    # The files are in input/train/ or input/test/
    full_train_df["image_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("train", f"{x}.tiff")
    )
    full_train_df["json_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("train", f"{x}.json")
    )
    full_train_df["anatomical_json_path"] = full_train_df["id"].apply(
        lambda x: os.path.join("train", f"{x}-anatomical-structure.json")
    )

    # Split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    groups = full_train_df["patient_number"]
    train_idx, val_idx = next(gss.split(full_train_df, groups=groups))

    train_metadata = full_train_df.iloc[train_idx].copy()
    val_metadata = full_train_df.iloc[val_idx].copy()

    # Test merge
    test_metadata = pd.merge(test_df[["id"]], info_df, on="id", how="left")
    test_metadata["image_path"] = test_metadata["id"].apply(
        lambda x: os.path.join("test", f"{x}.tiff")
    )
    test_metadata["anatomical_json_path"] = test_metadata["id"].apply(
        lambda x: os.path.join("test", f"{x}-anatomical-structure.json")
    )
    test_metadata["json_path"] = test_metadata["id"].apply(
        lambda x: os.path.join("test", f"{x}.json")
    )  # Might not exist/be useful but consistent

    # Save
    train_metadata.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_metadata.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_metadata.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")

    # 2. Verify
    print("\n--- Starting Verification ---")

    # Load back
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Stats
    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    # Check Paths
    # We need to check if ./input/ + path exists
    paths_to_check = []

    # Collect paths
    for df in [df_train, df_val]:
        paths_to_check.extend(df["image_path"].tolist())
        paths_to_check.extend(df["json_path"].tolist())
        paths_to_check.extend(df["anatomical_json_path"].tolist())

    for df in [df_test]:
        paths_to_check.extend(df["image_path"].tolist())
        paths_to_check.extend(df["anatomical_json_path"].tolist())
        # Not checking json_path for test as it might not be guaranteed/used for inference in same way

    # Sample 1000
    if len(paths_to_check) > 1000:
        check_sample = np.random.choice(paths_to_check, 1000, replace=False)
    else:
        check_sample = paths_to_check

    missing_files = []
    for p in check_sample:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_files.append(p)

    missing_ratio = (
        len(missing_files) / len(check_sample) if len(check_sample) > 0 else 0
    )
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample of missing files:")
        for mp in missing_files[:5]:
            print(mp)
        raise FileNotFoundError("More than 50% of sampled file paths do not resolve.")

    # Check Split
    train_pats = set(df_train["patient_number"])
    val_pats = set(df_val["patient_number"])

    if not train_pats.isdisjoint(val_pats):
        raise AssertionError(
            f"Patient overlap detected between train and val: {train_pats & val_pats}"
        )

    print("Verification successful.")
