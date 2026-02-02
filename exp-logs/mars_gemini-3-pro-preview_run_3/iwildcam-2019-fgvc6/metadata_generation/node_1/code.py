import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw datasets...")
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Fallback for checking sample_submission if test.csv is missing/empty (though task says it exists)
        raise

    # Clean column names (remove whitespace)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Map common variations to expected names
    column_mapping = {
        "id": "Id",
        "image_id": "Id",
        "category": "Category",
        "category_id": "Category",
        "label": "Category",
    }
    train_df = train_df.rename(columns=column_mapping)
    test_df = test_df.rename(columns=column_mapping)

    # Remove 'Unnamed: 0' column if it exists
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    # Verify essential columns
    if "Id" not in train_df.columns or "Category" not in train_df.columns:
        raise ValueError("train.csv must contain 'Id' and 'Category' columns.")
    if "Id" not in test_df.columns:
        raise ValueError("test.csv must contain 'Id' column.")

    # Construct relative file paths
    # Assuming images are .jpg based on dataset information
    train_df["file_path"] = train_df["Id"].apply(
        lambda x: os.path.join("train_images", f"{x}.jpg")
    )
    test_df["file_path"] = test_df["Id"].apply(
        lambda x: os.path.join("test_images", f"{x}.jpg")
    )

    print(f"Original Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Split training data into Train and Validation
    print("Performing stratified split...")

    # Identify rare classes that might break stratification
    class_counts = train_df["Category"].value_counts()
    rare_classes = class_counts[class_counts < 2].index

    if not rare_classes.empty:
        print(
            f"Warning: {len(rare_classes)} classes have fewer than 2 samples. Assigning them to train set to avoid split errors."
        )
        is_rare = train_df["Category"].isin(rare_classes)
        train_rare = train_df[is_rare]
        train_common = train_df[~is_rare]

        # Stratified split on common classes
        train_split, val_split = train_test_split(
            train_common,
            test_size=VAL_SIZE,
            stratify=train_common["Category"],
            random_state=RANDOM_STATE,
        )

        # Add rare classes back to training set
        train_split = pd.concat([train_split, train_rare])

        # Shuffle training set to mix rare classes in
        train_split = train_split.sample(frac=1, random_state=RANDOM_STATE).reset_index(
            drop=True
        )
        val_split = val_split.reset_index(drop=True)
    else:
        # Standard stratified split
        train_split, val_split = train_test_split(
            train_df,
            test_size=VAL_SIZE,
            stratify=train_df["Category"],
            random_state=RANDOM_STATE,
        )

    # Save metadata files
    print("Saving metadata to ./metadata/ ...")
    train_split.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # --- Verification Section ---
    print("\n--- Verification ---")

    # Reload datasets to verify integrity
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    datasets = {"Train": train_meta, "Validation": val_meta, "Test": test_meta}

    # 1. Summary Statistics
    for name, df in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {df.shape}")
        if "Category" in df.columns:
            print(f"Unique classes: {df['Category'].nunique()}")
            print("Top 5 classes distribution:")
            print(df["Category"].value_counts(normalize=True).head())

    # 2. File Path Check
    print("\nChecking file path validity (sampling 1000 files per dataset)...")
    for name, df in datasets.items():
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Path is relative to ./input
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        missing_ratio = missing_count / sample_size
        print(f"{name}: Missing Ratio = {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Examples of missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} are invalid."
            )

    # 3. Validation Split Verification
    print("\nVerifying validation split requirements...")

    # Check Split Ratio
    total_train_val = len(train_meta) + len(val_meta)
    actual_val_ratio = len(val_meta) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f}")

    # Allow small tolerance (0.01)
    if not (0.19 <= actual_val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates significantly from 0.2"
        )

    # Check Stratification
    # Compare class distributions
    train_dist = train_meta["Category"].value_counts(normalize=True)
    val_dist = val_meta["Category"].value_counts(normalize=True)

    # Align indices to handle potential missing rare classes in validation
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_classes, fill_value=0)
    val_dist = val_dist.reindex(all_classes, fill_value=0)

    # Calculate Mean Absolute Error between distributions
    mae = (train_dist - val_dist).abs().mean()
    print(f"Class Distribution MAE: {mae:.6f}")

    # Threshold for stratification failure (0.05 is generous but catches major skew)
    if mae > 0.05:
        raise AssertionError(
            "Stratification failed: Class distributions in Train and Validation differ significantly."
        )

    print("\nSuccess: Metadata generated and verified.")


if __name__ == "__main__":
    main()
