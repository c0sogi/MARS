import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Create metadata directory
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Generating metadata...")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    df_train_raw = pd.read_csv(train_csv_path)

    # Add relative file path
    # Using forward slashes for consistency, though os.path.join is also valid
    df_train_raw["file_path"] = df_train_raw["Image"].apply(
        lambda x: os.path.join("train", x)
    )

    # Analyze Class Distribution
    class_counts = df_train_raw["Id"].value_counts()

    # Identify classes with too few samples for a 20% split (requires at least 5 samples)
    # Strategy: Put these few-shot classes in Train to ensure model sees them. Stratify the rest.
    # Threshold changed from 2 to 5 to prevent "test_size < n_classes" error.
    singletons = class_counts[class_counts < 5].index

    df_singletons = df_train_raw[df_train_raw["Id"].isin(singletons)].copy()
    df_multi = df_train_raw[~df_train_raw["Id"].isin(singletons)].copy()

    # Stratified Split on multi-sample classes
    # Using 0.2 test size for the 80:20 ratio requirement on the splitable portion
    if len(df_multi) > 0:
        train_multi, val_multi = train_test_split(
            df_multi, test_size=0.2, stratify=df_multi["Id"], random_state=RANDOM_STATE
        )
    else:
        train_multi = pd.DataFrame(columns=df_train_raw.columns)
        val_multi = pd.DataFrame(columns=df_train_raw.columns)

    # Combine to form final Train/Val sets
    # Train = Stratified Multi Train + All Singletons
    df_train_final = pd.concat([train_multi, df_singletons], ignore_index=True)
    df_val_final = val_multi.copy().reset_index(drop=True)

    # Shuffle training set to mix singletons and multi-samples
    df_train_final = df_train_final.sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Save Train/Val Metadata
    df_train_final.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val_final.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    test_dir = os.path.join(INPUT_DIR, "test")
    # List all jpg files
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    else:
        test_files = []
        print("Warning: Test directory not found.")

    df_test = pd.DataFrame(
        {
            "Image": test_files,
            "file_path": [os.path.join("test", f) for f in test_files],
        }
    )

    # Save Test Metadata
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")

    # ---------------------------------------------------------
    # 3. Validation & Checks
    # ---------------------------------------------------------
    print("\nPerforming validation checks...")

    # Load generated metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # A. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(
        f"Train Set: {len(train_meta)} samples, {train_meta['Id'].nunique()} unique Ids"
    )
    print(f"Val Set:   {len(val_meta)} samples, {val_meta['Id'].nunique()} unique Ids")
    print(f"Test Set:  {len(test_meta)} samples")
    print("--------------------------")

    # B. File Path Verification
    def verify_paths(df, dataset_name):
        if df.empty:
            print(f"[{dataset_name}] is empty, skipping path check.")
            return

        paths = df["file_path"].values
        total = len(paths)
        sample_size = min(1000, total)

        # Random sample
        np.random.seed(RANDOM_STATE)
        indices = np.random.choice(total, sample_size, replace=False)
        sample_paths = paths[indices]

        missing_count = 0
        missing_examples = []

        for p in sample_paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        ratio = missing_count / sample_size
        print(
            f"[{dataset_name}] Missing file ratio: {ratio:.4f} ({missing_count}/{sample_size})"
        )

        if ratio > 0.5:
            print(f"Examples of missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Validation failed: Too many missing files in {dataset_name} dataset."
            )

    verify_paths(train_meta, "Train")
    verify_paths(val_meta, "Validation")
    verify_paths(test_meta, "Test")

    # C. Split Verification
    # Requirement: Assert stratification success.

    # Check 1: No data leakage
    train_imgs = set(train_meta["Image"])
    val_imgs = set(val_meta["Image"])
    intersection = train_imgs.intersection(val_imgs)
    if intersection:
        raise AssertionError(
            f"Data Leakage detected: {len(intersection)} images found in both Train and Validation sets."
        )

    # Check 2: All classes in Validation must exist in Training
    # (Since we put singletons in Train, this should hold true)
    train_classes = set(train_meta["Id"])
    val_classes = set(val_meta["Id"])

    if not val_classes.issubset(train_classes):
        missing_in_train = val_classes - train_classes
        raise AssertionError(
            f"Stratification Error: Found classes in Validation that are not in Training: {list(missing_in_train)[:5]}"
        )

    # Check 3: Check ratio roughly
    # Note: Because singletons (which are many) are forced to train, Train % will be > 80%.
    # We just want to ensure we didn't accidentally empty the validation set.
    total_train_val = len(train_meta) + len(val_meta)
    if total_train_val > 0:
        val_ratio = len(val_meta) / total_train_val
        print(f"Effective Validation Ratio: {val_ratio:.4f}")

    if len(val_meta) == 0 and len(train_meta) > 0:
        print("Warning: Validation set is empty (likely all classes were few-shots).")
    elif len(train_meta) == 0:
        raise AssertionError("Train set is empty!")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
