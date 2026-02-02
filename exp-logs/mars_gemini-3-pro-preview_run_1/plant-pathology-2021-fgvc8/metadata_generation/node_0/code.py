import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_IMAGES_DIR = "train_images"
TEST_IMAGES_DIR = "test_images"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Could not find {sample_sub_path}")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(sample_sub_path)

    # 2. Process Training Data (Add file paths)
    # The 'image' column contains the filename (e.g., 'abc.jpg')
    df_train_full["file_path"] = df_train_full["image"].apply(
        lambda x: os.path.join(TRAIN_IMAGES_DIR, x)
    )

    # 3. Create Validation Split
    # We stratify based on the 'labels' string. This treats each unique combination
    # of diseases as a distinct class, which is standard for this dataset.

    # Handle singleton classes (classes with only 1 sample cannot be stratified)
    label_counts = df_train_full["labels"].value_counts()
    singletons = label_counts[label_counts < 2].index.tolist()

    # Separate singletons from the rest
    df_singletons = df_train_full[df_train_full["labels"].isin(singletons)]
    df_stratify_candidates = df_train_full[~df_train_full["labels"].isin(singletons)]

    # Split the stratifiable data
    train_split, val_split = train_test_split(
        df_stratify_candidates,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_stratify_candidates["labels"],
    )

    # Add singletons to the training set to avoid data loss (or could be random split)
    # Adding to train ensures the model sees them at least once.
    df_train_final = (
        pd.concat([train_split, df_singletons], axis=0)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    df_val_final = val_split.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    # 4. Process Test Data
    df_test["file_path"] = df_test["image"].apply(
        lambda x: os.path.join(TEST_IMAGES_DIR, x)
    )

    # 5. Save Metadata
    df_train_final.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val_final.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print(f"Metadata generated in {METADATA_DIR}")
    print(f"Train shape: {df_train_final.shape}")
    print(f"Val shape: {df_val_final.shape}")
    print(f"Test shape: {df_test.shape}")


def check_file_existence(df, name):
    """Checks if a random sample of file paths exist."""
    sample_size = min(1000, len(df))
    if sample_size == 0:
        return

    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        # Path in metadata is relative to INPUT_DIR
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(row["file_path"])

    missing_ratio = missing_count / sample_size
    print(
        f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
    )

    if missing_ratio > 0.5:
        print(f"Sample missing files: {missing_samples}")
        raise FileNotFoundError(
            f"Too many missing files in {name} dataset. Ratio: {missing_ratio}"
        )


def validate_metadata():
    print("\nStarting Validation...")

    # Load generated metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"\nDataset: {name}")
        print(f"Total samples: {len(df)}")
        print(f"Unique labels: {df['labels'].nunique()}")
        print("Top 5 Label Classes:")
        print(df["labels"].value_counts().head(5))

    # 2. Check File Existence
    print("\n--- Checking File Existence ---")
    check_file_existence(df_train, "Train")
    check_file_existence(df_val, "Val")
    # We check test files as well. Based on the problem description,
    # the files corresponding to sample_submission should exist in the directory.
    check_file_existence(df_test, "Test")

    # 3. Verify Validation Split
    print("\n--- Verifying Split Requirements ---")

    # Check Ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Validation Split Ratio: {val_ratio:.4f}")

    # Allow small deviation due to singletons handling
    if not (0.18 <= val_ratio <= 0.22):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is outside acceptable range (0.18-0.22)."
        )

    # Check Stratification
    # We compare the relative frequency of the top classes
    train_dist = df_train["labels"].value_counts(normalize=True)
    val_dist = df_val["labels"].value_counts(normalize=True)

    print("\nClass Distribution Comparison (Top 5):")
    common_labels = train_dist.head(5).index
    for label in common_labels:
        t_prop = train_dist.get(label, 0)
        v_prop = val_dist.get(label, 0)
        diff = abs(t_prop - v_prop)
        print(
            f"Label: {label:<30} | Train: {t_prop:.4f} | Val: {v_prop:.4f} | Diff: {diff:.4f}"
        )

        # Assert that the difference in proportion is not extreme for common classes
        if diff > 0.05:
            raise AssertionError(
                f"Stratification failed for label '{label}'. Diff: {diff:.4f}"
            )

    print("\nValidation passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
