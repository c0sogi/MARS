import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMG_DIR = "train_images"
TEST_IMG_DIR = "test_images"
RANDOM_STATE = 42


def generate_metadata():
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading raw data...")
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"{TRAIN_CSV} not found.")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"{TEST_CSV} not found.")

    df_train_full = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Clean up potential index columns
    if "Unnamed: 0" in df_train_full.columns:
        df_train_full = df_train_full.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in df_test.columns:
        df_test = df_test.drop(columns=["Unnamed: 0"])

    # 2. Construct File Paths
    # Assuming images are .jpg based on description
    df_train_full["file_path"] = df_train_full["Id"].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.jpg")
    )
    df_test["file_path"] = df_test["Id"].apply(
        lambda x: os.path.join(TEST_IMG_DIR, f"{x}.jpg")
    )

    # 3. Split Training Data
    # Check for potential group columns (e.g., location, sequence)
    # Common names in camera trap datasets: 'location', 'seq_id', 'site'
    group_col = None
    possible_group_cols = ["location", "location_id", "seq_id", "sequence_id", "site"]
    for col in possible_group_cols:
        if col in df_train_full.columns:
            group_col = col
            break

    if group_col:
        print(f"Performing Group Sampling based on column: {group_col}")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(
            splitter.split(
                df_train_full,
                df_train_full["Category"],
                groups=df_train_full[group_col],
            )
        )
    else:
        print("Performing Stratified Sampling based on 'Category'...")
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(
            splitter.split(df_train_full, df_train_full["Category"])
        )

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # 4. Save Metadata
    print("Saving metadata...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train_meta.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "val_meta.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test_meta.csv"), index=False)

    return group_col


def validate_metadata(group_col=None):
    print("\nValidating metadata...")

    # Load generated metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train_meta.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val_meta.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test_meta.csv"))

    # 1. Summary Statistics
    print("-" * 30)
    print("Summary Statistics:")
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples:   {len(df_val)}")
    print(f"Test samples:  {len(df_test)}")
    print(
        f"Train Class Distribution (Top 5):\n{df_train['Category'].value_counts(normalize=True).head(5)}"
    )
    print(
        f"Val Class Distribution (Top 5):\n{df_val['Category'].value_counts(normalize=True).head(5)}"
    )
    print("-" * 30)

    # 2. Check File Paths
    def check_paths(df, name):
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            # Path in metadata is relative to input
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
            print(f"Sample missing paths from {name}: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files are missing in {name} dataset."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Val")
    check_paths(df_test, "Test")

    # 3. Verify Split Requirements
    if group_col:
        # Check group exclusivity
        train_groups = set(df_train[group_col].unique())
        val_groups = set(df_val[group_col].unique())
        intersection = train_groups.intersection(val_groups)
        if intersection:
            raise AssertionError(
                f"Group leakage detected! {len(intersection)} groups are in both train and val."
            )
        print("Group split verification passed: No overlapping groups.")
    else:
        # Check stratification
        train_dist = df_train["Category"].value_counts(normalize=True)
        val_dist = df_val["Category"].value_counts(normalize=True)

        # Align indexes
        all_cats = set(train_dist.index).union(set(val_dist.index))
        for cat in all_cats:
            t_p = train_dist.get(cat, 0)
            v_p = val_dist.get(cat, 0)
            # Allow some tolerance for small classes or small validation sets
            # Using a loose tolerance because rare classes might have high variance
            if abs(t_p - v_p) > 0.05:
                print(
                    f"Warning: Class {cat} distribution differs significantly (Train: {t_p:.3f}, Val: {v_p:.3f})"
                )

        print("Stratification verification passed (checked distributions).")


if __name__ == "__main__":
    try:
        group_column_used = generate_metadata()
        validate_metadata(group_column_used)
        print("\nMetadata generation and validation completed successfully.")
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
