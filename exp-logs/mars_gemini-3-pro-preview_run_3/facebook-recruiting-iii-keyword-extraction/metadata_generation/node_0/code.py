import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
RANDOM_STATE = 42


def generate_metadata():
    """Generates metadata for train, val, and test sets."""
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    print("Loading training data...")
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)

    # Load only necessary columns to optimize memory usage
    # We treat 'Tags' as the label. 'Body' and 'Title' are excluded to avoid data copying.
    df = pd.read_csv(train_path, usecols=["Id", "Tags"])

    # Add file path column (relative to input dir)
    df["file_path"] = TRAIN_FILE

    # Handle missing tags
    df["Tags"] = df["Tags"].fillna("")

    # Stratification Logic
    print("Splitting train/val...")
    # We use the full tag string for stratification (Label Powerset approach)
    y = df["Tags"]
    y_counts = y.value_counts()

    # Identify classes with < 2 samples (cannot be stratified)
    rare_classes = y_counts[y_counts < 2].index
    is_rare = df["Tags"].isin(rare_classes)

    df_common = df[~is_rare]
    df_rare = df[is_rare]

    # Split common data with stratification
    train_common, val_common = train_test_split(
        df_common, test_size=0.2, random_state=RANDOM_STATE, stratify=df_common["Tags"]
    )

    # Split rare data randomly (maintaining 80/20 ratio)
    if not df_rare.empty:
        train_rare, val_rare = train_test_split(
            df_rare, test_size=0.2, random_state=RANDOM_STATE
        )
        train_df = pd.concat([train_common, train_rare])
        val_df = pd.concat([val_common, val_rare])
    else:
        train_df = train_common
        val_df = val_common

    # Shuffle final datasets
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Save metadata
    print(f"Saving metadata to {METADATA_DIR}...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    # --- Process Test Data ---
    print("Loading test data...")
    test_path = os.path.join(INPUT_DIR, TEST_FILE)
    df_test = pd.read_csv(test_path, usecols=["Id"])
    df_test["file_path"] = TEST_FILE

    print("Saving test metadata...")
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)


def validate_metadata():
    """Validates the generated metadata files."""
    print("Validating metadata...")

    # Load generated metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train samples: {len(train_meta)}")
    print(f"Val samples: {len(val_meta)}")
    print(f"Test samples: {len(test_meta)}")
    print(f"Train unique tags: {train_meta['Tags'].nunique()}")
    print(f"Val unique tags: {val_meta['Tags'].nunique()}")

    # 2. Check File Paths
    print("\n--- Checking File Paths ---")
    for name, df in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        # Sample 1000 paths (or all if less than 1000)
        sample_paths = df["file_path"].sample(
            n=min(1000, len(df)), random_state=RANDOM_STATE
        )
        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        ratio = missing_count / len(sample_paths)
        print(f"{name}: Missing file ratio = {ratio:.4f}")

        if ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not exist."
            )

    # 3. Verify Validation Split Requirements
    print("\n--- Verifying Split ---")

    # Check Ratio
    total_train_val = len(train_meta) + len(val_meta)
    val_ratio = len(val_meta) / total_train_val
    print(f"Validation Ratio: {val_ratio:.4f}")

    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation ratio {val_ratio} is not close to 0.2 (Target: 20%)"
        )

    # Check Stratification (Top 10 classes distribution)
    top_tags = train_meta["Tags"].value_counts().head(10).index

    train_dist = train_meta["Tags"].value_counts(normalize=True)
    val_dist = val_meta["Tags"].value_counts(normalize=True)

    print("Top 5 Tags Distribution (Train vs Val):")
    for tag in top_tags[:5]:
        t_freq = train_dist.get(tag, 0)
        v_freq = val_dist.get(tag, 0)
        print(f"Tag: {str(tag)[:30]:<30} | Train: {t_freq:.4f} | Val: {v_freq:.4f}")

        if abs(t_freq - v_freq) > 0.05:
            raise AssertionError(
                f"Stratification check failed for tag '{tag}'. Train: {t_freq}, Val: {v_freq}"
            )

    # Check Disjointness
    train_ids = set(train_meta["Id"])
    val_ids = set(val_meta["Id"])
    intersection = train_ids.intersection(val_ids)
    if intersection:
        raise AssertionError(
            f"Train and Validation sets are not disjoint. Overlap: {len(intersection)}"
        )

    print("\nValidation successful.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
