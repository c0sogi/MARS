import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"Could not find {train_labels_path}")

    df_train_full = pd.read_csv(train_labels_path)

    # Construct file paths: train/{first_char}/{id}.npy
    # We assume the folder structure follows the first character of the ID based on the dataset description
    df_train_full["file_path"] = df_train_full["id"].apply(
        lambda x: os.path.join("train", str(x)[0], f"{x}.npy")
    )

    # Split into Train and Validation
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        stratify=df_train_full["target"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save Train and Val Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    print(f"Saved training metadata to {train_meta_path}")
    print(f"Saved validation metadata to {val_meta_path}")

    # --- Process Test Data ---
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Could not find {sample_sub_path}")

    df_test = pd.read_csv(sample_sub_path)

    # Construct file paths: test/{first_char}/{id}.npy
    df_test["file_path"] = df_test["id"].apply(
        lambda x: os.path.join("test", str(x)[0], f"{x}.npy")
    )

    # We only need id and file_path for test metadata (target in sample_sub is dummy)
    # However, keeping the dummy target or removing it is fine. Let's keep columns consistent with id, file_path.
    # The submission format requires id, target. For metadata, we usually just need id and path.
    # Let's keep 'id' and 'file_path'.
    test_meta_df = df_test[["id", "file_path"]].copy()

    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    test_meta_df.to_csv(test_meta_path, index=False)
    print(f"Saved test metadata to {test_meta_path}")


def verify_metadata():
    print("\nStarting metadata verification...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Print Summary Statistics
    print("-" * 30)
    print("Summary Statistics")
    print("-" * 30)

    print(f"Train set size: {len(train_df)}")
    print(
        f"Train target distribution:\n{train_df['target'].value_counts(normalize=True)}"
    )

    print(f"\nValidation set size: {len(val_df)}")
    print(
        f"Validation target distribution:\n{val_df['target'].value_counts(normalize=True)}"
    )

    print(f"\nTest set size: {len(test_df)}")
    print("-" * 30)

    # 2. Check File Existence (Random Sample)
    def check_files(df, name):
        print(f"Checking file paths for {name}...")
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            rel_path = row["file_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("  Sample missing files:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} metadata."
            )

    check_files(train_df, "Train")
    check_files(val_df, "Validation")
    check_files(test_df, "Test")

    # 3. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check Split Ratio (approximate check is fine, exact check: len(val) / (len(train)+len(val)))
    total_train_val = len(train_df) + len(val_df)
    actual_val_ratio = len(val_df) / total_train_val
    print(f"Actual validation ratio: {actual_val_ratio:.4f}")

    # Allow small floating point deviation
    if not (0.19 < actual_val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} is not close to 0.2"
        )

    # Check Stratification
    train_pos_rate = train_df["target"].mean()
    val_pos_rate = val_df["target"].mean()

    print(f"Train positive rate: {train_pos_rate:.4f}")
    print(f"Val positive rate: {val_pos_rate:.4f}")

    # Check if distributions are reasonably close (within 1% absolute difference)
    if abs(train_pos_rate - val_pos_rate) > 0.01:
        raise AssertionError(
            "Stratification failed: Target distributions in Train and Val differ significantly."
        )

    print("\nVerification passed successfully.")


if __name__ == "__main__":
    try:
        generate_metadata()
        verify_metadata()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
