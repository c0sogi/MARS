import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob


def generate_metadata():
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Generating metadata...")

    # ---------------------------------------------------------
    # 1. Process Training and Validation Data
    # ---------------------------------------------------------
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"Could not find {train_labels_path}")

    df_train_full = pd.read_csv(train_labels_path)

    # Construct file paths
    # Structure: train/<first_char>/<id>.npy
    df_train_full["file_path"] = df_train_full["id"].apply(
        lambda x: os.path.join("train", str(x)[0], f"{x}.npy")
    )

    # Stratified Split
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=0.2,
        stratify=df_train_full["target"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save to metadata
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    print(f"Saved train.csv with {len(train_df)} samples.")
    print(f"Saved val.csv with {len(val_df)} samples.")

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Could not find {sample_sub_path}")

    df_test = pd.read_csv(sample_sub_path)

    # Construct file paths
    # Structure: test/<first_char>/<id>.npy
    df_test["file_path"] = df_test["id"].apply(
        lambda x: os.path.join("test", str(x)[0], f"{x}.npy")
    )

    # We don't need the target column from sample_submission for the metadata file usually,
    # but keeping it or dropping it is fine. We'll keep the ID and file_path mainly.
    # The prompt implies predicting target, so test.csv usually just needs ID and path.
    # However, keeping the structure consistent is fine.

    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)
    print(f"Saved test.csv with {len(df_test)} samples.")


def validate_metadata():
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    print("\nValidating metadata...")

    metadata_files = {
        "train": os.path.join(METADATA_DIR, "train.csv"),
        "val": os.path.join(METADATA_DIR, "val.csv"),
        "test": os.path.join(METADATA_DIR, "test.csv"),
    }

    dfs = {}

    for name, path in metadata_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file missing: {path}")

        df = pd.read_csv(path)
        dfs[name] = df

        print(f"\n--- {name.upper()} Set Statistics ---")
        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        if "target" in df.columns:
            print("Target Distribution:")
            print(df["target"].value_counts(normalize=True))
            print(f"Target Mean: {df['target'].mean():.4f}")

        # Check file paths
        print(f"Checking file existence for {name} set...")
        sample_size = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=sample_size, random_state=42).tolist()

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of files are missing in {name} set."
            )

    # Verify Stratification
    print("\nVerifying Stratification...")
    train_mean = dfs["train"]["target"].mean()
    val_mean = dfs["val"]["target"].mean()
    diff = abs(train_mean - val_mean)

    print(f"Train Target Mean: {train_mean:.5f}")
    print(f"Val Target Mean:   {val_mean:.5f}")
    print(f"Difference:        {diff:.5f}")

    # Allow a small tolerance for stratification differences
    if diff > 0.01:
        raise AssertionError(
            "Stratification failed: Target distribution differs significantly between train and val."
        )

    print("Stratification verification passed.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
