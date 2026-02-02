import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CURATED_CSV = "train_curated.csv"
TRAIN_NOISY_CSV = "train_noisy.csv"
SAMPLE_SUBMISSION_CSV = "sample_submission.csv"
RANDOM_STATE = 42

# Known data issues
CORRUPTED_FILES = ["1d44b0bd.wav"]
MISLABELED_FILES = [
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
]


def main():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading datasets...")
    # Load raw CSVs
    curated_df = pd.read_csv(os.path.join(INPUT_DIR, TRAIN_CURATED_CSV))
    noisy_df = pd.read_csv(os.path.join(INPUT_DIR, TRAIN_NOISY_CSV))

    # Preprocessing: Clean data
    # Remove corrupted file (contains no signal)
    curated_df = curated_df[~curated_df["fname"].isin(CORRUPTED_FILES)].copy()
    # Remove mislabeled files to prevent training on noise
    curated_df = curated_df[~curated_df["fname"].isin(MISLABELED_FILES)].copy()

    # Add relative filepath column
    curated_df["filepath"] = curated_df["fname"].apply(
        lambda x: os.path.join("train_curated", x)
    )
    noisy_df["filepath"] = noisy_df["fname"].apply(
        lambda x: os.path.join("train_noisy", x)
    )

    # Combine datasets for splitting
    # We combine them to ensure the validation set represents the full training distribution
    full_train_df = pd.concat([curated_df, noisy_df], ignore_index=True)

    # Shuffle the dataframe to ensure randomness before splitting
    # (iterative_train_test_split is deterministic based on input order)
    full_train_df = full_train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    print("Preparing for stratified split (this may take a moment)...")
    # Prepare labels for stratification
    # Labels are comma-separated strings, convert to list
    full_train_df["label_list"] = full_train_df["labels"].apply(lambda x: x.split(","))

    # Create Multi-Hot Encoding for labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(full_train_df["label_list"])

    # Prepare X (metadata) for splitting
    # iterative_train_test_split expects numpy arrays
    X = full_train_df[["fname", "filepath", "labels"]].values

    # Perform iterative stratified split
    # test_size = 0.2 (20% validation)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X, y, test_size=0.2)

    # Convert results back to DataFrames
    train_metadata = pd.DataFrame(X_train, columns=["fname", "filepath", "labels"])
    val_metadata = pd.DataFrame(X_val, columns=["fname", "filepath", "labels"])

    print(
        f"Split complete. Train samples: {len(train_metadata)}, Validation samples: {len(val_metadata)}"
    )

    # Save Training and Validation Metadata
    train_metadata.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_metadata.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    # Process Test Set
    sample_sub = pd.read_csv(os.path.join(INPUT_DIR, SAMPLE_SUBMISSION_CSV))
    test_metadata = sample_sub[["fname"]].copy()
    test_metadata["filepath"] = test_metadata["fname"].apply(
        lambda x: os.path.join("test", x)
    )
    test_metadata.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")

    # --- Verification Section ---
    print("\n--- Verifying Datasets ---")

    # 1. Summary Statistics
    for name, df in [
        ("Train", train_metadata),
        ("Val", val_metadata),
        ("Test", test_metadata),
    ]:
        print(f"Dataset: {name}")
        print(f"  Shape: {df.shape}")
        if "labels" in df.columns:
            # Count unique labels to ensure coverage
            all_labels = [
                lbl
                for sublist in df["labels"].apply(lambda x: x.split(","))
                for lbl in sublist
            ]
            print(f"  Unique Labels: {len(set(all_labels))}")
        print("-" * 20)

    # 2. File Path Verification
    def verify_paths(df, dataset_name):
        # Check up to 1000 random paths
        n_check = min(1000, len(df))
        paths_to_check = (
            df["filepath"].sample(n=n_check, random_state=RANDOM_STATE).values
        )

        missing_count = 0
        missing_examples = []

        for p in paths_to_check:
            # Paths in metadata are relative to input dir
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        ratio = missing_count / n_check
        print(f"Missing file ratio for {dataset_name}: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Sample missing files: {missing_examples}")
            raise FileNotFoundError(
                f"Verification Failed: >50% files missing in {dataset_name} metadata."
            )

    print("Checking file existence...")
    verify_paths(train_metadata, "Train")
    verify_paths(val_metadata, "Val")
    verify_paths(test_metadata, "Test")

    # 3. Stratification Verification
    print("Verifying Stratification...")
    # Re-transform labels to binary matrix for calculation
    y_train_check = mlb.transform(
        train_metadata["labels"].apply(lambda x: x.split(","))
    )
    y_val_check = mlb.transform(val_metadata["labels"].apply(lambda x: x.split(",")))

    # Calculate mean occurrence of each class (distribution)
    train_dist = y_train_check.mean(axis=0)
    val_dist = y_val_check.mean(axis=0)

    # Check correlation between class probabilities in Train and Val
    correlation = np.corrcoef(train_dist, val_dist)[0, 1]
    print(f"Class distribution correlation (Train vs Val): {correlation:.4f}")

    # Assert that the split is reasonably stratified
    if correlation < 0.9:
        raise AssertionError(
            f"Stratification verification failed. Correlation {correlation:.4f} is too low, indicating mismatched distributions."
        )

    print("All checks passed successfully.")


if __name__ == "__main__":
    main()
