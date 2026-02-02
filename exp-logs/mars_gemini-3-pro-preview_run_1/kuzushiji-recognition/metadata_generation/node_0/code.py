import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit
import ast


def generate_metadata():
    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
    RANDOM_STATE = 42

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(SAMPLE_SUB_CSV)

    # Handle missing labels (images with no characters)
    train_df["labels"] = train_df["labels"].fillna("")

    # --- Process Training Data ---
    # Construct relative file paths
    # Assuming images are .jpg based on dataset info
    train_df["file_path"] = "train_images/" + train_df["image_id"] + ".jpg"

    # Extract Group ID for splitting
    # IDs look like '100241706_00004_2' or 'umgy007-028'
    # We split by '_' or '-' and take the first part as the book/document ID
    def extract_group(img_id):
        return re.split(r"[_-]", img_id)[0]

    train_df["group_id"] = train_df["image_id"].apply(extract_group)

    # Perform Group Split (80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(train_df, groups=train_df["group_id"]))

    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    print(f"Split complete. Train: {len(train_split)}, Val: {len(val_split)}")

    # Save Train/Val Metadata
    train_split.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)

    # --- Process Test Data ---
    test_df["file_path"] = "test_images/" + test_df["image_id"] + ".jpg"
    # Test data doesn't have meaningful labels for stats, but we keep the column structure
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    print("Metadata generation complete.")

    # --- Verification & Stats ---
    validate_and_summarize(INPUT_DIR, METADATA_DIR)


def validate_and_summarize(input_dir, metadata_dir):
    print("\nStarting Validation and Summary...")

    # Load generated metadata
    train_meta = pd.read_csv(os.path.join(metadata_dir, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(metadata_dir, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(metadata_dir, "test_metadata.csv"))

    # 1. Verify Split Requirements
    # Check Ratio (approximate due to group split)
    total_train_val = len(train_meta) + len(val_meta)
    val_ratio = len(val_meta) / total_train_val
    print(f"Validation Ratio: {val_ratio:.4f} (Target ~0.2)")

    # Check Group Leakage
    train_groups = set(train_meta["group_id"])
    val_groups = set(val_meta["group_id"])
    intersection = train_groups.intersection(val_groups)

    if intersection:
        raise AssertionError(
            f"Group leakage detected! Groups {intersection} are in both train and val."
        )
    print("Group split verification passed: No leakage detected.")

    # 2. Check File Paths
    def check_paths(df, name):
        paths = df["file_path"].values
        # Sample 1000 or all if less
        n_sample = min(1000, len(paths))
        sampled_paths = np.random.choice(paths, n_sample, replace=False)

        missing_count = 0
        missing_samples = []

        for p in sampled_paths:
            full_path = os.path.join(input_dir, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / n_sample
        print(
            f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_sample})"
        )

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(f"Too many missing files in {name} dataset!")

    check_paths(train_meta, "Train")
    check_paths(val_meta, "Val")
    check_paths(test_meta, "Test")

    # 3. Summary Statistics
    def get_stats(df, name, is_labeled=True):
        print(f"\n--- {name} Dataset Statistics ---")
        print(f"Total Images: {len(df)}")
        print(
            f"Unique Groups: {df['group_id'].nunique() if 'group_id' in df.columns else 'N/A'}"
        )

        if is_labeled:
            # Parse labels to get class distribution
            # Format: Unicode X Y W H ...
            all_labels = []
            total_annotations = 0

            for label_str in df["labels"].dropna():
                if not isinstance(label_str, str) or not label_str.strip():
                    continue
                parts = label_str.strip().split(" ")
                # Unicode char is every 5th element: 0, 5, 10...
                chars = parts[0::5]
                all_labels.extend(chars)
                total_annotations += len(chars)

            print(f"Total Annotations: {total_annotations}")
            if total_annotations > 0:
                unique_labels = pd.Series(all_labels).value_counts()
                print(f"Unique Character Classes: {len(unique_labels)}")
                print("Top 5 Classes:")
                print(unique_labels.head(5))
            else:
                print("No annotations found (or empty labels).")
        else:
            print("Labels: Placeholder/Hidden (Test Set)")

    get_stats(train_meta, "Training")
    get_stats(val_meta, "Validation")
    get_stats(test_meta, "Test", is_labeled=False)


if __name__ == "__main__":
    generate_metadata()
