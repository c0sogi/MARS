import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def generate_metadata():
    # Ensure metadata directory exists
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Scanning input directories...")

    # 1. Identify all unique image IDs from the Cover directory
    # We use the Cover directory as the source of truth for image IDs.
    cover_dir = os.path.join(INPUT_DIR, "Cover")
    cover_files = glob.glob(os.path.join(cover_dir, "*.jpg"))

    # Extract just the filenames (e.g., '00001.jpg') to use as IDs
    all_ids = [os.path.basename(f) for f in cover_files]

    # Sort IDs to ensure the random split is deterministic given the seed
    all_ids.sort()

    print(f"Found {len(all_ids)} unique image IDs in training data.")

    # 2. Split IDs into Train and Validation sets (Group Split)
    # We split by ID so that all versions of an image (Cover + 3 Stego) stay together.
    train_ids, val_ids = train_test_split(
        all_ids, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    # 3. Helper function to build DataFrame rows
    def build_dataset(image_ids):
        data = []
        # Define the 4 variants for each image ID
        # Algo Name, Label, Folder Name
        variants = [
            ("Cover", 0, "Cover"),
            ("JMiPOD", 1, "JMiPOD"),
            ("JUNIWARD", 1, "JUNIWARD"),
            ("UERD", 1, "UERD"),
        ]

        for img_id in image_ids:
            for algo, label, folder in variants:
                data.append(
                    {
                        "image_id": img_id,
                        "file_path": f"{folder}/{img_id}",
                        "algo": algo,
                        "label": label,
                    }
                )
        return pd.DataFrame(data)

    print("Generating training metadata...")
    train_df = build_dataset(train_ids)

    print("Generating validation metadata...")
    val_df = build_dataset(val_ids)

    # 4. Generate Test Metadata
    print("Generating test metadata...")
    test_dir = os.path.join(INPUT_DIR, "Test")
    test_files = glob.glob(os.path.join(test_dir, "*.jpg"))
    test_ids = [os.path.basename(f) for f in test_files]
    test_ids.sort()

    test_data = []
    for img_id in test_ids:
        test_data.append(
            {
                "image_id": img_id,
                "file_path": f"Test/{img_id}",
                "algo": "Test",
                "label": 0,  # Placeholder for test set
            }
        )
    test_df = pd.DataFrame(test_data)

    # 5. Save Metadata
    print("Saving metadata to CSV...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # 6. Validation Checks
    print("\n=== Performing Validation Checks ===")

    # 6a. Print Summary Statistics
    datasets = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
    for name, df in datasets:
        print(f"\n{name} Set Statistics:")
        print(f"  Total Samples: {len(df)}")
        print(f"  Unique Images: {df['image_id'].nunique()}")
        print(f"  Label Distribution:\n{df['label'].value_counts().to_string()}")
        if "algo" in df.columns:
            print(f"  Algorithm Distribution:\n{df['algo'].value_counts().to_string()}")

    # 6b. Check File Existence (Random Sample)
    def validate_paths(df, name):
        if df.empty:
            return
        # Sample 1000 paths or all if less than 1000
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Construct full path
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        missing_ratio = missing_count / sample_size
        print(f"\n{name} Set - Missing File Ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Example missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Error: More than 50% of sampled file paths in {name} set are invalid."
            )

    validate_paths(train_df, "Train")
    validate_paths(val_df, "Validation")
    validate_paths(test_df, "Test")

    # 6c. Verify Split Integrity
    print("\nVerifying split integrity...")
    train_img_set = set(train_df["image_id"])
    val_img_set = set(val_df["image_id"])

    # Check for leakage
    intersection = train_img_set.intersection(val_img_set)
    if intersection:
        raise AssertionError(
            f"Data Leakage Detected: {len(intersection)} image IDs found in both Train and Validation sets."
        )

    # Check split ratio (approximate check on IDs)
    total_ids = len(all_ids)
    train_ratio = len(train_ids) / total_ids
    print(f"Split Ratio (by ID): Train={train_ratio:.2f}, Val={1-train_ratio:.2f}")

    # Verify stratification/grouping logic
    # Each ID in train/val should have exactly 4 entries (Cover, JMiPOD, JUNIWARD, UERD)
    train_counts = train_df["image_id"].value_counts()
    if not (train_counts == 4).all():
        raise AssertionError(
            "Structure Error: Not all training IDs have exactly 4 records (Cover + 3 Stego)."
        )

    val_counts = val_df["image_id"].value_counts()
    if not (val_counts == 4).all():
        raise AssertionError(
            "Structure Error: Not all validation IDs have exactly 4 records (Cover + 3 Stego)."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
