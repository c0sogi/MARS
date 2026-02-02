import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


def generate_metadata():
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- 1. Identify Training Data ---
    cover_dir = os.path.join(INPUT_DIR, "Cover")
    # List all files in Cover to get unique Image IDs
    # Sorting ensures reproducibility
    try:
        cover_files = sorted(
            [f for f in os.listdir(cover_dir) if f.lower().endswith(".jpg")]
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {cover_dir}")

    print(f"Found {len(cover_files)} unique cover images.")

    # Construct the full training dataset (before split)
    # We assume for every Cover image, there exists a corresponding image in JMiPOD, JUNIWARD, UERD

    # We will build lists first for efficiency
    ids = []
    file_paths = []
    labels = []
    algos = []

    # Define the structure: Algo Name -> (Folder Name, Label)
    # Task is binary classification: Cover=0, Stego=1
    sources = [
        ("Cover", "Cover", 0),
        ("JMiPOD", "JMiPOD", 1),
        ("JUNIWARD", "JUNIWARD", 1),
        ("UERD", "UERD", 1),
    ]

    # Iterate through each unique image ID and add all 4 variants
    for img_id in cover_files:
        for algo_name, folder, label in sources:
            ids.append(img_id)
            file_paths.append(os.path.join(folder, img_id))
            labels.append(label)
            algos.append(algo_name)

    full_df = pd.DataFrame(
        {"image_id": ids, "file_path": file_paths, "label": labels, "algo": algos}
    )

    print(f"Total training samples generated: {len(full_df)}")

    # --- 2. Split Training/Validation ---
    # We must split by 'image_id' to prevent data leakage (same content in train and val)
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=TRAIN_RATIO, random_state=RANDOM_STATE
    )

    # The split method returns indices
    train_idx, val_idx = next(splitter.split(full_df, groups=full_df["image_id"]))

    train_df = full_df.iloc[train_idx].copy()
    val_df = full_df.iloc[val_idx].copy()

    # --- 3. Identify Test Data ---
    test_dir = os.path.join(INPUT_DIR, "Test")
    try:
        test_files = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
        )
    except FileNotFoundError:
        # Fallback if Test dir doesn't exist (though it should)
        print(f"Warning: {test_dir} not found. Creating empty test metadata.")
        test_files = []

    test_df = pd.DataFrame(
        {
            "image_id": test_files,
            "file_path": [os.path.join("Test", f) for f in test_files],
            # Test set usually doesn't have labels, but we can add a placeholder if needed.
            # We will omit 'label' and 'algo' for test to strictly follow "unknown" nature,
            # or we can add them as NaNs/0s. Let's keep it minimal.
        }
    )

    # --- 4. Save Metadata ---
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")
    return train_path, val_path, test_path


def check_file_existence(df, name):
    """Checks a random sample of file paths to ensure they exist."""
    if df.empty:
        print(f"Dataset {name} is empty. Skipping file check.")
        return

    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        # Paths in metadata are relative to ./input
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(row["file_path"])

    ratio = missing_count / sample_size
    print(f"[{name}] Missing file ratio: {ratio:.4f} ({missing_count}/{sample_size})")

    if ratio > 0.5:
        print(f"Sample missing files from {name}: {missing_samples}")
        raise FileNotFoundError(
            f"Error: More than 50% of sampled files are missing in {name} dataset."
        )


def validate_metadata():
    print("\n--- Validating Metadata ---")

    # Load datasets
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print(f"Train Shape: {train_df.shape}")
    print(f"Val Shape:   {val_df.shape}")
    print(f"Test Shape:  {test_df.shape}")

    print("\nTrain Class Distribution:")
    print(train_df["label"].value_counts())
    print("\nTrain Algo Distribution:")
    print(train_df["algo"].value_counts())

    print("\nVal Class Distribution:")
    print(val_df["label"].value_counts())

    # 2. File Existence Checks
    check_file_existence(train_df, "Train")
    check_file_existence(val_df, "Validation")
    check_file_existence(test_df, "Test")

    # 3. Validation Logic Verification
    # Check for Data Leakage (Intersection of Image IDs)
    train_ids = set(train_df["image_id"].unique())
    val_ids = set(val_df["image_id"].unique())

    intersection = train_ids.intersection(val_ids)
    if intersection:
        raise AssertionError(
            f"Data Leakage Detected! {len(intersection)} image IDs found in both Train and Validation sets."
        )
    else:
        print("Leakage Check: Passed (No overlap in Image IDs).")

    # Check Split Ratio
    total_ids = len(train_ids) + len(val_ids)
    if total_ids > 0:
        actual_train_ratio = len(train_ids) / total_ids
        print(f"Actual Train Split Ratio (by ID): {actual_train_ratio:.4f}")

        # Allow small floating point tolerance
        if not (0.79 <= actual_train_ratio <= 0.81):
            raise AssertionError(
                f"Split ratio mismatch! Expected ~0.8, got {actual_train_ratio:.4f}"
            )

    # Check Stratification/Group Balance
    # Since we grouped by ID and every ID has 4 variants (1 Cover, 3 Stego),
    # the label distribution should be exactly 25% 0 and 75% 1 in both sets.
    train_label_mean = train_df["label"].mean()
    val_label_mean = val_df["label"].mean()

    print(f"Train Label Mean: {train_label_mean:.4f} (Expected 0.75)")
    print(f"Val Label Mean:   {val_label_mean:.4f} (Expected 0.75)")

    if not np.isclose(train_label_mean, 0.75, atol=0.01):
        raise AssertionError(
            "Train label distribution is not preserved (Expected 75% Stego)."
        )
    if not np.isclose(val_label_mean, 0.75, atol=0.01):
        raise AssertionError(
            "Validation label distribution is not preserved (Expected 75% Stego)."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
