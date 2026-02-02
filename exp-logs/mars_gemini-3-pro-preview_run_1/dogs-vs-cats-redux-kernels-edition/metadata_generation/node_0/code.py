import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Scanning input directories...")

    # --- Process Training Data ---
    train_files = glob.glob(os.path.join(TRAIN_DIR, "*.jpg"))
    train_data = []

    print(f"Found {len(train_files)} training files.")

    for filepath in train_files:
        filename = os.path.basename(filepath)
        # Filename format: label.id.jpg (e.g., cat.0.jpg)
        # We need to handle potential variations, but standard is cat.x.jpg
        parts = filename.split(".")
        label_str = parts[0]

        if label_str == "dog":
            label = 1
        elif label_str == "cat":
            label = 0
        else:
            # Skip or handle unexpected files
            continue

        # Store relative path
        rel_path = os.path.join("train", filename)
        train_data.append({"filepath": rel_path, "filename": filename, "label": label})

    df_full_train = pd.DataFrame(train_data)

    # --- Process Test Data ---
    test_files = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    test_data = []

    print(f"Found {len(test_files)} test files.")

    for filepath in test_files:
        filename = os.path.basename(filepath)
        # Filename format: id.jpg (e.g., 1.jpg)
        try:
            file_id = int(filename.split(".")[0])
        except ValueError:
            continue

        rel_path = os.path.join("test", filename)
        test_data.append({"filepath": rel_path, "filename": filename, "id": file_id})

    df_test = pd.DataFrame(test_data)
    # Sort test data by ID for consistency
    if not df_test.empty:
        df_test = df_test.sort_values("id").reset_index(drop=True)

    # --- Split Training Data ---
    print("Splitting data into train and validation sets...")

    if df_full_train.empty:
        raise ValueError("No training data found.")

    # Stratified split
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.2,
        stratify=df_full_train["label"],
        random_state=42,
        shuffle=True,
    )

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # --- Save Metadata ---
    print("Saving metadata files...")
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print(f"Saved {train_csv_path}")
    print(f"Saved {val_csv_path}")
    print(f"Saved {test_csv_path}")

    # --- Verification ---
    print("\n--- Verifying Datasets ---")

    # 1. Load datasets
    df_train_loaded = pd.read_csv(train_csv_path)
    df_val_loaded = pd.read_csv(val_csv_path)
    df_test_loaded = pd.read_csv(test_csv_path)

    # 2. Print Summary Statistics
    print(f"Train Set: {len(df_train_loaded)} samples")
    print(
        f"Train Label Distribution:\n{df_train_loaded['label'].value_counts(normalize=True)}"
    )

    print(f"Validation Set: {len(df_val_loaded)} samples")
    print(
        f"Validation Label Distribution:\n{df_val_loaded['label'].value_counts(normalize=True)}"
    )

    print(f"Test Set: {len(df_test_loaded)} samples")

    # 3. Check File Paths
    def check_paths(df, name):
        if df.empty:
            return

        sample_size = min(1000, len(df))
        sample_paths = df["filepath"].sample(n=sample_size, random_state=42).tolist()

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
            f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Sample missing files:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    check_paths(df_train_loaded, "Train")
    check_paths(df_val_loaded, "Validation")
    check_paths(df_test_loaded, "Test")

    # 4. Verify Stratification
    train_mean = df_train_loaded["label"].mean()
    val_mean = df_val_loaded["label"].mean()
    diff = abs(train_mean - val_mean)

    print(f"Train Label Mean: {train_mean:.4f}")
    print(f"Val Label Mean: {val_mean:.4f}")
    print(f"Difference: {diff:.4f}")

    # Allow a small margin of error for stratification (e.g., due to rounding in small datasets,
    # though here datasets are large)
    if diff > 0.01:
        raise AssertionError(
            f"Stratification failed. Label distribution difference {diff} is too high."
        )

    print("Verification passed successfully.")


if __name__ == "__main__":
    generate_metadata()
