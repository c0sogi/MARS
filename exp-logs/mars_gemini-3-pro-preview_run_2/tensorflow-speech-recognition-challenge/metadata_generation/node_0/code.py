import os
import glob
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def generate_metadata():
    # ==========================================
    # 1. Configuration and Setup
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    TRAIN_AUDIO_REL = os.path.join("train", "audio")
    TEST_AUDIO_REL = os.path.join("test", "audio")

    TRAIN_AUDIO_FULL = os.path.join(INPUT_DIR, TRAIN_AUDIO_REL)
    TEST_AUDIO_FULL = os.path.join(INPUT_DIR, TEST_AUDIO_REL)

    TARGET_LABELS = {
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    }
    RANDOM_STATE = 42

    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # ==========================================
    # 2. Process Training Data
    # ==========================================
    train_records = []

    # Check if train dir exists
    if not os.path.exists(TRAIN_AUDIO_FULL):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_AUDIO_FULL}")

    # Walk through the training directory
    # Structure: train/audio/<label>/<file.wav>
    for folder_name in os.listdir(TRAIN_AUDIO_FULL):
        folder_path = os.path.join(TRAIN_AUDIO_FULL, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Determine label
        if folder_name == "_background_noise_":
            label = "silence"
            is_background = True
        elif folder_name in TARGET_LABELS:
            label = folder_name
            is_background = False
        else:
            label = "unknown"
            is_background = False

        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            if not filename.endswith(".wav"):
                continue

            file_path_rel = os.path.join(TRAIN_AUDIO_REL, folder_name, filename)

            user_id = None
            if not is_background:
                # Format: <user_id>_nohash_<repetition>.wav
                parts = filename.split("_")
                if len(parts) > 0:
                    user_id = parts[0]

            train_records.append(
                {
                    "file_path": file_path_rel,
                    "fname": filename,
                    "label": label,
                    "user_id": user_id,
                    "is_background": is_background,
                    "original_folder": folder_name,
                }
            )

    df_all_train = pd.DataFrame(train_records)
    print(f"Total training files found: {len(df_all_train)}")

    # ==========================================
    # 3. Split Train/Validation
    # ==========================================
    # Separate background noise (keep in train) and regular commands
    df_background = df_all_train[df_all_train["is_background"]].copy()
    df_commands = df_all_train[~df_all_train["is_background"]].copy()

    # Use GroupShuffleSplit for commands based on user_id
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    train_idx, val_idx = next(
        splitter.split(df_commands, groups=df_commands["user_id"])
    )

    df_train_commands = df_commands.iloc[train_idx].copy()
    df_val_commands = df_commands.iloc[val_idx].copy()

    # Combine background noise into training set
    df_train_final = pd.concat([df_train_commands, df_background], ignore_index=True)
    df_val_final = df_val_commands.copy()

    # Save to CSV
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    df_train_final.to_csv(train_csv_path, index=False)
    df_val_final.to_csv(val_csv_path, index=False)

    print(f"Saved train metadata to {train_csv_path} ({len(df_train_final)} samples)")
    print(f"Saved val metadata to {val_csv_path} ({len(df_val_final)} samples)")

    # ==========================================
    # 4. Process Test Data
    # ==========================================
    test_records = []

    if os.path.exists(TEST_AUDIO_FULL):
        for filename in os.listdir(TEST_AUDIO_FULL):
            if not filename.endswith(".wav"):
                continue

            file_path_rel = os.path.join(TEST_AUDIO_REL, filename)

            test_records.append(
                {
                    "file_path": file_path_rel,
                    "fname": filename,
                    "label": "unknown",  # Placeholder for test
                }
            )

    df_test = pd.DataFrame(test_records)
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test.to_csv(test_csv_path, index=False)
    print(f"Saved test metadata to {test_csv_path} ({len(df_test)} samples)")

    # ==========================================
    # 5. Validation and Checks
    # ==========================================
    print("\nRunning validation checks...")

    # Load datasets
    df_train_check = pd.read_csv(train_csv_path)
    df_val_check = pd.read_csv(val_csv_path)
    df_test_check = pd.read_csv(test_csv_path)

    # Check 1: Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {df_train_check.shape}")
    print(f"Val set shape: {df_val_check.shape}")
    print(f"Test set shape: {df_test_check.shape}")

    print("\nTrain Label Distribution:")
    print(df_train_check["label"].value_counts())
    print("\nVal Label Distribution:")
    print(df_val_check["label"].value_counts())

    train_users = set(df_train_check["user_id"].dropna().unique())
    val_users = set(df_val_check["user_id"].dropna().unique())
    print(f"\nUnique users in Train: {len(train_users)}")
    print(f"Unique users in Val: {len(val_users)}")

    # Check 2: File Path Resolution
    print("\n--- Checking File Paths ---")
    all_dfs = [
        ("train", df_train_check),
        ("val", df_val_check),
        ("test", df_test_check),
    ]

    for name, df in all_dfs:
        if len(df) == 0:
            continue

        # Sample 1000 paths (or all if less than 1000)
        sample_size = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE)

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
                f"Too many missing files in {name} dataset! Ratio: {missing_ratio}"
            )

    # Check 3: Validation Split Logic (No User Leakage)
    print("\n--- Checking Split Logic ---")

    # Intersection of users
    user_overlap = train_users.intersection(val_users)

    if user_overlap:
        print(f"Found {len(user_overlap)} overlapping users.")
        # Raise error as per requirements
        raise AssertionError(
            f"Data leakage detected! {len(user_overlap)} users found in both train and validation sets."
        )
    else:
        print("Success: No user overlap between train and validation sets.")

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    generate_metadata()
