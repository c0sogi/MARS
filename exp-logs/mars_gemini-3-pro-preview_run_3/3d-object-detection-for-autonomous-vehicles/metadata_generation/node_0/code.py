import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import random

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def load_json_table(path):
    """Helper to load a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def get_lidar_map(sample_data_json):
    """
    Creates a mapping from sample_token to LIDAR filename.
    Assumes LIDAR files end with .bin.
    """
    mapping = {}
    for item in sample_data_json:
        if item["filename"].endswith(".bin"):
            # We use the basename of the file and assume it resides in the corresponding lidar folder
            # This handles cases where JSON path is 'samples/LIDAR_TOP/file.bin' but data is in 'train_lidar/file.bin'
            mapping[item["sample_token"]] = os.path.basename(item["filename"])
    return mapping


def parse_class_counts(df):
    """
    Parses the label string to count classes for summary statistics.
    Assumes format: center_x center_y center_z width length height yaw class_name
    (8 tokens per box)
    """
    class_counts = {}
    total_objects = 0

    for label_str in df["label"]:
        if pd.isna(label_str) or label_str == "":
            continue

        parts = str(label_str).strip().split()
        # Each object has 8 parameters in train.csv (no confidence score in GT usually)
        # If the format matches the submission example with confidence, it would be 9.
        # Based on description: "annotations in train.csv... center_x ... class_name" (8 items)
        # We will try to detect stride based on length.

        stride = 8
        if len(parts) % 8 != 0:
            # Fallback or check if it's 9 (though description says 8 for train.csv)
            if len(parts) % 9 == 0:
                stride = 9
            else:
                # Irregular format, skip detailed parsing for this row
                continue

        num_objects = len(parts) // stride
        total_objects += num_objects

        for i in range(num_objects):
            # class_name is the last element in the block
            class_name = parts[i * stride + (stride - 1)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return total_objects, class_counts


def generate_metadata():
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Process Training Data
    # ---------------------------------------------------------
    print("Processing Train Data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    # Load relationships
    train_sample_json = load_json_table(
        os.path.join(INPUT_DIR, "train_data", "sample.json")
    )
    train_sample_data_json = load_json_table(
        os.path.join(INPUT_DIR, "train_data", "sample_data.json")
    )

    # Map sample_token -> scene_token
    sample_to_scene = {item["token"]: item["scene_token"] for item in train_sample_json}

    # Map sample_token -> lidar_filename
    sample_to_lidar = get_lidar_map(train_sample_data_json)

    # Build Metadata List
    data = []
    for _, row in train_df.iterrows():
        s_token = row["Id"]
        label = row["PredictionString"]

        # Skip if no lidar data found (should not happen for valid samples)
        if s_token not in sample_to_lidar:
            continue

        lidar_file = sample_to_lidar[s_token]
        scene_token = sample_to_scene.get(s_token, "unknown")

        # Construct path relative to input directory
        lidar_path = os.path.join("train_lidar", lidar_file)

        data.append(
            {
                "sample_token": s_token,
                "lidar_path": lidar_path,
                "label": label,
                "scene_token": scene_token,
            }
        )

    full_train_df = pd.DataFrame(data)

    # Stratified Group Split
    # We group by scene_token to avoid leakage
    print(f"Splitting {len(full_train_df)} samples into Train/Val (80:20) by Scene...")
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    # We need to handle cases where scene_token might be missing (unlikely), filtering them out or treating as separate groups
    valid_group_df = full_train_df[full_train_df["scene_token"] != "unknown"]

    train_idx, val_idx = next(
        gss.split(valid_group_df, groups=valid_group_df["scene_token"])
    )

    train_metadata = valid_group_df.iloc[train_idx].copy()
    val_metadata = valid_group_df.iloc[val_idx].copy()

    # Save to CSV
    train_metadata.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_metadata.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)

    # ---------------------------------------------------------
    # Process Test Data
    # ---------------------------------------------------------
    print("Processing Test Data...")
    test_csv_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    test_df = pd.read_csv(test_csv_path)

    test_sample_data_json = load_json_table(
        os.path.join(INPUT_DIR, "test_data", "sample_data.json")
    )
    test_sample_to_lidar = get_lidar_map(test_sample_data_json)

    test_data = []
    for _, row in test_df.iterrows():
        s_token = row["Id"]
        if s_token in test_sample_to_lidar:
            lidar_file = test_sample_to_lidar[s_token]
            lidar_path = os.path.join("test_lidar", lidar_file)
            test_data.append({"sample_token": s_token, "lidar_path": lidar_path})

    test_metadata = pd.DataFrame(test_data)
    test_metadata.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    return train_metadata, val_metadata, test_metadata


def validate_metadata(train_df, val_df, test_df):
    print("\n" + "=" * 30)
    print("PERFORMING VALIDATION CHECKS")
    print("=" * 30)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Samples: {len(train_df)}")
    print(f"Val Samples:   {len(val_df)}")
    print(f"Test Samples:  {len(test_df)}")

    train_obj, train_classes = parse_class_counts(train_df)
    val_obj, val_classes = parse_class_counts(val_df)

    print(f"Train Objects: {train_obj}")
    print(f"Val Objects:   {val_obj}")
    print("Train Class Distribution (Top 5):", dict(list(train_classes.items())[:5]))

    # 2. Path Validation
    print("\n--- Checking File Paths (Random Sample 1000) ---")

    def check_paths(df, name):
        if df.empty:
            return
        sample_n = min(1000, len(df))
        samples = df.sample(n=sample_n, random_state=RANDOM_STATE)
        missing = 0
        missing_examples = []

        for _, row in samples.iterrows():
            # Path in metadata is relative to INPUT_DIR
            full_path = os.path.join(INPUT_DIR, row["lidar_path"])
            if not os.path.exists(full_path):
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["lidar_path"])

        ratio = missing / sample_n
        print(f"[{name}] Missing Ratio: {ratio:.4f} ({missing}/{sample_n})")

        if missing_examples:
            print(f"[{name}] Example missing: {missing_examples}")

        if ratio > 0.5:
            raise FileNotFoundError(
                f"CRITICAL: >50% of files missing in {name} dataset!"
            )

    check_paths(train_df, "Train")
    check_paths(val_df, "Val")
    check_paths(test_df, "Test")

    # 3. Split Validation
    print("\n--- Verifying Split Integrity ---")
    train_scenes = set(train_df["scene_token"].unique())
    val_scenes = set(val_df["scene_token"].unique())

    intersection = train_scenes.intersection(val_scenes)
    print(f"Unique Scenes Train: {len(train_scenes)}")
    print(f"Unique Scenes Val:   {len(val_scenes)}")
    print(f"Scene Overlap:       {len(intersection)}")

    if len(intersection) > 0:
        raise AssertionError(
            f"DATA LEAKAGE: {len(intersection)} scenes found in both Train and Val sets!"
        )

    split_ratio = len(val_df) / (len(train_df) + len(val_df))
    print(f"Actual Validation Split Ratio: {split_ratio:.4f}")

    # Assert ratio is within reasonable bounds (allow some variance due to scene length differences)
    if not (0.15 <= split_ratio <= 0.25):
        print(
            "WARNING: Split ratio deviates from 0.2. This is expected if scene lengths vary significantly."
        )

    print("\nValidation Complete. Metadata generated successfully.")


if __name__ == "__main__":
    train_meta, val_meta, test_meta = generate_metadata()
    validate_metadata(train_meta, val_meta, test_meta)
