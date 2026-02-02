import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def load_json_table(json_path):
    """Loads a JSON file containing a list of records into a DataFrame."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def get_file_paths_map(data_dir, split_name):
    """
    Creates a mapping from sample_token to a dictionary of sensor paths.
    Resolves paths based on the provided directory structure (flattened images/lidar folders).
    """
    sample_data_path = os.path.join(data_dir, "sample_data.json")
    calib_path = os.path.join(data_dir, "calibrated_sensor.json")

    df_sd = load_json_table(sample_data_path)
    df_cs = load_json_table(calib_path)

    # Map calibrated_sensor_token to channel name (e.g., LIDAR_TOP, CAM_FRONT)
    # Fallback to sensor_token if channel is missing, though channel is standard in NuScenes
    channel_col = "channel" if "channel" in df_cs.columns else "token"
    sensor_map = dict(zip(df_cs["token"], df_cs[channel_col]))

    # Function to resolve relative path based on filename extension
    def resolve_path(filename):
        basename = os.path.basename(filename)
        ext = os.path.splitext(basename)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            subdir = f"{split_name}_images"
        elif ext in [".bin", ".pcd"]:
            subdir = f"{split_name}_lidar"
        else:
            subdir = f"{split_name}_maps"
        return os.path.join(subdir, basename)

    # Apply mappings
    # We use a dictionary comprehension approach for speed and clarity
    df_sd["channel"] = df_sd["calibrated_sensor_token"].map(sensor_map)
    df_sd["resolved_path"] = df_sd["filename"].apply(resolve_path)

    # Construct the map: sample_token -> {channel: path}
    paths_map = {}
    for st, ch, rp in zip(
        df_sd["sample_token"], df_sd["channel"], df_sd["resolved_path"]
    ):
        if st not in paths_map:
            paths_map[st] = {}
        paths_map[st][ch] = rp

    return paths_map


def parse_annotation_string(ann_str):
    """
    Parses space-delimited annotation string into a list of dicts.
    Expected format chunks of 8: center_x center_y center_z width length height yaw class_name
    """
    if pd.isna(ann_str) or str(ann_str).strip() == "":
        return []

    parts = str(ann_str).split()
    num_fields = 8

    # If the string is not a multiple of 8, it might be malformed or empty
    if len(parts) % num_fields != 0:
        return []

    objects = []
    for i in range(0, len(parts), num_fields):
        try:
            obj = {
                "center_x": float(parts[i]),
                "center_y": float(parts[i + 1]),
                "center_z": float(parts[i + 2]),
                "width": float(parts[i + 3]),
                "length": float(parts[i + 4]),
                "height": float(parts[i + 5]),
                "yaw": float(parts[i + 6]),
                "class_name": parts[i + 7],
            }
            objects.append(obj)
        except ValueError:
            continue
    return objects


def process_dataset():
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ==========================================
    # 1. Process Training Data
    # ==========================================
    print("Processing Training Data...")
    train_data_dir = os.path.join(INPUT_DIR, "train_data")

    # Load Sample Info (Master List of Samples)
    # This ensures we have all samples, even those without annotations
    df_samples = load_json_table(os.path.join(train_data_dir, "sample.json"))

    # Ensure scene_token exists for Group Split
    if "scene_token" not in df_samples.columns:
        # Fallback: Treat each sample as its own group (Random Split)
        print("Warning: 'scene_token' not found. Using 'token' for grouping.")
        df_samples["scene_token"] = df_samples["token"]

    # Load File Paths
    train_paths_map = get_file_paths_map(train_data_dir, "train")
    df_samples["file_paths"] = df_samples["token"].map(train_paths_map)

    # Load Annotations from train.csv
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    df_train_csv = pd.read_csv(train_csv_path)

    # Identify ID and Annotation columns
    cols = df_train_csv.columns
    # Heuristic to find ID column
    id_col = (
        "Id"
        if "Id" in cols
        else ("sample_token" if "sample_token" in cols else cols[0])
    )
    # Heuristic to find Annotation column (usually the other one)
    ann_col = "PredictionString" if "PredictionString" in cols else None
    if ann_col is None:
        remaining = [c for c in cols if c != id_col]
        if remaining:
            ann_col = remaining[0]

    # Create Annotation Map: sample_token -> JSON string of objects
    ann_map = {}
    if ann_col:
        for _, row in df_train_csv.iterrows():
            sid = row[id_col]
            ann_str = row[ann_col]
            parsed_objs = parse_annotation_string(ann_str)
            ann_map[sid] = json.dumps(parsed_objs)

    df_samples["annotations"] = df_samples["token"].map(ann_map).fillna("[]")

    # Perform Group Shuffle Split (80:20)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    df_samples = df_samples.reset_index(drop=True)

    try:
        train_idx, val_idx = next(
            splitter.split(df_samples, groups=df_samples["scene_token"])
        )
    except ValueError:
        # Fallback if groups are insufficient
        print("Group split failed, falling back to random split.")
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            df_samples.index, test_size=0.2, random_state=RANDOM_STATE
        )

    df_train = df_samples.iloc[train_idx].copy()
    df_val = df_samples.iloc[val_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"

    # Serialize file_paths to JSON string for CSV storage
    df_train["file_paths"] = df_train["file_paths"].apply(
        lambda x: json.dumps(x) if isinstance(x, dict) else "{}"
    )
    df_val["file_paths"] = df_val["file_paths"].apply(
        lambda x: json.dumps(x) if isinstance(x, dict) else "{}"
    )

    # Save Train/Val Metadata
    out_cols = ["token", "scene_token", "file_paths", "annotations", "split"]
    # Filter columns if they exist
    out_cols = [c for c in out_cols if c in df_train.columns]

    df_train[out_cols].to_csv(
        os.path.join(METADATA_DIR, "train_metadata.csv"), index=False
    )
    df_val[out_cols].to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)

    # ==========================================
    # 2. Process Test Data
    # ==========================================
    print("Processing Test Data...")
    test_data_dir = os.path.join(INPUT_DIR, "test_data")

    # Load Submission IDs
    sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    df_sub = pd.read_csv(sub_path)
    test_ids = df_sub["Id"].unique()

    # Load Test Paths
    test_paths_map = get_file_paths_map(test_data_dir, "test")

    df_test = pd.DataFrame({"token": test_ids})
    df_test["file_paths"] = (
        df_test["token"]
        .map(test_paths_map)
        .apply(lambda x: json.dumps(x) if isinstance(x, dict) else "{}")
    )
    df_test["split"] = "test"
    df_test["annotations"] = "[]"

    df_test.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    return df_train, df_val, df_test


def verify_metadata(df_train, df_val, df_test):
    print("\n=== Verifying Metadata ===")

    # 1. Summary Statistics
    print(f"Train Samples: {len(df_train)}")
    print(f"Val Samples:   {len(df_val)}")
    print(f"Test Samples:  {len(df_test)}")

    total_tv = len(df_train) + len(df_val)
    if total_tv > 0:
        val_ratio = len(df_val) / total_tv
        print(f"Validation Ratio: {val_ratio:.4f}")
        if not (0.15 <= val_ratio <= 0.25):
            raise AssertionError(
                f"Validation split ratio {val_ratio:.2f} is outside required range (0.15-0.25)."
            )

    # 2. Check for Data Leakage (Scene Overlap)
    if "scene_token" in df_train.columns:
        train_scenes = set(df_train["scene_token"])
        val_scenes = set(df_val["scene_token"])
        overlap = train_scenes.intersection(val_scenes)
        print(f"Scene Overlap Count: {len(overlap)}")
        if len(overlap) > 0:
            raise AssertionError(
                f"Data leakage detected! {len(overlap)} scenes appear in both train and val."
            )

    # 3. Check File Paths
    def check_files(df, name):
        print(f"Checking file paths for {name}...")
        n_samples = min(1000, len(df))
        sample_df = df.sample(n_samples, random_state=RANDOM_STATE)

        missing_count = 0
        total_checked = 0
        missing_examples = []

        for _, row in sample_df.iterrows():
            # Parse JSON if it's a string (it should be based on previous steps)
            paths = row["file_paths"]
            if isinstance(paths, str):
                paths = json.loads(paths)

            if not paths:
                # If a sample has no paths, it counts as missing data
                missing_count += 1
                total_checked += 1
                continue

            for _, rel_path in paths.items():
                full_path = os.path.join(INPUT_DIR, rel_path)
                total_checked += 1
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_examples) < 3:
                        missing_examples.append(full_path)

        if total_checked == 0:
            return

        ratio = missing_count / total_checked
        print(f"  Missing File Ratio: {ratio:.4f} ({missing_count}/{total_checked})")
        if len(missing_examples) > 0:
            print(f"  Example missing file: {missing_examples[0]}")

        if ratio > 0.5:
            raise FileNotFoundError(
                f"Missing file ratio {ratio:.2f} exceeds threshold 0.5 for {name} dataset."
            )

    check_files(df_train, "Train")
    check_files(df_val, "Val")
    check_files(df_test, "Test")

    print("Verification Passed Successfully.")


if __name__ == "__main__":
    df_train, df_val, df_test = process_dataset()
    verify_metadata(df_train, df_val, df_test)
