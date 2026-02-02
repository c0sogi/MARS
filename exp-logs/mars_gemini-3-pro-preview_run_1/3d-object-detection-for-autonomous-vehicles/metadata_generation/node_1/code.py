import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import ast

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def load_json_table(path):
    """Helper to load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_file_paths(data_dir, sample_tokens, subset_name):
    """
    Parses sample_data.json to get file paths for given sample_tokens.
    Returns a DataFrame with sample_token, lidar_path, image_paths.
    """
    sample_data_path = os.path.join(data_dir, "sample_data.json")
    if not os.path.exists(sample_data_path):
        raise FileNotFoundError(f"Could not find {sample_data_path}")

    sample_data = load_json_table(sample_data_path)

    # Convert to dataframe for easier filtering
    sd_df = pd.DataFrame(sample_data)

    # Filter for relevant samples
    sd_df = sd_df[sd_df["sample_token"].isin(sample_tokens)]

    def construct_path(filename):
        # filename in json might be 'images/host-...' or just 'host-...'
        # We need to map it to 'input/{subset}_images/...' or 'input/{subset}_lidar/...'
        base_name = os.path.basename(filename)
        ext = os.path.splitext(base_name)[1].lower()

        if ext in [".jpeg", ".jpg", ".png"]:
            folder = f"{subset_name}_images"
        elif ext in [".bin", ".pcd"]:
            folder = f"{subset_name}_lidar"
        else:
            folder = f"{subset_name}_other"

        return os.path.join(INPUT_DIR, folder, base_name)

    sd_df["file_path"] = sd_df["filename"].apply(construct_path)

    # Separate Lidar and Images
    # We assume .bin is Lidar and image extensions are Camera
    lidar_mask = sd_df["filename"].str.lower().str.endswith((".bin", ".pcd"))
    image_mask = sd_df["filename"].str.lower().str.endswith((".jpeg", ".jpg", ".png"))

    lidar_df = sd_df[lidar_mask]
    image_df = sd_df[image_mask]

    # For Lidar, we take the first one found for the sample (typically LIDAR_TOP)
    lidar_paths = (
        lidar_df.groupby("sample_token")["file_path"]
        .first()
        .reset_index()
        .rename(columns={"file_path": "lidar_path"})
    )

    # For Images, we aggregate all available camera images into a list
    image_paths = (
        image_df.groupby("sample_token")["file_path"]
        .apply(list)
        .reset_index()
        .rename(columns={"file_path": "image_paths"})
    )

    # Merge
    paths_df = pd.DataFrame({"sample_token": list(sample_tokens)})
    paths_df = paths_df.merge(lidar_paths, on="sample_token", how="left")
    paths_df = paths_df.merge(image_paths, on="sample_token", how="left")

    return paths_df


def process_train_data():
    print("Processing train data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    # Ensure 'Id' column exists (it corresponds to sample_token)
    if "Id" not in train_df.columns and "sample_token" in train_df.columns:
        train_df.rename(columns={"sample_token": "Id"}, inplace=True)

    # 1. Aggregate annotations
    # Fix: Parse PredictionString instead of expecting exploded columns
    def parse_annotations(pred_str):
        if pd.isna(pred_str):
            return json.dumps([])

        tokens = str(pred_str).split()
        if not tokens:
            return json.dumps([])

        # Determine step size (8 or 9)
        # Format: center_x center_y center_z width length height yaw class_name (8)
        # Or with confidence: conf x y z w l h yaw class (9)

        step = 8
        if len(tokens) % 9 == 0 and len(tokens) % 8 != 0:
            step = 9
        elif len(tokens) % 8 == 0 and len(tokens) % 9 != 0:
            step = 8
        elif len(tokens) % 8 == 0 and len(tokens) % 9 == 0:
            # Ambiguity check: check if token[7] is a number (yaw) or string (class)
            try:
                float(tokens[7])
                step = 9
            except ValueError:
                step = 8

        annotations = []
        for i in range(0, len(tokens), step):
            try:
                if step == 9:
                    # conf = tokens[i]
                    center_x = tokens[i + 1]
                    center_y = tokens[i + 2]
                    center_z = tokens[i + 3]
                    width = tokens[i + 4]
                    length = tokens[i + 5]
                    height = tokens[i + 6]
                    yaw = tokens[i + 7]
                    class_name = tokens[i + 8]
                else:
                    center_x = tokens[i]
                    center_y = tokens[i + 1]
                    center_z = tokens[i + 2]
                    width = tokens[i + 3]
                    length = tokens[i + 4]
                    height = tokens[i + 5]
                    yaw = tokens[i + 6]
                    class_name = tokens[i + 7]

                ann = {
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "center_z": float(center_z),
                    "width": float(width),
                    "length": float(length),
                    "height": float(height),
                    "yaw": float(yaw),
                    "class_name": class_name,
                }
                annotations.append(ann)
            except (ValueError, IndexError):
                continue

        return json.dumps(annotations)

    if "PredictionString" in train_df.columns:
        train_df["annotations"] = train_df["PredictionString"].apply(parse_annotations)
        annotations_df = train_df[["Id", "annotations"]]
    else:
        # Fallback if column name is different
        print(f"Columns in train.csv: {train_df.columns.tolist()}")
        raise KeyError("PredictionString column not found in train.csv")

    unique_ids = annotations_df["Id"].unique()

    # 2. Get Scene Tokens for splitting (Group Sampling)
    sample_json_path = os.path.join(INPUT_DIR, "train_data", "sample.json")
    sample_json = load_json_table(sample_json_path)
    sample_scene_map = {s["token"]: s["scene_token"] for s in sample_json}

    # Create a DataFrame for samples
    samples_df = pd.DataFrame({"sample_token": unique_ids})
    samples_df["scene_token"] = samples_df["sample_token"].map(sample_scene_map)

    # Merge annotations
    samples_df = samples_df.merge(
        annotations_df, left_on="sample_token", right_on="Id", how="left"
    ).drop(columns=["Id"])

    # 3. Get File Paths
    paths_df = get_file_paths(
        os.path.join(INPUT_DIR, "train_data"), unique_ids, "train"
    )
    samples_df = samples_df.merge(paths_df, on="sample_token", how="left")

    # 4. Split Train/Val
    # We use GroupShuffleSplit on scene_token to avoid data leakage
    samples_df = samples_df.dropna(subset=["scene_token"])

    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(samples_df, groups=samples_df["scene_token"]))

    train_set = samples_df.iloc[train_idx].copy()
    val_set = samples_df.iloc[val_idx].copy()

    return train_set, val_set


def process_test_data():
    print("Processing test data...")
    submission_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    sub_df = pd.read_csv(submission_path)

    test_ids = sub_df["Id"].unique()

    # Get File Paths
    paths_df = get_file_paths(os.path.join(INPUT_DIR, "test_data"), test_ids, "test")

    # Map scene tokens if available (useful for consistency, though not strictly needed for test)
    sample_json_path = os.path.join(INPUT_DIR, "test_data", "sample.json")
    if os.path.exists(sample_json_path):
        sample_json = load_json_table(sample_json_path)
        sample_scene_map = {s["token"]: s["scene_token"] for s in sample_json}
        paths_df["scene_token"] = paths_df["sample_token"].map(sample_scene_map)
    else:
        paths_df["scene_token"] = None

    return paths_df


def validate_metadata(train_df, val_df, test_df):
    print("Validating metadata...")

    # 1. Summary Stats
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # 2. Check Scene Overlap
    train_scenes = set(train_df["scene_token"])
    val_scenes = set(val_df["scene_token"])
    overlap = train_scenes.intersection(val_scenes)
    if overlap:
        raise AssertionError(f"Train and Validation sets share scenes: {overlap}")
    print("Verification Passed: No scene overlap between train and val.")

    # 3. Check File Existence
    all_paths = []

    def collect_paths(df):
        paths = []
        if "lidar_path" in df.columns:
            paths.extend(df["lidar_path"].dropna().tolist())
        if "image_paths" in df.columns:
            # image_paths contains lists
            for p_list in df["image_paths"].dropna():
                # If loaded from csv it might be string, but here it is list
                if isinstance(p_list, str):
                    p_list = ast.literal_eval(p_list)
                paths.extend(p_list)
        return paths

    all_paths.extend(collect_paths(train_df))
    all_paths.extend(collect_paths(val_df))
    all_paths.extend(collect_paths(test_df))

    if not all_paths:
        print("Warning: No file paths found to validate.")
        return

    # Sample 1000 paths randomly
    sample_size = min(1000, len(all_paths))
    sampled_paths = np.random.choice(all_paths, sample_size, replace=False)

    missing_count = 0
    missing_samples = []

    for p in sampled_paths:
        if not os.path.exists(p):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / sample_size
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing files:", missing_samples)
        raise FileNotFoundError(f"Too many missing files. Ratio: {missing_ratio}")
    print("Verification Passed: File existence check.")


def main():
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # Process Data
    train_df, val_df = process_train_data()
    test_df = process_test_data()

    # Save Metadata
    # We save annotations and image_paths as strings (JSON or stringified list) automatically by pandas to_csv
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # Validate
    validate_metadata(train_df, val_df, test_df)

    print("Metadata generation complete.")


if __name__ == "__main__":
    main()
