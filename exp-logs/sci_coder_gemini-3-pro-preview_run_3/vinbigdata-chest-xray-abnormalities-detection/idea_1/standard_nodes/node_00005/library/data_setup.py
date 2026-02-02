import os
import pandas as pd
import numpy as np
import pydicom
import yaml
import shutil
from concurrent.futures import ProcessPoolExecutor
from library.config import (
    INPUT_DIR,
    IDEA_DIR,
    YOLO_DATASET_DIR,
    YOLO_CLASSES,
    IMG_SIZE,
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
)
from library.dicom_utils import process_dicom_image, save_image


def process_single_image(args):
    """
    Worker function to process a single image and its labels.
    """
    img_id, rel_path, group_records, input_root, images_dir, labels_dir, img_size = args

    dicom_path = os.path.join(input_root, rel_path)

    if not os.path.exists(dicom_path):
        return None

    # 1. Read Original Dimensions
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        orig_h, orig_w = ds.Rows, ds.Columns
    except Exception as e:
        print(f"Error reading header for {dicom_path}: {e}")
        return None

    # 2. Process and Save Image
    try:
        img_array = process_dicom_image(dicom_path, target_size=img_size)
        save_path = os.path.join(images_dir, f"{img_id}.jpg")
        save_image(img_array, save_path)
    except Exception as e:
        print(f"Error processing image {dicom_path}: {e}")
        return None

    # 3. Process Labels
    label_path = os.path.join(labels_dir, f"{img_id}.txt")
    yolo_labels = []

    for row in group_records:
        class_id = int(row["class_id"])
        if class_id == 14:
            continue

        # Inline get_yolo_bbox logic here to avoid import circularity
        xmin = row["x_min"]
        ymin = row["y_min"]
        xmax = row["x_max"]
        ymax = row["y_max"]

        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = xmin + (box_w / 2.0)
        y_center = ymin + (box_h / 2.0)

        x_center /= orig_w
        y_center /= orig_h
        width = box_w / orig_w
        height = box_h / orig_h

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    with open(label_path, "w") as f:
        if yolo_labels:
            f.write("\n".join(yolo_labels))

    split_name = os.path.basename(images_dir)
    return {
        "image_id": img_id,
        "split": split_name,
        "image_path": save_path,
        "label_path": label_path,
    }


def create_data_yaml(output_dir):
    """
    Creates the data.yaml file required for YOLO training.
    """
    # YOLO expects absolute paths or paths relative to the execution directory.
    # We will use absolute paths to be safe.
    train_path = os.path.abspath(os.path.join(output_dir, "images", "train"))
    val_path = os.path.abspath(os.path.join(output_dir, "images", "val"))

    # Create dictionary structure
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": train_path,
        "val": val_path,
        "names": YOLO_CLASSES,  # Dictionary mapping class_id to name
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Created data.yaml at {yaml_path}")


def get_yolo_bbox(row, img_w, img_h):
    """
    Converts bounding box to YOLO format: x_center, y_center, width, height (normalized).
    """
    # Original coordinates
    xmin = row["x_min"]
    ymin = row["y_min"]
    xmax = row["x_max"]
    ymax = row["y_max"]

    # Calculate center and dimensions
    box_w = xmax - xmin
    box_h = ymax - ymin
    x_center = xmin + (box_w / 2.0)
    y_center = ymin + (box_h / 2.0)

    # Normalize
    x_center /= img_w
    y_center /= img_h
    width = box_w / img_w
    height = box_h / img_h

    # Clip to [0, 1] to ensure numerical stability
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def prepare_split(df, split_name, output_dir, input_root):
    """
    Processes a specific split (train/val), saving images and labels.
    """
    images_dir = os.path.join(output_dir, "images", split_name)
    labels_dir = os.path.join(output_dir, "labels", split_name)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    unique_images = df["image_id"].unique()
    print(
        f"Processing {len(unique_images)} images for {split_name} set using parallel processing..."
    )

    # Prepare tasks
    tasks = []
    # Group by image_id to gather all labels for an image
    grouped = df.groupby("image_id")

    for img_id, group in grouped:
        rel_path = group.iloc[0]["file_path"]
        group_records = group.to_dict("records")
        tasks.append(
            (
                img_id,
                rel_path,
                group_records,
                input_root,
                images_dir,
                labels_dir,
                IMG_SIZE,
            )
        )

    processed_records = []

    # Use ProcessPoolExecutor for CPU-bound tasks (DICOM read + Resize)
    # We have 12 vCPUs available.
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_single_image, tasks)

        for res in results:
            if res is not None:
                processed_records.append(res)

    return processed_records


def prepare_yolo_data(
    train_metadata_path=TRAIN_METADATA_PATH,
    val_metadata_path=VAL_METADATA_PATH,
    output_dir=YOLO_DATASET_DIR,
    sample_size=None,
    load_cached_data=True,
):
    """
    Main function to prepare data for YOLOv8.

    Args:
        train_metadata_path (str): Path to training metadata CSV.
        val_metadata_path (str): Path to validation metadata CSV.
        output_dir (str): Root directory for YOLO dataset.
        sample_size (int, optional): If set, limits the number of images per split for debugging.
        load_cached_data (bool): If True, attempts to load from cache before processing.

    Returns:
        pd.DataFrame: DataFrame containing paths to processed images and labels.
    """
    # Ensure working directory for idea exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    cache_file = os.path.join(IDEA_DIR, "processed_data_cache.parquet")
    config_file = os.path.join(IDEA_DIR, "processed_data_config.yaml")

    # Current configuration to validate against cache
    current_config = {"sample_size": sample_size, "seed": SEED}

    # --- Cache Logic ---
    if load_cached_data and os.path.exists(cache_file) and os.path.exists(config_file):
        print(f"Checking cache at {cache_file}...")
        try:
            # Load cached config
            with open(config_file, "r") as f:
                cached_config = yaml.safe_load(f)

            # Validate config (Cite solution_lesson_node_00001)
            if cached_config == current_config:
                print("Cache configuration matches current request.")
                df_cache = pd.read_parquet(cache_file)

                # Basic validation: check if the first image file actually exists
                if not df_cache.empty:
                    first_file = df_cache.iloc[0]["image_path"]
                    if os.path.exists(first_file):
                        print("Cache validation successful. Loading data...")
                        return df_cache
                    else:
                        print("Cached files missing on disk. Regenerating...")
                else:
                    print("Cache is empty. Regenerating...")
            else:
                print(
                    f"Cache configuration mismatch (Cached: {cached_config}, Current: {current_config}). Regenerating..."
                )

        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating...")

    # --- Data Processing ---
    print("Starting data preparation for YOLOv8...")

    # Load Metadata
    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)

    # Apply Sampling if requested
    if sample_size is not None:
        print(f"Debugging: Limiting dataset to {sample_size} images per split.")
        train_ids = df_train["image_id"].unique()[:sample_size]
        val_ids = df_val["image_id"].unique()[:sample_size]
        df_train = df_train[df_train["image_id"].isin(train_ids)]
        df_val = df_val[df_val["image_id"].isin(val_ids)]

    # Clean output directory to ensure no stale files
    if os.path.exists(output_dir):
        # If we are not using cache, we should probably clean up to avoid mixing data
        # But removing it might be dangerous if we have partial runs.
        # For safety in this environment, we'll overwrite files but keep the dir.
        pass
    else:
        os.makedirs(output_dir)

    all_records = []

    # Process Train
    train_records = prepare_split(df_train, "train", output_dir, INPUT_DIR)
    all_records.extend(train_records)

    # Process Val
    val_records = prepare_split(df_val, "val", output_dir, INPUT_DIR)
    all_records.extend(val_records)

    # Create YAML configuration
    create_data_yaml(output_dir)

    # Compile results
    result_df = pd.DataFrame(all_records)

    # Save Cache
    print(f"Saving processed data cache to {cache_file}...")
    result_df.to_parquet(cache_file, index=False)

    # Save Config (Cite solution_lesson_node_00001)
    print(f"Saving data configuration to {config_file}...")
    with open(config_file, "w") as f:
        yaml.dump(current_config, f)

    print("Data preparation complete.")
    return result_df
