import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import WORKING_DIR, SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_megadetector_data(json_path, load_cached_data=True):
    """
    Parses the MegaDetector JSON file to create a mapping from image_id to the
    bounding box [x, y, w, h] of the highest confidence detection.

    If an image has no detections, the bounding box defaults to [0.0, 0.0, 1.0, 1.0].

    Args:
        json_path (str): Path to the MegaDetector results JSON file.
        load_cached_data (bool): If True, attempts to load from a parquet cache.

    Returns:
        dict: A dictionary mapping image_id (str) to bbox (list of [x, y, w, h]).
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, "megadetector_boxes.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading MegaDetector data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            # Convert DataFrame back to dictionary: image_id -> [x, y, w, h]
            # Using zip is efficient for iteration
            bbox_map = {
                img_id: [x, y, w, h]
                for img_id, x, y, w, h in zip(
                    df["image_id"], df["x"], df["y"], df["w"], df["h"]
                )
            }
            return bbox_map
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from source...")

    # 2. Process from scratch
    print(f"Loading MegaDetector results from {json_path}...")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return {}

    bbox_data = []
    bbox_map = {}

    # Iterate through images in the JSON
    for img in data.get("images", []):
        img_id = img["id"]
        detections = img.get("detections", [])

        # Default to full image [0, 0, 1, 1] if no detections found
        best_bbox = [0.0, 0.0, 1.0, 1.0]

        if detections:
            # Find detection with max confidence
            # Each detection has "category", "bbox", "conf"
            best_det = max(detections, key=lambda x: x["conf"])
            best_bbox = best_det["bbox"]

        bbox_map[img_id] = best_bbox

        # Prepare for DataFrame
        bbox_data.append(
            {
                "image_id": img_id,
                "x": best_bbox[0],
                "y": best_bbox[1],
                "w": best_bbox[2],
                "h": best_bbox[3],
            }
        )

    # 3. Save to cache
    if bbox_data:
        try:
            df = pd.DataFrame(bbox_data)
            df.to_parquet(cache_path, index=False)
            print(f"Saved MegaDetector data to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

    return bbox_map


def calculate_class_weights(train_df, num_classes):
    """
    Computes class weights for CrossEntropyLoss based on inverse class frequency.
    Weight = Total_Samples / (Num_Classes * Class_Count)

    Args:
        train_df (pd.DataFrame): DataFrame containing a 'category_id' column.
        num_classes (int): Total number of classes.

    Returns:
        torch.FloatTensor: Tensor of shape (num_classes,) containing weights.
    """
    # Get counts for each class present in the training data
    class_counts = train_df["category_id"].value_counts()

    # Initialize counts for all classes
    counts = np.zeros(num_classes)

    # Map the counts to the correct indices
    # Filter valid indices to ensure safety against out-of-bounds IDs
    valid_indices = class_counts.index[class_counts.index < num_classes]
    counts[valid_indices] = class_counts[valid_indices].values

    total_samples = len(train_df)

    # Avoid division by zero. If a class has 0 samples, we treat it as having 1
    # for the denominator. These classes generally won't appear in the loss calculation anyway.
    safe_counts = np.maximum(counts, 1)

    # Compute balanced weights
    weights = total_samples / (num_classes * safe_counts)

    return torch.FloatTensor(weights)
