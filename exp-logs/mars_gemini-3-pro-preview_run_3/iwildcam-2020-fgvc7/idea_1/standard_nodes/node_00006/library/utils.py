import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_detector_bboxes(json_path=Config.MEGADETECTOR_PATH, load_cached_data=True):
    """
    Parses the MegaDetector results JSON to extract the bounding box with the highest
    confidence for each image. Caches the result to a parquet file to speed up subsequent runs.

    Args:
        json_path (str): Path to the MegaDetector JSON file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary mapping image_id (str) to bbox [x, y, w, h].
    """
    cache_path = os.path.join(Config.CACHE_DIR, "bboxes.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure image_id is string
            df["image_id"] = df["image_id"].astype(str)

            # Convert DataFrame back to dictionary for O(1) access
            bbox_dict = {}
            for row in df.itertuples(index=False):
                bbox_dict[row.image_id] = [row.x, row.y, row.w, row.h]
            return bbox_dict
        except Exception as e:
            print(f"Error loading cache: {e}. Reprocessing data.")
            # Fall through to processing logic if cache load fails

    # 2. Process from scratch
    with open(json_path, "r") as f:
        data = json.load(f)

    extracted_data = []
    images = data.get("images", [])

    for img in images:
        img_id = str(img["id"])
        detections = img.get("detections", [])

        if not detections:
            continue

        # Find the detection with the highest confidence
        best_detection = max(detections, key=lambda x: x["conf"])
        bbox = best_detection["bbox"]  # Format: [x, y, w, h] (normalized 0-1)

        extracted_data.append(
            {"image_id": img_id, "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
        )

    # Create DataFrame
    df = pd.DataFrame(extracted_data)

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    # 4. Return dictionary
    bbox_dict = {}
    for row in df.itertuples(index=False):
        bbox_dict[row.image_id] = [row.x, row.y, row.w, row.h]

    return bbox_dict
