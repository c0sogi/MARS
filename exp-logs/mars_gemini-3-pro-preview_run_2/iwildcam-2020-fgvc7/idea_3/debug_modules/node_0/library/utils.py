import os
import json
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets seeds for random, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_megadetector_data(json_path=None, cache_dir=None, load_cached_data=True):
    """
    Parses the MegaDetector JSON file to extract the best bounding box for each image.
    Caches the result as a Parquet file for faster subsequent loading.

    Args:
        json_path (str): Path to the MegaDetector JSON file.
        cache_dir (str): Directory to store the cached Parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping image_id (str) to bbox [x, y, w, h] (list of floats).
    """
    if json_path is None:
        json_path = Config.MEGADETECTOR_FILE
    if cache_dir is None:
        cache_dir = Config.WORKING_DIR

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "megadetector_boxes.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            # Convert DataFrame back to dictionary
            # Assuming columns: id, x, y, w, h
            box_dict = dict(zip(df["id"], df[["x", "y", "w", "h"]].values.tolist()))
            return box_dict
        except Exception:
            # If loading fails (e.g. corrupt file), proceed to recompute
            pass

    # 2. Compute from scratch
    with open(json_path, "r") as f:
        data = json.load(f)

    records = []
    images = data.get("images", [])

    for img in images:
        img_id = img["id"]
        detections = img.get("detections", [])

        # Default to full image [x, y, w, h] if no animal detected
        best_bbox = [0.0, 0.0, 1.0, 1.0]
        max_conf = -1.0

        for det in detections:
            # Category "1" represents an animal
            if det.get("category") == "1":
                conf = det.get("conf", 0.0)
                if conf > max_conf:
                    max_conf = conf
                    best_bbox = det.get("bbox", [0.0, 0.0, 1.0, 1.0])

        records.append(
            {
                "id": img_id,
                "x": best_bbox[0],
                "y": best_bbox[1],
                "w": best_bbox[2],
                "h": best_bbox[3],
            }
        )

    df = pd.DataFrame(records)

    # 3. Save to cache
    df.to_parquet(cache_file, index=False)

    # Return dictionary
    box_dict = dict(zip(df["id"], df[["x", "y", "w", "h"]].values.tolist()))
    return box_dict
