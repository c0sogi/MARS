import os
import json
import random
import numpy as np
import torch
import pandas as pd
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device configured in config.py.
    """
    return torch.device(config.DEVICE)


def ensure_directory(path):
    """
    Ensures that the directory for the given path exists.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_json(path):
    """
    Loads a JSON file from the specified path.
    """
    with open(path, "r") as f:
        return json.load(f)


def get_megadetector_boxes(
    json_path=config.MEGADETECTOR_PATH,
    cache_path=config.BBOX_CACHE_PATH,
    load_cached_data=True,
):
    """
    Parses MegaDetector JSON to extract the highest confidence bounding box for 'animal' (category '1').
    Returns a pandas DataFrame with columns: ['image_id', 'x', 'y', 'w', 'h', 'conf'].

    Implements caching logic:
    1. If load_cached_data is True and cache exists, load from Parquet.
    2. Otherwise, parse JSON, create DataFrame, save to Parquet, and return.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails (e.g., corrupt file), proceed to recompute
            pass

    # 2. Process from scratch
    # Ensure the directory for the cache file exists
    ensure_directory(os.path.dirname(cache_path))

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MegaDetector file not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    # Iterate through images in the JSON
    # The JSON structure is {"images": [{"id": "...", "detections": [...]}, ...]}
    for img in data.get("images", []):
        img_id = img["id"]
        detections = img.get("detections", [])

        # Filter for animal category (category "1")
        animal_detections = [d for d in detections if d.get("category") == "1"]

        if not animal_detections:
            continue

        # Find max confidence detection
        # Detection format: {"category": "1", "bbox": [x, y, w, h], "conf": 0.99}
        best_det = max(animal_detections, key=lambda x: x.get("conf", 0))
        bbox = best_det.get(
            "bbox", [0, 0, 1, 1]
        )  # Default to full image if bbox missing

        rows.append(
            {
                "image_id": img_id,
                "x": bbox[0],
                "y": bbox[1],
                "w": bbox[2],
                "h": bbox[3],
                "conf": best_det.get("conf", 0),
            }
        )

    df = pd.DataFrame(rows)

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df
