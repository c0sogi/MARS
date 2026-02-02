import os
import json
import pandas as pd
import numpy as np
from library import config


def load_megadetector_data(
    json_path=config.MEGADETECTOR_FILE,
    cache_path=config.BBOX_CACHE_PATH,
    load_cached_data=True,
):
    """
    Parses the MegaDetector JSON file to extract the highest confidence bounding box
    for each image. Implements caching to Parquet to speed up subsequent runs.

    Args:
        json_path (str): Path to the raw MegaDetector JSON file.
        cache_path (str): Path to save/load the processed DataFrame (Parquet format).
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing ['image_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'conf'].
                      bbox values are normalized [0, 1].
                      If no detection is found, bbox columns are NaN and conf is 0.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading MegaDetector data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data.")

    # 2. Process from scratch
    print(f"Loading MegaDetector JSON from: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MegaDetector file not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    images = data.get("images", [])

    print(f"Processing {len(images)} images from MegaDetector results...")

    for img in images:
        img_id = img["id"]
        detections = img.get("detections", [])

        # Filter for animal category '1'
        # The schema indicates category is a string "1"
        animal_detections = [d for d in detections if d.get("category") == "1"]

        if animal_detections:
            # Sort by confidence descending
            animal_detections.sort(key=lambda x: x.get("conf", 0), reverse=True)
            best_det = animal_detections[0]

            bbox = best_det.get("bbox", [0, 0, 0, 0])  # [x, y, w, h] normalized
            conf = best_det.get("conf", 0.0)

            rows.append(
                {
                    "image_id": img_id,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "conf": conf,
                }
            )
        else:
            # No animal detection found
            rows.append(
                {
                    "image_id": img_id,
                    "bbox_x": np.nan,
                    "bbox_y": np.nan,
                    "bbox_w": np.nan,
                    "bbox_h": np.nan,
                    "conf": 0.0,
                }
            )

    df = pd.DataFrame(rows)

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print(f"Saving processed MegaDetector data to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def get_context_square_crop(bbox, img_w, img_h, margin=config.CROP_MARGIN):
    """
    Calculates the coordinates for a square crop centered on the bounding box,
    expanded by a margin to include context.

    Args:
        bbox (list or np.array): Normalized bounding box [x, y, w, h].
        img_w (int): Width of the original image in pixels.
        img_h (int): Height of the original image in pixels.
        margin (float): Fraction to expand the max dimension by (e.g., 0.2 for 20%).

    Returns:
        tuple: (x_min, y_min, x_max, y_max) in absolute pixel coordinates.
               Note: Coordinates can be negative or larger than image dimensions
               (requires padding in downstream processing).
    """
    if bbox is None or np.isnan(bbox).any():
        # Fallback to center crop of the whole image if no bbox
        # Or return the whole image coordinates
        # Here we return the full image coordinates as a square centered on image
        cx, cy = img_w / 2, img_h / 2
        side = max(img_w, img_h)
        x_min = int(cx - side / 2)
        y_min = int(cy - side / 2)
        x_max = int(cx + side / 2)
        y_max = int(cy + side / 2)
        return x_min, y_min, x_max, y_max

    # Unpack normalized bbox
    norm_x, norm_y, norm_w, norm_h = bbox

    # Convert to absolute pixels
    x = norm_x * img_w
    y = norm_y * img_h
    w = norm_w * img_w
    h = norm_h * img_h

    # Calculate center of the bbox
    cx = x + w / 2
    cy = y + h / 2

    # Determine the side length of the square
    # Take the largest dimension of the bbox and expand by margin
    long_side = max(w, h)
    side_length = long_side * (1 + margin)

    # Calculate new coordinates
    half_side = side_length / 2
    x_min = int(cx - half_side)
    y_min = int(cy - half_side)
    x_max = int(cx + half_side)
    y_max = int(cy + half_side)

    return x_min, y_min, x_max, y_max
