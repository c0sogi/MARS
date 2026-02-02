import os
import json
import pandas as pd
import numpy as np
from library.config import Config


def load_megadetector_data(json_path=Config.MEGADETECTOR_JSON, load_cached_data=True):
    """
    Parses the MegaDetector JSON file to extract the best detection for each image.
    Uses Parquet for caching to avoid re-parsing the large JSON file.

    Args:
        json_path (str): Path to the MegaDetector results JSON.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping image_id (str) to a dict containing:
              - 'bbox': list of [x, y, w, h] (normalized) or None
              - 'conf': float (confidence score)
    """
    cache_file = os.path.join(Config.CACHE_DIR, "megadetector_boxes.parquet")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    df = None

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
        except Exception:
            # If cache load fails, fall back to processing
            df = None

    # 2. Process from scratch if needed
    if df is None:
        with open(json_path, "r") as f:
            data = json.load(f)

        processed_rows = []

        # The JSON contains a list of images, each with a list of detections
        for img in data.get("images", []):
            img_id = img.get("id")
            detections = img.get("detections", [])

            best_bbox = None
            best_conf = 0.0

            # Find the detection with the highest confidence
            if detections:
                # Sort by confidence descending
                detections.sort(key=lambda x: x.get("conf", 0), reverse=True)
                top_det = detections[0]
                best_bbox = top_det.get("bbox")  # [x, y, w, h]
                best_conf = top_det.get("conf", 0.0)

            if best_bbox:
                processed_rows.append(
                    {
                        "id": img_id,
                        "x": best_bbox[0],
                        "y": best_bbox[1],
                        "w": best_bbox[2],
                        "h": best_bbox[3],
                        "conf": best_conf,
                        "has_detection": True,
                    }
                )
            else:
                # No detection found for this image
                processed_rows.append(
                    {
                        "id": img_id,
                        "x": 0.0,
                        "y": 0.0,
                        "w": 1.0,
                        "h": 1.0,
                        "conf": 0.0,
                        "has_detection": False,
                    }
                )

        df = pd.DataFrame(processed_rows)
        # Save to cache using Parquet (no pickle allowed)
        df.to_parquet(cache_file, index=False)

    # 3. Convert to dictionary for fast O(1) access
    # We manually build the dict to ensure the structure is exactly what we need
    # Iterating numpy arrays is generally faster than DataFrame rows
    ids = df["id"].values
    xs = df["x"].values
    ys = df["y"].values
    ws = df["w"].values
    hs = df["h"].values
    confs = df["conf"].values
    has_dets = df["has_detection"].values

    result_dict = {}
    for i in range(len(ids)):
        img_id = ids[i]
        if has_dets[i]:
            result_dict[img_id] = {
                "bbox": [xs[i], ys[i], ws[i], hs[i]],
                "conf": confs[i],
            }
        else:
            result_dict[img_id] = {"bbox": None, "conf": 0.0}

    return result_dict


def get_crop_coordinates(image_width, image_height, detection_info, conf_threshold=0.0):
    """
    Calculates absolute crop coordinates based on detection info and image dimensions.

    Args:
        image_width (int): Width of the original image.
        image_height (int): Height of the original image.
        detection_info (dict or None): Dictionary with keys 'bbox' and 'conf'.
        conf_threshold (float): Minimum confidence required to crop.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) integers.
               Returns full image coordinates if no valid detection.
    """
    # Default to full image
    full_crop = (0, 0, image_width, image_height)

    if detection_info is None:
        return full_crop

    bbox = detection_info.get("bbox")
    conf = detection_info.get("conf", 0.0)

    # If no bbox exists or confidence is too low, use full image
    if bbox is None or conf < conf_threshold:
        return full_crop

    # bbox is [x_norm, y_norm, w_norm, h_norm]
    rel_x, rel_y, rel_w, rel_h = bbox

    # Convert to absolute pixel coordinates
    x_min = int(rel_x * image_width)
    y_min = int(rel_y * image_height)
    box_w = int(rel_w * image_width)
    box_h = int(rel_h * image_height)

    x_max = x_min + box_w
    y_max = y_min + box_h

    # Clip coordinates to image boundaries
    x_min = max(0, min(x_min, image_width))
    y_min = max(0, min(y_min, image_height))
    x_max = max(0, min(x_max, image_width))
    y_max = max(0, min(y_max, image_height))

    # Validate crop dimensions (avoid zero-width/height crops)
    if x_max <= x_min or y_max <= y_min:
        return full_crop

    return (x_min, y_min, x_max, y_max)
