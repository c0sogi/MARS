import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config

# Set seed for reproducibility
np.random.seed(Config.SEED)


def extract_geometric_features(image_path):
    """
    Extracts 7 geometric features from a single image file.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        list: A list of 5 float values corresponding to Config.GEOMETRIC_FEATURES.
              Returns a list of 0.0s if the image cannot be processed.
    """
    # Default zero-vector for failures
    default_feats = [0.0] * 5

    if not os.path.exists(image_path):
        return default_feats

    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return default_feats

    # Polarity Correction: Leaf=White(255), Background=Black(0)
    # The dataset description says "binary black leaves against white backgrounds"
    # We invert so the leaf is the foreground for contour detection.
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find Contours (Lossless)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return default_feats

    # Get largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area <= 0:
        return default_feats

    # 1. Absolute Scale: Area
    feat_area = float(area)

    # 2. Elongation: Eccentricity (via Ellipse Fit)
    # e = sqrt(1 - (min_axis/max_axis)^2)
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # Sort axes to ensure min/max
            axes = sorted([MA, ma])
            min_axis, max_axis = axes[0], axes[1]
            if max_axis > 0:
                feat_eccentricity = np.sqrt(1 - (min_axis / max_axis) ** 2)
            else:
                feat_eccentricity = 0.0
        except:
            feat_eccentricity = 0.0
    else:
        feat_eccentricity = 0.0

    # 3. Roughness: Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    feat_solidity = area / hull_area if hull_area > 0 else 0.0

    # 4. Rectangularity: Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    feat_extent = area / rect_area if rect_area > 0 else 0.0

    # 5. Orientation: Aspect Ratio
    feat_aspect_ratio = float(w) / h if h > 0 else 0.0

    return [
        feat_area,
        feat_eccentricity,
        feat_solidity,
        feat_extent,
        feat_aspect_ratio,
    ]


def process_dataset(metadata_df, input_dir, cache_path=None, load_cached_data=True):
    """
    Extracts geometric features for all images in the provided metadata DataFrame.
    Implements caching to Parquet to avoid re-computation.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path' and 'id'.
        input_dir (str): Base directory where images are located.
        cache_path (str, optional): Path to save/load the parquet cache.
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        pd.DataFrame: DataFrame containing the extracted features.
    """
    # 1. Try Loading Cache
    if load_cached_data and cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. Process Data
    features = []
    file_paths = metadata_df["file_path"].values

    for rel_path in file_paths:
        full_path = os.path.join(input_dir, rel_path)
        feats = extract_geometric_features(full_path)
        features.append(feats)

    # Create DataFrame
    feat_df = pd.DataFrame(features, columns=Config.GEOMETRIC_FEATURES)

    # Include ID for alignment safety if present in metadata
    if "id" in metadata_df.columns:
        feat_df.insert(0, "id", metadata_df["id"])

    # 3. Save Cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feat_df.to_parquet(cache_path, index=False)

    return feat_df
