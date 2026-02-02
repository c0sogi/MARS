import os
import cv2
import numpy as np
import pandas as pd
import hashlib
from library.config import GEOMETRIC_FEATURES, FLOAT_PRECISION, WORKING_DIR


def extract_geometry(image_path):
    """
    Extracts robust geometric descriptors from a binary leaf image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the calculated geometric features.
              All values are strictly cast to FLOAT_PRECISION (float64).
    """
    # Initialize default values (zeros)
    features = {k: FLOAT_PRECISION(0.0) for k in GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Threshold to ensure binary (0 or 255)
    # Use THRESH_BINARY_INV because leaves are black (0) on white (255) background.
    # Inversion makes leaves white (255), which is what findContours expects as foreground.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Basic Moments and Dimensions
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area == 0:
        return features

    features["Area"] = FLOAT_PRECISION(area)
    features["Perimeter"] = FLOAT_PRECISION(perimeter)

    # 2. Equivalent Diameter
    # sqrt(4 * Area / pi)
    features["Equivalent_Diameter"] = FLOAT_PRECISION(np.sqrt(4 * area / np.pi))

    # 3. Convex Hull & Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["Solidity"] = FLOAT_PRECISION(area / hull_area)
    else:
        features["Solidity"] = FLOAT_PRECISION(0.0)

    # 4. Bounding Rectangle (AABB) & Extent & Aspect Ratio
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        features["Extent"] = FLOAT_PRECISION(area / rect_area)
    else:
        features["Extent"] = FLOAT_PRECISION(0.0)

    # Aspect Ratio (Bounded: min_dim / max_dim) to strictly satisfy [0, 1]
    if w > 0 and h > 0:
        features["Aspect_Ratio"] = FLOAT_PRECISION(min(w, h) / max(w, h))
    else:
        features["Aspect_Ratio"] = FLOAT_PRECISION(0.0)

    # 5. Roundness (Circularity)
    # 4 * pi * Area / (Perimeter^2)
    if perimeter > 0:
        features["Roundness"] = FLOAT_PRECISION((4 * np.pi * area) / (perimeter**2))
    else:
        features["Roundness"] = FLOAT_PRECISION(0.0)

    # 6. Ellipse Fitting (Major/Minor Axis, Eccentricity)
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            # MA and ma are lengths of the axes (diameters)
            # fitEllipse returns (MA, ma) but order depends on angle.
            # Usually sorted as (minor, major) or we sort them explicitly.
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)

            features["Major_Axis_Length"] = FLOAT_PRECISION(major_axis)
            features["Minor_Axis_Length"] = FLOAT_PRECISION(minor_axis)

            # Eccentricity: sqrt(1 - (minor/major)^2)
            if major_axis > 0:
                ratio_sq = (minor_axis / major_axis) ** 2
                # Clamp to avoid numerical errors making it negative
                features["Eccentricity"] = FLOAT_PRECISION(
                    np.sqrt(max(0, 1 - ratio_sq))
                )
            else:
                features["Eccentricity"] = FLOAT_PRECISION(0.0)
        except Exception:
            # Fallback if fitEllipse fails numerically
            features["Major_Axis_Length"] = FLOAT_PRECISION(0.0)
            features["Minor_Axis_Length"] = FLOAT_PRECISION(0.0)
            features["Eccentricity"] = FLOAT_PRECISION(0.0)
    else:
        # Fallback for small contours
        features["Major_Axis_Length"] = FLOAT_PRECISION(0.0)
        features["Minor_Axis_Length"] = FLOAT_PRECISION(0.0)
        features["Eccentricity"] = FLOAT_PRECISION(0.0)

    return features


def batch_extract(image_paths, load_cached_data=True):
    """
    Extracts geometry features for a list of images with caching support.

    Args:
        image_paths (list): List of file paths to process.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        pd.DataFrame: DataFrame containing the geometric features.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Generate a unique cache key based on the list of paths
    # We sort the paths to ensure the hash is deterministic regardless of input order
    # provided the set of files is the same.
    path_str = "".join(sorted(image_paths))
    path_hash = hashlib.md5(path_str.encode("utf-8")).hexdigest()
    cache_path = os.path.join(WORKING_DIR, f"geo_features_{path_hash}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Verify columns match expected features
            if all(col in df.columns for col in GEOMETRIC_FEATURES):
                # Ensure correct precision
                return df.astype(FLOAT_PRECISION)
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute from scratch
    data_list = []
    for path in image_paths:
        features = extract_geometry(path)
        data_list.append(features)

    df = pd.DataFrame(data_list)

    # Ensure column order and precision
    df = df[GEOMETRIC_FEATURES].astype(FLOAT_PRECISION)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df
