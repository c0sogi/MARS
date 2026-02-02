import os
import cv2
import numpy as np
import pandas as pd
from library import config


def _process_single_image(image_path):
    """
    Extracts the 9 robust-integral geometric features from a single image.

    Features:
    - Area, Perimeter, Major_Axis_Length, Minor_Axis_Length
    - Mean_Thickness (Internal Topology)
    - Solidity, Extent, Eccentricity (Invariant Shape)
    - Roundness (Non-Linear Ratio)

    Returns a dictionary with keys matching config.GEOMETRIC_FEATURES.
    """
    # Initialize default result (zeros)
    result = {k: 0.0 for k in config.GEOMETRIC_FEATURES}

    # Check existence
    if not os.path.exists(image_path):
        return result

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return result

    # Polarity Correction: Leaf becomes white (255), background black (0)
    # The dataset description says "binary black leaves against white backgrounds"
    # So we invert it using THRESH_BINARY_INV.
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Lossless Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return result

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Area (Integral)
    area = cv2.contourArea(cnt)
    if area <= 0:
        return result
    result["Area"] = float(area)

    # 2. Perimeter (Used for Roundness but not stored as a raw feature)
    # Cite solution_lesson_node_00140: Avoid noise-sensitive descriptors like raw perimeter.
    perimeter = cv2.arcLength(cnt, True)

    # 3. Roundness (4 * pi * Area / P^2)
    if perimeter > 0:
        result["Roundness"] = (4 * np.pi * area) / (perimeter**2)

    # 4. Extent (Area / Upright Bounding Rect Area)
    # Cite solution_lesson_node_00120: Extent is part of the robust triad.
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        result["Extent"] = area / rect_area

    # 5. Aspect Ratio (Rotated, Bounded)
    # Cite solution_lesson_node_00142: Prefer bounded descriptors.
    # Cite solution_lesson_node_00119: Prioritize rotation-invariant geometric ratios.
    rect = cv2.minAreaRect(cnt)
    (center), (dim1, dim2), angle = rect
    min_dim, max_dim = sorted([dim1, dim2])
    if max_dim > 0:
        result["Aspect_Ratio"] = min_dim / max_dim

    # 6. Solidity (Area / Convex Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        result["Solidity"] = area / hull_area

    # 7. Eccentricity
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            axes = sorted([MA, ma])
            minor_axis = axes[0]
            major_axis = axes[1]

            if major_axis > 0:
                # Eccentricity = sqrt(1 - (b/a)^2)
                ratio_sq = (minor_axis / major_axis) ** 2
                result["Eccentricity"] = np.sqrt(1 - ratio_sq)
        except Exception:
            pass

    return result


def extract_integral_features(metadata_df, dataset_name, load_cached_data=True):
    """
    Extracts geometric features for all images in the provided metadata DataFrame.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path' or 'id'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing the geometric features, aligned with input index.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(
        config.CACHE_DIR, f"{dataset_name}_geometric_features.parquet"
    )

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached geometric features from {cache_path}...")
        try:
            df_features = pd.read_parquet(cache_path)
            # Validate shape
            if len(df_features) == len(metadata_df):
                return df_features
            else:
                print("Cached file length mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(
        f"Extracting geometric features for {dataset_name} ({len(metadata_df)} images)..."
    )

    features_list = []

    # We iterate over the dataframe.
    # The metadata contains 'file_path' which is relative to INPUT_DIR.
    # We need to construct the full path.
    # The 'file_path' column in metadata/train.csv is like "images/123.jpg".
    # INPUT_DIR is "./input".
    # So full path is "./input/images/123.jpg".

    for idx, row in metadata_df.iterrows():
        # Construct full path
        # Note: metadata 'file_path' already includes 'images/' prefix based on the description
        # but let's be robust. config.INPUT_DIR is "./input".
        # If file_path is "images/1.jpg", join yields "./input/images/1.jpg".
        rel_path = row["file_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Extract features
        feats = _process_single_image(full_path)
        features_list.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(features_list, dtype=config.FLOAT_PRECISION)

    # Ensure column order matches config
    df_features = df_features[config.GEOMETRIC_FEATURES]

    # 3. Save to cache
    print(f"Saving geometric features to {cache_path}...")
    df_features.to_parquet(cache_path, index=False)

    return df_features
