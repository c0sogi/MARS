import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMAGE_DERIVED_FEATURES,
    FLOAT_PRECISION,
    SEED,
)

# Set random seed for reproducibility where applicable
np.random.seed(SEED)


def get_geometry(mask, contour):
    """
    Computes standard geometric shape descriptors from the contour.
    Optimized for robust, integral-based ratios (Cite solution_lesson_node_00140).

    Args:
        mask (numpy.ndarray): Binary mask.
        contour (numpy.ndarray): The largest contour found in the mask.

    Returns:
        dict: Dictionary containing solidity, extent, aspect_ratio, eccentricity, roundness, equivalent_diameter.
    """
    # Basic moments
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Convex Hull & Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Bounding Rectangle & Extent
    x, y, w, h = cv2.boundingRect(contour)
    extent = area / (w * h) if w * h > 0 else 0.0
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Ellipse Fitting for Eccentricity
    major_axis = 0.0
    minor_axis = 0.0
    eccentricity = 0.0

    # fitEllipse requires at least 5 points
    if len(contour) >= 5:
        try:
            (center, (MA, ma), angle) = cv2.fitEllipse(contour)
            # MA and ma are the lengths of the major and minor axes (diameters)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)

            if major_axis > 0:
                # Eccentricity formula: e = sqrt(1 - (b^2 / a^2))
                eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
        except Exception:
            # Fallback if ellipse fitting fails due to collinear points etc.
            pass

    # Roundness (Cite solution_lesson_node_00151)
    # 4 * pi * Area / Perimeter^2
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

    # Equivalent Diameter (Cite solution_lesson_node_00118)
    # sqrt(4 * Area / pi)
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area >= 0 else 0.0

    return {
        "solidity": float(solidity),
        "extent": float(extent),
        "aspect_ratio": float(aspect_ratio),
        "eccentricity": float(eccentricity),
        "roundness": float(roundness),
        "equivalent_diameter": float(equivalent_diameter),
    }


def extract_image_features(image_path):
    """
    Orchestrates feature extraction for a single image file.

    Args:
        image_path (str): Relative path to the image (e.g., 'images/1.jpg').

    Returns:
        dict: Combined dictionary of geometric features.
    """
    full_path = os.path.join(INPUT_DIR, image_path)

    # Default features in case of failure
    default_features = {k: 0.0 for k in IMAGE_DERIVED_FEATURES}

    if not os.path.exists(full_path):
        return default_features

    # Read image as grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return default_features

    # Polarity Correction:
    # Dataset description: "binary black leaves against white backgrounds"
    # Leaf = 0 (Black), Background = 255 (White).
    # We invert this so Leaf = 255 (Foreground) for correct contour/distance processing.
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    # RETR_EXTERNAL: Only outer contours
    # CHAIN_APPROX_NONE: Store all contour points (lossless)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return default_features

    # Assume the largest contour by area corresponds to the leaf
    c = max(contours, key=cv2.contourArea)

    # Extract features
    geom_feats = get_geometry(mask, c)

    return geom_feats


def process_dataset(metadata_df, dataset_name, load_cached_data=True):
    """
    Processes a dataset defined by a metadata DataFrame, extracting features for all images.
    Implements caching using Parquet.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and all extracted image features.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_image_features.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached image features for '{dataset_name}' from {cache_path}")
        try:
            df_features = pd.read_parquet(cache_path)
            return df_features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting image features for '{dataset_name}'...")

    features_list = []
    ids = metadata_df["id"].values
    paths = metadata_df["file_path"].values

    # Iterate through all images
    for i, path in enumerate(paths):
        feats = extract_image_features(path)
        feats["id"] = ids[i]  # Preserve ID for joining
        features_list.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(features_list)

    # Ensure all expected columns exist and are float64
    for col in IMAGE_DERIVED_FEATURES:
        if col not in df_features.columns:
            df_features[col] = 0.0
        df_features[col] = df_features[col].astype(FLOAT_PRECISION)

    # Ensure 'id' is present and correct type
    df_features["id"] = df_features["id"].astype(int)

    # Reorder columns: id first, then features alphabetically or as defined
    cols = ["id"] + IMAGE_DERIVED_FEATURES
    df_features = df_features[cols]

    # 3. Save to cache
    print(f"Saving image features for '{dataset_name}' to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
