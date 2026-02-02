import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import save_to_cache, load_from_cache, ensure_float64


def _process_single_image(image_path):
    """
    Extracts morphological features from a single binary leaf image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    # Initialize default values (zeros) in case of failure
    features = {k: 0.0 for k in Config.MORPHOLOGICAL_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Threshold: Leaf is black (0), Background is white (255).
    # We want Leaf to be Foreground (255/1) for contour detection.
    # Use THRESH_BINARY_INV to invert: 0 -> 255, 255 -> 0.
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Select the largest contour by area (assuming it's the leaf)
    cnt = max(contours, key=cv2.contourArea)

    # 1. Absolute Scale Features
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    convex_area = cv2.contourArea(hull)
    convex_perimeter = cv2.arcLength(hull, True)

    # Fit Ellipse (requires at least 5 points)
    major_axis = 0.0
    minor_axis = 0.0
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        except:
            pass  # Fallback to 0.0

    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    # 2. Rotated Envelope (Minimum Area Rectangle)
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    # Sort w, h to be rotation invariant (width < height)
    min_box_width = min(w, h)
    min_box_height = max(w, h)
    min_box_area = min_box_width * min_box_height

    # 3. Internal Morphology (Distance Transform)
    # distanceTransform calculates the distance to the nearest zero pixel for each pixel.
    # Since mask has leaf=255 and bg=0, this gives distance to background (thickness).
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    inscribed_circle_radius = dist_transform.max()

    # 4. Relative Shape / Dimensionless Ratios
    # Add epsilon to denominators to avoid division by zero
    eps = 1e-15

    solidity = area / (convex_area + eps)
    extent = area / (min_box_area + eps)
    min_box_aspect_ratio = min_box_width / (min_box_height + eps)
    convexity = convex_perimeter / (perimeter + eps)

    # Populate dictionary
    features["Area"] = area
    features["Perimeter"] = perimeter
    features["Convex_Perimeter"] = convex_perimeter
    features["Major_Axis_Length"] = major_axis
    features["Minor_Axis_Length"] = minor_axis
    features["Equivalent_Diameter"] = equiv_diameter

    features["Min_Box_Width"] = min_box_width
    features["Min_Box_Height"] = min_box_height
    features["Min_Box_Area"] = min_box_area

    features["Inscribed_Circle_Radius"] = inscribed_circle_radius

    features["Solidity"] = solidity
    features["Extent"] = extent
    features["Min_Box_Aspect_Ratio"] = min_box_aspect_ratio
    features["Convexity"] = convexity

    return features


def extract_morphological_features(metadata_df, dataset_name, load_cached_data=True):
    """
    Extracts morphological features for a dataset defined by the metadata DataFrame.
    Handles caching to avoid re-computation.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path' column.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing the extracted features.
    """
    cache_filename = f"morphology_{dataset_name}"

    # 1. Try loading from cache
    if load_cached_data:
        cached_df = load_from_cache(cache_filename, expected_type="dataframe")
        if cached_df is not None:
            print(f"Loaded morphological features for '{dataset_name}' from cache.")
            return ensure_float64(cached_df)

    # 2. Compute from scratch
    print(f"Extracting morphological features for '{dataset_name}'...")

    extracted_rows = []

    # Iterate over metadata
    for _, row in metadata_df.iterrows():
        # Construct full path
        # metadata 'file_path' is relative, e.g., 'images/123.jpg'
        full_path = os.path.join(Config.INPUT_DIR, row[Config.IMAGE_PATH_COL])

        # Extract features
        features = _process_single_image(full_path)

        # Add ID for join safety (optional, but good practice, though we return aligned DF)
        # We will just return the features corresponding to the row order.
        extracted_rows.append(features)

    # Create DataFrame
    feature_df = pd.DataFrame(extracted_rows)

    # Ensure column order matches Config
    feature_df = feature_df[Config.MORPHOLOGICAL_FEATURES]

    # Enforce float64 precision
    feature_df = ensure_float64(feature_df)

    # 3. Save to cache
    save_to_cache(cache_filename, feature_df)
    print(f"Saved morphological features for '{dataset_name}' to cache.")

    return feature_df
