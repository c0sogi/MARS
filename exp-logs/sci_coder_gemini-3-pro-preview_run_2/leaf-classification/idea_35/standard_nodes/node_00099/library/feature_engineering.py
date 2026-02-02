import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    FLOAT_PRECISION,
    RANDOM_SEED,
)


def extract_hu_moments(contour):
    """
    Calculates the 7 invariant Hu Moments for a given contour.

    Args:
        contour: A numpy array representing the contour points.

    Returns:
        np.ndarray: A 1D array of 7 float64 values.
    """
    try:
        moments = cv2.moments(contour)
        # cv2.HuMoments returns a (7, 1) array, we flatten it
        hu_moments = cv2.HuMoments(moments).flatten()
        return hu_moments.astype(FLOAT_PRECISION)
    except Exception:
        return np.zeros(7, dtype=FLOAT_PRECISION)


def extract_geometric_properties(contour):
    """
    Calculates geometric scalar properties: Aspect Ratio, Solidity, Extent, Eccentricity.

    Args:
        contour: A numpy array representing the contour points.

    Returns:
        np.ndarray: A 1D array of 4 float64 values.
    """
    try:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 1. Aspect Ratio
        aspect_ratio = float(w) / h if h > 0 else 0.0

        # 2. Extent (Ratio of contour area to bounding rect area)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0.0

        # 3. Solidity (Ratio of contour area to convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # 4. Eccentricity
        # fitEllipse requires at least 5 points
        if len(contour) >= 5:
            # fitEllipse returns ((center_x, center_y), (width, height), angle)
            # Note: width and height here refer to the axes lengths of the rotated rect
            (_, _), (d1, d2), _ = cv2.fitEllipse(contour)

            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2) where b is semi-minor, a is semi-major
                # ratio of lengths is same as ratio of semi-axes
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
        else:
            eccentricity = 0.0

        return np.array(
            [aspect_ratio, solidity, extent, eccentricity], dtype=FLOAT_PRECISION
        )

    except Exception:
        return np.zeros(4, dtype=FLOAT_PRECISION)


def process_single_image(image_rel_path):
    """
    Reads an image from disk and extracts all macro features.

    Args:
        image_rel_path (str): Relative path to the image (e.g., 'images/1.jpg').

    Returns:
        np.ndarray: A 1D array of 11 float64 features (7 Hu + 4 Geometric).
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Return zeros if file doesn't exist
    if not os.path.exists(full_path):
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # Thresholding
    # Dataset description: "binary black leaves against white backgrounds"
    # Leaf pixels are low values (black), Background is high (white).
    # We invert this so Leaf is Foreground (255) and Background is (0).
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # Assume the largest contour is the leaf
    c = max(contours, key=cv2.contourArea)

    hu_moments = extract_hu_moments(c)
    geometric_props = extract_geometric_properties(c)

    return np.concatenate([hu_moments, geometric_props])


def get_macro_features(dataset_split, load_cached_data=True):
    """
    Retrieves macro features for a specific dataset split.
    Implements caching logic to avoid re-computation.

    Args:
        dataset_split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and the extracted features.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(CACHE_DIR, f"macro_features_{dataset_split}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached macro features for '{dataset_split}' from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute from scratch
    print(f"Computing macro features for '{dataset_split}'...")

    # Load metadata to get image paths
    metadata_path = os.path.join(METADATA_DIR, f"{dataset_split}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    feature_rows = []
    ids = []

    # Define column names
    hu_cols = [f"hu_moment_{i}" for i in range(1, 8)]
    geom_cols = ["aspect_ratio", "solidity", "extent", "eccentricity"]
    all_cols = hu_cols + geom_cols

    # Iterate and process
    for _, row in df_meta.iterrows():
        img_id = row["id"]
        img_path = row["image_path"]

        features = process_single_image(img_path)
        feature_rows.append(features)
        ids.append(img_id)

    # Construct DataFrame
    X = np.array(feature_rows, dtype=FLOAT_PRECISION)
    df_features = pd.DataFrame(X, columns=all_cols)
    df_features.insert(0, "id", ids)

    # 3. Save to cache
    print(f"Saving macro features for '{dataset_split}' to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
