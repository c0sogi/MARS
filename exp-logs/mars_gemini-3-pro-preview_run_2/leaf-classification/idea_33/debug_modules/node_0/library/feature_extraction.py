import os
import cv2
import numpy as np
import pandas as pd
from library import config


def extract_hu_moments(contour):
    """
    Extracts the 7 invariant Hu Moments from a contour.

    Args:
        contour: Numpy array of contour points.

    Returns:
        Numpy array of shape (7,) containing the Hu moments.
    """
    moments = cv2.moments(contour)
    # HuMoments returns a (7, 1) array, flatten to (7,)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def extract_geometric_properties(contour):
    """
    Extracts geometric properties: Aspect Ratio, Solidity, Extent, Eccentricity.

    Args:
        contour: Numpy array of contour points.

    Returns:
        Numpy array of shape (4,) containing the geometric scalars.
    """
    # 1. Aspect Ratio & Extent (Bounding Rect)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    rect_area = w * h

    area = cv2.contourArea(contour)
    extent = area / rect_area if rect_area > 0 else 0.0

    # 2. Solidity (Convex Hull)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # 3. Eccentricity (Fit Ellipse)
    # fitEllipse requires at least 5 points
    eccentricity = 0.0
    if len(contour) >= 5:
        try:
            # fitEllipse returns ((center_x, center_y), (width, height), angle)
            # Note: width and height here refer to the axes of the ellipse, not necessarily x/y aligned
            (center, (axis1, axis2), angle) = cv2.fitEllipse(contour)

            ma = max(axis1, axis2)
            mi = min(axis1, axis2)

            if ma > 0:
                # Eccentricity = sqrt(1 - (b/a)^2) where a is semi-major (ma/2), b is semi-minor (mi/2)
                # (mi/ma)^2 is equivalent to (b/a)^2
                eccentricity = np.sqrt(1 - (mi / ma) ** 2)
        except Exception:
            # Fallback if ellipse fitting fails (e.g. collinear points)
            eccentricity = 0.0

    return np.array([aspect_ratio, solidity, extent, eccentricity])


def process_single_image(image_path):
    """
    Loads and processes a single binary leaf image to extract macro features.

    Args:
        image_path: Relative path to the image (e.g., 'images/1.jpg').

    Returns:
        Numpy array of shape (11,) containing extracted features.
    """
    full_path = os.path.join(config.INPUT_DIR, image_path)

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return np.zeros(11, dtype=config.FLOAT_PRECISION)

    # Dataset description: "binary black leaves against white backgrounds".
    # OpenCV findContours assumes white object on black background.
    # We invert the image to make the leaf white (255) and background black (0).
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(11, dtype=config.FLOAT_PRECISION)

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)

    # Extract features
    hu = extract_hu_moments(cnt)
    geom = extract_geometric_properties(cnt)

    # Combine: 7 Hu moments + 4 Geometric properties = 11 features
    features = np.concatenate([hu, geom])

    return features.astype(config.FLOAT_PRECISION)


def extract_macro_features(metadata_df, dataset_key, load_cached_data=True):
    """
    Extracts macro features for a given dataset split.
    Handles caching to ./working/idea_33/

    Args:
        metadata_df: DataFrame containing 'image_path' and 'id'.
        dataset_key: String identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data: Boolean, whether to load from cache.

    Returns:
        DataFrame containing 'id' and the macro features.
    """
    cache_path = os.path.join(config.CACHE_DIR, f"{dataset_key}_macro.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached macro features for {dataset_key} from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Extracting macro features for {dataset_key}...")

    feature_list = []
    ids = []

    for idx, row in metadata_df.iterrows():
        img_id = row["id"]
        img_path = row["image_path"]

        feats = process_single_image(img_path)
        feature_list.append(feats)
        ids.append(img_id)

    # Create DataFrame
    cols = config.MACRO_FEATURE_NAMES

    # Safety check for column count
    if len(cols) != 11:
        cols = [f"macro_{i}" for i in range(11)]

    df_features = pd.DataFrame(feature_list, columns=cols)
    df_features["id"] = ids

    # Reorder columns to put 'id' first
    cols_ordered = ["id"] + [c for c in df_features.columns if c != "id"]
    df_features = df_features[cols_ordered]

    # 3. Save Cache
    try:
        df_features.to_parquet(cache_path, index=False)
        print(f"Saved macro features to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df_features
