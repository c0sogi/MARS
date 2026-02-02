import os
import cv2
import numpy as np
import pandas as pd
from library import config


def extract_single_image_features(image_path):
    """
    Extracts geometric features (Aspect Ratio, Solidity) from a single binary image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing 'Aspect_Ratio' and 'Solidity'.
              Returns NaNs if processing fails.
    """
    # Initialize default values
    features = {
        "Aspect_Ratio": np.nan,
        "Solidity": np.nan,
        "Equivalent_Diameter": np.nan,
        "Eccentricity": np.nan,
        "Roundness": np.nan,
    }

    if not os.path.exists(image_path):
        return features

    # Read image
    # The dataset consists of binary black leaves against white backgrounds.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return features

    # Invert the image so the leaf is white (255) and background is black (0)
    # for correct contour detection.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: if no contours found, return 0s or NaNs.
        # 0 is safer for linear models than NaN if imputation isn't guaranteed later.
        features["Aspect_Ratio"] = 0.0
        features["Solidity"] = 0.0
        return features

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Aspect Ratio
    # Width / Height of the bounding rect
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 0:
        features["Aspect_Ratio"] = float(w) / float(h)
    else:
        features["Aspect_Ratio"] = 0.0

    # 2. Solidity
    # Contour Area / Convex Hull Area
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        features["Solidity"] = float(area) / float(hull_area)
    else:
        features["Solidity"] = 0.0

    # 3. Equivalent Diameter (Scale feature)
    # Cite EDA: Image dimensions significantly vary across different species.
    features["Equivalent_Diameter"] = np.sqrt(4 * area / np.pi)

    # 4. Roundness (Shape feature)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0:
        features["Roundness"] = 4 * np.pi * area / (perimeter**2)
    else:
        features["Roundness"] = 0.0

    # 5. Eccentricity (Shape feature)
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            if MA > 0 and ma > 0:
                major = max(MA, ma)
                minor = min(MA, ma)
                features["Eccentricity"] = np.sqrt(1 - (minor / major) ** 2)
            else:
                features["Eccentricity"] = 0.0
        except Exception:
            features["Eccentricity"] = 0.0
    else:
        features["Eccentricity"] = 0.0

    return features


def extract_geometry(df, dataset_key, load_cached_data=True):
    """
    Extracts geometric features for all images in the provided DataFrame.
    Implements caching to disk using Parquet format.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'file_path' columns.
        dataset_key (str): Unique identifier for the dataset (e.g., 'train', 'val', 'test')
                           used for naming the cache file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and the extracted geometric features.
    """
    # Construct cache filename
    cache_filename = f"geometry_features_{dataset_key}.parquet"
    cache_path = config.get_cache_path(cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached geometric features from {cache_path}")
        try:
            cached_df = pd.read_parquet(cache_path)
            # Verify alignment
            if len(cached_df) == len(df) and np.all(
                cached_df["id"].values == df["id"].values
            ):
                return cached_df
            else:
                print("Cache mismatch (ID alignment or length). Recomputing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Extracting geometric features for {dataset_key} set...")

    # 2. Compute from scratch
    results = []

    # Ensure config directory exists for safety (though config.py does it)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    for idx, row in df.iterrows():
        # Construct full path using config.INPUT_DIR
        # Metadata 'file_path' is relative, e.g., 'images/123.jpg'
        full_path = os.path.join(config.INPUT_DIR, row["file_path"])

        feats = extract_single_image_features(full_path)

        # Add ID for safety/joining
        row_result = {"id": row["id"]}
        row_result.update(feats)
        results.append(row_result)

    # Create DataFrame
    feature_df = pd.DataFrame(results)

    # Ensure correct types (float64)
    cols_to_cast = [c for c in feature_df.columns if c != "id"]
    feature_df[cols_to_cast] = feature_df[cols_to_cast].astype(config.FLOAT_PRECISION)

    # 3. Save to cache
    try:
        feature_df.to_parquet(cache_path, index=False)
        print(f"Saved geometric features to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return feature_df
