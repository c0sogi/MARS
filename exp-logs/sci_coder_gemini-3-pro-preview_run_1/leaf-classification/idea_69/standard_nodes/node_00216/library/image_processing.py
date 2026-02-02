import os
import cv2
import numpy as np
import pandas as pd
from library.utils import set_seed

# Set fixed seed for reproducibility
set_seed(42)

# Constants
CACHE_DIR = "./working/idea_69"
INPUT_DIR = "./input"


def extract_geometric_features(image_path):
    """
    Extracts 6 scalar geometric features from a binary leaf image.

    Logic:
    1. Load as grayscale.
    2. Threshold (BINARY_INV, 127).
    3. Find largest contour (Area).
    4. Compute: Area, Major_Axis_Length, Eccentricity, Solidity, Extent, Aspect_Ratio.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the 6 geometric features (float64).
    """
    # Default zero-vector in case of failure/noise
    features = {
        "Area": 0.0,
        "Major_Axis_Length": 0.0,
        "Eccentricity": 0.0,
        "Solidity": 0.0,
        "Extent": 0.0,
        "Aspect_Ratio": 0.0,
    }

    if not os.path.exists(image_path):
        return features

    try:
        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return features

        # Apply fixed thresholding
        # Leaf is black on white, so BINARY_INV makes leaf white (foreground)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not cnts:
            return features

        # Select largest contour by Area (Implicit Denoising)
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area == 0:
            return features

        features["Area"] = float(area)

        # 1. Ellipse Features (Major Axis, Eccentricity)
        # fitEllipse requires at least 5 points
        if len(c) >= 5:
            try:
                # fitEllipse returns ((x,y), (width, height), angle)
                # The axes lengths are the width and height of the rotated rect
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)

                # Major axis is the larger of the two
                major_axis = max(MA, ma)
                minor_axis = min(MA, ma)

                features["Major_Axis_Length"] = float(major_axis)

                if major_axis > 0:
                    # Eccentricity = sqrt(1 - (minor/major)^2)
                    ratio_sq = (minor_axis / major_axis) ** 2
                    features["Eccentricity"] = float(np.sqrt(1 - ratio_sq))
            except Exception:
                # Fallback if fitEllipse fails numerically
                pass

        # 2. Hull Features (Solidity)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            features["Solidity"] = float(area / hull_area)

        # 3. Bounding Rect Features (Extent, Aspect Ratio)
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h

        if rect_area > 0:
            features["Extent"] = float(area / rect_area)

        if h > 0:
            features["Aspect_Ratio"] = float(w / h)

    except Exception as e:
        # In case of any unexpected OpenCV error, return default zeros
        pass

    return features


def process_dataset(metadata_path, load_cached_data=True):
    """
    Processes a dataset defined by a metadata CSV file.
    Extracts geometric features for all images and handles caching.

    Args:
        metadata_path (str): Path to the metadata CSV (e.g., './metadata/train.csv').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and the extracted features.
    """
    # Determine cache filename based on metadata filename
    base_name = os.path.splitext(os.path.basename(metadata_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{base_name}_geometric_features.parquet")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df_features = pd.read_parquet(cache_file)
            # Verify columns exist
            required_cols = [
                "id",
                "Area",
                "Major_Axis_Length",
                "Eccentricity",
                "Solidity",
                "Extent",
                "Aspect_Ratio",
            ]
            if all(col in df_features.columns for col in required_cols):
                return df_features
        except Exception:
            # If load fails, proceed to re-compute
            pass

    # 2. Compute from Scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    results = []

    for idx, row in df_meta.iterrows():
        image_id = row["id"]
        # Metadata contains relative path, e.g., 'images/1.jpg'
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        feats = extract_geometric_features(full_path)
        feats["id"] = image_id
        results.append(feats)

    df_features = pd.DataFrame(results)

    # Ensure column order and types
    cols_order = [
        "id",
        "Area",
        "Major_Axis_Length",
        "Eccentricity",
        "Solidity",
        "Extent",
        "Aspect_Ratio",
    ]
    df_features = df_features[cols_order]

    # Enforce float64 for feature columns
    for col in cols_order:
        if col != "id":
            df_features[col] = df_features[col].astype(np.float64)

    # 3. Save to Cache
    df_features.to_parquet(cache_file, index=False)

    return df_features
