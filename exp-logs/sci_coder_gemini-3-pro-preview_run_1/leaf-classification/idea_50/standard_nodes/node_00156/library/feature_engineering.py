import os
import numpy as np
import pandas as pd
import cv2
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    GEOMETRIC_FEATURES,
    SHAPE_COLS_TO_DROP,
    CV_THRESH_BINARY_INV,
    CV_CHAIN_APPROX_NONE,
    ID_COL,
    TARGET_COL,
    IMAGE_PATH_COL,
)


def extract_single_image_features(image_relative_path):
    """
    Extracts robust geometric scalar features from a binary leaf image.

    Args:
        image_relative_path (str): Relative path to the image (e.g., 'images/1.jpg').

    Returns:
        dict: A dictionary containing the calculated geometric features.
    """
    full_path = os.path.join(INPUT_DIR, image_relative_path)

    # Initialize default values (NaN) in case of failure
    features = {k: np.nan for k in GEOMETRIC_FEATURES}

    if not os.path.exists(full_path):
        return features

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Invert image: Leaf becomes white (255), background black (0)
    # The dataset description says "binary black leaves against white backgrounds".
    # Using THRESH_BINARY_INV ensures the object of interest (leaf) is foreground (white).
    _, thresh = cv2.threshold(img, 127, 255, CV_THRESH_BINARY_INV)

    # Find contours
    # RETR_EXTERNAL: We only care about the outer boundary of the leaf
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, CV_CHAIN_APPROX_NONE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Basic Measures
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area == 0 or perimeter == 0:
        return features

    # 2. Bounding Rectangle (Axis Aligned)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    # Aspect Ratio (width / height)
    features["Aspect_Ratio"] = float(w) / h if h > 0 else 0.0

    # Extent (Object Area / Bounding Box Area)
    features["Extent"] = float(area) / rect_area if rect_area > 0 else 0.0

    # 3. Convex Hull
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    # Solidity (Object Area / Convex Hull Area)
    features["Solidity"] = float(area) / hull_area if hull_area > 0 else 0.0

    # 4. Ellipse Fitting (Requires at least 5 points)
    if len(cnt) >= 5:
        try:
            # fitEllipse returns ((center_x, center_y), (axis1, axis2), angle)
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)

            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)

            # Eccentricity: sqrt(1 - (b/a)^2) where a is semi-major, b is semi-minor
            if major_axis > 0:
                features["Eccentricity"] = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                features["Eccentricity"] = 0.0

        except Exception:
            features["Eccentricity"] = 0.0
    else:
        features["Eccentricity"] = 0.0

    # 5. Roundness (4 * pi * Area / Perimeter^2)
    features["Roundness"] = (4 * np.pi * area) / (perimeter**2)

    # 6. Equivalent Diameter (sqrt(4 * Area / pi))
    features["Equivalent_Diameter"] = np.sqrt(4 * area / np.pi)

    return features


def load_and_process_data(metadata_path, load_cached_data=True):
    """
    Loads metadata, extracts geometric features, replaces shape features,
    and returns the processed dataframe. Implements strict caching.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The processed dataframe with float64 precision.
    """
    # Determine cache path
    base_name = os.path.basename(metadata_path)
    cache_name = os.path.splitext(base_name)[0] + "_processed.parquet"
    cache_path = os.path.join(CACHE_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Extract geometric features for all images
    print("Extracting geometric features...")
    geo_features_list = []

    # Iterate over rows (using simple loop as requested, no progress bar)
    for idx, row in df.iterrows():
        rel_path = row[IMAGE_PATH_COL]
        feats = extract_single_image_features(rel_path)
        geo_features_list.append(feats)

    # Create DataFrame from new features
    geo_df = pd.DataFrame(geo_features_list)

    # Enforce Alphanumeric Column Ordering on new features
    # This ensures deterministic memory layout
    geo_df = geo_df.reindex(sorted(geo_df.columns), axis=1)

    # 3. Subtractive Fusion
    # Drop original shape columns
    # Strict validation to ensure we are actually dropping the intended columns (Cite 00154)
    missing_cols = [c for c in SHAPE_COLS_TO_DROP if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Critical Error: Target columns for dropping not found: {missing_cols[:5]}... "
            "Check column naming conventions."
        )

    df_reduced = df.drop(columns=SHAPE_COLS_TO_DROP)

    # Concatenate: Original (minus shape) + New Geometric
    # We reset index to ensure alignment, though iterrows preserves order
    df_processed = pd.concat(
        [df_reduced.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1
    )

    # 4. Enforce float64 precision for feature columns
    # Identify all feature columns (exclude ID, Target, FilePath)
    exclude_cols = [ID_COL, TARGET_COL, IMAGE_PATH_COL]
    feature_cols = [c for c in df_processed.columns if c not in exclude_cols]

    for col in feature_cols:
        df_processed[col] = df_processed[col].astype(np.float64)

    # 5. Save to cache
    print(f"Saving processed data to {cache_path}")
    try:
        df_processed.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")

    return df_processed
