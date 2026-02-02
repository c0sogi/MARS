import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config


def extract_inertial_features(image_path):
    """
    Extracts Integral-Inertial geometric features from a binary leaf image.
    Uses float64 precision for all calculations.
    """
    # Initialize default features in case of failure (e.g., empty image)
    features = {k: 0.0 for k in Config.EXTRACTED_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Invert polarity if configured (Leaf should be white/255, Background black/0)
    if Config.INVERT_IMAGES:
        # Assuming input is white background (255) and black leaf (0)
        # Inversion makes leaf 255, background 0
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 1. Image Moments (Integral Primitives)
    moments = cv2.moments(img, binaryImage=True)
    m00 = moments["m00"]

    # If image is empty or effectively empty
    if m00 <= 1e-9:
        return features

    # Area
    features["Area"] = float(m00)

    # 2. Inertial Axes (from Covariance Matrix of Moments)
    # Central moments
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    # Covariance matrix elements (normalized by mass)
    # Cov = [ mu20/m00  mu11/m00 ]
    #       [ mu11/m00  mu02/m00 ]
    a = mu20 / m00
    b = mu11 / m00
    c = mu02 / m00

    # Eigenvalues of 2x2 symmetric matrix
    # lambda = ( (a+c) +/- sqrt((a-c)^2 + 4b^2) ) / 2
    discriminant = np.sqrt((a - c) ** 2 + 4 * b**2)
    lambda1 = (a + c + discriminant) / 2.0
    lambda2 = (a + c - discriminant) / 2.0

    # Ensure non-negative (numerical noise protection)
    lambda1 = max(lambda1, 0.0)
    lambda2 = max(lambda2, 0.0)

    # Inertial Axes Lengths (Analogy to uniform ellipse: Length = 4 * sqrt(lambda))
    features["Inertial_Major_Axis"] = 4.0 * np.sqrt(lambda1)
    features["Inertial_Minor_Axis"] = 4.0 * np.sqrt(lambda2)

    # Eccentricity
    if lambda1 > 1e-9:
        features["Eccentricity"] = np.sqrt(max(0.0, 1.0 - lambda2 / lambda1))
    else:
        features["Eccentricity"] = 0.0

    # 3. Macro-Geometry (Contours)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Assume the largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)

        # Perimeter
        perimeter = cv2.arcLength(cnt, True)
        features["Perimeter"] = float(perimeter)

        # Convex Hull
        hull = cv2.convexHull(cnt)
        convex_perimeter = cv2.arcLength(hull, True)
        convex_area = cv2.contourArea(hull)
        features["Convex_Perimeter"] = float(convex_perimeter)

        # Solidity (Area / ConvexArea)
        if convex_area > 1e-9:
            features["Solidity"] = m00 / convex_area
        else:
            features["Solidity"] = 0.0

        # Convexity (ConvexPerimeter / Perimeter)
        if perimeter > 1e-9:
            features["Convexity"] = convex_perimeter / perimeter
        else:
            features["Convexity"] = 0.0

        # AABB (Axis Aligned Bounding Box)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        # AABB Aspect Ratio
        if h > 0:
            features["AABB_Aspect_Ratio"] = float(w) / float(h)
        else:
            features["AABB_Aspect_Ratio"] = 0.0

        # AABB Extent
        if rect_area > 0:
            features["AABB_Extent"] = m00 / rect_area
        else:
            features["AABB_Extent"] = 0.0

    return features


def _process_subset(metadata_path, cache_name, load_cached_data=True):
    """
    Internal function to process a specific dataset split (train/val/test).
    Handles caching, feature extraction, and merging.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {cache_name} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    print(f"Processing {cache_name} data from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    meta_df = pd.read_csv(metadata_path)

    # Prepare list to collect extracted features
    extracted_data = []

    # Iterate over images
    total = len(meta_df)
    for idx, row in meta_df.iterrows():
        # Construct full image path
        # Metadata contains relative path 'images/123.jpg'
        # INPUT_DIR is './input'
        # Full path should be './input/images/123.jpg'
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Extract features
        feats = extract_inertial_features(full_path)
        extracted_data.append(feats)

    # Create DataFrame from extracted features
    extracted_df = pd.DataFrame(extracted_data)

    # Merge with original metadata
    # The metadata contains: id, species (optional), raw tabular features, file_path
    # We want to keep: id, species (if exists), raw tabular features, extracted features

    # 1. Identify raw tabular features columns present in metadata
    raw_cols = [c for c in Config.RAW_TABULAR_FEATURES if c in meta_df.columns]

    # 2. Identify ID and Target
    id_col = "id"
    target_col = "species"

    cols_to_keep = [id_col]
    if target_col in meta_df.columns:
        cols_to_keep.append(target_col)

    # Concatenate parts
    # meta_df[cols_to_keep + raw_cols] + extracted_df
    # We reset index to ensure alignment, though iterrows preserves order
    df_final = pd.concat(
        [
            meta_df[cols_to_keep + raw_cols].reset_index(drop=True),
            extracted_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Enforce Alphanumeric Column Ordering for features
    # This ensures deterministic memory layout
    # We separate ID and Target from features for sorting
    feature_cols = raw_cols + list(extracted_df.columns)
    feature_cols = sorted(feature_cols)

    final_cols = cols_to_keep + feature_cols
    df_final = df_final[final_cols]

    # Save to cache
    print(f"Saving {cache_name} data to cache: {cache_path}")
    df_final.to_parquet(cache_path, index=False)

    return df_final


def get_train_data(load_cached_data=True):
    """
    Returns (X_train, y_train, ids_train)
    """
    df = _process_subset(Config.TRAIN_METADATA_PATH, "train_data", load_cached_data)

    y = df["species"].values
    ids = df["id"].values
    X = df.drop(columns=["id", "species"])

    return X, y, ids


def get_val_data(load_cached_data=True):
    """
    Returns (X_val, y_val, ids_val)
    """
    df = _process_subset(Config.VAL_METADATA_PATH, "val_data", load_cached_data)

    y = df["species"].values
    ids = df["id"].values
    X = df.drop(columns=["id", "species"])

    return X, y, ids


def get_test_data(load_cached_data=True):
    """
    Returns (X_test, ids_test)
    """
    df = _process_subset(Config.TEST_METADATA_PATH, "test_data", load_cached_data)

    ids = df["id"].values
    # Test set has no species column
    X = df.drop(columns=["id"])

    return X, ids
