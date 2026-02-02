import os
import cv2
import numpy as np
import pandas as pd
from library import config


def extract_geometric_features(image_path):
    """
    Extracts 5 parsimonious geometric features from a binary leaf image.

    Features:
    1. Area (Scale)
    2. Eccentricity (Elongation)
    3. Solidity (Compactness)
    4. Extent (Rectangularity)
    5. Aspect_Ratio (Orientation)

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the 5 geometric features.
    """
    # Default values in case of failure/empty image
    features = {feat: 0.0 for feat in config.GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Polarity Correction: Leaf is black on white bg -> Invert to make leaf white (foreground)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours with high fidelity
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return features

    # Assume largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Area
    area = cv2.contourArea(cnt)
    features["Area"] = float(area)

    if area == 0:
        return features

    # 2. Eccentricity
    # Needs at least 5 points for fitEllipse
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # MA, ma are axis lengths (minor, major) or vice versa.
            # fitEllipse returns (width, height) of rotated rect.
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            if a > 0:
                # e = sqrt(1 - b^2/a^2)
                features["Eccentricity"] = np.sqrt(1 - (b**2 / a**2))
            else:
                features["Eccentricity"] = 0.0
        except Exception:
            features["Eccentricity"] = 0.0
    else:
        features["Eccentricity"] = 0.0

    # 3. Solidity = Area / ConvexHull Area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["Solidity"] = area / hull_area
    else:
        features["Solidity"] = 0.0

    # 4. Extent = Area / BoundingRect Area
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        features["Extent"] = area / rect_area
    else:
        features["Extent"] = 0.0

    # 5. Aspect Ratio = Bounding Rect Width / Height
    if h > 0:
        features["Aspect_Ratio"] = float(w) / h
    else:
        features["Aspect_Ratio"] = 0.0

    return features


def get_processed_data(metadata_path, split_name, load_cached_data=True):
    """
    Loads data, extracts geometric features, combines with tabular features,
    and returns X (features), y (targets), and ids.

    Implements caching using parquet/npy in config.WORKING_DIR.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        split_name (str): Name of the split ('train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_df, y_array, ids_array)
            X_df: pandas DataFrame of features (float64)
            y_array: numpy array of targets (strings) or None for test
            ids_array: numpy array of image IDs
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_X_path = os.path.join(cache_dir, f"X_{split_name}.parquet")
    cache_y_path = os.path.join(cache_dir, f"y_{split_name}.npy")
    cache_ids_path = os.path.join(cache_dir, f"ids_{split_name}.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
            # Check y cache existence only if not test set (test set might not have y, though metadata format has it as optional/missing)
            # Based on metadata description, test.csv doesn't have 'species'.
            has_y = os.path.exists(cache_y_path)

            # If it's train/val, we need y. If test, we don't.
            if split_name in ["train", "val"] and not has_y:
                pass  # Cache invalid
            else:
                # print(f"Loading {split_name} data from cache...")
                X = pd.read_parquet(cache_X_path)
                ids = np.load(cache_ids_path)
                y = np.load(cache_y_path, allow_pickle=True) if has_y else None
                return X, y, ids

    # 2. Process from Scratch
    # print(f"Processing {split_name} data from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Identify tabular columns
    # We select columns starting with the defined prefixes
    tabular_cols = [
        c
        for c in df_meta.columns
        if any(c.startswith(p) for p in config.TABULAR_FEATURE_PREFIXES)
    ]

    feature_rows = []
    ids_list = []
    targets_list = []

    # Construct full image paths
    # Metadata contains relative path in 'file_path'

    for idx, row in df_meta.iterrows():
        img_id = row[config.ID_COL]
        rel_path = row[config.FILE_PATH_COL]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Extract Geometric Features
        geo_feats = extract_geometric_features(full_path)

        # Get Tabular Features
        tab_feats = row[tabular_cols].to_dict()

        # Combine
        combined = {**geo_feats, **tab_feats}
        feature_rows.append(combined)

        ids_list.append(img_id)

        if config.TARGET_COL in row:
            targets_list.append(row[config.TARGET_COL])

    # Create DataFrame
    X_df = pd.DataFrame(feature_rows)

    # Deterministic Column Ordering (Alphanumeric)
    sorted_cols = sorted(X_df.columns)
    X_df = X_df[sorted_cols]

    # Enforce Precision
    X_df = X_df.astype(config.FLOAT_TYPE)

    ids_array = np.array(ids_list)

    y_array = None
    if targets_list:
        y_array = np.array(targets_list)

    # 3. Save to Cache
    X_df.to_parquet(cache_X_path)
    np.save(cache_ids_path, ids_array)
    if y_array is not None:
        np.save(cache_y_path, y_array)

    return X_df, y_array, ids_array
