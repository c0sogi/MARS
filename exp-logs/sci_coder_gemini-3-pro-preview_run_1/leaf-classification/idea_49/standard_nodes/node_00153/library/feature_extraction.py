import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config


def extract_inertial_features(image_path):
    """
    Extracts Boundary-Fitted Geometric features from a binary leaf image.
    Replaces inertial moments with explicit boundary fitting (fitEllipse) and ratios.
    Cite solution_lesson_node_00151: Prefer boundary-fitted geometric descriptors.
    """
    # Initialize default features
    features = {k: 0.0 for k in Config.EXTRACTED_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Invert polarity (Cite solution_lesson_node_00145)
    if Config.INVERT_IMAGES:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find Contours with Lossless Retrieval (Cite solution_lesson_node_00149)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return features

    # Assume largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # Basic Integral Properties
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area <= 1e-9:
        return features

    features["Area"] = float(area)
    features["Perimeter"] = float(perimeter)

    # Equivalent Diameter (Cite solution_lesson_node_00118)
    features["Equivalent_Diameter"] = np.sqrt(4 * area / np.pi)

    # Roundness (Cite solution_lesson_node_00151)
    if perimeter > 0:
        features["Roundness"] = (4 * np.pi * area) / (perimeter**2)

    # Convex Hull & Solidity (Cite solution_lesson_node_00120)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["Solidity"] = area / hull_area

    # Bounding Box & Extent (Cite solution_lesson_node_00120)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        features["Extent"] = area / rect_area

    # Aspect Ratio (Cite solution_lesson_node_00120)
    if h > 0:
        features["Aspect_Ratio"] = float(w) / float(h)

    # Ellipse Fitting & Eccentricity (Cite solution_lesson_node_00151)
    # Requires at least 5 points
    if len(cnt) >= 5:
        try:
            (center, (ma, MA), angle) = cv2.fitEllipse(cnt)
            # MA is major axis, ma is minor axis
            if MA > 0:
                # Eccentricity = sqrt(1 - (minor/major)^2)
                # Ensure ratio is <= 1
                ratio = (ma / MA) ** 2
                features["Eccentricity"] = np.sqrt(max(0.0, 1.0 - ratio))
        except Exception:
            pass

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
