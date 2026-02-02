import os
import cv2
import numpy as np
import pandas as pd
import library.config as config
import library.utils as utils


def extract_features_from_image(image_path):
    """
    Extracts a comprehensive suite of geometric and moment-based features from a binary leaf image.

    Features include:
    - Absolute Moments: Area, Major/Minor Axis Lengths.
    - Relative Moments: Eccentricity.
    - Invariant Moments: Hu Moments 1-7 (Log-Modulus transformed).
    - Boundary Integrals: Perimeter, Convex Perimeter, Solidity, Roundness.
    - Stable Extrema: Aspect Ratio, Extent.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: A dictionary mapping feature names (from config) to float64 values.
    """
    # Initialize all features to 0.0
    features = {col: 0.0 for col in config.IMAGE_FEATURE_COLS}

    if not os.path.exists(image_path):
        return features

    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Binarize
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour by area is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # ---------------------------------------------------------
    # Base Moments & Area
    # ---------------------------------------------------------
    M = cv2.moments(cnt)
    area = M["m00"]

    # Filter extremely small noise
    if area < 1e-6:
        return features

    # Equivalent Diameter: sqrt(4 * Area / pi)
    # Cite solution_lesson_node_00118: Absolute size signal is critical.
    features[config.FEAT_IMG_EQUIV_DIAMETER] = np.sqrt(4 * area / np.pi)

    # ---------------------------------------------------------
    # Eccentricity (derived from 2nd Moments)
    # ---------------------------------------------------------
    mu20 = M["mu20"] / area
    mu02 = M["mu02"] / area
    mu11 = M["mu11"] / area

    common_term = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
    lambda1 = (mu20 + mu02 + common_term) / 2
    lambda2 = (mu20 + mu02 - common_term) / 2

    major_axis = 4 * np.sqrt(lambda1) if lambda1 > 0 else 0.0
    minor_axis = 4 * np.sqrt(lambda2) if lambda2 > 0 else 0.0

    if major_axis > 1e-9:
        eccentricity = np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2))
    else:
        eccentricity = 0.0
    features[config.FEAT_IMG_ECCENTRICITY] = float(eccentricity)

    # ---------------------------------------------------------
    # Boundary Integrals (Solidity, Roundness)
    # ---------------------------------------------------------
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    convex_area = cv2.contourArea(hull)

    if convex_area > 1e-9:
        solidity = area / convex_area
    else:
        solidity = 0.0
    features[config.FEAT_IMG_SOLIDITY] = float(solidity)

    if perimeter > 1e-9:
        # Roundness: 4 * pi * Area / Perimeter^2
        roundness = (4 * np.pi * area) / (perimeter**2)
    else:
        roundness = 0.0
    features[config.FEAT_IMG_ROUNDNESS] = float(roundness)

    # ---------------------------------------------------------
    # Stable Extrema (Aspect Ratio, Extent)
    # ---------------------------------------------------------
    x, y, w, h = cv2.boundingRect(cnt)

    if h > 0:
        aspect_ratio = float(w) / h
    else:
        aspect_ratio = 0.0
    features[config.FEAT_IMG_ASPECT_RATIO] = aspect_ratio

    rect_area = w * h
    if rect_area > 0:
        extent = area / rect_area
    else:
        extent = 0.0
    features[config.FEAT_IMG_EXTENT] = extent

    return features


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Loads metadata, extracts image features (or loads from cache), merges with
    tabular features, and returns a float64 DataFrame.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_path (str): Path to the Parquet cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with all features.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Attempt Load from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing dataset from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Extract image features
    img_features_list = []

    # Iterate through metadata to process images
    for idx, row in df_meta.iterrows():
        # Construct full path. Metadata contains relative path e.g. "images/1.jpg"
        rel_path = row[config.FILE_PATH_COL]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Extract features
        feats = extract_features_from_image(full_path)
        img_features_list.append(feats)

    # Create DataFrame from extracted features
    df_img = pd.DataFrame(img_features_list)

    # Merge with original metadata (which contains tabular features and labels)
    # We concat horizontally. Indices should align perfectly.
    df_combined = pd.concat([df_meta, df_img], axis=1)

    # Define columns to keep
    # We need ID, Target (if present), Tabular Features, and Image Features
    cols_to_keep = [config.ID_COL]
    if config.TARGET_COL in df_combined.columns:
        cols_to_keep.append(config.TARGET_COL)

    all_features = config.TABULAR_FEATURE_COLS + config.IMAGE_FEATURE_COLS

    # Verify all tabular features exist
    missing_tabular = [
        c for c in config.TABULAR_FEATURE_COLS if c not in df_combined.columns
    ]
    if missing_tabular:
        raise ValueError(
            f"Missing tabular features in metadata: {missing_tabular[:5]}..."
        )

    cols_to_keep.extend(all_features)

    # Filter DataFrame
    df_final = df_combined[cols_to_keep].copy()

    # Enforce float64 precision and fill NaNs
    for col in all_features:
        df_final[col] = df_final[col].astype(config.FLOAT_PRECISION)

    # Fill any potential NaNs (e.g. from division by zero) with 0.0
    df_final[all_features] = df_final[all_features].fillna(0.0)

    # Save to Cache
    print(f"Saving features to {cache_path}")
    df_final.to_parquet(cache_path, index=False)

    return df_final


def load_datasets(load_cached_data=True):
    """
    Convenience function to load Train, Validation, and Test datasets.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = process_dataset(
        config.TRAIN_METADATA_PATH, config.CACHE_TRAIN_PATH, load_cached_data
    )
    val_df = process_dataset(
        config.VAL_METADATA_PATH, config.CACHE_VAL_PATH, load_cached_data
    )
    test_df = process_dataset(
        config.TEST_METADATA_PATH, config.CACHE_TEST_PATH, load_cached_data
    )

    return train_df, val_df, test_df
