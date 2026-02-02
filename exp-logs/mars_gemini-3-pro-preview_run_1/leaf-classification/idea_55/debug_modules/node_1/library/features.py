import os
import cv2
import numpy as np
import pandas as pd
from library import config, utils


def extract_geometric_features(image_path):
    """
    Extracts integral-geometric descriptors from a binary leaf image.

    Implements the 'Corrected Integral-Geometric Fusion' logic:
    - Polarity correction (white leaf, black background).
    - Lossless contour extraction.
    - Computation of integral, bounding, convex, elliptical, and internal structure metrics.
    """
    # Initialize default features with 0.0 (float64)
    features = {k: 0.0 for k in config.GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Polarity Correction: Dataset is black leaf on white background.
    # Apply THRESH_BINARY_INV to get white leaf (foreground) on black background.
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Contours: Use CHAIN_APPROX_NONE for lossless boundary
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return features

    # Assume largest contour is the leaf
    c = max(contours, key=cv2.contourArea)

    # 1. Integral Measures
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    features["area"] = float(area)
    features["perimeter"] = float(perimeter)

    # 2. Bounding Geometry
    x, y, w, h = cv2.boundingRect(c)
    rect_area = w * h

    features["aspect_ratio"] = float(w) / h if h > 0 else 0.0
    features["extent"] = float(area) / rect_area if rect_area > 0 else 0.0

    # 3. Convex Geometry
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = float(area) / hull_area if hull_area > 0 else 0.0

    # 4. Elliptical Geometry
    # fitEllipse requires at least 5 points
    if len(c) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)
            # fitEllipse returns axes lengths (diameters)
            major = max(MA, ma)
            minor = min(MA, ma)

            features["major_axis_length"] = float(major)
            features["minor_axis_length"] = float(minor)

            if major > 0:
                # Eccentricity: sqrt(1 - (b/a)^2)
                features["eccentricity"] = np.sqrt(1 - (minor / major) ** 2)
            else:
                features["eccentricity"] = 0.0
        except Exception:
            # Fallback if fitEllipse fails (e.g., collinear points)
            pass

    # 5. Non-Linear Ratios
    if perimeter > 0:
        # Roundness: 4 * pi * Area / Perimeter^2
        features["roundness"] = (4 * np.pi * area) / (perimeter**2)

    # 6. Internal Structure (Distance Transform)
    # Euclidean distance to nearest zero pixel (background)
    # This captures the "fleshiness" or width distribution of the leaf
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # We are interested in the mean thickness of the leaf body
    leaf_mask = dist_transform > 0
    if np.any(leaf_mask):
        features["mean_thickness"] = np.mean(dist_transform[leaf_mask])

    return features


def process_dataset(metadata_path, dataset_type, load_cached_data=True):
    """
    Loads metadata, extracts geometric features, merges with tabular features,
    and returns a clean DataFrame with float64 precision.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        dataset_type (str): 'train', 'val', or 'test' (used for caching naming).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe with all features.
    """
    logger = utils.setup_logger()
    cache_file = os.path.join(config.CACHE_DIR, f"{dataset_type}_features.parquet")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading {dataset_type} features from cache: {cache_file}")
        df = pd.read_parquet(cache_file)
        return utils.enforce_float64(df)

    logger.info(f"Processing {dataset_type} dataset from scratch...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Debugging: Limit dataset size if configured
    if config.DEBUG_SAMPLE_SIZE is not None:
        logger.info(
            f"Debug mode enabled: Subsetting {dataset_type} to {config.DEBUG_SAMPLE_SIZE} rows."
        )
        df_meta = df_meta.head(config.DEBUG_SAMPLE_SIZE)

    # 3. Extract Geometry
    geo_features_list = []

    # Iterate rows
    total_rows = len(df_meta)
    for idx, row in df_meta.iterrows():
        # Construct full path. Metadata contains relative path e.g. "images/1.jpg"
        # config.INPUT_DIR is "./input"
        full_path = os.path.join(config.INPUT_DIR, row["file_path"])

        feats = extract_geometric_features(full_path)
        geo_features_list.append(feats)

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{total_rows} images for {dataset_type}")

    # Create DataFrame from new features
    df_geo = pd.DataFrame(geo_features_list)

    # 4. Merge
    # Ensure indices align before concatenation
    df_geo.index = df_meta.index

    # Combine original metadata (features + ids) with new geometric features
    df_final = pd.concat([df_meta, df_geo], axis=1)

    # 5. Enforce Precision
    # Ensure all floating point calculations and columns are float64
    df_final = utils.enforce_float64(df_final)

    # 6. Save Cache
    logger.info(f"Saving {dataset_type} features to cache: {cache_file}")
    df_final.to_parquet(cache_file, index=False)

    return df_final


def get_train_data(load_cached_data=True):
    """Wrapper to get training data."""
    return process_dataset(config.TRAIN_CSV, "train", load_cached_data)


def get_val_data(load_cached_data=True):
    """Wrapper to get validation data."""
    return process_dataset(config.VAL_CSV, "val", load_cached_data)


def get_test_data(load_cached_data=True):
    """Wrapper to get test data."""
    return process_dataset(config.TEST_CSV, "test", load_cached_data)
