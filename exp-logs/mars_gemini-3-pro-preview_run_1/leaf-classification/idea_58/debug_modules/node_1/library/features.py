import os
import cv2
import numpy as np
import pandas as pd
import logging
from library.utils import load_metadata, get_cache_dir, set_seed

# Set seed for reproducibility across operations
set_seed(42)


def extract_geometric_features(image_path: str) -> dict:
    """
    Extracts 7 parsimonious geometric scalars from a binary leaf image.

    Features:
        - Area (Absolute scale)
        - Eccentricity (Elongation)
        - Solidity (Roughness)
        - Extent (Rectangularity)
        - Aspect_Ratio (Orientation)
        - Roundness (Compactness)
        - Mean_Thickness (Internal Structure via Distance Transform)

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the 7 extracted features (float64).
    """
    # Default feature vector for failures
    features = {
        "Area": 0.0,
        "Eccentricity": 0.0,
        "Solidity": 0.0,
        "Extent": 0.0,
        "Aspect_Ratio": 0.0,
        "Roundness": 0.0,
        "Mean_Thickness": 0.0,
    }

    if not os.path.exists(image_path):
        logging.warning(f"Image path does not exist: {image_path}")
        return features

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.warning(f"Failed to read image: {image_path}")
        return features

    # Polarity Correction: The dataset consists of black leaves on white backgrounds.
    # We invert this to get white leaves (foreground) on black backgrounds.
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find Contours: Use CHAIN_APPROX_NONE for lossless boundary retrieval
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return features

    # Select the largest contour by area (assuming it's the leaf)
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))

    if area <= 0:
        return features

    features["Area"] = area

    # 1. Eccentricity via Ellipse Fitting
    # Requires at least 5 points
    if len(cnt) >= 5:
        try:
            (center, (MA, ma), angle) = cv2.fitEllipse(cnt)
            # MA and ma are the lengths of the major and minor axes (diameters)
            # a = semi-major, b = semi-minor
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            if a > 0:
                # e = sqrt(1 - (b/a)^2)
                features["Eccentricity"] = np.sqrt(1 - (b / a) ** 2)
        except Exception:
            pass  # Keep default 0.0

    # 2. Solidity = Area / Convex Hull Area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["Solidity"] = area / hull_area

    # 3. Extent = Area / Bounding Rect Area
    # 4. Aspect Ratio = Bounding Rect Width / Height
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        features["Extent"] = area / rect_area
    if h > 0:
        features["Aspect_Ratio"] = float(w) / h

    # 5. Roundness = 4 * pi * Area / Perimeter^2
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0:
        features["Roundness"] = (4 * np.pi * area) / (perimeter**2)

    # 6. Mean Thickness via Euclidean Distance Transform
    # Compute distance from every foreground pixel to the nearest background pixel
    # Mask is where binary > 0
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    mask = binary > 0
    if np.any(mask):
        features["Mean_Thickness"] = np.mean(dist_transform[mask])

    return features


def get_dataset(
    split: str, debug: bool = False, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads the dataset for the specified split, extracting geometric features and
    merging them with the tabular metadata. Implements caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, processes a small subset of data.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        pd.DataFrame: The combined dataset with tabular and geometric features.
    """
    cache_dir = get_cache_dir()
    cache_filename = f"dataset_{split}_debug_{debug}.parquet"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logging.info(f"Loading cached dataset from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            logging.info(f"Loaded {split} set with shape {df.shape}")
            return df
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    logging.info(f"Generating features for {split} split (debug={debug})...")
    meta_df = load_metadata(split, debug=debug)

    # 3. Extract Features
    input_dir = "./input"
    extracted_rows = []

    total_images = len(meta_df)
    log_interval = max(1, total_images // 10)

    for i, row in meta_df.iterrows():
        # Construct full path. Metadata 'file_path' is relative (e.g., "images/1.jpg")
        full_path = os.path.join(input_dir, row["file_path"])

        # Extract geometric features
        feats = extract_geometric_features(full_path)
        feats["id"] = row["id"]  # Key for merging
        extracted_rows.append(feats)

        if (i + 1) % log_interval == 0:
            logging.info(f"Processed {i + 1}/{total_images} images")

    # Create DataFrame from extracted features
    geo_df = pd.DataFrame(extracted_rows)

    # 4. Merge with Metadata
    # meta_df contains 'id', 'species' (if train/val), and 192 tabular features
    combined_df = pd.merge(meta_df, geo_df, on="id", how="left")

    # Remove file_path as it is no longer needed for modeling
    if "file_path" in combined_df.columns:
        combined_df = combined_df.drop(columns=["file_path"])

    # 5. Enforce Deterministic Column Ordering
    # We separate ID and Species (if present) from features
    meta_cols = {"id", "species"}
    feature_cols = [c for c in combined_df.columns if c not in meta_cols]
    feature_cols.sort()  # Alphanumeric sort

    # Construct final column order: id, [sorted features], species
    final_order = ["id"] + feature_cols
    if "species" in combined_df.columns:
        final_order.append("species")

    combined_df = combined_df[final_order]

    # 6. Save to Cache
    try:
        combined_df.to_parquet(cache_path, index=False)
        logging.info(f"Cached dataset saved to {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")

    return combined_df
