import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SPATIAL_FEATURES,
    BINARY_THRESHOLD,
    FLOAT_PRECISION,
)
from library.utils import get_config_hash


def get_image_path(relative_path):
    """Constructs the full path to an image."""
    return os.path.join(INPUT_DIR, relative_path)


def extract_spatial_features(contour):
    """
    Extracts macro-geometric spatial features from a contour.
    """
    features = {}

    # Basic moments
    M = cv2.moments(contour)
    area = M["m00"]
    perimeter = cv2.arcLength(contour, True)

    # Convex Hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # Bounding Rect
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Ellipse fit (requires at least 5 points)
    if len(contour) >= 5:
        (center, (axis1, axis2), angle) = cv2.fitEllipse(contour)
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)
    else:
        major_axis, minor_axis, angle = 0.0, 0.0, 0.0

    # Derived metrics
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / rect_area if rect_area > 0 else 0.0
    equiv_diameter = np.sqrt(4 * area / np.pi) if area >= 0 else 0.0
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    eccentricity = (
        np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0
    )

    # Populate dictionary based on config
    # Note: We calculate all and filter/map to config names
    calc_map = {
        "Area": area,
        "Perimeter": perimeter,
        "Major_Axis": major_axis,
        "Minor_Axis": minor_axis,
        "Solidity": solidity,
        "Extent": extent,
        "Aspect_Ratio": aspect_ratio,
        "Equivalent_Diameter": equiv_diameter,
        "Roundness": roundness,
        "Eccentricity": eccentricity,
    }

    for feat_name in SPATIAL_FEATURES:
        features[f"spatial_{feat_name}"] = FLOAT_PRECISION(calc_map.get(feat_name, 0.0))

    return features, (major_axis, minor_axis, angle)


def process_single_image(image_path):
    """
    Loads an image and extracts all features.
    """
    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Return zeros if image load fails
        dummy_spatial = {f"spatial_{k}": FLOAT_PRECISION(0.0) for k in SPATIAL_FEATURES}
        return dummy_spatial

    # Threshold (Leaves are black on white, so invert)
    # BINARY_THRESHOLD is typically 127
    _, thresh = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        dummy_spatial = {f"spatial_{k}": FLOAT_PRECISION(0.0) for k in SPATIAL_FEATURES}
        return dummy_spatial

    # Take largest contour
    c = max(contours, key=cv2.contourArea)

    # Extract Features
    spatial_feats, _ = extract_spatial_features(c)

    return spatial_feats


def get_dataset(split_name, load_cached_data=True):
    """
    Main function to get the processed dataset.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: The dataset with ID, original features, and new spectral-spatial features.
    """
    # 1. Determine Metadata Path
    if split_name == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split_name == "val":
        meta_path = VAL_METADATA_PATH
    elif split_name == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split_name}")

    # 2. Check Cache
    config_hash = get_config_hash()
    cache_filename = f"features_{split_name}_{config_hash}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features for {split_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    # 3. Process from scratch
    print(f"Processing {split_name} dataset from scratch...")

    # Load metadata
    df_meta = pd.read_csv(meta_path)

    # List to hold new features
    new_features_list = []

    # Iterate through images
    # Using simple loop to avoid tqdm dependency as requested
    for idx, row in df_meta.iterrows():
        img_rel_path = row["file_path"]
        full_path = get_image_path(img_rel_path)

        # Extract
        feats = process_single_image(full_path)
        feats["id"] = row["id"]  # Keep ID for merging/verification

        new_features_list.append(feats)

    # Create DataFrame from new features
    df_new = pd.DataFrame(new_features_list)

    # Merge with original metadata
    # The metadata already contains margin/shape/texture features.
    # We merge on 'id'.
    df_final = pd.merge(df_meta, df_new, on="id", how="left")

    # Ensure precision
    # Convert all feature columns to FLOAT_PRECISION
    exclude_cols = ["id", "species", "file_path"]
    feature_cols = [c for c in df_final.columns if c not in exclude_cols]
    df_final[feature_cols] = df_final[feature_cols].astype(FLOAT_PRECISION)

    # 4. Save to Cache
    print(f"Saving features to {cache_path}")
    df_final.to_parquet(cache_path, index=False)

    return df_final
