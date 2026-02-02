import os
import cv2
import numpy as np
import pandas as pd
import library.config as config
import library.utils as utils

# Ensure reproducible behavior across operations
utils.set_seed(config.SEED)


def extract_geometric_features(image_path):
    """
    Extracts 6 geometric features from a binary leaf image.

    Pipeline:
    1. Load Image (Grayscale).
    2. Polarity Correction (Invert so leaf is foreground/white).
    3. Denoising (Select largest contour).
    4. Feature Computation (Mass, Density, Shape).

    Returns:
        dict: Dictionary containing the 6 scalar features.
    """
    # Default return vector for failures or empty images
    default_features = {
        "Area": 0.0,
        "Mean_Thickness": 0.0,
        "Eccentricity": 0.0,
        "Solidity": 0.0,
        "Extent": 0.0,
        "Aspect_Ratio": 0.0,
    }

    if not os.path.exists(image_path):
        return default_features

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return default_features

    # Polarity Correction:
    # Dataset is black leaves on white background.
    # We need white leaves (foreground) on black background.
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours (Lossless)
    # RETR_EXTERNAL: Only outer contours
    # CHAIN_APPROX_NONE: Store all boundary points
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return default_features

    # Implicit Denoising: Select largest contour by Area
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Safety check for empty/noise contours
    if area == 0:
        return default_features

    # --- Feature 1: Absolute Mass (2D) ---
    feat_area = float(area)

    # --- Feature 2: Absolute Density (3D) ---
    # Computed via Euclidean Distance Transform on the mask
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [c], -1, 255, -1)  # Fill contour

    # Calculate distance to nearest zero pixel
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Mean of non-zero values (average thickness/depth)
    inside_pixels = dist_transform[mask > 0]
    if inside_pixels.size > 0:
        # Use float64 accumulator to prevent overflow and cast to native float
        feat_thickness = float(np.mean(inside_pixels, dtype=np.float64))
    else:
        feat_thickness = 0.0

    # --- Feature 3: Eccentricity (Elongation) ---
    # Requires at least 5 points to fit ellipse
    feat_eccentricity = 0.0
    if len(c) >= 5:
        try:
            # fitEllipse returns ((x,y), (MA, ma), angle)
            # Note: OpenCV does not guarantee MA > ma in the return tuple
            _, (d1, d2), _ = cv2.fitEllipse(c)
            a = max(d1, d2)  # Major axis
            b = min(d1, d2)  # Minor axis
            if a > 0:
                # e = sqrt(1 - (b/a)^2)
                feat_eccentricity = float(np.sqrt(1 - (b / a) ** 2))
        except Exception:
            feat_eccentricity = 0.0

    # --- Feature 4: Solidity (Roughness) ---
    # Ratio of Contour Area to Convex Hull Area
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        feat_solidity = float(area / hull_area)
    else:
        feat_solidity = 0.0

    # --- Feature 5: Extent (Rectangularity) ---
    # Ratio of Contour Area to Bounding Rectangle Area
    _, _, w, h = cv2.boundingRect(c)
    rect_area = w * h
    if rect_area > 0:
        feat_extent = float(area / rect_area)
    else:
        feat_extent = 0.0

    # --- Feature 6: Aspect Ratio (Orientation) ---
    # Ratio of Bounding Rect Width to Height
    if h > 0:
        feat_aspect_ratio = float(w) / h
    else:
        feat_aspect_ratio = 0.0

    return {
        "Area": feat_area,
        "Mean_Thickness": feat_thickness,
        "Eccentricity": feat_eccentricity,
        "Solidity": feat_solidity,
        "Extent": feat_extent,
        "Aspect_Ratio": feat_aspect_ratio,
    }


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Loads metadata, extracts geometric features from images, merges with
    existing tabular features, and returns the complete dataset.

    Implements caching using Parquet.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    print(f"Processing {len(df_meta)} images from {metadata_path}...")

    # 3. Extract Geometric Features
    geo_features_list = []
    file_paths = df_meta[config.FILE_PATH_COL].values

    for rel_path in file_paths:
        full_path = os.path.join(config.IMAGES_BASE_DIR, rel_path)
        feats = extract_geometric_features(full_path)
        geo_features_list.append(feats)

    # Create DataFrame from new features
    df_geo = pd.DataFrame(geo_features_list)

    # 4. Merge with Metadata
    # Reset indices to ensure safe concatenation
    df_meta = df_meta.reset_index(drop=True)
    df_geo = df_geo.reset_index(drop=True)

    df_combined = pd.concat([df_meta, df_geo], axis=1)

    # 5. Validation and Type Casting
    # Ensure all configured features exist
    missing_cols = [c for c in config.ALL_FEATURES if c not in df_combined.columns]
    if missing_cols:
        raise ValueError(f"Missing columns after extraction: {missing_cols}")

    # Cast all feature columns to high precision float64
    df_combined[config.ALL_FEATURES] = df_combined[config.ALL_FEATURES].astype(
        config.FLOAT_PRECISION
    )

    # 6. Save to Cache
    print(f"Saving features to {cache_path}")
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_combined.to_parquet(cache_path, index=False)

    return df_combined


def get_train_data(load_cached_data=True):
    """Wrapper to get processed training data."""
    return process_dataset(
        config.TRAIN_METADATA_PATH, config.CACHE_TRAIN_PATH, load_cached_data
    )


def get_val_data(load_cached_data=True):
    """Wrapper to get processed validation data."""
    return process_dataset(
        config.VAL_METADATA_PATH, config.CACHE_VAL_PATH, load_cached_data
    )


def get_test_data(load_cached_data=True):
    """Wrapper to get processed test data."""
    return process_dataset(
        config.TEST_METADATA_PATH, config.CACHE_TEST_PATH, load_cached_data
    )
