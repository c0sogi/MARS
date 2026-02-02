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
    EFD_HARMONICS,
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

    if major_axis > 0:
        ratio = minor_axis / major_axis
        if ratio > 1.0:
            ratio = 1.0
        eccentricity = np.sqrt(1 - ratio**2)
    else:
        eccentricity = 0.0

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


def extract_efd_features(contour):
    """
    Computes Elliptical Fourier Descriptors for a contour.
    Normalizes for rotation and starting point, but NOT scale.
    """
    if EFD_HARMONICS == 0:
        return {}

    # 1. Pre-processing for Normalization
    if len(contour) < 5:
        return {
            f"efd_{n}_{c}": FLOAT_PRECISION(0.0)
            for n in range(1, EFD_HARMONICS + 1)
            for c in ["a", "b", "c", "d"]
        }

    # Convert contour to float for precision
    contour = contour.astype(FLOAT_PRECISION).reshape(-1, 2)

    # Center the contour
    centroid = np.mean(contour, axis=0)
    contour_centered = contour - centroid

    # Rotation Invariance: Align major axis to X-axis
    # We use PCA/Covariance to find orientation robustly
    cov = np.cov(contour_centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    # Sort eigenvectors by eigenvalues (largest first)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # Rotate
    contour_rotated = np.dot(contour_centered, evecs)

    # Starting Point Invariance: Start at point furthest from centroid
    distances = np.linalg.norm(contour_rotated, axis=1)
    start_idx = np.argmax(distances)
    contour_final = np.roll(contour_rotated, -start_idx, axis=0)

    # 2. EFD Calculation (Kuhl & Giardina)
    # x and y projections
    x = contour_final[:, 0]
    y = contour_final[:, 1]

    # Deltas
    dx = np.diff(x)
    dy = np.diff(y)
    # Close the loop
    dx = np.append(dx, x[0] - x[-1])
    dy = np.append(dy, y[0] - y[-1])

    dt = np.sqrt(dx**2 + dy**2)
    t_len = np.sum(dt)

    # Cumulative distance
    t = np.concatenate(([0], np.cumsum(dt)))

    # Normalize t to [0, T] -> used in formula
    # Formula terms:
    # a_n = T / (2*pi^2*n^2) * sum( (dx/dt) * (cos(2n*pi*t_p/T) - cos(2n*pi*t_{p-1}/T)) )

    coeffs = {}

    # Precompute constants
    two_pi = 2 * np.pi
    T = t_len
    if T == 0:
        T = 1.0  # Avoid div by zero

    inv_dt = np.where(dt > 1e-9, 1.0 / dt, 0.0)

    for n in range(1, EFD_HARMONICS + 1):
        factor = T / (2 * (np.pi * n) ** 2)

        # Arguments for trig functions
        arg_curr = (two_pi * n * t[1:]) / T
        arg_prev = (two_pi * n * t[:-1]) / T

        cos_diff = np.cos(arg_curr) - np.cos(arg_prev)
        sin_diff = np.sin(arg_curr) - np.sin(arg_prev)

        an = factor * np.sum(dx * inv_dt * cos_diff)
        bn = factor * np.sum(dx * inv_dt * sin_diff)
        cn = factor * np.sum(dy * inv_dt * cos_diff)
        dn = factor * np.sum(dy * inv_dt * sin_diff)

        coeffs[f"efd_{n}_a"] = FLOAT_PRECISION(an)
        coeffs[f"efd_{n}_b"] = FLOAT_PRECISION(bn)
        coeffs[f"efd_{n}_c"] = FLOAT_PRECISION(cn)
        coeffs[f"efd_{n}_d"] = FLOAT_PRECISION(dn)

    return coeffs


def process_single_image(image_path):
    """
    Loads an image and extracts all features.
    """
    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Return zeros if image load fails
        dummy_spatial = {f"spatial_{k}": FLOAT_PRECISION(0.0) for k in SPATIAL_FEATURES}
        dummy_efd = {
            f"efd_{n}_{c}": FLOAT_PRECISION(0.0)
            for n in range(1, EFD_HARMONICS + 1)
            for c in ["a", "b", "c", "d"]
        }
        return {**dummy_spatial, **dummy_efd}

    # Threshold (Leaves are black on white, so invert)
    # BINARY_THRESHOLD is typically 127
    _, thresh = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        dummy_spatial = {f"spatial_{k}": FLOAT_PRECISION(0.0) for k in SPATIAL_FEATURES}
        dummy_efd = {
            f"efd_{n}_{c}": FLOAT_PRECISION(0.0)
            for n in range(1, EFD_HARMONICS + 1)
            for c in ["a", "b", "c", "d"]
        }
        return {**dummy_spatial, **dummy_efd}

    # Take largest contour
    c = max(contours, key=cv2.contourArea)

    # Extract Features
    spatial_feats, _ = extract_spatial_features(c)
    efd_feats = extract_efd_features(c)

    return {**spatial_feats, **efd_feats}


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
