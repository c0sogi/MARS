import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, CACHE_DIR, GEOMETRIC_FEATURES, SEED
from library.utils import set_seed

# Ensure reproducibility
set_seed(SEED)


def extract_geometric_features(image_path: str) -> dict:
    """
    Extracts robust geometric features from a binary leaf image.

    Implements the 'Sanitized Dictionary-Assembled' strategy:
    1. Polarity Correction (THRESH_BINARY_INV)
    2. Lossless Contours (CHAIN_APPROX_NONE)
    3. Dictionary Assembly (Name-based mapping)

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the calculated scalar features.
    """
    # Initialize default values (0.0) for all expected features
    features = {k: 0.0 for k in GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Polarity Correction: The dataset has black leaves on white background.
    # We invert this so leaves are white (foreground) and background is black.
    # Using OTSU for dynamic thresholding, combined with BINARY_INV.
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Check for empty mask
    if cv2.countNonZero(mask) == 0:
        return features

    # --- Contour-based Features ---
    # Find contours with exact approximation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # --- Feature 2: Area ---
    area = cv2.contourArea(cnt)
    features["Area"] = float(area)

    if area == 0:
        return features

    # --- Feature 3: Solidity ---
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["Solidity"] = float(area / hull_area)

    # --- Feature 4: Extent & Feature 5: Aspect Ratio ---
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        features["Extent"] = float(area / rect_area)

    if h > 0:
        features["Aspect_Ratio"] = float(w / h)

    # --- Feature 6: Eccentricity ---
    # Requires at least 5 points to fit an ellipse
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are lengths of the axes (major/minor depends on orientation)
            ma = max(axis1, axis2)
            ma_min = min(axis1, axis2)

            if ma > 0:
                # e = sqrt(1 - (b^2 / a^2))
                # a is semi-major (ma/2), b is semi-minor (ma_min/2)
                # ratio (b/a) is same as (ma_min/ma)
                features["Eccentricity"] = float(np.sqrt(1 - (ma_min / ma) ** 2))
        except:
            # Fallback if fitEllipse fails numerically
            features["Eccentricity"] = 0.0

    return features


def process_dataset(metadata_path: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata, extracts geometric features, and merges them with tabular features.
    Handles caching to avoid re-computation.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The fused dataset containing ID, original features, and new geometric features.
    """
    # Determine cache file path based on metadata filename
    # e.g., metadata/train.csv -> working/idea_63/train_features.parquet
    base_name = os.path.splitext(os.path.basename(metadata_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{base_name}_features.parquet")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing dataset: {metadata_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    meta_df = pd.read_csv(metadata_path)

    # List to store extracted feature dicts
    extracted_data = []

    # Iterate over images
    # Note: Using simple loop to avoid progress bar clutter as requested
    for idx, row in meta_df.iterrows():
        # Construct full image path
        # Metadata contains relative path 'images/123.jpg'
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Extract features
        feats = extract_geometric_features(full_path)

        # Add ID for safety check/merging (though we rely on row order)
        feats["id"] = row["id"]

        extracted_data.append(feats)

    # Create DataFrame from new features
    geo_df = pd.DataFrame(extracted_data)

    # Merge with original metadata
    # We want to keep the original 192 features + the new 6 features
    # We drop 'file_path' as it's not a feature.
    # We keep 'species' if it exists (train/val) for the model to use later.

    # Ensure alignment on 'id'
    merged_df = pd.merge(meta_df, geo_df, on="id", how="left")

    # Drop file_path as it is no longer needed for modeling
    if "file_path" in merged_df.columns:
        merged_df = merged_df.drop(columns=["file_path"])

    # Enforce alphanumeric column ordering for deterministic memory layout
    # Move 'id' and 'species' (if present) to front, rest sorted
    cols = merged_df.columns.tolist()
    special_cols = ["id", "species"]
    feature_cols = sorted([c for c in cols if c not in special_cols])

    final_cols = [c for c in special_cols if c in cols] + feature_cols
    merged_df = merged_df[final_cols]

    # 3. Save to cache
    print(f"Saving features to {cache_file}")
    merged_df.to_parquet(cache_file, index=False)

    return merged_df
