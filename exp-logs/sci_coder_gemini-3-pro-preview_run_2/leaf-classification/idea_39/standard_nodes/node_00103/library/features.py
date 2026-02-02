import os
import cv2
import numpy as np
import pandas as pd
from library import config


def extract_morphometrics(image_path):
    """
    Extracts deterministic morphological descriptors from a binary leaf image.

    Features:
    - Hu Moments (7 invariants)
    - Geometric Scalars: Aspect Ratio, Solidity, Extent, Eccentricity

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the extracted features. Returns zeros if failure.
    """
    # Initialize default (zero) vector in case of failure
    features = {f"hu_{i}": 0.0 for i in range(1, 8)}
    features.update(
        {"aspect_ratio": 0.0, "solidity": 0.0, "extent": 0.0, "eccentricity": 0.0}
    )

    try:
        # Read image
        if not os.path.exists(image_path):
            return features

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return features

        # Threshold: Leaf is black, background is white.
        # Invert so leaf is white (foreground) for contour detection.
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return features

        # Assume largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area == 0:
            return features

        # 1. Hu Moments
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i, hu in enumerate(hu_moments):
            features[f"hu_{i+1}"] = hu

        # 2. Geometric Scalars
        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect Ratio
        if h > 0:
            features["aspect_ratio"] = float(w) / h

        # Extent
        rect_area = w * h
        if rect_area > 0:
            features["extent"] = area / rect_area

        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            features["solidity"] = area / hull_area

        # Eccentricity
        if len(cnt) >= 5:  # fitEllipse requires at least 5 points
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # MA, ma are axis lengths. Sort them.
            major = max(MA, ma)
            minor = min(MA, ma)
            if major > 0:
                # e = sqrt(1 - (b/a)^2)
                ratio_sq = (minor / major) ** 2
                features["eccentricity"] = np.sqrt(max(0, 1 - ratio_sq))

    except Exception as e:
        # In case of any processing error, return the zero-initialized features
        # This ensures the pipeline doesn't crash on a single bad image
        pass

    return features


def process_image_batch(metadata_df, cache_path, load_cached_data=True):
    """
    Process a batch of images defined in metadata_df to extract macro features.
    Handles caching mechanism using Parquet.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'image_path'.
        cache_path (str): Path to save/load the parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame of extracted features aligned with metadata_df.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached macro features from {cache_path}")
        try:
            df_features = pd.read_parquet(cache_path)
            # specific check to ensure cache matches current metadata length/ids
            if len(df_features) == len(metadata_df) and np.all(
                df_features.index == metadata_df.index
            ):
                return df_features
            else:
                print("Cache mismatch (length or index). Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting macro features for {len(metadata_df)} images...")

    results = []

    # Iterate over the dataframe
    # We assume metadata_df has 'image_path' relative to INPUT_DIR
    for idx, row in metadata_df.iterrows():
        rel_path = row["image_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        feats = extract_morphometrics(full_path)
        # Add ID for safety, though we rely on index alignment
        feats["id"] = row["id"]
        results.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(results)

    # Set index to match metadata_df for easy concatenation later
    df_features.index = metadata_df.index

    # Ensure float precision as per config (Critical for Metric Floor)
    numeric_cols = [c for c in df_features.columns if c != "id"]
    df_features[numeric_cols] = df_features[numeric_cols].astype(config.FLOAT_PRECISION)

    # 3. Save to cache
    print(f"Saving macro features to {cache_path}")
    df_features.to_parquet(cache_path)

    return df_features
