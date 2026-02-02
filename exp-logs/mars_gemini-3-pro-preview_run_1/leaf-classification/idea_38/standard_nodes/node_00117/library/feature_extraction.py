import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    GEOMETRIC_FEATURES,
    SEED,
)


def compute_hu_moments(contour):
    """
    Computes the 7 invariant Hu Moments for a given contour and applies
    the Log-Modulus Transform: sign(x) * log(|x|).
    """
    try:
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log-Modulus Transform to handle high dynamic range
        # Adding a small epsilon to avoid log(0)
        epsilon = 1e-20
        transformed_hu = []
        for h in hu_moments:
            val = np.sign(h) * np.log(np.abs(h) + epsilon)
            transformed_hu.append(val)

        return transformed_hu
    except Exception:
        # Return zeros if computation fails (e.g., degenerate contour)
        return [0.0] * 7


def extract_geometric_features(image_path):
    """
    Extracts geometric features from a binary leaf image.

    Features:
    - Aspect Ratio
    - Solidity
    - Extent
    - Eccentricity
    - Hu Moments (1-7)
    """
    # Initialize default values
    features = {feat: 0.0 for feat in GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Find contours
    # The images are binary black leaves on white background or vice versa.
    # We usually invert to get white leaf on black background for contour finding if needed.
    # Assuming standard binary leaf images where leaf is foreground or distinct.
    # We use simple thresholding to ensure binary nature.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Try without inversion if no contours found (in case leaf is black on white)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # 1. Aspect Ratio & Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = area / rect_area if rect_area > 0 else 0.0

    features["aspect_ratio"] = aspect_ratio
    features["extent"] = extent

    # 2. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    features["solidity"] = solidity

    # 3. Eccentricity
    # Fit ellipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are lengths of the axes (diameters)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2) where a is semi-major, b is semi-minor
                # (b/a)^2 is equivalent to (minor_axis/major_axis)^2
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
        except Exception:
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    features["eccentricity"] = eccentricity

    # 4. Hu Moments
    hu_moments = compute_hu_moments(cnt)
    for i in range(7):
        features[f"hu_moment_{i+1}"] = hu_moments[i]

    return features


def process_dataset(metadata_path, dataset_name, load_cached_data=True, limit=None):
    """
    Process a specific dataset (train/val/test), extracting geometric features
    for all images referenced in the metadata.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to load from cache if available.
        limit (int, optional): Limit number of rows for debugging.

    Returns:
        pd.DataFrame: DataFrame containing extracted features, indexed by 'id'.
    """
    cache_file = os.path.join(WORKING_DIR, f"{dataset_name}_geometric_features.parquet")

    # Try loading from cache
    if load_cached_data and limit is None and os.path.exists(cache_file):
        print(
            f"Loading cached geometric features for {dataset_name} from {cache_file}..."
        )
        return pd.read_parquet(cache_file)

    print(f"Extracting geometric features for {dataset_name}...")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)

    if limit is not None:
        df_meta = df_meta.head(limit)
        print(f"  [DEBUG] Limiting to {limit} samples.")

    # Prepare list to collect features
    results = []

    for _, row in df_meta.iterrows():
        image_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        feats = extract_geometric_features(full_path)
        feats["id"] = image_id
        results.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(results)

    # Set ID as index to facilitate merging later
    if not df_features.empty:
        df_features.set_index("id", inplace=True)

    # Save to cache (only if not debugging with a limit)
    if limit is None:
        print(f"Saving {dataset_name} geometric features to cache...")
        df_features.to_parquet(cache_file)

    return df_features


def get_geometric_datasets(load_cached_data=True, debug_limit=None):
    """
    Main entry point to get geometric features for all splits.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_limit (int, optional): Limit samples for debugging.

    Returns:
        tuple: (df_train_feats, df_val_feats, df_test_feats)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    df_train = process_dataset(TRAIN_CSV, "train", load_cached_data, debug_limit)
    df_val = process_dataset(VAL_CSV, "val", load_cached_data, debug_limit)
    df_test = process_dataset(TEST_CSV, "test", load_cached_data, debug_limit)

    return df_train, df_val, df_test
