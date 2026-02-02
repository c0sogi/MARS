import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, CACHE_DIR, VISUAL_FEATURES, PRECISION_TYPE, SEED


def extract_single_image_features(image_path):
    """
    Extracts geometric features from a single binary leaf image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the extracted features with float64 precision.
    """
    # Initialize default values (zeros) in case of failure
    features = {k: 0.0 for k in VISUAL_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Threshold to binary (assuming white background or black background,
    # dataset desc says "binary black leaves against white backgrounds")
    # We need the leaf to be white (255) for contour detection usually,
    # or we detect on the black part.
    # If leaves are black (0) and background is white (255):
    # Inverting so leaf is foreground (255)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Absolute Scale
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area == 0:
        return features

    features["area"] = float(area)
    features["perimeter"] = float(perimeter)
    features["equivalent_diameter"] = np.sqrt(4 * area / np.pi)

    # Fit Ellipse (requires at least 5 points)
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            features["major_axis_length"] = float(max(MA, ma))
            features["minor_axis_length"] = float(min(MA, ma))
        except:
            pass  # Keep defaults

    # Calculate Eccentricity
    if features["major_axis_length"] > 0:
        # e = sqrt(1 - (b^2/a^2))
        ratio_sq = (features["minor_axis_length"] ** 2) / (
            features["major_axis_length"] ** 2
        )
        features["eccentricity"] = np.sqrt(max(0.0, 1.0 - ratio_sq))

    # Calculate Roundness
    if features["perimeter"] > 0:
        # R = 4 * pi * Area / Perimeter^2
        features["roundness"] = (4 * np.pi * features["area"]) / (
            features["perimeter"] ** 2
        )

    # 2. Axis-Aligned Envelope (Bounding Rect)
    x, y, w, h = cv2.boundingRect(cnt)
    features["aspect_ratio"] = float(w) / float(h) if h > 0 else 0.0
    features["extent"] = float(area) / (float(w) * float(h)) if (w * h) > 0 else 0.0

    # 3. Intrinsic Envelope (Min Area Rect)
    rect = cv2.minAreaRect(cnt)
    (center), (dim1, dim2), angle = rect
    # Sort dimensions to be rotation invariant (width <= height)
    min_box_w = min(dim1, dim2)
    min_box_h = max(dim1, dim2)

    features["min_box_aspect_ratio"] = (
        float(min_box_w) / float(min_box_h) if min_box_h > 0 else 0.0
    )

    # Solidity & Topology
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    features["solidity"] = float(area) / float(hull_area) if hull_area > 0 else 0.0
    features["convexity"] = (
        float(hull_perimeter) / float(perimeter) if perimeter > 0 else 0.0
    )

    # 4. Internal Morphology (Distance Transform)
    # Distance transform calculates distance to nearest zero pixel.
    # Since we inverted, leaf is 255. We want distance from inside leaf to background (0).
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
    features["inscribed_circle_radius"] = float(max_val)

    return features


def extract_geometric_features(metadata_df, split_name, load_cached_data=True):
    """
    Extracts geometric features for a dataset split, with caching.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path' and 'id'.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing only the extracted geometric features,
                      indexed to match the input dataframe.
    """
    cache_file = os.path.join(CACHE_DIR, f"geometric_features_{split_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(
            f"Loading cached geometric features for {split_name} from {cache_file}..."
        )
        try:
            df_features = pd.read_parquet(cache_file)
            # Verify length matches
            if len(df_features) == len(metadata_df):
                return df_features
            else:
                print(
                    f"Cache length mismatch ({len(df_features)} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(
        f"Extracting geometric features for {split_name} ({len(metadata_df)} images)..."
    )

    extracted_rows = []

    # Iterate over metadata
    for idx, row in metadata_df.iterrows():
        # Construct full path
        # metadata 'file_path' is relative, e.g., 'images/1.jpg'
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Extract features
        feats = extract_single_image_features(full_path)
        extracted_rows.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(extracted_rows)

    # Ensure columns are in the correct order (alphanumeric sort or config order)
    # Using config order to match VISUAL_FEATURES list
    df_features = df_features[VISUAL_FEATURES]

    # Enforce Precision
    df_features = df_features.astype(PRECISION_TYPE)

    # Save to cache
    print(f"Saving geometric features to {cache_file}...")
    df_features.to_parquet(cache_file)

    return df_features
