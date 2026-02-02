import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    GEOMETRIC_FEATURES,
    NUMERIC_DTYPE,
    IMAGES_DIR_NAME,
    ID_COL,
    IMAGE_PATH_COL,
)


def extract_visual_features_for_row(row):
    """
    Extracts geometric features for a single image row from metadata.

    Args:
        row (pd.Series): A row from the metadata dataframe containing 'file_path'.

    Returns:
        dict: A dictionary of extracted features.
    """
    # Construct full path
    # Metadata file_path is relative, e.g., "images/123.jpg"
    full_path = os.path.join(INPUT_DIR, row[IMAGE_PATH_COL])

    # Initialize default values (zeros) in case of failure/empty image
    features = {k: 0.0 for k in GEOMETRIC_FEATURES}

    if not os.path.exists(full_path):
        return features

    # Load image in grayscale
    # Dataset: Black leaves on white background
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Invert image: make leaf white (255), background black (0)
    # This is required for correct contour finding in OpenCV
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf (handling noise)
    cnt = max(contours, key=cv2.contourArea)

    # --- 1. Absolute Scale (Size) ---
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    # Convex Hull
    hull = cv2.convexHull(cnt)
    convex_area = float(cv2.contourArea(hull))
    convex_perimeter = float(cv2.arcLength(hull, True))

    # Ellipse fit (requires at least 5 points)
    major_axis = 0.0
    minor_axis = 0.0
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        except:
            pass

    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    features["Area"] = area
    features["Perimeter"] = perimeter
    features["Convex_Perimeter"] = convex_perimeter
    features["Major_Axis_Length"] = major_axis
    features["Minor_Axis_Length"] = minor_axis
    features["Equivalent_Diameter"] = equiv_diameter

    # --- 2. Scanner Frame (AABB) ---
    x, y, w, h = cv2.boundingRect(cnt)
    aabb_area = float(w * h)

    features["AABB_Width"] = float(w)
    features["AABB_Height"] = float(h)
    features["AABB_Aspect_Ratio"] = float(w) / h if h > 0 else 0.0
    features["AABB_Extent"] = area / aabb_area if aabb_area > 0 else 0.0

    # --- 3. Object Frame (MinBox) ---
    # minAreaRect returns ((cx, cy), (w, h), angle)
    # The order of w, h is dependent on angle. To ensure rotation invariance for
    # our "Width" and "Height" features, we sort them.
    rect = cv2.minAreaRect(cnt)
    (rect_w, rect_h) = rect[1]

    # Define Width as min dimension, Height as max dimension
    mb_width = min(rect_w, rect_h)
    mb_height = max(rect_w, rect_h)
    mb_area = mb_width * mb_height

    features["MinBox_Width"] = mb_width
    features["MinBox_Height"] = mb_height
    features["MinBox_Aspect_Ratio"] = mb_width / mb_height if mb_height > 0 else 0.0
    features["MinBox_Extent"] = area / mb_area if mb_area > 0 else 0.0

    # --- 4. Internal Morphology (Distance Transform) ---
    # Compute Distance Transform on the binary mask
    # dist_map values are distance to nearest zero pixel (background)
    dist_map = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    inscribed_radius = float(np.max(dist_map)) if dist_map.size > 0 else 0.0

    features["Inscribed_Circle_Radius"] = inscribed_radius

    # --- 5. Explicit Invariants ---
    # Solidity: Area / Convex Area
    features["Solidity"] = area / convex_area if convex_area > 0 else 0.0

    # Convexity: Convex Perimeter / Perimeter
    features["Convexity"] = convex_perimeter / perimeter if perimeter > 0 else 0.0

    # Roundness: 4 * pi * Area / Perimeter^2
    # (1.0 for circle, lower for irregular shapes)
    features["Roundness"] = (
        (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    )

    # Compactness: Perimeter^2 / Area (Inverse of Roundness approx)
    features["Compactness"] = (perimeter**2) / area if area > 0 else 0.0

    return features


def process_dataset_images(metadata_path, load_cached_data=True):
    """
    Processes all images listed in the metadata file and extracts geometric features.
    Handles caching to avoid re-computation.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and extracted geometric features.
    """
    # Determine cache filename based on metadata filename
    # e.g. "train.csv" -> "features_train.parquet"
    base_name = os.path.basename(metadata_path).replace(".csv", "")
    cache_file = os.path.join(CACHE_DIR, f"features_{base_name}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached geometric features from {cache_file}...")
        try:
            df_features = pd.read_parquet(cache_file)
            return df_features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Extracting geometric features for {base_name}...")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Check required columns
    if ID_COL not in df_meta.columns or IMAGE_PATH_COL not in df_meta.columns:
        raise ValueError(f"Metadata must contain '{ID_COL}' and '{IMAGE_PATH_COL}'")

    # Extract features for each row
    extracted_data = []

    # Iterate with index to keep track
    # Using a simple loop. For very large datasets, multiprocessing would be better,
    # but N < 2000 here, so single thread is fine and safer for determinism.
    for _, row in df_meta.iterrows():
        feats = extract_visual_features_for_row(row)
        # Add ID to the dict for merging later
        feats[ID_COL] = row[ID_COL]
        extracted_data.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(extracted_data)

    # Ensure column order matches config and ID is first
    cols = [ID_COL] + GEOMETRIC_FEATURES
    df_features = df_features[cols]

    # Enforce float64 for feature columns
    for col in GEOMETRIC_FEATURES:
        df_features[col] = df_features[col].astype(NUMERIC_DTYPE)

    # 3. Save Cache
    print(f"Saving geometric features to {cache_file}...")
    df_features.to_parquet(cache_file, index=False)

    return df_features
