import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config


def extract_single_image_features(image_path):
    """
    Extracts geometric features from a single binary leaf image.
    Applies polarity correction to ensure the leaf is the foreground.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing calculated geometric features in float64.
    """
    # Initialize default feature vector with zeros
    features = {feat: 0.0 for feat in Config.GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Polarity Correction: Ensure leaf is white (255) and background is black (0)
    # Dataset is Black Leaf on White Background. findContours needs White on Black.
    if Config.INVERT_IMAGES:
        img = cv2.bitwise_not(img)

    # Threshold to ensure strict binary (0 or 255)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Absolute Geometry
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    if area == 0:
        return features

    # Convex Hull
    hull = cv2.convexHull(cnt)
    convex_area = float(cv2.contourArea(hull))
    convex_perimeter = float(cv2.arcLength(hull, True))

    # Bounding Rectangle (Axis Aligned)
    x, y, w, h = cv2.boundingRect(cnt)

    # Fit Ellipse (Requires at least 5 points)
    major_axis = 0.0
    minor_axis = 0.0
    if len(cnt) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        except:
            pass

    # 2. Derived Non-Linear Ratios (Dimensionless)
    # Use float64 for precision

    # Equivalent Diameter: Diameter of a circle with the same area
    equivalent_diameter = np.sqrt(4 * area / np.pi)

    # Aspect Ratio: Width to Height of bounding box
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Extent: Ratio of contour area to bounding rectangle area
    rect_area = float(w * h)
    extent = area / rect_area if rect_area > 0 else 0.0

    # Solidity: Ratio of contour area to its convex hull area
    solidity = area / convex_area if convex_area > 0 else 0.0

    # Convexity: Ratio of convex hull perimeter to contour perimeter
    # (Note: Sometimes defined inversely, here we use Hull_P / Contour_P or Contour_P / Hull_P.
    # Usually Convexity is defined as Convex_Perimeter / Perimeter.
    # A perfectly convex shape has convexity 1. Rough shapes < 1.)
    convexity = convex_perimeter / perimeter if perimeter > 0 else 0.0

    # Compactness: Perimeter^2 / Area (Isoperimetric quotient related)
    compactness = (perimeter**2) / area if area > 0 else 0.0

    # Roundness: 4 * pi * Area / Perimeter^2 (Inverse of compactness scaled)
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

    # Eccentricity: sqrt(1 - (b/a)^2) for ellipse
    eccentricity = 0.0
    if major_axis > 0:
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)

    # Computed values
    computed = {
        "area": area,
        "perimeter": perimeter,
        "convex_perimeter": convex_perimeter,
        "major_axis_length": major_axis,
        "minor_axis_length": minor_axis,
        "equivalent_diameter": equivalent_diameter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "convexity": convexity,
        "compactness": compactness,
        "roundness": roundness,
        "eccentricity": eccentricity,
    }

    # Populate only configured features (Cite Lesson 00140: Feature Parsimony)
    for k, v in computed.items():
        if k in features:
            features[k] = v

    return features


def process_dataset(metadata_path, dataset_name, load_cached_data=True, df_meta=None):
    """
    Loads metadata, extracts geometric features for all images, and returns a DataFrame.
    Implements caching mechanism.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        dataset_name (str): Name identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.
        df_meta (pd.DataFrame, optional): Pre-loaded/filtered metadata DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing IDs and extracted geometric features.
    """
    Config.setup()

    # Prevent cache collision between debug and full runs (Cite debug_lesson_14)
    suffix = "_debug" if Config.DEBUG else ""
    cache_filename = f"X_{dataset_name}{suffix}_geometric.parquet"
    cache_path = Config.get_cache_path(cache_filename)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached geometric features from {cache_path}")
        try:
            df_features = pd.read_parquet(cache_path)
            return df_features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Extracting geometric features for {dataset_name}...")

    if df_meta is None:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Debugging: Sample if configured and no explicit df_meta was passed
        if Config.DEBUG:
            df_meta = df_meta.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    feature_list = []
    ids = []

    for idx, row in df_meta.iterrows():
        # Construct full path
        # Metadata contains relative path like "images/123.jpg"
        # Config.INPUT_DIR is "./input"
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Extract features
        feats = extract_single_image_features(full_path)
        feature_list.append(feats)
        ids.append(row["id"])

    # Create DataFrame
    df_features = pd.DataFrame(feature_list)
    df_features["id"] = ids

    # Ensure 'id' is first, then alphabetical features
    cols = ["id"] + sorted([c for c in df_features.columns if c != "id"])
    df_features = df_features[cols]

    # Enforce float64
    for col in cols:
        if col != "id":
            df_features[col] = df_features[col].astype(Config.DTYPE)

    # 3. Save Cache
    try:
        df_features.to_parquet(cache_path)
        print(f"Saved geometric features to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df_features
