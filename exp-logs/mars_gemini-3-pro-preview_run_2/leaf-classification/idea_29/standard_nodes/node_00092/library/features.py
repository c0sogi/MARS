import os
import cv2
import numpy as np
import pandas as pd
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_29"


def extract_morphometrics(image_path):
    """
    Extracts deterministic morphometric features from a binary leaf image.

    Features extracted:
    - Hu Moments (7 invariants): Capture global shape structure.
    - Aspect Ratio: Width / Height of bounding rect.
    - Solidity: Contour Area / Convex Hull Area.
    - Extent: Contour Area / Bounding Rect Area.
    - Eccentricity: Derived from fitted ellipse.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the extracted features.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Default zero-vector in case of read failure
    default_feats = {f"hu_{i}": 0.0 for i in range(7)}
    default_feats.update(
        {"aspect_ratio": 0.0, "solidity": 0.0, "extent": 0.0, "eccentricity": 0.0}
    )

    if img is None:
        return default_feats

    # Binarize to ensure clean contours (Otsu's method)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return default_feats

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return default_feats

    # 1. Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 2. Geometric Scalars

    # Aspect Ratio & Extent
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Eccentricity
    # Requires fitting an ellipse (needs at least 5 points)
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (x_e, y_e), (MA, ma), angle = cv2.fitEllipse(cnt)
            # MA and ma are lengths of axes.
            # fitEllipse returns (MA, ma) as (minor, major) or vice versa depending on orientation,
            # but usually we treat them as axis lengths.
            a = ma / 2.0
            b = MA / 2.0

            # Ensure a is major axis
            if a < b:
                a, b = b, a

            if a > 0:
                eccentricity = np.sqrt(1 - (b**2 / a**2))
        except:
            # Fallback if ellipse fitting fails numerically
            eccentricity = 0.0

    # Construct result dictionary
    features = {}
    for i in range(7):
        features[f"hu_{i}"] = hu_moments[i]

    features["aspect_ratio"] = aspect_ratio
    features["solidity"] = solidity
    features["extent"] = extent
    features["eccentricity"] = eccentricity

    return features


def generate_macro_features(metadata_path, cache_tag, load_cached_data=True):
    """
    Generates or loads macro-resolution features for a given dataset split.

    Args:
        metadata_path (str): Path to the metadata CSV file (e.g., './metadata/train.csv').
        cache_tag (str): Identifier for the cache file (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and the extracted features.
    """
    set_seed(42)  # Ensure deterministic behavior

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"macro_features_{cache_tag}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached macro features from {cache_file}...")
        return pd.read_parquet(cache_file)

    # 2. Compute Features
    print(f"Computing macro features for {cache_tag} (Source: {metadata_path})...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    feature_list = []

    # Iterate over images
    for idx, row in df_meta.iterrows():
        # Construct full path. Metadata 'image_path' is relative to input dir (e.g. 'images/1.jpg')
        full_path = os.path.join(INPUT_DIR, row["image_path"])

        feats = extract_morphometrics(full_path)
        feats["id"] = row["id"]  # Keep ID for merging
        feature_list.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(feature_list)

    # Reorder columns to put ID first
    cols = ["id"] + [c for c in df_features.columns if c != "id"]
    df_features = df_features[cols]

    # Enforce float64 precision for all feature columns
    for col in df_features.columns:
        if col != "id":
            df_features[col] = df_features[col].astype(np.float64)

    # 3. Save Cache
    print(f"Saving macro features to {cache_file}...")
    df_features.to_parquet(cache_file, index=False)

    return df_features
