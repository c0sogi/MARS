import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    INPUT_DIR,
    SEED,
    set_seed,
)

# Ensure reproducibility
set_seed(SEED)


def extract_single_image_features(image_rel_path):
    """
    Extracts the 7-scalar geometric basis from a single binary leaf image.

    Features:
    1. Area (Absolute Mass)
    2. Major_Axis_Length (Absolute Length)
    3. Mean_Thickness (Absolute Density)
    4. Eccentricity (Elongation)
    5. Solidity (Roughness)
    6. Extent (Rectangularity)
    7. Aspect_Ratio (Orientation)
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Default return vector in case of failure
    default_features = {
        "geo_Area": 0.0,
        "geo_Major_Axis_Length": 0.0,
        "geo_Mean_Thickness": 0.0,
        "geo_Eccentricity": 0.0,
        "geo_Solidity": 0.0,
        "geo_Extent": 0.0,
        "geo_Aspect_Ratio": 0.0,
    }

    if not os.path.exists(full_path):
        return default_features

    # Load image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return default_features

    # Polarity Correction: Leaf is black (0), Background is white (255).
    # Invert so leaf is foreground (255) for contour detection.
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return default_features

    # Implicit Denoising: Select largest contour by Area
    cnt = max(contours, key=cv2.contourArea)

    # 1. Area
    area = cv2.contourArea(cnt)
    if area == 0:
        return default_features

    # 2. Major Axis Length & 4. Eccentricity (via fitEllipse)
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # OpenCV returns (MA, ma) as (width, height) of rotated rect.
            # Major axis is the larger of the two.
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)

            # Eccentricity: e = sqrt(1 - (b/a)^2)
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
        except:
            major_axis = 0.0
            eccentricity = 0.0
    else:
        # Fallback for very small contours
        rect = cv2.minAreaRect(cnt)
        major_axis = max(rect[1])
        eccentricity = 0.0

    # 3. Mean Thickness (Euclidean Distance Transform)
    # Compute distance from every foreground pixel to the nearest zero pixel
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    # Mean of distances for pixels inside the leaf
    # We use binary_img > 0 mask
    if np.count_nonzero(binary_img) > 0:
        mean_thickness = np.mean(dist_transform[binary_img > 0])
    else:
        mean_thickness = 0.0

    # 5. Solidity (Area / ConvexHullArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # 6. Extent (Area / BoundingRectArea)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # 7. Aspect Ratio (BoundingWidth / BoundingHeight)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    return {
        "geo_Area": float(area),
        "geo_Major_Axis_Length": float(major_axis),
        "geo_Mean_Thickness": float(mean_thickness),
        "geo_Eccentricity": float(eccentricity),
        "geo_Solidity": float(solidity),
        "geo_Extent": float(extent),
        "geo_Aspect_Ratio": float(aspect_ratio),
    }


def process_dataset(meta_path, dataset_name, load_cached_data):
    """
    Processes a specific dataset (train/val/test), extracts features,
    and handles caching.
    """
    cache_file = os.path.join(CACHE_DIR, f"{dataset_name}_geometric.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(
            f"Loading cached geometric features for {dataset_name} from {cache_file}..."
        )
        return pd.read_parquet(cache_file)

    # 2. Process from Scratch
    print(f"Extracting geometric features for {dataset_name}...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    features_list = []
    ids = []

    for _, row in df_meta.iterrows():
        img_id = row["id"]
        file_path = row["file_path"]

        feats = extract_single_image_features(file_path)
        features_list.append(feats)
        ids.append(img_id)

    # Create DataFrame
    df_features = pd.DataFrame(features_list)
    df_features["id"] = ids

    # Reorder to put id first
    cols = ["id"] + [c for c in df_features.columns if c != "id"]
    df_features = df_features[cols]

    # 3. Save Cache
    print(f"Saving {dataset_name} features to {cache_file}...")
    df_features.to_parquet(cache_file, index=False)

    return df_features


def get_geometric_features(load_cached_data=True):
    """
    Main entry point to get geometric features for all splits.
    Returns: (train_df, val_df, test_df)
    """
    train_feat = process_dataset(TRAIN_META_PATH, "train", load_cached_data)
    val_feat = process_dataset(VAL_META_PATH, "val", load_cached_data)
    test_feat = process_dataset(TEST_META_PATH, "test", load_cached_data)

    return train_feat, val_feat, test_feat
