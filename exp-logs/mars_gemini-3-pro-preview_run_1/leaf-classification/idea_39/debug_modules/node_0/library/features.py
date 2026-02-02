import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure
import hashlib

from library.config import GEOMETRIC_FEATURES, CONFIG_HASH_DICT, CACHE_DIR
from library.utils import generate_config_hash


def extract_geometric_properties(image_path):
    """
    Extracts a suite of absolute and relative geometric descriptors from a binary leaf image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary mapping feature names to calculated float64 values.
    """
    # Initialize zero-filled dictionary
    features = {feat: 0.0 for feat in GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Invert: Leaf (black=0) becomes 1, Background (white=255) becomes 0
    # Using threshold 128 to separate
    _, binary = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV)

    # Label connected components
    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)

    if not regions:
        return features

    # Assume the leaf is the largest connected component by area
    leaf = max(regions, key=lambda r: r.area)

    # --- Absolute Scale Features ---
    area = float(leaf.area)
    perimeter = float(leaf.perimeter)
    minr, minc, maxr, maxc = leaf.bbox
    bbox_width = float(maxc - minc)
    bbox_height = float(maxr - minr)
    major_axis = float(leaf.major_axis_length)
    minor_axis = float(leaf.minor_axis_length)
    convex_area = float(leaf.convex_area)
    equiv_diameter = float(leaf.equivalent_diameter)

    # --- Relative Shape Features ---
    # Aspect Ratio (Width / Height)
    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0.0

    # Solidity (Area / Convex Area)
    solidity = float(leaf.solidity)

    # Extent (Area / Bounding Box Area)
    extent = float(leaf.extent)

    # Eccentricity
    eccentricity = float(leaf.eccentricity)

    # Roundness: 4 * pi * Area / Perimeter^2
    # Measures how close the shape is to a perfect circle (1.0)
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

    # Compactness: Perimeter^2 / Area
    # Inverse related to roundness, often used in shape analysis
    compactness = (perimeter**2) / area if area > 0 else 0.0

    # Populate dictionary
    features["Area"] = area
    features["Perimeter"] = perimeter
    features["Bounding_Width"] = bbox_width
    features["Bounding_Height"] = bbox_height
    features["Major_Axis_Length"] = major_axis
    features["Minor_Axis_Length"] = minor_axis
    features["Convex_Area"] = convex_area
    features["Equivalent_Diameter"] = equiv_diameter
    features["Aspect_Ratio"] = aspect_ratio
    features["Solidity"] = solidity
    features["Extent"] = extent
    features["Roundness"] = roundness
    features["Compactness"] = compactness
    features["Eccentricity"] = eccentricity

    return features


def batch_extract_features(metadata_df, input_dir, load_cached_data=True):
    """
    Iterates over a metadata DataFrame, extracts geometric features for each image,
    and returns a DataFrame of features. Implements strict caching.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
        input_dir (str): Base directory where images are stored.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and all geometric feature columns.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Generate Cache Key
    # 1. Hash of the Data IDs to ensure we are processing the exact same set of images
    ids_hash = hashlib.md5(
        pd.util.hash_pandas_object(metadata_df["id"]).values
    ).hexdigest()

    # 2. Hash of the Configuration to ensure feature logic hasn't changed
    config_hash = generate_config_hash(CONFIG_HASH_DICT)

    cache_filename = f"features_{ids_hash}_{config_hash}.parquet"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached geometric features from {cache_path}")
        try:
            df_features = pd.read_parquet(cache_path)
            # Verify length matches
            if len(df_features) == len(metadata_df):
                return df_features
            else:
                print("Cache length mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # Compute Features
    print("Computing geometric features...")
    feature_list = []

    # Iterate over metadata
    for _, row in metadata_df.iterrows():
        # Construct full path
        # Metadata file_path is relative (e.g., 'images/123.jpg')
        full_path = os.path.join(input_dir, row["file_path"])

        # Extract
        feats = extract_geometric_properties(full_path)

        # Add ID for merging later
        feats["id"] = row["id"]
        feature_list.append(feats)

    # Create DataFrame
    # Ensure column order matches GEOMETRIC_FEATURES + id
    cols = ["id"] + GEOMETRIC_FEATURES
    df_features = pd.DataFrame(feature_list)

    # Reorder columns to ensure consistency
    df_features = df_features[cols]

    # Save to Cache
    print(f"Saving features to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
