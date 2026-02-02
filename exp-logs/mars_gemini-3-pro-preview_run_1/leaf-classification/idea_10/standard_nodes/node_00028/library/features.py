import os
import cv2
import numpy as np
import pandas as pd
from library import config


def extract_geometric_props(image_path):
    """
    Extracts geometric properties from a binary leaf image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: A dictionary containing 'area' and 'aspect_ratio'.
              Returns defaults if image cannot be processed.
    """
    # Defaults
    props = {"area": 0.0, "aspect_ratio": 0.0}

    if not os.path.exists(image_path):
        return props

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return props

    # The dataset description says "binary black leaves against white backgrounds".
    # For contour detection and area counting, we usually want White Foreground, Black Background.
    # So we invert the image.
    # Leaf pixels (originally 0/black) become 255/white.
    # Background pixels (originally 255/white) become 0/black.
    img_inv = cv2.bitwise_not(img)

    # 1. Calculate Area (number of foreground pixels)
    area = cv2.countNonZero(img_inv)
    props["area"] = float(area)

    # 2. Calculate Aspect Ratio via Bounding Rectangle
    # Find contours on the inverted image
    contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the leaf
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if h > 0:
            props["aspect_ratio"] = float(w) / float(h)
        else:
            props["aspect_ratio"] = 0.0

    return props


def augment_dataframe(df):
    """
    Augments the dataframe with geometric features extracted from images.

    Args:
        df (pd.DataFrame): DataFrame containing a 'file_path' column.

    Returns:
        pd.DataFrame: DataFrame with added 'area' and 'aspect_ratio' columns.
    """
    areas = []
    aspect_ratios = []

    # Iterate over rows to process images
    # Using a loop is generally acceptable here given N is small (~1000 total images)
    for _, row in df.iterrows():
        # Construct full path. Metadata contains relative path e.g., "images/1.jpg"
        full_path = os.path.join(config.INPUT_DIR, row["file_path"])

        props = extract_geometric_props(full_path)
        areas.append(props["area"])
        aspect_ratios.append(props["aspect_ratio"])

    # Assign new columns
    df_aug = df.copy()
    df_aug["area"] = areas
    df_aug["aspect_ratio"] = aspect_ratios

    return df_aug


def get_augmented_dataset(split_name, load_cached_data=True):
    """
    Loads the dataset for a specific split, augments it with geometric features,
    and handles caching.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The augmented dataset.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_path = os.path.join(config.WORKING_DIR, f"{split_name}_augmented.parquet")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Determine source path based on split name
    if split_name == "train":
        source_path = config.TRAIN_DATA_PATH
    elif split_name == "val":
        source_path = config.VAL_DATA_PATH
    elif split_name == "test":
        source_path = config.TEST_DATA_PATH
    else:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'."
        )

    print(f"Processing {split_name} data from {source_path}...")

    # Load metadata
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Augment data
    df_augmented = augment_dataframe(df)

    # Save to cache
    print(f"Saving augmented {split_name} data to {cache_path}...")
    df_augmented.to_parquet(cache_path, index=False)

    return df_augmented
