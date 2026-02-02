import os
import cv2
import numpy as np
import pandas as pd
from library.utils import load_dataset

# Directory for caching processed/loaded data
CACHE_DIR = "./working/idea_69"


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata dataframe for a specific split using the provided utility.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe containing features and image paths.
    """
    return load_dataset(split)


def load_raw_images(image_paths: list, base_dir: str = "./input") -> list:
    """
    Loads raw binary images from disk given a list of relative paths.

    Args:
        image_paths (list): List of relative paths to images (e.g., 'images/1.jpg').
        base_dir (str): Base directory where input data is located.

    Returns:
        list: List of numpy arrays representing the images.
    """
    images = []
    for rel_path in image_paths:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Read image in unchanged mode to preserve binary/grayscale nature
        # The dataset consists of binary black leaves on white backgrounds
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to read image: {full_path}")

        images.append(img)
    return images


def get_data_split(split: str, load_cached_data: bool = True, max_samples: int = None):
    """
    Retrieves the metadata and raw images for a specific split.
    Implements caching for the raw images to speed up subsequent runs.

    Logic:
        1. Loads full metadata.
        2. Checks for cached full image set.
        3. If not cached, loads from disk and saves to cache (npz format).
        4. If max_samples is provided, slices the dataframes and image lists accordingly.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): Limit the number of samples (for debugging).

    Returns:
        tuple: (pd.DataFrame, list of np.ndarray) -> (metadata, images)
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Load Full Metadata
    df_full = load_metadata(split)

    # 2. Load Full Images (Cache handling)
    cache_file = os.path.join(CACHE_DIR, f"images_{split}.npz")
    images_full = None

    if load_cached_data and os.path.exists(cache_file):
        try:
            # Load from .npz file
            # np.savez stores arrays as arr_0, arr_1, etc.
            # We must ensure we read them back in the correct order.
            with np.load(cache_file) as data:
                # Sort keys numerically by the suffix (arr_0 -> 0)
                keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
                images_full = [data[k] for k in keys]
        except Exception:
            # If cache loading fails, fall back to raw loading
            images_full = None

    if images_full is None:
        # Load from disk
        image_paths = df_full["image_path"].tolist()
        images_full = load_raw_images(image_paths)

        # Save to cache
        # We use *images_full to pass them as separate arguments,
        # which np.savez saves as arr_0, arr_1...
        try:
            np.savez(cache_file, *images_full)
        except Exception:
            pass

    # 3. Apply Limit if requested
    # We slice AFTER loading/caching to ensure cache consistency (cache always holds full dataset)
    if max_samples is not None and max_samples < len(df_full):
        df_subset = df_full.iloc[:max_samples].reset_index(drop=True)
        images_subset = images_full[:max_samples]
        return df_subset, images_subset

    return df_full, images_full
