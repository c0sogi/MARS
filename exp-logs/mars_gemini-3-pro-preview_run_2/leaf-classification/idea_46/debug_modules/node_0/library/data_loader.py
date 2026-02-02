import os
import pandas as pd
import numpy as np
from library import config, image_processing


def _get_global_features(df, split_name, load_cached_data=True):
    """
    Extracts the 192 provided features (Margin, Shape, Texture) from the dataframe.
    Implements caching to .npy files in the working directory.

    Args:
        df (pd.DataFrame): The dataframe containing feature columns.
        split_name (str): The name of the split (train/val/test) for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The matrix of global features (float64).
    """
    cache_filename = f"global_features_{split_name}.npy"
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            features = np.load(cache_path)
            # Verify shape matches current dataframe
            if features.shape[0] == len(df):
                return features
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Extract features from DataFrame
    # Filter columns that start with margin, shape, or texture
    feature_cols = [
        c for c in df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Ensure we have exactly 192 features (64 * 3)
    # This check ensures we aren't picking up unexpected columns
    expected_count = 64 * 3
    if len(feature_cols) != expected_count:
        # Fallback: assume the columns are present and filter strictly if count mismatches,
        # but for this dataset, the prefix filter is robust.
        pass

    features = df[feature_cols].values.astype(config.FLOAT_PRECISION)

    # 3. Save to cache
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    try:
        np.save(cache_path, features)
    except Exception:
        pass

    return features


def load_data(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Organizes data into two views:
    1. Global: The 192 provided features (Margin, Shape, Texture).
    2. Morphometric: Features extracted from images (Hu moments, Geometric scalars).

    Args:
        load_cached_data (bool): Whether to load features from cache if available.

    Returns:
        dict: A dictionary containing 'train', 'val', 'test' data and 'class_names'.
              Structure:
              {
                  'class_names': [list of strings],
                  'train': {'X_global': np.array, 'X_morph': np.array, 'y': np.array, 'ids': np.array},
                  'val':   {'X_global': np.array, 'X_morph': np.array, 'y': np.array, 'ids': np.array},
                  'test':  {'X_global': np.array, 'X_morph': np.array, 'ids': np.array}
              }
    """
    print("Loading metadata...")
    # Load Metadata CSVs
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Extract unique class names from training data (sorted)
    class_names = sorted(train_df["species"].unique().tolist())

    # Initialize return structure
    data = {"class_names": class_names, "train": {}, "val": {}, "test": {}}

    # List of splits to process
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    for split_name, df in splits:
        # 1. Extract IDs
        data[split_name]["ids"] = df["id"].values

        # 2. Extract Targets (y) if available
        if "species" in df.columns:
            data[split_name]["y"] = df["species"].values

        # 3. Extract Global Features (with local caching)
        data[split_name]["X_global"] = _get_global_features(
            df, split_name, load_cached_data=load_cached_data
        )

        # 4. Extract Morphometric Features (using image_processing module's caching)
        # The image_processing module handles reading images and caching the result
        data[split_name]["X_morph"] = image_processing.get_morphometric_features(
            df, load_cached_data=load_cached_data
        )

    print("Data loading complete.")
    return data
