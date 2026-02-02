import os
import numpy as np
import pandas as pd
import logging
from library.config import Config
from library.utils import setup_logging
from library.feature_extraction import extract_features

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def load_dataset(split_name: str, load_cached_data: bool = True):
    """
    Loads and aligns the complete dataset for a given split.

    Aggregates:
    1. Global Geometry Features (DINOv2)
    2. Local Texture Features (ConvNeXt)
    3. Handcrafted Tabular Features (Margin, Shape, Texture)
    4. Target Labels (Species) - if available

    Ensures that all arrays are strictly aligned by Image ID.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load processed tabular data from cache.

    Returns:
        dict: A dictionary containing:
            - 'dino': np.ndarray of shape (N, D_dino)
            - 'conv': np.ndarray of shape (N, D_conv)
            - 'tab':  np.ndarray of shape (N, 192)
            - 'y':    np.ndarray of shape (N,) or None
            - 'ids':  np.ndarray of shape (N,)
    """
    # 1. Resolve Metadata Path
    if split_name == "train":
        meta_path = Config.TRAIN_META_PATH
    elif split_name == "val":
        meta_path = Config.VAL_META_PATH
    elif split_name == "test":
        meta_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    # 2. Load Visual Features
    # This function handles image processing, inference, and its own caching.
    # It returns the IDs of images that were successfully processed.
    logger.info(f"Retrieving visual features for {split_name}...")
    dino_feats, conv_feats, visual_ids = extract_features(
        meta_path, split_name, load_cached_data=load_cached_data
    )

    if len(visual_ids) == 0:
        logger.warning(
            f"No features returned for {split_name}. Returning empty dataset."
        )
        return {
            "dino": np.array([]),
            "conv": np.array([]),
            "tab": np.array([]),
            "y": np.array([]),
            "ids": np.array([]),
        }

    # 3. Load/Align Tabular Data and Labels
    # We cache the aligned tabular data to avoid re-parsing CSVs and re-indexing.
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    tab_cache_path = os.path.join(cache_dir, f"{split_name}_tab_features.npy")
    y_cache_path = os.path.join(cache_dir, f"{split_name}_labels.npy")
    ids_cache_path = os.path.join(cache_dir, f"{split_name}_aligned_ids.npy")

    # Check if valid cache exists
    cache_valid = False
    if (
        load_cached_data
        and os.path.exists(tab_cache_path)
        and os.path.exists(ids_cache_path)
    ):
        try:
            # Verify that the cached tabular data corresponds to the exact same IDs
            # as the visual features we just loaded.
            cached_ids = np.load(ids_cache_path)
            if np.array_equal(visual_ids, cached_ids):
                # Check label cache if needed
                if split_name != "test":
                    if os.path.exists(y_cache_path):
                        cache_valid = True
                else:
                    cache_valid = True
        except Exception as e:
            logger.warning(f"Failed to verify tabular cache: {e}")

    if cache_valid:
        logger.info(f"Loading cached tabular data for {split_name}...")
        tab_feats = np.load(tab_cache_path)
        y = np.load(y_cache_path, allow_pickle=True) if split_name != "test" else None

        return {
            "dino": dino_feats,
            "conv": conv_feats,
            "tab": tab_feats,
            "y": y,
            "ids": visual_ids,
        }

    # 4. Process from Source CSV
    logger.info(f"Processing tabular data for {split_name} from {meta_path}...")
    df = pd.read_csv(meta_path)

    # Ensure IDs are integers for matching
    df["id"] = df["id"].astype(int)

    # Set index to ID for efficient alignment
    df.set_index("id", inplace=True)

    # Select only the rows corresponding to the successfully processed visual features
    # This aligns the tabular data order to the visual data order
    try:
        df_aligned = df.loc[visual_ids]
    except KeyError as e:
        logger.error(f"One or more Visual IDs not found in Metadata CSV: {e}")
        raise

    # Extract Tabular Features
    # Construct column list: margin_1..64, shape_1..64, texture_1..64
    feature_cols = []
    for prefix in Config.TABULAR_PREFIXES:
        for i in range(1, 65):
            feature_cols.append(f"{prefix}{i}")

    # Verify columns exist
    missing = [c for c in feature_cols if c not in df_aligned.columns]
    if missing:
        raise ValueError(f"Missing tabular columns in metadata: {missing[:5]}...")

    tab_feats = df_aligned[feature_cols].values.astype(np.float32)

    # Extract Labels (if not test set)
    y = None
    if split_name != "test":
        if "species" not in df_aligned.columns:
            raise ValueError(f"Column 'species' missing in {split_name} metadata.")
        y = df_aligned["species"].values

    # 5. Save to Cache
    np.save(tab_cache_path, tab_feats)
    np.save(ids_cache_path, visual_ids)
    if y is not None:
        np.save(y_cache_path, y)

    logger.info(f"Tabular data processed and cached for {split_name}.")

    return {
        "dino": dino_feats,
        "conv": conv_feats,
        "tab": tab_feats,
        "y": y,
        "ids": visual_ids,
    }
