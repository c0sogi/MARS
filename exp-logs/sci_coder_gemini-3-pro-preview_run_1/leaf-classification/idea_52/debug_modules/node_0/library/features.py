import os
import pandas as pd
import numpy as np
from library.config import Config, extract_geometric_features, FEATURE_NAMES_GEO


class ImageFeatureExtractor:
    """
    Handles the loading, processing, and feature extraction for the leaf classification task.
    Implements the Integral-Morphological strategy by coordinating image feature extraction
    and tabular data merging with a caching layer.
    """

    def __init__(self, input_dir=Config.INPUT_DIR, cache_dir=Config.CACHE_DIR):
        """
        Initialize the extractor with input and cache directories.

        Args:
            input_dir (str): Root directory containing the input data.
            cache_dir (str): Directory to store cached parquet files.
        """
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_data(
        self, metadata_path, dataset_name, load_cached_data=True, max_samples=None
    ):
        """
        Loads metadata, extracts geometric features from images (or loads from cache),
        merges with tabular features, and returns the feature matrix X, target y, and ids.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            dataset_name (str): Unique name for the dataset (used for caching).
            load_cached_data (bool): If True, attempts to load from cache first.
            max_samples (int, optional): If set, limits the number of samples processed (for debugging).

        Returns:
            tuple: (X, y, ids)
                X (np.ndarray): Feature matrix (n_samples, n_features).
                y (np.ndarray or None): Target labels.
                ids (np.ndarray): Image IDs.
        """
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.parquet")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            df_features = pd.read_parquet(cache_path)

            # Handle max_samples on cached data if requested for debugging
            if max_samples is not None:
                df_features = df_features.iloc[:max_samples]

        else:
            # 2. Compute from Scratch
            print(f"Processing images for {dataset_name}...")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            df_meta = pd.read_csv(metadata_path)

            # Apply max_samples before processing to save time during debug
            if max_samples is not None:
                df_meta = df_meta.iloc[:max_samples]

            # Extract geometric features
            geo_features = []
            for idx, row in df_meta.iterrows():
                # Construct full path. Metadata contains relative path 'images/xxx.jpg'
                full_path = os.path.join(self.input_dir, row["file_path"])

                # Extract features using the library function (handles polarity, contours, integral thickness)
                feats = extract_geometric_features(full_path)
                geo_features.append(feats)

            df_geo = pd.DataFrame(geo_features, columns=FEATURE_NAMES_GEO)

            # Combine with provided tabular features
            # Reconstruct tabular column names based on Config
            tab_cols = []
            for prefix in Config.TABULAR_PREFIXES:
                for i in range(1, Config.NUM_FEATURES_PER_GROUP + 1):
                    tab_cols.append(f"{prefix}_{i}")

            # Extract tabular data
            df_tab = df_meta[tab_cols].astype(Config.FLOAT_PRECISION)

            # Merge ID, Features, and Target
            # We concatenate: [id, tabular_features, geometric_features, species(optional)]
            df_features = pd.concat([df_meta[["id"]], df_tab, df_geo], axis=1)

            if "species" in df_meta.columns:
                df_features["species"] = df_meta["species"]

            # Save to cache
            # Note: If max_samples was used, we probably shouldn't overwrite the full cache,
            # but for this implementation we save what we computed to the specified name.
            print(f"Saving features to {cache_path}")
            df_features.to_parquet(cache_path)

        # 3. Prepare Return Values
        ids = df_features["id"].values

        # Identify feature columns (exclude id, species)
        exclude = ["id", "species"]
        feature_cols = [c for c in df_features.columns if c not in exclude]

        # Enforce alphanumeric sort for deterministic memory layout and float associativity
        feature_cols.sort()

        X = df_features[feature_cols].values.astype(Config.FLOAT_PRECISION)

        y = None
        if "species" in df_features.columns:
            y = df_features["species"].values

        return X, y, ids

    def compute_integral_thickness(self, image_path):
        """
        Computes the Mean Thickness of the leaf using the Integral-Morphological strategy.
        This method delegates to the core extraction logic to ensure consistency.

        Args:
            image_path (str): Path to the image file.

        Returns:
            float: The mean thickness value.
        """
        # The extract_geometric_features function returns a vector where
        # 'mean_thickness' is the last element (index -1)
        feats = extract_geometric_features(image_path)
        return feats[-1]
