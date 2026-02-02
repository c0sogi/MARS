import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from library.config import Config
from library.extractors import FeatureExtractor


class DataProcessor:
    """
    Handles data loading and feature engineering for Level-0 experts.
    Bridges the gap between raw embeddings (from FeatureExtractor) and
    model-ready feature matrices (scaled/reduced).
    """

    def __init__(self, device=Config.DEVICE):
        """
        Args:
            device (str): Device to use for feature extraction if needed.
        """
        self.device = device
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def get_raw_features(
        self, backbone_name, split, load_cached_data=True, debug=False
    ):
        """
        Retrieves raw features using the FeatureExtractor.

        Args:
            backbone_name (str): Hugging Face model ID.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use disk caching.
            debug (bool): Whether to run in debug mode (subset of data).

        Returns:
            tuple: (features, ids, meta, targets)
        """
        extractor = FeatureExtractor(model_name=backbone_name, device=self.device)
        return extractor.extract(split, load_cached_data=load_cached_data, debug=debug)

    def prepare_linear_features(self, features, meta, scaler=None):
        """
        Prepares features for Linear and Kernel experts (Ridge, SVR).
        Strategy: Concatenate Embeddings + Metadata, then Apply StandardScaler.

        Args:
            features (np.ndarray): Image embeddings (N, D).
            meta (np.ndarray): Binary metadata (N, 12).
            scaler (StandardScaler, optional): Fitted scaler. If None, a new one is fitted.

        Returns:
            tuple: (X_scaled, scaler)
        """
        # Concatenate embeddings and metadata
        # features: (N, D), meta: (N, 12) -> X: (N, D+12)
        X = np.hstack([features, meta])

        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

        return X_scaled, scaler

    def prepare_tree_features(
        self, features, meta, pca=None, n_components=Config.PCA_COMPONENTS
    ):
        """
        Prepares features for the Partitioning expert (ExtraTrees).
        Strategy: Apply PCA to Embeddings, then Concatenate with Raw Metadata.

        Args:
            features (np.ndarray): Image embeddings (N, D).
            meta (np.ndarray): Binary metadata (N, 12).
            pca (PCA, optional): Fitted PCA object. If None, a new one is fitted.
            n_components (int): Number of PCA components to keep.

        Returns:
            tuple: (X_processed, pca)
        """
        # 1. Apply PCA to image embeddings
        if pca is None:
            pca = PCA(n_components=n_components, random_state=Config.SEED)
            features_pca = pca.fit_transform(features)
        else:
            features_pca = pca.transform(features)

        # 2. Concatenate with raw metadata
        # Trees handle binary features well, so we don't need to scale metadata or PCA components
        X_processed = np.hstack([features_pca, meta])

        return X_processed, pca

    def save_processed_data(self, data_dict, prefix):
        """
        Caches processed numpy arrays to disk.
        Strictly uses .npy format, avoiding pickle.

        Args:
            data_dict (dict): Dictionary of arrays to save (e.g., {'X': ..., 'y': ...}).
            prefix (str): Filename prefix.
        """
        for key, array in data_dict.items():
            filename = f"{prefix}_{key}.npy"
            path = os.path.join(self.working_dir, filename)
            np.save(path, array)

    def load_processed_data(self, keys, prefix):
        """
        Loads processed numpy arrays from disk.

        Args:
            keys (list): List of keys to load (e.g., ['X', 'y']).
            prefix (str): Filename prefix.

        Returns:
            dict: Dictionary containing loaded arrays, or None if any file is missing.
        """
        loaded_data = {}
        for key in keys:
            filename = f"{prefix}_{key}.npy"
            path = os.path.join(self.working_dir, filename)
            if not os.path.exists(path):
                return None
            loaded_data[key] = np.load(path)
        return loaded_data
