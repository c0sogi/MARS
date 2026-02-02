import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from library.config import Config


class FeatureEngineer:
    """
    Handles feature engineering for Level-0 experts.
    Transforms raw embeddings and metadata into algorithm-specific formats.
    Implements caching for both processed datasets and fitted transformers.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.pca_components = Config.PCA_COMPONENTS
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_paths(self, backbone, split, method):
        """
        Generates file paths for cached data and transformers.
        """
        data_path = os.path.join(
            self.cache_dir, f"{backbone}_{split}_{method}_features.npy"
        )

        # Transformer names depend only on backbone and method, not split
        if method == "linear":
            transformer_path = os.path.join(
                self.cache_dir, f"{backbone}_{method}_scaler.joblib"
            )
        else:
            transformer_path = os.path.join(
                self.cache_dir, f"{backbone}_{method}_pca.joblib"
            )

        return data_path, transformer_path

    def prepare_linear_features(
        self, features, meta, backbone_name, split_name, load_cached_data=True
    ):
        """
        Prepares features for Linear and Kernel experts (Ridge, SVR).
        Strategy: Concatenate [Embeddings, Metadata] -> StandardScaler.

        Args:
            features (np.ndarray): Image embeddings.
            meta (np.ndarray): Binary metadata features.
            backbone_name (str): Name of the backbone (e.g., 'siglip').
            split_name (str): Name of the split ('train', 'val', 'test').
            load_cached_data (bool): Whether to use cached files.

        Returns:
            np.ndarray: Scaled feature matrix.
        """
        method = "linear"
        data_path, scaler_path = self._get_paths(backbone_name, split_name, method)

        # 1. Try to load processed data from cache
        if load_cached_data and os.path.exists(data_path):
            print(
                f"Loading cached linear features for {backbone_name} ({split_name})..."
            )
            return np.load(data_path)

        print(f"Generating linear features for {backbone_name} ({split_name})...")

        # 2. Concatenate features and metadata
        # Ensure dimensions match
        if features.shape[0] != meta.shape[0]:
            raise ValueError(
                f"Mismatch in samples: features {features.shape[0]}, meta {meta.shape[0]}"
            )

        X_raw = np.hstack([features, meta])

        # 3. Handle Scaling
        if split_name == "train":
            print(f"Fitting StandardScaler for {backbone_name}...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            # Save the fitted scaler
            joblib.dump(scaler, scaler_path)
        else:
            # For val/test, load the scaler fitted on train
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at {scaler_path}. "
                    "Please process the 'train' split first to fit the scaler."
                )
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X_raw)

        # 4. Save processed data to cache
        np.save(data_path, X_scaled)

        return X_scaled

    def prepare_tree_features(
        self, features, meta, backbone_name, split_name, load_cached_data=True
    ):
        """
        Prepares features for Tree and Boosting experts (ExtraTrees, LightGBM).
        Strategy: PCA on Embeddings -> Concatenate [PCA_Embeddings, Metadata].

        Args:
            features (np.ndarray): Image embeddings.
            meta (np.ndarray): Binary metadata features.
            backbone_name (str): Name of the backbone.
            split_name (str): Name of the split.
            load_cached_data (bool): Whether to use cached files.

        Returns:
            np.ndarray: Feature matrix (PCA-reduced embeddings + raw metadata).
        """
        method = "tree"
        data_path, pca_path = self._get_paths(backbone_name, split_name, method)

        # 1. Try to load processed data from cache
        if load_cached_data and os.path.exists(data_path):
            print(f"Loading cached tree features for {backbone_name} ({split_name})...")
            return np.load(data_path)

        print(f"Generating tree features for {backbone_name} ({split_name})...")

        # 2. Handle PCA on Embeddings
        if split_name == "train":
            print(f"Fitting PCA for {backbone_name}...")
            # Ensure we don't request more components than samples (relevant for debugging)
            n_samples = features.shape[0]
            n_components = min(n_samples, self.pca_components)

            pca = PCA(n_components=n_components, random_state=Config.SEED)
            features_pca = pca.fit_transform(features)

            # Save the fitted PCA
            joblib.dump(pca, pca_path)
        else:
            # For val/test, load the PCA fitted on train
            if not os.path.exists(pca_path):
                raise FileNotFoundError(
                    f"PCA model not found at {pca_path}. "
                    "Please process the 'train' split first to fit the PCA."
                )
            pca = joblib.load(pca_path)
            features_pca = pca.transform(features)

        # 3. Concatenate PCA features and raw metadata
        # Metadata is kept binary/raw for tree-based models as they handle it well
        X_final = np.hstack([features_pca, meta])

        # 4. Save processed data to cache
        np.save(data_path, X_final)

        return X_final
