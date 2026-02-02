import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from library.config import Config
from library.utils import seed_everything


class FeatureProcessor:
    """
    Handles feature transformation for Level-0 models.
    Manages state for StandardScaler and PCA to ensure correct fitting on training sets
    and transformation on validation/test sets.
    """

    def __init__(self):
        """
        Initializes the FeatureProcessor with random seeds and empty transformers.
        """
        seed_everything(Config.SEED)

        # Transformer for Linear Models (Ridge, SVR)
        self.scaler = StandardScaler()
        self.linear_fitted = False

        # Transformer for Tree Models (ExtraTrees)
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        self.tree_fitted = False

    def prepare_data_for_linear(
        self, features: np.ndarray, meta: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        Prepares data for linear models (Ridge, SVR).

        Strategy:
            1. Concatenate image embeddings and metadata.
            2. Apply StandardScaler to the full vector.

        Args:
            features (np.ndarray): Image embeddings.
            meta (np.ndarray): Metadata features.
            fit (bool): Whether to fit the StandardScaler on this data.

        Returns:
            np.ndarray: Scaled and concatenated features.
        """
        # Ensure inputs are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if meta.ndim == 1:
            meta = meta.reshape(1, -1)

        # Concatenate embeddings and metadata
        combined = np.hstack([features, meta])

        if fit:
            self.scaler.fit(combined)
            self.linear_fitted = True

        if not self.linear_fitted:
            raise RuntimeError(
                "StandardScaler is not fitted. Call prepare_data_for_linear with fit=True on training data first."
            )

        return self.scaler.transform(combined)

    def prepare_data_for_tree(
        self, features: np.ndarray, meta: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        Prepares data for tree-based models (ExtraTrees).

        Strategy:
            1. Apply PCA to image embeddings to reduce dimensionality.
            2. Concatenate PCA-reduced embeddings with RAW metadata.

        Args:
            features (np.ndarray): Image embeddings.
            meta (np.ndarray): Metadata features.
            fit (bool): Whether to fit the PCA on this data.

        Returns:
            np.ndarray: Concatenated PCA features and raw metadata.
        """
        # Ensure inputs are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if meta.ndim == 1:
            meta = meta.reshape(1, -1)

        if fit:
            self.pca.fit(features)
            self.tree_fitted = True

        if not self.tree_fitted:
            raise RuntimeError(
                "PCA is not fitted. Call prepare_data_for_tree with fit=True on training data first."
            )

        # Transform embeddings using PCA
        features_pca = self.pca.transform(features)

        # Concatenate with raw metadata (trees handle binary features well without scaling)
        combined = np.hstack([features_pca, meta])

        return combined
