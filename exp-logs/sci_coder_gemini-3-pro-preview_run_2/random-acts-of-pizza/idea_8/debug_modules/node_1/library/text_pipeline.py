import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from library.config import Config, get_text_embeddings
from library.utils import set_seed


class ProjectedTextEmbedder:
    """
    Implements the text processing branch of the Projected Semantic-Interaction Ensemble.

    This class handles:
    1. Retrieval of raw MPNet embeddings (via library helper).
    2. Dimensionality reduction using PCA (768 -> 256).
    3. L2 Normalization of the projected vectors.
    """

    def __init__(self):
        self.pca_path = os.path.join(Config.WORKING_DIR, "pca_model.joblib")
        self.pca = None
        self.normalizer = Normalizer(norm="l2")
        set_seed(Config.RANDOM_SEED)

    def fit(self, df_train: pd.DataFrame, load_cached_data: bool = True):
        """
        Generates embeddings for the training data, fits the PCA model, and saves it.

        Args:
            df_train (pd.DataFrame): Training data containing text columns.
            load_cached_data (bool): Whether to use cached raw embeddings.

        Returns:
            self
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Get raw MPNet embeddings (768 dimensions)
        # We explicitly use split="train" as we only fit on training data
        X_raw = get_text_embeddings(
            df_train, split="train", load_cached_data=load_cached_data
        )

        # Initialize and fit PCA
        # Dynamically adjust n_components for small datasets (e.g. demo/debugging)
        n_samples = X_raw.shape[0]
        n_components = min(Config.N_PCA_COMPONENTS, n_samples)

        self.pca = PCA(n_components=n_components, random_state=Config.RANDOM_SEED)
        self.pca.fit(X_raw)

        # Save the fitted PCA model for consistency across inference
        joblib.dump(self.pca, self.pca_path)

        return self

    def transform(self, df: pd.DataFrame, split: str, load_cached_data: bool = True):
        """
        Transforms text data into the projected and normalized feature space.

        Args:
            df (pd.DataFrame): Data containing text columns.
            split (str): Dataset split ('train', 'val', 'test') for cache management.
            load_cached_data (bool): Whether to use cached raw embeddings.

        Returns:
            np.ndarray: The processed feature matrix (N_samples, 256).
        """
        # Load PCA model if not currently in memory
        if self.pca is None:
            if os.path.exists(self.pca_path):
                self.pca = joblib.load(self.pca_path)
            else:
                raise FileNotFoundError(
                    f"PCA model not found at {self.pca_path}. "
                    "Please call fit() on training data first."
                )

        # Get raw MPNet embeddings
        X_raw = get_text_embeddings(df, split=split, load_cached_data=load_cached_data)

        # Apply PCA Projection (768 -> 256)
        X_pca = self.pca.transform(X_raw)

        # Apply L2 Normalization (Project to Hypersphere)
        X_final = self.normalizer.transform(X_pca)

        return X_final

    def fit_transform(self, df_train: pd.DataFrame, load_cached_data: bool = True):
        """
        Fits the pipeline on training data and returns the transformed training features.

        Args:
            df_train (pd.DataFrame): Training data.
            load_cached_data (bool): Whether to use cached raw embeddings.

        Returns:
            np.ndarray: Transformed training features.
        """
        self.fit(df_train, load_cached_data=load_cached_data)
        return self.transform(
            df_train, split="train", load_cached_data=load_cached_data
        )
