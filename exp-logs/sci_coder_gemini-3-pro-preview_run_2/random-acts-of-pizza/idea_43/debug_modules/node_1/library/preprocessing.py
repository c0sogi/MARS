import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import Config
from library.utils import setup_logger


class HAMFPreprocessor:
    """
    Implements the learnable transformations for the HAMF-ADBE architecture.

    This preprocessor handles:
    1. L2 Normalization of Anchor Views (Title, Body).
    2. PCA Projection + L2 Normalization of Auxiliary Views (Global, Hook).
    3. RankGauss (Quantile) Transformation of Metadata.
    4. Feature Concatenation.

    This class is designed to be instantiated within each cross-validation fold
    to prevent data leakage.
    """

    def __init__(self):
        self.logger = setup_logger("hamf_preprocessor")

        # Initialize PCA for Global Context (View 3)
        self.pca_global = PCA(
            n_components=Config.PCA_GLOBAL_COMPONENTS, random_state=Config.SEED
        )

        # Initialize PCA for Deep Hook (View 4)
        self.pca_hook = PCA(
            n_components=Config.PCA_HOOK_COMPONENTS, random_state=Config.SEED
        )

        # Initialize QuantileTransformer for Metadata (View 5)
        self.scaler_meta = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        self.is_fitted = False

    def fit(self, X_dict, y=None):
        """
        Fits the internal transformers (PCA and Scaler) on the provided data.

        Args:
            X_dict (dict): Dictionary containing the feature views:
                           - 'aux_global': MPNet Global embeddings
                           - 'aux_hook': MPNet Title embeddings
                           - 'metadata': Numerical metadata
            y (array-like, optional): Target labels (unused, for API consistency).

        Returns:
            self
        """
        self.logger.info("Fitting HAMFPreprocessor...")

        # Fit PCA on Global Context
        if "aux_global" in X_dict:
            self.logger.info(
                f"Fitting Global PCA (n={Config.PCA_GLOBAL_COMPONENTS})..."
            )
            self.pca_global.fit(X_dict["aux_global"])
        else:
            raise ValueError("Key 'aux_global' missing from input dictionary.")

        # Fit PCA on Deep Hook
        if "aux_hook" in X_dict:
            self.logger.info(f"Fitting Hook PCA (n={Config.PCA_HOOK_COMPONENTS})...")
            self.pca_hook.fit(X_dict["aux_hook"])
        else:
            raise ValueError("Key 'aux_hook' missing from input dictionary.")

        # Fit Scaler on Metadata
        if "metadata" in X_dict:
            self.logger.info("Fitting Metadata QuantileTransformer...")
            self.scaler_meta.fit(X_dict["metadata"])
        else:
            raise ValueError("Key 'metadata' missing from input dictionary.")

        self.is_fitted = True
        return self

    def transform(self, X_dict):
        """
        Applies transformations to the input data and concatenates features.

        Args:
            X_dict (dict): Dictionary containing the feature views:
                           - 'anchor_title': MiniLM Title embeddings
                           - 'anchor_body': MiniLM Body embeddings
                           - 'aux_global': MPNet Global embeddings
                           - 'aux_hook': MPNet Title embeddings
                           - 'metadata': Numerical metadata

        Returns:
            np.ndarray: Concatenated feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "HAMFPreprocessor must be fitted before calling transform."
            )

        # 1. View 1: Title Anchor (L2 Normalized)
        # We normalize explicitly to ensure unit length
        feat_anchor_title = normalize(X_dict["anchor_title"], norm="l2")

        # 2. View 2: Body Anchor (L2 Normalized)
        feat_anchor_body = normalize(X_dict["anchor_body"], norm="l2")

        # 3. View 3: Global Context (PCA -> L2 Normalized)
        # Project using PCA
        feat_aux_global_raw = self.pca_global.transform(X_dict["aux_global"])
        # Normalize after PCA to align scale with Anchors
        feat_aux_global = normalize(feat_aux_global_raw, norm="l2")

        # 4. View 4: Deep Hook (PCA -> L2 Normalized)
        # Project using PCA
        feat_aux_hook_raw = self.pca_hook.transform(X_dict["aux_hook"])
        # Normalize after PCA
        feat_aux_hook = normalize(feat_aux_hook_raw, norm="l2")

        # 5. View 5: Robust Metadata (RankGauss)
        feat_metadata = self.scaler_meta.transform(X_dict["metadata"])

        # Concatenate all views
        # Expected dim: 384 + 384 + 50 + 20 + ~10 = ~848
        X_combined = np.hstack(
            [
                feat_anchor_title,
                feat_anchor_body,
                feat_aux_global,
                feat_aux_hook,
                feat_metadata,
            ]
        )

        return X_combined

    def fit_transform(self, X_dict, y=None):
        """
        Fits the preprocessor and transforms the data in one step.
        """
        return self.fit(X_dict, y).transform(X_dict)
