import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from library.config import Config


class MultiModalTransformer(BaseEstimator, TransformerMixin):
    """
    Implements the Affect-Augmented Asymmetric Early Fusion (AAAEF) logic.
    Handles dimensionality reduction, normalization, and fusion of 4 feature views.
    """

    def __init__(self):
        self.pca_components = Config.PCA_COMPONENTS
        self.seed = Config.SEED

        # View 2: Semantic Aux (Deep Semantics) -> PCA
        self.pca = PCA(n_components=self.pca_components, random_state=self.seed)

        # View 3: Affective Aux (Orthogonal Signal) -> RankGauss
        self.qt_affective = QuantileTransformer(
            output_distribution="normal", random_state=self.seed
        )

        # View 4: Metadata -> RankGauss
        self.qt_metadata = QuantileTransformer(
            output_distribution="normal", random_state=self.seed
        )

    def fit(self, X, y=None):
        """
        Fits the internal transformers on the provided feature views.

        Args:
            X (dict): A dictionary containing:
                - 'anchor': np.ndarray (N, 384) [Not used in fit, stateless]
                - 'semantic_aux': np.ndarray (N, 768)
                - 'affective_aux': np.ndarray (N, 28)
                - 'metadata': pd.DataFrame or np.ndarray (N, F)
            y (array-like, optional): Target values (ignored).

        Returns:
            self
        """
        # 1. Fit PCA on Semantic Aux
        self.pca.fit(X["semantic_aux"])

        # 2. Fit QuantileTransformer on Affective Aux logits
        self.qt_affective.fit(X["affective_aux"])

        # 3. Fit QuantileTransformer on Metadata
        meta_data = self._to_numpy(X["metadata"])
        self.qt_metadata.fit(meta_data)

        return self

    def transform(self, X):
        """
        Applies transformations and fuses the views into a single feature matrix.

        Args:
            X (dict): A dictionary containing the feature views.

        Returns:
            np.ndarray: The fused feature matrix (N, Total_Dims).
        """
        # View 1: Semantic Anchor (High-Res)
        # Strategy: L2 Normalize only (keep high dimensionality)
        v1 = normalize(X["anchor"], norm="l2", axis=1)

        # View 2: Semantic Aux (Deep Semantics)
        # Strategy: PCA -> L2 Normalize (Asymmetric reduction)
        v2_projected = self.pca.transform(X["semantic_aux"])
        v2 = normalize(v2_projected, norm="l2", axis=1)

        # View 3: Affective Aux (Orthogonal Signal)
        # Strategy: RankGauss (Normalize unbounded logits)
        v3 = self.qt_affective.transform(X["affective_aux"])

        # View 4: Metadata
        # Strategy: RankGauss (Normalize diverse numerical scales)
        meta_data = self._to_numpy(X["metadata"])
        v4 = self.qt_metadata.transform(meta_data)

        # Early Fusion: Concatenate all processed views
        X_fused = np.hstack([v1, v2, v3, v4])

        return X_fused

    def _to_numpy(self, data):
        """Helper to convert DataFrame to numpy array if necessary."""
        if isinstance(data, pd.DataFrame):
            return data.values
        return data
