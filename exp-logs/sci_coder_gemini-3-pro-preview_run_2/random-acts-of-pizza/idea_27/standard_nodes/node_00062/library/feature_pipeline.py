import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
import library.config as config


class ADBEFTransformer:
    """
    Implements the Asymmetric Dual-Backbone Early Fusion (ADBEF) strategy.

    This transformer manages the stateful transformations required for the three feature views:
    1. Primary View (MiniLM): Stateless L2 Normalization.
    2. Auxiliary View (MPNet): PCA compression -> L2 Normalization.
    3. Metadata View: QuantileTransformer (RankGauss).

    Finally, it performs Early Fusion by concatenating the processed views.
    """

    def __init__(self):
        """
        Initialize the transformer with PCA and QuantileTransformer using config hyperparameters.
        """
        # Initialize PCA for the Auxiliary View (MPNet)
        # Reduces 768d -> 32d (as per config)
        self.pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.SEED)

        # Initialize QuantileTransformer for the Metadata View
        # Transforms arbitrary distributions to Normal distribution
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=config.SEED
        )

        self.is_fitted = False

    def fit(self, X_primary, X_aux, X_meta):
        """
        Fit the stateful transformers (PCA and QuantileTransformer) on the training data.

        Args:
            X_primary (np.ndarray): Primary embeddings (MiniLM). Not used for fitting (stateless).
            X_aux (np.ndarray): Auxiliary embeddings (MPNet). Used to fit PCA.
            X_meta (np.ndarray): Numerical metadata. Used to fit QuantileTransformer.

        Returns:
            self: The fitted transformer instance.
        """
        # Ensure inputs are numpy arrays
        X_aux = np.asarray(X_aux)
        X_meta = np.asarray(X_meta)

        # Fit PCA on Auxiliary View
        # Captures top variance components of the deeper semantic model
        self.pca.fit(X_aux)

        # Fit QuantileTransformer on Metadata View
        # Learns the cumulative distribution function of each feature
        self.qt.fit(X_meta)

        self.is_fitted = True
        return self

    def transform(self, X_primary, X_aux, X_meta):
        """
        Apply transformations and fuse the features.

        Args:
            X_primary (np.ndarray): Primary embeddings (MiniLM).
            X_aux (np.ndarray): Auxiliary embeddings (MPNet).
            X_meta (np.ndarray): Numerical metadata.

        Returns:
            np.ndarray: The fused feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before calling transform.")

        # Ensure inputs are numpy arrays
        X_primary = np.asarray(X_primary)
        X_aux = np.asarray(X_aux)
        X_meta = np.asarray(X_meta)

        # 1. Process Primary View (MiniLM)
        # Strategy: L2 Normalize to project onto hypersphere
        # This is the "Anchor View" (High Res)
        v1 = normalize(X_primary, norm="l2")

        # 2. Process Auxiliary View (MPNet)
        # Strategy: PCA -> L2 Normalize
        # This is the "Asymmetric Dimensionality Reduction" step
        # Normalization happens AFTER projection to ensure unit norm in the reduced space
        v2_pca = self.pca.transform(X_aux)
        v2 = normalize(v2_pca, norm="l2")

        # 3. Process Metadata View
        # Strategy: QuantileTransformer (RankGauss)
        # Aligns metadata distribution with the normalized embeddings
        v3 = self.qt.transform(X_meta)

        # 4. Early Fusion
        # Concatenate all views horizontally
        X_fused = np.hstack([v1, v2, v3])

        return X_fused

    def fit_transform(self, X_primary, X_aux, X_meta):
        """
        Fit and transform in one step.

        Args:
            X_primary (np.ndarray): Primary embeddings.
            X_aux (np.ndarray): Auxiliary embeddings.
            X_meta (np.ndarray): Metadata.

        Returns:
            np.ndarray: Fused feature matrix.
        """
        self.fit(X_primary, X_aux, X_meta)
        return self.transform(X_primary, X_aux, X_meta)
