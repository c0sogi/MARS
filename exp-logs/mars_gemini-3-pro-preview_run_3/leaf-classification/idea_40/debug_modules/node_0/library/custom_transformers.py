import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from library.config import Config


class DualStreamPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom Scikit-Learn transformer that implements the 'Selective Feature Topology'
    strategy. It processes heterogeneous data modalities (DINOv2, ConvNeXt, Tabular)
    with specific transformations tailored to their geometric properties.

    Pipeline:
    1. Splits input X into DINO, ConvNeXt, and Tabular components.
    2. Visual Streams (DINO, ConvNeXt): Applies Independent PCA (Linear) to retain
       specified variance (e.g., 99%). No non-linear distortion is applied.
    3. Tabular Stream: Applies QuantileTransformer (Normal) to enforce Gaussian
       distribution on engineered histograms.
    4. Fusion: Concatenates processed streams.
    5. Global Alignment: Applies StandardScaler to the fused vector to ensure
       uniform shrinkage in the downstream LDA classifier.
    """

    def __init__(
        self,
        pca_variance=Config.PCA_VARIANCE,
        dino_dim=1024,
        conv_dim=1536,
        tab_dim=Config.N_TABULAR_FEATURES,
    ):
        """
        Args:
            pca_variance (float): Variance to retain in PCA (0.0 to 1.0).
            dino_dim (int): Dimensionality of DINOv2 features (default 1024 for ViT-L).
            conv_dim (int): Dimensionality of ConvNeXt features (default 1536 for Large).
            tab_dim (int): Dimensionality of tabular features (default 192).
        """
        self.pca_variance = pca_variance
        self.dino_dim = dino_dim
        self.conv_dim = conv_dim
        self.tab_dim = tab_dim

        # Transformers
        self.pca_dino = None
        self.pca_conv = None
        self.qt_tab = None
        self.global_scaler = None

        # State check
        self._is_fitted = False

    def _split_features(self, X):
        """
        Splits the concatenated feature matrix X into its three components.
        Assumes X is constructed as [DINO, ConvNeXt, Tabular].
        """
        total_dim = self.dino_dim + self.conv_dim + self.tab_dim
        if X.shape[1] != total_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {total_dim}, got {X.shape[1]}. "
                f"Ensure X is concatenated as [DINO({self.dino_dim}), "
                f"ConvNeXt({self.conv_dim}), Tabular({self.tab_dim})]."
            )

        # Calculate indices
        idx_dino_end = self.dino_dim
        idx_conv_end = self.dino_dim + self.conv_dim

        # Slice
        X_dino = X[:, :idx_dino_end]
        X_conv = X[:, idx_dino_end:idx_conv_end]
        X_tab = X[:, idx_conv_end:]

        return X_dino, X_conv, X_tab

    def fit(self, X, y=None):
        """
        Fits the independent transformers on the provided data.
        """
        # 1. Split Data
        X_dino, X_conv, X_tab = self._split_features(X)

        # 2. Fit Independent PCA on Visual Streams
        # DINO Stream
        self.pca_dino = PCA(n_components=self.pca_variance, random_state=Config.SEED)
        self.pca_dino.fit(X_dino)

        # ConvNeXt Stream
        self.pca_conv = PCA(n_components=self.pca_variance, random_state=Config.SEED)
        self.pca_conv.fit(X_conv)

        # 3. Fit QuantileTransformer on Tabular Stream
        # Enforce Gaussian distribution for LDA compliance
        self.qt_tab = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )
        self.qt_tab.fit(X_tab)

        # 4. Transform and Fuse for Global Scaling
        X_dino_t = self.pca_dino.transform(X_dino)
        X_conv_t = self.pca_conv.transform(X_conv)
        X_tab_t = self.qt_tab.transform(X_tab)

        X_fused = np.concatenate([X_dino_t, X_conv_t, X_tab_t], axis=1)

        # 5. Fit Global StandardScaler
        # Ensures Ledoit-Wolf shrinkage is applied uniformly across modalities
        self.global_scaler = StandardScaler()
        self.global_scaler.fit(X_fused)

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Transformer has not been fitted yet. Call 'fit' before 'transform'."
            )

        # 1. Split Data
        X_dino, X_conv, X_tab = self._split_features(X)

        # 2. Apply Independent Transformations
        X_dino_t = self.pca_dino.transform(X_dino)
        X_conv_t = self.pca_conv.transform(X_conv)
        X_tab_t = self.qt_tab.transform(X_tab)

        # 3. Fuse
        X_fused = np.concatenate([X_dino_t, X_conv_t, X_tab_t], axis=1)

        # 4. Global Scaling
        X_final = self.global_scaler.transform(X_fused)

        return X_final
