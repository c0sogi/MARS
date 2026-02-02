import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class PipelineMember:
    """
    Represents a single member of the Bagging Ensemble.
    Encapsulates the feature transformation and classification pipeline:
    1. PCA for Shape Stream (DINOv2)
    2. PCA for Texture Stream (ConvNeXt)
    3. QuantileTransformer for Tabular Data
    4. Linear Discriminant Analysis (LDA) with Ledoit-Wolf Shrinkage
    """

    def __init__(self):
        # Stream 1: Shape (PCA)
        self.pca_shape = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

        # Stream 2: Texture (PCA)
        self.pca_texture = PCA(
            n_components=Config.PCA_VARIANCE, random_state=Config.SEED
        )

        # Tabular Data (Quantile Transformer -> Gaussian)
        self.qt_tabular = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        # Classifier: LDA with Shrinkage
        # solver='lsqr' is required for shrinkage='auto' (Ledoit-Wolf)
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, X_shape, X_texture, X_tabular, y):
        """
        Fits the transformers and classifier on the provided training data.

        Args:
            X_shape (np.ndarray): Shape embeddings (N, D_shape).
            X_texture (np.ndarray): Texture embeddings (N, D_texture).
            X_tabular (np.ndarray): Tabular features (N, 192).
            y (np.ndarray): Target labels (N,).

        Returns:
            self: The fitted instance.
        """
        # 1. Fit and Transform Shape Stream
        # PCA requires 2D array
        X_shape_trans = self.pca_shape.fit_transform(X_shape)

        # 2. Fit and Transform Texture Stream
        X_texture_trans = self.pca_texture.fit_transform(X_texture)

        # 3. Fit and Transform Tabular Data
        # Adjust n_quantiles if sample size is smaller than default (1000)
        n_samples = X_tabular.shape[0]
        n_quantiles = min(n_samples, 1000)
        self.qt_tabular.set_params(n_quantiles=n_quantiles)
        X_tabular_trans = self.qt_tabular.fit_transform(X_tabular)

        # 4. Feature Fusion
        # Concatenate along the feature axis
        X_final = np.concatenate(
            [X_shape_trans, X_texture_trans, X_tabular_trans], axis=1
        )

        # 5. Train Classifier
        self.lda.fit(X_final, y)

        return self

    def predict_proba(self, X_shape, X_texture, X_tabular):
        """
        Predicts class probabilities for new data.

        Args:
            X_shape (np.ndarray): Shape embeddings.
            X_texture (np.ndarray): Texture embeddings.
            X_tabular (np.ndarray): Tabular features.

        Returns:
            np.ndarray: Class probabilities of shape (N, n_classes).
        """
        # 1. Transform Shape Stream
        X_shape_trans = self.pca_shape.transform(X_shape)

        # 2. Transform Texture Stream
        X_texture_trans = self.pca_texture.transform(X_texture)

        # 3. Transform Tabular Data
        X_tabular_trans = self.qt_tabular.transform(X_tabular)

        # 4. Feature Fusion
        X_final = np.concatenate(
            [X_shape_trans, X_texture_trans, X_tabular_trans], axis=1
        )

        # 5. Predict
        return self.lda.predict_proba(X_final)
