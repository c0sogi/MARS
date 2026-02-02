import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config


class FidelityBranch:
    """
    Implements a single branch of the Multi-Fidelity Dimensionality Ensemble.

    This pipeline performs:
    1. Quantile Transformation on tabular features (Gaussian output).
    2. Independent PCA reduction on DINOv2 and ConvNeXt visual streams based on a variance threshold.
    3. Early fusion of the three feature sets.
    4. Classification using Linear Discriminant Analysis (LDA) with Ledoit-Wolf shrinkage.
    """

    def __init__(self, pca_variance, quantile_dist="normal"):
        """
        Args:
            pca_variance (float): Variance threshold for PCA (e.g., 0.99, 0.95, 0.90).
            quantile_dist (str): Output distribution for QuantileTransformer ('normal' or 'uniform').
        """
        self.pca_variance = pca_variance
        self.quantile_dist = quantile_dist

        # Initialize transformers
        # Random state is fixed for reproducibility
        self.scaler_tab = QuantileTransformer(
            output_distribution=self.quantile_dist, random_state=config.RANDOM_SEED
        )

        # PCA for DINOv2 stream
        self.pca_dino = PCA(
            n_components=self.pca_variance, random_state=config.RANDOM_SEED
        )

        # PCA for ConvNeXt stream
        self.pca_conv = PCA(
            n_components=self.pca_variance, random_state=config.RANDOM_SEED
        )

        # LDA Classifier with Ledoit-Wolf shrinkage (requires lsqr or eigen solver)
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        self.is_fitted = False

    def fit(self, X_dino, X_conv, X_tab, y):
        """
        Fits the transformers and classifier on the provided data.

        Args:
            X_dino (np.ndarray): DINOv2 features, shape (N, D_dino).
            X_conv (np.ndarray): ConvNeXt features, shape (N, D_conv).
            X_tab (np.ndarray): Tabular features, shape (N, D_tab).
            y (np.ndarray): Target labels, shape (N,).

        Returns:
            self: The fitted instance.
        """
        # 1. Transform Tabular Features
        # QuantileTransformer is robust to outliers and enforces Gaussianity for LDA
        X_tab_trans = self.scaler_tab.fit_transform(X_tab)

        # 2. Transform Visual Features (Independent Reduction)
        # PCA retains 'pca_variance' amount of information
        X_dino_trans = self.pca_dino.fit_transform(X_dino)
        X_conv_trans = self.pca_conv.fit_transform(X_conv)

        # 3. Early Fusion
        # Concatenate all features into a single vector
        X_fused = np.hstack([X_tab_trans, X_dino_trans, X_conv_trans])

        # 4. Fit LDA Classifier
        self.lda.fit(X_fused, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X_dino, X_conv, X_tab):
        """
        Predicts class probabilities for the given data.

        Args:
            X_dino (np.ndarray): DINOv2 features.
            X_conv (np.ndarray): ConvNeXt features.
            X_tab (np.ndarray): Tabular features.

        Returns:
            np.ndarray: Probability matrix of shape (N, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before calling predict_proba.")

        # 1. Transform Tabular Features
        X_tab_trans = self.scaler_tab.transform(X_tab)

        # 2. Transform Visual Features
        X_dino_trans = self.pca_dino.transform(X_dino)
        X_conv_trans = self.pca_conv.transform(X_conv)

        # 3. Early Fusion
        X_fused = np.hstack([X_tab_trans, X_dino_trans, X_conv_trans])

        # 4. Predict
        return self.lda.predict_proba(X_fused)

    def save(self, filepath):
        """
        Saves the fitted model pipeline to a file using pickle.

        Args:
            filepath (str): Destination path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Loads a fitted model pipeline from a file.

        Args:
            filepath (str): Source path.

        Returns:
            FidelityBranch: The loaded model instance.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
