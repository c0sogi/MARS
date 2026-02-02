import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.configuration import Config
from library.utilities import setup_logger

# Initialize logger
logger = setup_logger()


class LeafSpeciesPipeline(BaseEstimator, ClassifierMixin):
    """
    Custom pipeline implementing the Independent Subspace Reduction strategy.

    This pipeline handles:
    1. Splitting concatenated image features into DINOv2 (Global) and ConvNeXt (Local) streams.
    2. Applying Independent PCA to each vision stream to retain specific variance (0.99).
    3. Applying QuantileTransformer (Normal distribution) to tabular features.
    4. Concatenating all transformed features (Early Fusion).
    5. Classification using Linear Discriminant Analysis with Ledoit-Wolf shrinkage.
    """

    def __init__(
        self,
        dino_dim: int = 1024,
        pca_variance: float = Config.PCA_VARIANCE,
        tabular_dist: str = Config.TABULAR_OUTPUT_DIST,
        lda_solver: str = Config.LDA_SOLVER,
        lda_shrinkage: str = Config.LDA_SHRINKAGE,
        random_state: int = Config.SEED,
    ):

        self.dino_dim = dino_dim
        self.pca_variance = pca_variance
        self.tabular_dist = tabular_dist
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage
        self.random_state = random_state

        # Transformers and Classifier
        self.pca_dino = None
        self.pca_conv = None
        self.scaler_tab = None
        self.lda = None

        # State flag
        self.is_fitted = False

    def _split_streams(self, X_img: np.ndarray):
        """
        Splits the concatenated image features into DINO and ConvNeXt streams.
        Assumes X_img is [N, D_dino + D_conv].
        """
        if X_img.shape[1] <= self.dino_dim:
            raise ValueError(
                f"Input image features dim {X_img.shape[1]} is smaller than expected DINO dim {self.dino_dim}"
            )

        dino_feats = X_img[:, : self.dino_dim]
        conv_feats = X_img[:, self.dino_dim :]
        return dino_feats, conv_feats

    def fit(self, X_img: np.ndarray, X_tab: np.ndarray, y: np.ndarray):
        """
        Fit the transformers and the classifier.

        Args:
            X_img: (N, D_vision) array of concatenated vision features.
            X_tab: (N, D_tab) array of tabular features.
            y: (N,) array of labels.
        """
        logger.info("Fitting LeafSpeciesPipeline...")

        # 1. Split Vision Streams
        dino_raw, conv_raw = self._split_streams(X_img)

        # 2. Independent PCA
        # DINO Stream
        logger.info(f"Fitting PCA on DINO stream (Input Dim: {dino_raw.shape[1]})...")
        self.pca_dino = PCA(
            n_components=self.pca_variance,
            svd_solver="full",
            random_state=self.random_state,
        )
        dino_pca = self.pca_dino.fit_transform(dino_raw)
        logger.info(f"DINO Stream reduced to {dino_pca.shape[1]} components.")

        # ConvNeXt Stream
        logger.info(
            f"Fitting PCA on ConvNeXt stream (Input Dim: {conv_raw.shape[1]})..."
        )
        self.pca_conv = PCA(
            n_components=self.pca_variance,
            svd_solver="full",
            random_state=self.random_state,
        )
        conv_pca = self.pca_conv.fit_transform(conv_raw)
        logger.info(f"ConvNeXt Stream reduced to {conv_pca.shape[1]} components.")

        # 3. Tabular Transformation
        logger.info("Fitting QuantileTransformer on Tabular features...")
        self.scaler_tab = QuantileTransformer(
            output_distribution=self.tabular_dist, random_state=self.random_state
        )
        tab_trans = self.scaler_tab.fit_transform(X_tab)

        # 4. Fusion
        X_final = np.hstack([dino_pca, conv_pca, tab_trans])
        logger.info(f"Final concatenated feature dimension: {X_final.shape[1]}")

        # 5. LDA Classifier
        logger.info(
            f"Training LDA Classifier (Solver: {self.lda_solver}, Shrinkage: {self.lda_shrinkage})..."
        )
        self.lda = LinearDiscriminantAnalysis(
            solver=self.lda_solver, shrinkage=self.lda_shrinkage
        )
        self.lda.fit(X_final, y)

        self.is_fitted = True
        logger.info("Pipeline fitting complete.")
        return self

    def predict_proba(self, X_img: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X_img: (N, D_vision) array of concatenated vision features.
            X_tab: (N, D_tab) array of tabular features.

        Returns:
            (N, n_classes) array of probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict_proba.")

        # 1. Split
        dino_raw, conv_raw = self._split_streams(X_img)

        # 2. Transform Vision
        dino_pca = self.pca_dino.transform(dino_raw)
        conv_pca = self.pca_conv.transform(conv_raw)

        # 3. Transform Tabular
        tab_trans = self.scaler_tab.transform(X_tab)

        # 4. Fusion
        X_final = np.hstack([dino_pca, conv_pca, tab_trans])

        # 5. Predict
        return self.lda.predict_proba(X_final)
