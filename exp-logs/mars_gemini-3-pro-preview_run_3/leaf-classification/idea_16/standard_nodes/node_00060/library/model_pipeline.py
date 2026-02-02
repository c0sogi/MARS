import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import QuantileTransformer
from library.config import Config
from library.utils import seed_everything


class HierarchicalLDA:
    """
    Hierarchical Discriminant Stacking architecture.

    Implements a two-stage stacked generalization:
    1. Modality-Specific Discriminant Projection:
       - Visual streams (DINOv2, ConvNeXt) are reduced via PCA (99% variance)
         and then projected via Base LDA (Ledoit-Wolf shrinkage) to C-1 components.
       - Tabular stream is Gaussianized via QuantileTransformer and projected via Base LDA.
    2. Discriminant Fusion:
       - Projections are concatenated and fed into a Meta-LDA (Ledoit-Wolf shrinkage).
    """

    def __init__(self):
        """
        Initialize the Hierarchical LDA pipeline components.
        """
        seed_everything()

        # --- Stage 1: Modality-Specific Transformers ---

        # DINOv2 Stream
        # PCA to reduce high-dimensional embeddings while retaining 99% variance
        self.dino_pca = PCA(n_components=Config.PCA_VARIANCE, svd_solver="full")
        # Base LDA to find discriminative subspace
        self.dino_lda = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=Config.LDA_COMPONENTS
        )

        # ConvNeXt Stream
        self.conv_pca = PCA(n_components=Config.PCA_VARIANCE, svd_solver="full")
        self.conv_lda = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=Config.LDA_COMPONENTS
        )

        # Tabular Stream
        # QuantileTransformer to enforce Gaussian distribution assumption of LDA
        self.tab_qt = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )
        self.tab_lda = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=Config.LDA_COMPONENTS
        )

        # --- Stage 2: Meta-Learner ---
        # Meta LDA to fuse the discriminative features
        self.meta_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        # Attribute to store class names after fitting
        self.classes_ = None

    def fit(self, X_dino, X_conv, X_tab, y):
        """
        Fit the hierarchical model on the provided training data.

        Args:
            X_dino (np.ndarray): DINOv2 features (N, D_dino).
            X_conv (np.ndarray): ConvNeXt features (N, D_conv).
            X_tab (np.ndarray): Tabular features (N, 192).
            y (np.ndarray): Target labels (N,).

        Returns:
            self: Returns the instance itself.
        """
        seed_everything()

        # --- Stage 1 Processing ---

        # 1. DINO Stream
        dino_pca_out = self.dino_pca.fit_transform(X_dino)
        dino_lda_out = self.dino_lda.fit_transform(dino_pca_out, y)

        # 2. ConvNeXt Stream
        conv_pca_out = self.conv_pca.fit_transform(X_conv)
        conv_lda_out = self.conv_lda.fit_transform(conv_pca_out, y)

        # 3. Tabular Stream
        tab_qt_out = self.tab_qt.fit_transform(X_tab)
        tab_lda_out = self.tab_lda.fit_transform(tab_qt_out, y)

        # --- Stage 2 Processing ---

        # Concatenate discriminative features
        # Shape: (N, 3 * (C-1))
        X_meta = np.hstack([dino_lda_out, conv_lda_out, tab_lda_out])

        # Fit Meta-LDA
        self.meta_lda.fit(X_meta, y)

        # Store classes for later use in prediction
        self.classes_ = self.meta_lda.classes_

        return self

    def predict_proba(self, X_dino, X_conv, X_tab):
        """
        Predict class probabilities for new data.

        Args:
            X_dino (np.ndarray): DINOv2 features (N, D_dino).
            X_conv (np.ndarray): ConvNeXt features (N, D_conv).
            X_tab (np.ndarray): Tabular features (N, 192).

        Returns:
            np.ndarray: Predicted probabilities (N, n_classes).
        """
        # --- Stage 1 Transform ---

        # DINO
        dino_pca_out = self.dino_pca.transform(X_dino)
        dino_lda_out = self.dino_lda.transform(dino_pca_out)

        # ConvNeXt
        conv_pca_out = self.conv_pca.transform(X_conv)
        conv_lda_out = self.conv_lda.transform(conv_pca_out)

        # Tabular
        tab_qt_out = self.tab_qt.transform(X_tab)
        tab_lda_out = self.tab_lda.transform(tab_qt_out)

        # --- Stage 2 Prediction ---

        # Concatenate
        X_meta = np.hstack([dino_lda_out, conv_lda_out, tab_lda_out])

        # Predict
        return self.meta_lda.predict_proba(X_meta)
