import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


def create_pipeline(dino_dim, conv_dim, tabular_dim=192):
    """
    Constructs the Stratified Independent-Subspace Linear Discriminant pipeline.

    Architecture:
    1. Independent Subspace Reduction (ColumnTransformer):
       - Stream A (DINOv2): PCA (Variance Retention = 99%)
       - Stream B (ConvNeXt): PCA (Variance Retention = 99%)
       - Stream C (Tabular): Passthrough
    2. Global Gaussianization: QuantileTransformer (Output = Normal)
    3. Classifier: Linear Discriminant Analysis (Ledoit-Wolf Shrinkage)

    Args:
        dino_dim (int): Number of features in the DINOv2 stream (e.g., 1024).
        conv_dim (int): Number of features in the ConvNeXt stream (e.g., 1536).
        tabular_dim (int): Number of tabular features (default 192).

    Returns:
        sklearn.pipeline.Pipeline: The constructed model pipeline.
    """

    # Calculate column indices for each stream based on concatenation order:
    # [DINO features | ConvNeXt features | Tabular features]
    dino_slice = slice(0, dino_dim)
    conv_slice = slice(dino_dim, dino_dim + conv_dim)
    tabular_slice = slice(dino_dim + conv_dim, dino_dim + conv_dim + tabular_dim)

    # 1. Independent Subspace Reduction
    # We apply PCA separately to visual streams to preserve their distinct manifold structures.
    # Tabular features are already low-dimensional and semantically distinct, so we keep them as is.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "pca_dino",
                PCA(n_components=Config.PCA_VARIANCE, svd_solver="full"),
                dino_slice,
            ),
            (
                "pca_conv",
                PCA(n_components=Config.PCA_VARIANCE, svd_solver="full"),
                conv_slice,
            ),
            (
                "passthrough_tabular",
                "passthrough",
                tabular_slice,
            ),
        ],
        verbose_feature_names_out=False,
    )

    # 2. Global Gaussianization & 3. Classifier
    # QuantileTransformer enforces normality for LDA.
    # LDA with shrinkage handles the HDLSS (High Dimension, Low Sample Size) condition.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "gaussianizer",
                QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=1000,
                    random_state=Config.SEED,
                ),
            ),
            (
                "classifier",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )

    return pipeline
