import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


def create_hybrid_pipeline(dino_dim: int, conv_dim: int, tab_dim: int) -> Pipeline:
    """
    Constructs the hybrid pipeline with independent PCA for visual streams
    and QuantileTransformer for the tabular stream, followed by LDA.

    The input matrix X is expected to be a horizontal concatenation of:
    [DINO_Features (dino_dim) | ConvNeXt_Features (conv_dim) | Tabular_Features (tab_dim)]

    Args:
        dino_dim (int): Number of features in the DINOv2 stream.
        conv_dim (int): Number of features in the ConvNeXt stream.
        tab_dim (int): Number of features in the tabular stream.

    Returns:
        Pipeline: The constructed Scikit-learn pipeline.
    """
    # Calculate column indices for slicing the concatenated input array
    start_dino = 0
    end_dino = dino_dim

    start_conv = end_dino
    end_conv = start_conv + conv_dim

    start_tab = end_conv
    end_tab = start_tab + tab_dim

    # Define slices
    dino_slice = slice(start_dino, end_dino)
    conv_slice = slice(start_conv, end_conv)
    tab_slice = slice(start_tab, end_tab)

    # 1. Preprocessing Step: Selective Feature Topology
    # - Visual Streams: Linear PCA (Preserve Manifold Geometry)
    # - Tabular Stream: Non-Linear Quantile Transform (Gaussianization)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "dino_pca",
                PCA(
                    n_components=Config.PCA_VARIANCE,
                    svd_solver="full",
                    random_state=Config.SEED,
                ),
                dino_slice,
            ),
            (
                "conv_pca",
                PCA(
                    n_components=Config.PCA_VARIANCE,
                    svd_solver="full",
                    random_state=Config.SEED,
                ),
                conv_slice,
            ),
            (
                "tab_qt",
                QuantileTransformer(
                    output_distribution=Config.TABULAR_TRANSFORM_DIST,
                    random_state=Config.SEED,
                ),
                tab_slice,
            ),
        ],
        remainder="drop",  # Drop any columns not explicitly sliced (safety)
        verbose_feature_names_out=False,
    )

    # 2. Classification Step: Linear Discriminant Analysis
    # Using Ledoit-Wolf shrinkage ('auto') with Least Squares solver ('lsqr')
    # This is robust for high-dimensional data where N < P or N ~ P
    classifier = LinearDiscriminantAnalysis(
        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
    )

    # Assemble the full pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    return pipeline
