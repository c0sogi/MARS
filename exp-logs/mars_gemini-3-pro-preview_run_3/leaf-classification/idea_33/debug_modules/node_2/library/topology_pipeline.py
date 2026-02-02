import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


def build_model_pipeline(dino_dim: int, conv_dim: int, tab_dim: int) -> Pipeline:
    """
    Constructs the Selective-Topology pipeline.

    Args:
        dino_dim (int): Number of features in the DINOv2 vector.
        conv_dim (int): Number of features in the ConvNeXt vector.
        tab_dim (int): Number of features in the Tabular vector.

    Returns:
        sklearn.pipeline.Pipeline: The untrained modeling pipeline.
    """
    # Define column indices based on the assumption of concatenated inputs:
    # [DINO (0..dino_dim), ConvNeXt (dino_dim..+conv_dim), Tabular (..+tab_dim)]
    dino_start = 0
    dino_end = dino_dim

    conv_start = dino_end
    conv_end = conv_start + conv_dim

    tab_start = conv_end
    tab_end = tab_start + tab_dim

    # Generate index lists for ColumnTransformer
    dino_indices = list(range(dino_start, dino_end))
    conv_indices = list(range(conv_start, conv_end))
    tab_indices = list(range(tab_start, tab_end))

    # 1. Selective Feature Topology via ColumnTransformer
    # - Visual Streams: Linear Subspace Reduction (PCA) to preserve geometry
    # - Tabular Stream: Non-linear Gaussianization (QuantileTransformer)

    pca_dino = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

    pca_conv = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

    qt_tabular = QuantileTransformer(
        output_distribution="normal", random_state=Config.SEED
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("dino_pca", pca_dino, dino_indices),
            ("conv_pca", pca_conv, conv_indices),
            ("tabular_qt", qt_tabular, tab_indices),
        ],
        remainder="drop",  # Drop any columns not explicitly handled (should be none)
    )

    # 2. Global Variance Alignment
    # Standardize the concatenated output of PCAs and QT to ensure uniform
    # regularization penalty in the subsequent LDA.
    global_scaler = StandardScaler()

    # 3. Classifier
    # LDA with Ledoit-Wolf shrinkage for robust covariance estimation in HDLSS.
    lda = LinearDiscriminantAnalysis(
        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
    )

    # Assemble the full pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("global_scaler", global_scaler),
            ("classifier", lda),
        ]
    )

    return pipeline
