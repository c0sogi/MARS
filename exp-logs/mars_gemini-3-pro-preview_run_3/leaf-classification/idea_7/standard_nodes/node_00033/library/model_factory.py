import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


def create_hybrid_pipeline(dino_dim, conv_dim, tabular_dim):
    """
    Constructs a scikit-learn Pipeline that applies specific transformations to
    different segments of the input feature vector and feeds the result into an
    LDA classifier.

    Input Feature Vector Structure:
    [ DINO_Features (dino_dim) | ConvNeXt_Features (conv_dim) | Tabular_Features (tabular_dim) ]

    Pipeline Steps:
    1. Preprocessing (ColumnTransformer):
       - Stream A (DINO): PCA (99% variance, Whitening)
       - Stream B (ConvNeXt): PCA (99% variance, Whitening)
       - Tabular: QuantileTransformer (Normal distribution)
    2. Classifier:
       - LinearDiscriminantAnalysis (Solver=lsqr, Shrinkage=auto/Ledoit-Wolf)

    Args:
        dino_dim (int): Number of features in the DINOv2 embedding.
        conv_dim (int): Number of features in the ConvNeXt embedding.
        tabular_dim (int): Number of tabular features.

    Returns:
        sklearn.pipeline.Pipeline: The constructed hybrid pipeline.
    """

    # Define column indices for each feature block
    # The input X is assumed to be a horizontal concatenation of [DINO, CONV, TABULAR]
    start_dino = 0
    end_dino = start_dino + dino_dim

    start_conv = end_dino
    end_conv = start_conv + conv_dim

    start_tab = end_conv
    end_tab = start_tab + tabular_dim

    # Define slices for ColumnTransformer
    # Using list of indices is robust for numpy arrays
    dino_indices = list(range(start_dino, end_dino))
    conv_indices = list(range(start_conv, end_conv))
    tab_indices = list(range(start_tab, end_tab))

    # 1. Define Transformers

    # Stream A: Global Geometry (DINOv2) -> PCA
    # Retain 99% variance (Config.PCA_VARIANCE) and apply whitening
    pca_dino = PCA(
        n_components=Config.PCA_VARIANCE, whiten=True, random_state=Config.SEED
    )

    # Stream B: Local Margin/Texture (ConvNeXt) -> PCA
    # Retain 99% variance and apply whitening
    pca_conv = PCA(
        n_components=Config.PCA_VARIANCE, whiten=True, random_state=Config.SEED
    )

    # Tabular Features -> Quantile Transformer
    # Transform to Normal distribution to satisfy LDA assumptions
    # n_quantiles defaults to 1000 or n_samples, which is safe here
    qt_tabular = QuantileTransformer(
        output_distribution="normal", random_state=Config.SEED
    )

    # 2. Compose Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("dino_pca", pca_dino, dino_indices),
            ("conv_pca", pca_conv, conv_indices),
            ("tabular_qt", qt_tabular, tab_indices),
        ],
        verbose_feature_names_out=False,
    )

    # 3. Define Classifier
    # LDA with Ledoit-Wolf shrinkage (shrinkage='auto' with solver='lsqr')
    lda = LinearDiscriminantAnalysis(
        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
    )

    # 4. Build Pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", lda)])

    return pipeline
