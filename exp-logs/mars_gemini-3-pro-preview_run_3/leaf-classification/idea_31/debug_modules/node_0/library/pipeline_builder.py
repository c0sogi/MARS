import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config


def build_selective_pipeline(
    dino_dim,
    conv_dim,
    tab_dim,
    pca_variance=config.PCA_VARIANCE_THRESHOLD,
    qt_dist=config.TABULAR_QT_DISTRIBUTION,
    lda_solver=config.LDA_SOLVER,
    lda_shrinkage=config.LDA_SHRINKAGE,
):
    """
    Constructs the Selective Feature Topology pipeline.

    Architecture:
    1. Split X into [DINO | CONV | TABULAR] based on dimensions.
    2. Visual Streams: Independent PCA (whiten=False) to reduce dimensions while preserving topology.
    3. Tabular Stream: QuantileTransformer to Gaussianize distributions.
    4. Concatenate processed streams.
    5. Global StandardScaler to align variance for shrinkage regularization.
    6. LDA Classifier with Ledoit-Wolf shrinkage.

    Args:
        dino_dim (int): Number of columns for DINO features.
        conv_dim (int): Number of columns for ConvNeXt features.
        tab_dim (int): Number of columns for Tabular features.
        pca_variance (float): Variance retention for PCA (default: from config).
        qt_dist (str): Output distribution for QuantileTransformer (default: from config).
        lda_solver (str): Solver for LDA (default: from config).
        lda_shrinkage (str): Shrinkage parameter for LDA (default: from config).

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """

    # Calculate slice indices assuming X is concatenated as [DINO, CONV, TABULAR]
    dino_end = dino_dim
    conv_end = dino_dim + conv_dim
    tab_end = conv_end + tab_dim

    dino_slice = slice(0, dino_end)
    conv_slice = slice(dino_end, conv_end)
    tab_slice = slice(conv_end, tab_end)

    # 1. Independent Subspace Reduction (Visual Streams)
    # strictly preserve linear topology (whiten=False)
    # svd_solver='full' required for fractional n_components
    dino_pca = PCA(
        n_components=pca_variance,
        whiten=False,
        svd_solver="full",
        random_state=config.SEED,
    )
    conv_pca = PCA(
        n_components=pca_variance,
        whiten=False,
        svd_solver="full",
        random_state=config.SEED,
    )

    # 2. Tabular Gaussianization (Tabular Stream)
    # Align arbitrary histograms to Normal distribution
    tab_qt = QuantileTransformer(output_distribution=qt_dist, random_state=config.SEED)

    # Define the parallel processing block
    preprocessor = ColumnTransformer(
        transformers=[
            ("dino_pca", dino_pca, dino_slice),
            ("conv_pca", conv_pca, conv_slice),
            ("tab_qt", tab_qt, tab_slice),
        ],
        verbose_feature_names_out=False,
    )

    # 3. Global Variance Alignment
    # Standardize the concatenated vector [PCA_DINO, PCA_CONV, QT_TAB]
    # This ensures Ledoit-Wolf shrinkage applies uniformly across modalities
    global_scaler = StandardScaler()

    # 4. Classifier
    # Linear Discriminant Analysis with shrinkage for HDLSS stability
    classifier = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)

    # Assemble Pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("global_scaler", global_scaler),
            ("classifier", classifier),
        ]
    )

    return pipeline
