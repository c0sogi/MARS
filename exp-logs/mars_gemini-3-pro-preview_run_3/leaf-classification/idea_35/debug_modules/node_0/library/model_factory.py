import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="model_factory")


def create_classifier():
    """
    Constructs the Stratified Selective-Topology Orthogonal Manifold-Densified LDA pipeline.

    Architecture:
    1. Feature Specific Transformations (ColumnTransformer):
       - DINOv2 Stream (0:1024): Independent PCA (99% Variance). No non-linear distortion.
       - ConvNeXt Stream (1024:2560): Independent PCA (99% Variance). No non-linear distortion.
       - Tabular Stream (2560:2752): QuantileTransformer (Normal). Gaussianization of histograms.
    2. Global Variance Alignment:
       - StandardScaler: Aligns variance of all reduced streams to ensure uniform Ledoit-Wolf shrinkage.
    3. Classifier:
       - LDA (Solver='lsqr', Shrinkage='auto'): Robust covariance estimation for HDLSS.

    Returns:
        sklearn.pipeline.Pipeline: The constructed model pipeline.
    """
    # 1. Define Feature Slices based on Config dimensions
    # Order matches LeafDataManager._densify_and_assemble: [DINO, Conv, Tabular]
    dino_dim = 1024
    conv_dim = 1536
    tabular_dim = Config.TABULAR_FEATURE_COUNT

    # Calculate slice indices
    dino_start = 0
    dino_end = dino_dim

    conv_start = dino_end
    conv_end = conv_start + conv_dim

    tab_start = conv_end
    tab_end = tab_start + tabular_dim

    # Slices for ColumnTransformer
    # Note: ColumnTransformer accepts lists of column indices
    dino_indices = list(range(dino_start, dino_end))
    conv_indices = list(range(conv_start, conv_end))
    tab_indices = list(range(tab_start, tab_end))

    logger.info(f"Pipeline Feature Configuration:")
    logger.info(
        f"  - DINOv2 Stream:   Cols {dino_start}:{dino_end} ({len(dino_indices)} dims) -> PCA({Config.PCA_VARIANCE})"
    )
    logger.info(
        f"  - ConvNeXt Stream: Cols {conv_start}:{conv_end} ({len(conv_indices)} dims) -> PCA({Config.PCA_VARIANCE})"
    )
    logger.info(
        f"  - Tabular Stream:  Cols {tab_start}:{tab_end} ({len(tab_indices)} dims) -> QuantileTransformer"
    )

    # 2. Define Transformers

    # Visual Stream Transformer: Independent Subspace Reduction
    # We use svd_solver='full' to allow float n_components (variance ratio)
    pca_dino = PCA(
        n_components=Config.PCA_VARIANCE, svd_solver="full", random_state=Config.SEED
    )
    pca_conv = PCA(
        n_components=Config.PCA_VARIANCE, svd_solver="full", random_state=Config.SEED
    )

    # Tabular Stream Transformer: Gaussianization
    # n_quantiles set to default 1000, or n_samples if smaller.
    # With densification (N~2100), default is fine.
    qt_tabular = QuantileTransformer(
        output_distribution="normal", random_state=Config.SEED
    )

    # 3. Construct ColumnTransformer
    # This applies specific transforms to specific columns and concatenates results
    preprocessor = ColumnTransformer(
        transformers=[
            ("dino_pca", pca_dino, dino_indices),
            ("conv_pca", pca_conv, conv_indices),
            ("tabular_qt", qt_tabular, tab_indices),
        ],
        n_jobs=None,  # Debugging safety
        verbose_feature_names_out=False,
    )

    # 4. Construct Final Pipeline
    pipeline = Pipeline(
        [
            # Step 1: Selective Feature Topology
            ("preprocessor", preprocessor),
            # Step 2: Global Variance Alignment
            # Ensures that the high-dimensional visual PCA components don't dominate
            # the regularization term in LDA simply due to scale differences vs Tabular.
            ("global_scaler", StandardScaler()),
            # Step 3: Classifier
            # Linear Discriminant Analysis with Ledoit-Wolf Shrinkage
            # Ideally suited for High-Dimension Low-Sample-Size (HDLSS) problems
            ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    return pipeline
