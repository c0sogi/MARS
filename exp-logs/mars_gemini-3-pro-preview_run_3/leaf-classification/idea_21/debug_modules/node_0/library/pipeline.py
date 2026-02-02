import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import Config


def create_expert_pipeline():
    """
    Constructs the scikit-learn pipeline for a single OS-LDE expert.

    Architecture:
    1. Input: Concatenated vector [DINO(1024) | ConvNeXt(1536) | Tabular(192)]
    2. Feature Processing (ColumnTransformer):
       - Stream 1 (DINO): PCA (99% variance)
       - Stream 2 (ConvNeXt): PCA (99% variance)
       - Stream 3 (Tabular): Passthrough
    3. Gaussianization: QuantileTransformer (Normal distribution)
    4. Classifier: Linear Discriminant Analysis (Ledoit-Wolf shrinkage)

    Returns:
        sklearn.pipeline.Pipeline: The un-fitted expert pipeline.
    """
    # Define feature dimensions based on extraction logic
    dim_dino = 1024
    dim_conv = 1536
    dim_tab = 192

    # Define slice indices for ColumnTransformer
    # Input X shape: (N, 2752)
    idx_dino = slice(0, dim_dino)
    idx_conv = slice(dim_dino, dim_dino + dim_conv)
    idx_tab = slice(dim_dino + dim_conv, dim_dino + dim_conv + dim_tab)

    # Step 1: Independent Subspace Reduction
    # We apply PCA independently to the visual streams to preserve their distinct manifolds.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "dino_pca",
                PCA(
                    n_components=Config.PCA_VARIANCE,
                    svd_solver="full",
                    random_state=Config.SEED,
                ),
                idx_dino,
            ),
            (
                "conv_pca",
                PCA(
                    n_components=Config.PCA_VARIANCE,
                    svd_solver="full",
                    random_state=Config.SEED,
                ),
                idx_conv,
            ),
            ("tabular_pass", "passthrough", idx_tab),
        ],
        verbose_feature_names_out=False,
    )

    # Step 2 & 3: Gaussianization and Classification
    # QuantileTransformer ensures features follow a normal distribution, optimizing LDA performance.
    pipeline = Pipeline(
        [
            ("feature_processing", preprocessor),
            (
                "gaussianization",
                QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                ),
            ),
            (
                "classifier",
                LinearDiscriminantAnalysis(
                    solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
                ),
            ),
        ]
    )

    return pipeline
