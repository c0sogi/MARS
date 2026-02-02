import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import RANDOM_SEED


def get_lda_estimator(shrinkage):
    """
    Creates an LDA estimator with the specified shrinkage.
    Handles solver selection and parameter mapping.

    Args:
        shrinkage: The shrinkage parameter (float, 'auto', 'ledoit_wolf', or None).

    Returns:
        LinearDiscriminantAnalysis: The configured estimator.
    """
    # Map 'ledoit_wolf' to 'auto' as sklearn uses 'auto' for Ledoit-Wolf lemma
    if shrinkage == "ledoit_wolf":
        shrinkage_param = "auto"
    else:
        shrinkage_param = shrinkage

    # Solver selection: 'svd' does not support shrinkage. 'lsqr' does.
    if shrinkage_param is None:
        solver = "svd"
    else:
        solver = "lsqr"

    return LinearDiscriminantAnalysis(
        solver=solver, shrinkage=shrinkage_param, store_covariance=False
    )


def get_rotation_block():
    """
    Constructs the rotation block: Power -> PCA(Full) -> Power.
    Used for Gaussianizing and aligning manifolds without dimensionality reduction.

    Returns:
        Pipeline: The rotation preprocessing pipeline.
    """
    return Pipeline(
        [
            ("pt_in", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("pca", PCA(n_components=None, whiten=False, random_state=RANDOM_SEED)),
            ("pt_out", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )


def get_global_pipeline(shrinkage):
    """
    Group A: Global Marginal Anchors.
    Pipeline: PowerTransformer -> LDA.

    Args:
        shrinkage: Shrinkage parameter for LDA.
    """
    return Pipeline(
        [
            ("pt", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("lda", get_lda_estimator(shrinkage)),
        ]
    )


def get_global_rotational_pipeline(shrinkage):
    """
    Group B: Global Rotational Experts.
    Pipeline: RotationBlock -> LDA.

    Args:
        shrinkage: Shrinkage parameter for LDA.
    """
    return Pipeline(
        [("rotation", get_rotation_block()), ("lda", get_lda_estimator(shrinkage))]
    )


def get_stratified_rotational_pipeline(shrinkage):
    """
    Group C: Stratified Rotational Experts.
    Input: Global View (192 features).
    Pipeline:
      - Split into Margin (0-64), Shape (64-128), Texture (128-192).
      - Apply RotationBlock to each independently.
      - Concatenate.
      - LDA.

    Args:
        shrinkage: Shrinkage parameter for LDA.
    """
    # Define indices for the 192-feature vector based on config order:
    # Margin: 0-64, Shape: 64-128, Texture: 128-192

    preprocessor = ColumnTransformer(
        transformers=[
            ("margin_rot", get_rotation_block(), slice(0, 64)),
            ("shape_rot", get_rotation_block(), slice(64, 128)),
            ("texture_rot", get_rotation_block(), slice(128, 192)),
        ],
        remainder="drop",  # Should be nothing left, but ensures safety
    )

    return Pipeline(
        [("stratified_rotation", preprocessor), ("lda", get_lda_estimator(shrinkage))]
    )


def get_morphometric_pipeline(shrinkage):
    """
    Group D: Physical Polynomial Experts.
    Input: Morphometric Features.
    Pipeline: Power -> Poly(2) -> Power -> LDA.

    Args:
        shrinkage: Shrinkage parameter for LDA.
    """
    return Pipeline(
        [
            ("pt_in", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pt_out", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("lda", get_lda_estimator(shrinkage)),
        ]
    )
