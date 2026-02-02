import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    PowerTransformer,
    PolynomialFeatures,
    FunctionTransformer,
)
from sklearn.decomposition import PCA


def to_float64(X):
    """
    Casts input to float64 to ensure double precision throughout the pipeline.
    """
    return np.array(X, dtype=np.float64)


def get_global_pipeline():
    """
    Returns the pipeline for the Global Linear Anchor (Marginal topology).

    Pipeline Steps:
    1. Cast to float64
    2. PowerTransformer (Yeo-Johnson)
    """
    return Pipeline(
        [
            ("cast", FunctionTransformer(to_float64, validate=False)),
            ("pt", PowerTransformer(method="yeo-johnson")),
        ]
    )


def get_stratified_rotational_pipeline(feature_groups):
    """
    Returns the pipeline for the Global Linear Anchor (Stratified-Rotational topology).

    Logic:
    1. Splits Global View into Margin, Shape, and Texture subsets.
    2. Applies Power -> PCA (Full Rank) -> Power to each subset independently.
    3. Recombines the processed subsets.

    Args:
        feature_groups (dict): Dictionary mapping group names ('margin', 'shape', 'texture')
                               to lists of column indices.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    transformers = []

    # The specific semantic groups to process independently
    target_groups = ["margin", "shape", "texture"]

    for group in target_groups:
        if group in feature_groups:
            indices = feature_groups[group]
            if len(indices) > 0:
                # Sub-pipeline for the specific group
                # PCA n_components=None ensures we keep the full rank (just rotation)
                sub_pipeline = Pipeline(
                    [
                        ("pt_pre", PowerTransformer(method="yeo-johnson")),
                        ("pca", PCA(n_components=None, whiten=False)),
                        ("pt_post", PowerTransformer(method="yeo-johnson")),
                    ]
                )
                transformers.append((f"{group}_rot", sub_pipeline, indices))

    # ColumnTransformer applies the sub-pipelines to specific columns and concatenates results.
    # remainder='drop' ensures we only output the processed stratified features,
    # ignoring any extra columns (like raw morphometrics) that might be in the input matrix.
    ct = ColumnTransformer(transformers, remainder="drop")

    return Pipeline(
        [
            ("cast", FunctionTransformer(to_float64, validate=False)),
            ("stratified_transform", ct),
        ]
    )


def get_poly_pipeline(degree=2):
    """
    Returns the pipeline for Polynomial Experts (Group B and C).

    Pipeline Steps:
    1. Cast to float64
    2. Gaussianization (PowerTransformer)
    3. Full-Rank Expansion (PolynomialFeatures)
    4. Re-Gaussianization (PowerTransformer)

    Args:
        degree (int): The degree of the polynomial features.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    return Pipeline(
        [
            ("cast", FunctionTransformer(to_float64, validate=False)),
            ("pt_pre", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("pt_post", PowerTransformer(method="yeo-johnson")),
        ]
    )
