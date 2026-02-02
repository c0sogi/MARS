import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.transformers import Float64Wrapper, GroupedLDAReducer


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from the input data based on indices.
    """

    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure indices are valid and return the subset
        return X[:, self.indices]


def get_expert_library(feature_indices):
    """
    Constructs the library of probabilistic experts based on the FBPGE strategy.

    Args:
        feature_indices (dict): Dictionary mapping semantic group names ('margin', 'shape',
                                'texture', 'physical') to lists of column indices.

    Returns:
        list: A list of tuples (expert_name, pipeline_object).
    """
    experts = []

    # 1. Define Feature Subsets
    # Group A (Global) uses Margin + Shape + Texture (The original 192 features)
    global_indices = (
        feature_indices["margin"]
        + feature_indices["shape"]
        + feature_indices["texture"]
    )
    global_indices.sort()  # Ensure deterministic order

    # Group B (Physical) uses Morphometrics
    physical_indices = feature_indices["physical"]

    # 2. Define Hyperparameters
    shrinkage_fixed = [0.001, 0.01]
    shrinkage_full = [0.001, 0.01, "auto"]

    # ==========================================================================
    # Group A: Global Statistical Anchors
    # ==========================================================================
    # Topologies: Marginal, Rotational, Robust

    # A1. Marginal Topology: Power -> LDA
    for s in shrinkage_fixed:
        name = f"Global_Marginal_LDA_s{s}"
        steps = [
            ("float64", Float64Wrapper()),
            ("selector", ColumnSelector(global_indices)),
            ("scaler", PowerTransformer(method="yeo-johnson")),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)),
        ]
        experts.append((name, Pipeline(steps)))

    # A2. Rotational Topology: Power -> PCA -> Power -> LDA
    # Note: PCA without whitening, followed by PowerTransformer approximates
    # multivariate normality better than whitening alone.
    for s in shrinkage_fixed:
        name = f"Global_Rotational_LDA_s{s}"
        steps = [
            ("float64", Float64Wrapper()),
            ("selector", ColumnSelector(global_indices)),
            ("pre_scaler", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(n_components=0.99, whiten=False, random_state=42)),
            ("post_scaler", PowerTransformer(method="yeo-johnson")),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)),
        ]
        experts.append((name, Pipeline(steps)))

    # A3. Robust Topology: Quantile -> LDA
    for s in shrinkage_fixed:
        name = f"Global_Robust_LDA_s{s}"
        steps = [
            ("float64", Float64Wrapper()),
            ("selector", ColumnSelector(global_indices)),
            (
                "scaler",
                QuantileTransformer(
                    output_distribution="normal", n_quantiles=50, random_state=42
                ),
            ),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)),
        ]
        experts.append((name, Pipeline(steps)))

    # ==========================================================================
    # Group B: Physical Polynomial Experts
    # ==========================================================================
    # Input: Physical Features
    # Pipeline: Power -> Poly(2) -> Power -> LDA(Ledoit-Wolf)

    name = "Physical_Poly_LDA_Auto"
    steps = [
        ("float64", Float64Wrapper()),
        ("selector", ColumnSelector(physical_indices)),
        ("pre_scaler", PowerTransformer(method="yeo-johnson")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("post_scaler", PowerTransformer(method="yeo-johnson")),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ]
    experts.append((name, Pipeline(steps)))

    # ==========================================================================
    # Group C: Factorized-Bottleneck Interaction Experts
    # ==========================================================================
    # Input: All Features (handled by GroupedLDAReducer via indices)
    # Pipeline: GroupedLDA -> Poly(2) -> Power -> LDA

    for s in shrinkage_full:
        name = f"Interaction_Factorized_LDA_s{s}"
        steps = [
            ("float64", Float64Wrapper()),
            # No ColumnSelector needed here; GroupedLDAReducer takes the full matrix
            # and uses feature_indices to slice internally.
            (
                "bottleneck",
                GroupedLDAReducer(
                    feature_indices=feature_indices,
                    n_components=5,
                    solver="svd",
                    shrinkage=None,
                ),
            ),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", PowerTransformer(method="yeo-johnson")),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s)),
        ]
        experts.append((name, Pipeline(steps)))

    return experts
