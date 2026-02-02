import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CrossDomainInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Computes pairwise cross-domain interactions between feature groups.
    Assumes input is a concatenation of [Group1, Group2, Group3] (Margin, Shape, Texture).
    """

    def __init__(self, group_sizes=[10, 10, 10]):
        self.group_sizes = group_sizes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Validate shape
        expected_cols = sum(self.group_sizes)
        if X.shape[1] != expected_cols:
            # Fallback or error; for safety in pipeline we raise error
            raise ValueError(f"Expected {expected_cols} columns, got {X.shape[1]}")

        # Split into groups based on known sizes
        g1_end = self.group_sizes[0]
        g2_end = g1_end + self.group_sizes[1]

        # G1: Margin, G2: Shape, G3: Texture
        G1 = X[:, :g1_end]
        G2 = X[:, g1_end:g2_end]
        G3 = X[:, g2_end:]

        # Compute Interactions (Outer products flattened)
        # We use broadcasting: (N, d1, 1) * (N, 1, d2) -> (N, d1, d2) -> reshape (N, d1*d2)

        # Margin x Shape
        I_MS = (G1[:, :, None] * G2[:, None, :]).reshape(X.shape[0], -1)

        # Margin x Texture
        I_MT = (G1[:, :, None] * G3[:, None, :]).reshape(X.shape[0], -1)

        # Shape x Texture
        I_ST = (G2[:, :, None] * G3[:, None, :]).reshape(X.shape[0], -1)

        # Concatenate: Linear Terms + Cross-Interactions
        # We explicitly model dependencies *between* domains while keeping domain signals.
        X_new = np.hstack([G1, G2, G3, I_MS, I_MT, I_ST])

        return X_new


def build_global_pipeline(feature_indices, shrinkage=0.01):
    """
    Group A: Global Statistical Anchors (Marginal).
    Pipeline: Select -> Power -> LDA.
    """
    steps = [
        (
            "selector",
            ColumnTransformer(
                [("sel", "passthrough", feature_indices)], remainder="drop"
            ),
        ),
        ("scaler", PowerTransformer(method="yeo-johnson")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)),
    ]
    return Pipeline(steps)


def build_global_rotational_pipeline(feature_indices, shrinkage=0.01):
    """
    Group A: Global Rotational.
    Pipeline: Select -> Power -> PCA(NoWhiten) -> Power -> LDA.
    """
    steps = [
        (
            "selector",
            ColumnTransformer(
                [("sel", "passthrough", feature_indices)], remainder="drop"
            ),
        ),
        ("scaler1", PowerTransformer(method="yeo-johnson")),
        ("pca", PCA(whiten=False)),  # Keep all components, just align
        ("scaler2", PowerTransformer(method="yeo-johnson")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)),
    ]
    return Pipeline(steps)


def build_stratified_rotational_pipeline(groups, shrinkage=0.01):
    """
    Group B: Stratified Rotational Experts.
    Applies independent rotation to Margin, Shape, and Texture subsets.
    """

    # Define the rotation sub-pipeline
    def make_rot_pipe():
        return Pipeline(
            [
                ("scaler1", PowerTransformer(method="yeo-johnson")),
                ("pca", PCA(whiten=False)),
                ("scaler2", PowerTransformer(method="yeo-johnson")),
            ]
        )

    # Apply to each group independently
    preprocessor = ColumnTransformer(
        [
            ("margin_rot", make_rot_pipe(), groups["margin"]),
            ("shape_rot", make_rot_pipe(), groups["shape"]),
            ("texture_rot", make_rot_pipe(), groups["texture"]),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)),
        ]
    )
    return pipeline


def build_factorized_interaction_pipeline(groups, shrinkage=0.01, n_lda_components=10):
    """
    Group C: Factorized Discriminative-Interaction Experts.
    Applies discriminative bottleneck to groups, then computes cross-interactions.
    """

    # Define the bottleneck sub-pipeline
    # LDA as transformer requires y, which ColumnTransformer handles.
    def make_bottleneck_pipe():
        return Pipeline(
            [
                ("scaler", PowerTransformer(method="yeo-johnson")),
                ("lda_dim", LinearDiscriminantAnalysis(n_components=n_lda_components)),
            ]
        )

    # 1. Discriminative Bottlenecks
    preprocessor = ColumnTransformer(
        [
            ("margin_lda", make_bottleneck_pipe(), groups["margin"]),
            ("shape_lda", make_bottleneck_pipe(), groups["shape"]),
            ("texture_lda", make_bottleneck_pipe(), groups["texture"]),
        ]
    )

    # 2. Interaction Synthesis & Re-Gaussianization & Final LDA
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "interactions",
                CrossDomainInteractionTransformer(
                    group_sizes=[n_lda_components, n_lda_components, n_lda_components]
                ),
            ),
            ("re_gaussian", PowerTransformer(method="yeo-johnson")),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)),
        ]
    )
    return pipeline


def build_morphometric_pipeline(morph_indices, shrinkage=0.01):
    """
    Group D: Physical Polynomial Experts.
    Pipeline: Select -> Power -> Poly(2) -> Power -> LDA.
    """
    steps = [
        (
            "selector",
            ColumnTransformer(
                [("sel", "passthrough", morph_indices)], remainder="drop"
            ),
        ),
        ("scaler1", PowerTransformer(method="yeo-johnson")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler2", PowerTransformer(method="yeo-johnson")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)),
    ]
    return Pipeline(steps)
