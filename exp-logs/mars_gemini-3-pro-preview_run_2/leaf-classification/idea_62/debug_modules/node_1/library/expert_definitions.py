import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import library.config as conf


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a specific range of columns from a numpy array.
    """

    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.start_idx : self.end_idx]


def get_expert_library():
    """
    Constructs and returns the library of HDB-PGE expert pipelines.

    Returns:
        dict: A dictionary where keys are expert names and values are sklearn Pipelines.
    """
    experts = {}

    # Indices based on the dataset structure:
    # 0-63: Margin, 64-127: Shape, 128-191: Texture (Total 192 Global Features)
    # 192-202: Morphometrics (11 Features)
    GLOBAL_START, GLOBAL_END = 0, 192
    MORPH_START, MORPH_END = 192, 203

    MARGIN_SLICE = slice(0, 64)
    SHAPE_SLICE = slice(64, 128)
    TEXTURE_SLICE = slice(128, 192)

    # Iterate over shrinkage candidates to create variations of experts
    for shrinkage in conf.LDA_SHRINKAGE_CANDIDATES:
        shrink_str = str(shrinkage).replace(".", "")

        # Common final estimator for all pipelines
        def get_final_estimator():
            return LinearDiscriminantAnalysis(
                solver=conf.LDA_SOLVER, shrinkage=shrinkage
            )

        # =========================================================================
        # Group A: Global Statistical Anchors
        # =========================================================================

        # A1. Marginal Topology: Power -> LDA
        name_a1 = f"GroupA_Marginal_Shrink{shrink_str}"
        experts[name_a1] = Pipeline(
            [
                ("selector", ColumnSelector(GLOBAL_START, GLOBAL_END)),
                ("scaler", PowerTransformer(method=conf.POWER_METHOD)),
                ("clf", get_final_estimator()),
            ]
        )

        # A2. Rotational Topology: Power -> PCA -> Power -> LDA
        name_a2 = f"GroupA_Rotational_Shrink{shrink_str}"
        experts[name_a2] = Pipeline(
            [
                ("selector", ColumnSelector(GLOBAL_START, GLOBAL_END)),
                ("scaler1", PowerTransformer(method=conf.POWER_METHOD)),
                ("pca", PCA(whiten=False)),  # Full rank PCA
                ("scaler2", PowerTransformer(method=conf.POWER_METHOD)),
                ("clf", get_final_estimator()),
            ]
        )

        # A3. Robust Topology: Quantile -> LDA
        name_a3 = f"GroupA_Robust_Shrink{shrink_str}"
        experts[name_a3] = Pipeline(
            [
                ("selector", ColumnSelector(GLOBAL_START, GLOBAL_END)),
                (
                    "scaler",
                    QuantileTransformer(
                        output_distribution=conf.QUANTILE_OUTPUT_DIST,
                        n_quantiles=conf.QUANTILE_N_QUANTILES,
                        random_state=conf.RANDOM_SEED,
                    ),
                ),
                ("clf", get_final_estimator()),
            ]
        )

        # =========================================================================
        # Group B: Physical Polynomial Experts
        # =========================================================================

        # B1. Poly-Physical: Morphometrics -> Power -> Poly -> Power -> LDA
        name_b = f"GroupB_Physical_Shrink{shrink_str}"
        experts[name_b] = Pipeline(
            [
                ("selector", ColumnSelector(MORPH_START, MORPH_END)),
                ("scaler1", PowerTransformer(method=conf.POWER_METHOD)),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=conf.POLY_DEGREE,
                        interaction_only=conf.POLY_INTERACTION_ONLY,
                        include_bias=conf.POLY_INCLUDE_BIAS,
                    ),
                ),
                ("scaler2", PowerTransformer(method=conf.POWER_METHOD)),
                ("clf", get_final_estimator()),
            ]
        )

        # =========================================================================
        # Group C: Global Discriminative-Interaction Experts
        # =========================================================================

        # C1. Global Bottleneck: Global -> Power -> LDA(15) -> Poly -> Power -> LDA
        name_c = f"GroupC_GlobalInter_Shrink{shrink_str}"
        experts[name_c] = Pipeline(
            [
                ("selector", ColumnSelector(GLOBAL_START, GLOBAL_END)),
                ("scaler1", PowerTransformer(method=conf.POWER_METHOD)),
                # Bottleneck projection using supervised LDA
                (
                    "bottleneck",
                    LinearDiscriminantAnalysis(n_components=conf.LDA_COMPONENTS_GLOBAL),
                ),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=conf.POLY_DEGREE,
                        interaction_only=conf.POLY_INTERACTION_ONLY,
                        include_bias=conf.POLY_INCLUDE_BIAS,
                    ),
                ),
                ("scaler2", PowerTransformer(method=conf.POWER_METHOD)),
                ("clf", get_final_estimator()),
            ]
        )

        # =========================================================================
        # Group D: Stratified Discriminative-Interaction Experts
        # =========================================================================

        # Define the stratified reducer using ColumnTransformer
        # Each branch: Power -> LDA(9)
        stratified_reducer = ColumnTransformer(
            [
                (
                    "margin_branch",
                    Pipeline(
                        [
                            ("pt", PowerTransformer(method=conf.POWER_METHOD)),
                            (
                                "lda",
                                LinearDiscriminantAnalysis(
                                    n_components=conf.LDA_COMPONENTS_STRATIFIED
                                ),
                            ),
                        ]
                    ),
                    MARGIN_SLICE,
                ),
                (
                    "shape_branch",
                    Pipeline(
                        [
                            ("pt", PowerTransformer(method=conf.POWER_METHOD)),
                            (
                                "lda",
                                LinearDiscriminantAnalysis(
                                    n_components=conf.LDA_COMPONENTS_STRATIFIED
                                ),
                            ),
                        ]
                    ),
                    SHAPE_SLICE,
                ),
                (
                    "texture_branch",
                    Pipeline(
                        [
                            ("pt", PowerTransformer(method=conf.POWER_METHOD)),
                            (
                                "lda",
                                LinearDiscriminantAnalysis(
                                    n_components=conf.LDA_COMPONENTS_STRATIFIED
                                ),
                            ),
                        ]
                    ),
                    TEXTURE_SLICE,
                ),
            ]
        )

        # D1. Stratified Bottleneck: Global -> Split(Power->LDA(9)) -> Concat -> Poly -> Power -> LDA
        name_d = f"GroupD_StratifiedInter_Shrink{shrink_str}"
        experts[name_d] = Pipeline(
            [
                ("selector", ColumnSelector(GLOBAL_START, GLOBAL_END)),
                ("stratified_bottleneck", stratified_reducer),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=conf.POLY_DEGREE,
                        interaction_only=conf.POLY_INTERACTION_ONLY,
                        include_bias=conf.POLY_INCLUDE_BIAS,
                    ),
                ),
                ("scaler", PowerTransformer(method=conf.POWER_METHOD)),
                ("clf", get_final_estimator()),
            ]
        )

    return experts
