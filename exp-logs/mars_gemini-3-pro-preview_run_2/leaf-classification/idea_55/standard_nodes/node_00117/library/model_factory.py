import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
    FunctionTransformer,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.compose import ColumnTransformer

from library.config import (
    SHRINKAGE_GRID,
    LDA_SOLVER,
    N_QUANTILES,
    POLY_DEGREE,
    FLOAT_PRECISION,
    YEO_JOHNSON_STANDARDIZE,
)
from library.custom_transformers import StratifiedDiscriminantProjector

# =============================================================================
# COLUMN SLICING CONFIGURATION
# =============================================================================
# The input matrix X contains 204 columns:
# - Indices 0-191: Global Features (Margin, Shape, Texture)
# - Indices 192-203: Morphometric Features (Hu Moments + Geometric Scalars)
GLOBAL_SLICE = slice(0, 192)
MORPH_SLICE = slice(192, 204)


def _get_column_selector(slice_obj):
    """
    Creates a ColumnTransformer to select specific feature slices.
    Using ColumnTransformer ensures compatibility with sklearn pipelines.
    """
    return ColumnTransformer(
        [("selector", "passthrough", slice_obj)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_expert_library():
    """
    Constructs the library of expert pipelines for the SDPGE ensemble.

    Returns:
        list of tuples: A list where each element is (expert_name, pipeline_object).
    """
    experts = []

    # =========================================================================
    # TOPOLOGY A: GLOBAL STATISTICAL ANCHORS (The Baseline)
    # =========================================================================
    # Input: Global View (192 features)
    # Algorithms: LDA with Fixed Shrinkage (0.001, 0.01)
    # Preprocessing: Marginal, Rotational, Robust

    # Define base preprocessing steps for Global View
    preprocessors_a = {
        "Marginal": [
            (
                "yeo_johnson",
                PowerTransformer(
                    method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE
                ),
            )
        ],
        "Rotational": [
            (
                "yeo_johnson_1",
                PowerTransformer(
                    method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE
                ),
            ),
            ("pca", PCA(whiten=False)),
            (
                "yeo_johnson_2",
                PowerTransformer(
                    method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE
                ),
            ),
        ],
        "Robust": [
            (
                "quantile",
                QuantileTransformer(
                    output_distribution="normal", n_quantiles=N_QUANTILES
                ),
            )
        ],
    }

    # Fixed shrinkage values for Topology A
    fixed_shrinkages = [0.001, 0.01]

    for prep_name, steps in preprocessors_a.items():
        for shrinkage in fixed_shrinkages:
            name = f"TopoA_{prep_name}_LDA_{shrinkage}"

            # Construct Pipeline
            # 1. Select Global Features
            # 2. Apply Preprocessing
            # 3. Apply LDA
            pipeline_steps = [("select_global", _get_column_selector(GLOBAL_SLICE))]
            pipeline_steps.extend(steps)
            pipeline_steps.append(
                (
                    "lda",
                    LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=shrinkage),
                )
            )

            experts.append((name, Pipeline(pipeline_steps)))

    # =========================================================================
    # TOPOLOGY B: PHYSICAL POLYNOMIAL EXPERTS (The Domain Signal)
    # =========================================================================
    # Input: Polarity-Corrected Morphometrics (12 features)
    # Algorithm: LDA (Ledoit-Wolf -> shrinkage='auto')
    # Preprocessing: PowerTransformer -> Poly(2) -> PowerTransformer

    name_b = "TopoB_Physical_Poly_LDA_Auto"

    pipeline_steps_b = [
        ("select_morph", _get_column_selector(MORPH_SLICE)),
        (
            "pt_1",
            PowerTransformer(method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE),
        ),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        (
            "pt_2",
            PowerTransformer(method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE),
        ),
        (
            "lda",
            LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage="auto"),
        ),
    ]

    experts.append((name_b, Pipeline(pipeline_steps_b)))

    # =========================================================================
    # TOPOLOGY C: STRATIFIED DISCRIMINATIVE-INTERACTION EXPERTS (The Innovation)
    # =========================================================================
    # Input: Global View (192 features)
    # Algorithm: LDA with Shrinkage Library (Grid)
    # Pipeline:
    #   1. Stratified Projection (Margin, Shape, Texture -> LDA subspaces)
    #   2. Interaction Expansion (Poly degree=2, interaction_only=True)
    #   3. Re-Gaussianization (PowerTransformer)

    # Note: StratifiedDiscriminantProjector handles the internal slicing of the 192 features.
    # We pass shrinkage=None to the projector to use standard LDA projection without shrinkage
    # for the dimensionality reduction step, or we can use a small fixed shrinkage.
    # The prompt implies the projector projects onto discriminative axes.
    # We'll use a small fixed shrinkage (0.001) for stability in the projection phase.

    projector_shrinkage = 0.001

    for shrinkage in SHRINKAGE_GRID:
        name = f"TopoC_Stratified_Interact_LDA_{shrinkage}"

        pipeline_steps_c = [
            ("select_global", _get_column_selector(GLOBAL_SLICE)),
            (
                "stratified_projector",
                StratifiedDiscriminantProjector(
                    shrinkage=projector_shrinkage, solver=LDA_SOLVER
                ),
            ),
            (
                "interaction",
                PolynomialFeatures(
                    degree=POLY_DEGREE, interaction_only=True, include_bias=False
                ),
            ),
            (
                "pt",
                PowerTransformer(
                    method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE
                ),
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=shrinkage),
            ),
        ]

        experts.append((name, Pipeline(pipeline_steps_c)))

    return experts
