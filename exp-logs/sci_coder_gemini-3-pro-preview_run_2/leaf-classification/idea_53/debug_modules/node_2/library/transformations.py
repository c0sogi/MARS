import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import (
    INTERACTION_N_COMPONENTS,
    INTERACTION_POLY_DEGREE,
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_rotational_sub_pipeline():
    """
    Creates the standard Rotational Pipeline:
    Gaussianization -> PCA (No Whitening) -> Gaussianization.

    This aligns the data manifold along principal axes while normalizing
    distributions, without the noise amplification of whitening.
    """
    return Pipeline(
        [
            ("pt_pre", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(whiten=False)),
            ("pt_post", PowerTransformer(method="yeo-johnson")),
        ]
    )


# =============================================================================
# PIPELINE FACTORIES
# =============================================================================


def make_global_marginal_pipeline():
    """
    Topology A: Global Marginal Anchors.

    Input: Global View (192 features).
    Transformation: Simple Gaussianization via PowerTransformer.
    """
    return Pipeline([("pt", PowerTransformer(method="yeo-johnson"))])


def make_global_rotational_pipeline():
    """
    Topology B: Global Rotational Experts.

    Input: Global View (192 features).
    Transformation: Rotational Pipeline applied to the full feature set.
    """
    return _get_rotational_sub_pipeline()


def make_stratified_rotational_pipeline():
    """
    Topology C: Stratified Rotational Experts.

    Input: Global View (192 features).
    Transformation: Applies the Rotational Pipeline independently to
    Margin, Shape, and Texture subsets, then concatenates them.

    This prevents high-variance feature groups (e.g., Texture) from
    dominating the principal components of lower-variance groups.
    """
    # Define column indices based on the fixed structure of the Global View
    # 0-63: Margin, 64-127: Shape, 128-191: Texture
    margin_indices = list(range(0, 64))
    shape_indices = list(range(64, 128))
    texture_indices = list(range(128, 192))

    # Create the column transformer
    # Note: We use the same pipeline structure for each subset
    stratified_transformer = ColumnTransformer(
        transformers=[
            ("margin_rot", _get_rotational_sub_pipeline(), margin_indices),
            ("shape_rot", _get_rotational_sub_pipeline(), shape_indices),
            ("texture_rot", _get_rotational_sub_pipeline(), texture_indices),
        ],
        n_jobs=1,  # Avoid multiprocessing overhead for these relatively small subsets
    )

    return Pipeline([("stratified_rot", stratified_transformer)])


def make_discriminative_interaction_pipeline():
    """
    Topology D: Discriminative-Interaction Experts.

    Input: Global View (192 features).
    Transformation:
    1. Gaussianization
    2. LDA Projection (Supervised Dimensionality Reduction) to N components
    3. Polynomial Expansion (Degree 2) to capture quadratic interactions
    4. Gaussianization

    Note: The LDA step requires 'y' during fit, which the Pipeline handles automatically.
    """
    return Pipeline(
        [
            ("pt_pre", PowerTransformer(method="yeo-johnson")),
            (
                "lda_proj",
                LinearDiscriminantAnalysis(n_components=INTERACTION_N_COMPONENTS),
            ),
            (
                "poly",
                PolynomialFeatures(degree=INTERACTION_POLY_DEGREE, include_bias=False),
            ),
            ("pt_post", PowerTransformer(method="yeo-johnson")),
        ]
    )


def make_poly_physical_pipeline():
    """
    Topology E: Polynomial Physical Experts.

    Input: Morphometric View (Hu Moments + Geometric Scalars).
    Transformation:
    1. Gaussianization
    2. Polynomial Expansion (Degree 2) to capture physical constraints
    3. Gaussianization
    """
    return Pipeline(
        [
            ("pt_pre", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("pt_post", PowerTransformer(method="yeo-johnson")),
        ]
    )
