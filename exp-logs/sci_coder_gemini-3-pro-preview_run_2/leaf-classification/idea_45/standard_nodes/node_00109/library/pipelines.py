from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from library.config import (
    RANDOM_SEED,
    ROTATION_WHITEN,
    ROTATION_COMPONENTS,
    PHYSICAL_POLY_DEGREE,
    PHYSICAL_POLY_INTERACTION_ONLY,
    PHYSICAL_POLY_INCLUDE_BIAS,
)


def get_marginal_pipeline():
    """
    Constructs the pipeline for Group A: Marginal Statistical Anchors.

    Strategy:
        - Apply PowerTransformer (Yeo-Johnson) to Gaussianize features marginally.
        - This prepares the data for LDA which assumes Gaussian class-conditional densities.

    Returns:
        sklearn.pipeline.Pipeline: The preprocessing pipeline.
    """
    steps = [("pt", PowerTransformer(method="yeo-johnson", standardize=True))]
    return Pipeline(steps)


def get_rotational_pipeline():
    """
    Constructs the pipeline for Group B: Rotational Statistical Experts.

    Strategy:
        1. Gaussianize original features to handle skewness.
        2. Apply PCA strictly for rotation (whiten=False). This aligns the data with
           its principal axes, satisfying the independence assumption of the LDA
           shrinkage target (Identity Matrix) more effectively.
        3. Gaussianize the rotated components again to approximate Multivariate Normality.

    Returns:
        sklearn.pipeline.Pipeline: The preprocessing pipeline.
    """
    steps = [
        # Step 1: Initial Gaussianization
        ("pt_1", PowerTransformer(method="yeo-johnson", standardize=True)),
        # Step 2: Variance-Preserving Rotation
        # We use PCA to rotate, but strictly disable whitening to avoid noise amplification.
        (
            "pca",
            PCA(
                n_components=ROTATION_COMPONENTS,  # None -> Keep all components
                whiten=ROTATION_WHITEN,  # False -> No scaling by singular values
                random_state=RANDOM_SEED,
            ),
        ),
        # Step 3: Gaussianize the Principal Components
        ("pt_2", PowerTransformer(method="yeo-johnson", standardize=True)),
    ]
    return Pipeline(steps)


def get_polynomial_pipeline():
    """
    Constructs the pipeline for Group C: Polynomial Physical Experts.

    Strategy:
        1. Expand physical features (Morphometrics) into polynomial space (degree 2).
           This allows linear models to capture non-linear constraints (e.g., area vs perimeter).
        2. Gaussianize the expanded features to stabilize the linear solver.

    Returns:
        sklearn.pipeline.Pipeline: The preprocessing pipeline.
    """
    steps = [
        # Step 1: Polynomial Expansion
        (
            "poly",
            PolynomialFeatures(
                degree=PHYSICAL_POLY_DEGREE,
                interaction_only=PHYSICAL_POLY_INTERACTION_ONLY,
                include_bias=PHYSICAL_POLY_INCLUDE_BIAS,
            ),
        ),
        # Step 2: Gaussianization
        # Polynomial expansion creates heavy-tailed distributions; PT is crucial here.
        ("pt", PowerTransformer(method="yeo-johnson", standardize=True)),
    ]
    return Pipeline(steps)
