import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import (
    PCA_VARIANCE_THRESHOLD,
    POLY_DEGREE,
    LDA_SUBSPACE_COMPONENTS,
    RANDOM_SEED,
)


def create_preprocessing_pipeline(
    pipeline_type,
    pca_variance=PCA_VARIANCE_THRESHOLD,
    poly_degree=POLY_DEGREE,
    lda_components=LDA_SUBSPACE_COMPONENTS,
):
    """
    Constructs an sklearn preprocessing pipeline based on the specified topology type.

    These pipelines correspond to the expert definitions in the CW-PPGE strategy.
    They handle feature transformation (Gaussianization, Rotation, Expansion) before
    the final estimator is applied.

    Args:
        pipeline_type (str): The type of pipeline to create.
                             Valid options: 'marginal_linear', 'rotational_linear',
                             'physical_poly', 'component_poly', 'subspace_poly'.
        pca_variance (float, optional): Variance threshold for PCA in component_poly.
                                        Defaults to config value.
        poly_degree (int, optional): Degree for PolynomialFeatures.
                                     Defaults to config value.
        lda_components (int, optional): Number of components for LDA dimensionality reduction.
                                        Defaults to config value.

    Returns:
        sklearn.pipeline.Pipeline: A constructed pipeline object ready to be fit.
    """

    # All pipelines start with an initial Gaussianization to stabilize input distributions
    steps = [("pt_initial", PowerTransformer(method="yeo-johnson"))]

    if pipeline_type == "marginal_linear":
        # Group A: Baseline Linear
        # Topology: PT -> [Estimator]
        # No additional preprocessing steps required after initial PT
        pass

    elif pipeline_type == "rotational_linear":
        # Group A: Rotational Linear
        # Topology: PT -> PCA(whiten=False) -> PT -> [Estimator]
        # Aligns data to principal axes without discarding variance, reducing noise
        steps.append(
            ("pca_rot", PCA(n_components=None, whiten=False, random_state=RANDOM_SEED))
        )
        steps.append(("pt_post_rot", PowerTransformer(method="yeo-johnson")))

    elif pipeline_type == "physical_poly":
        # Group B: Physical Polynomial Experts
        # Topology: PT -> Poly(2) -> PT -> [Estimator]
        # Captures non-linear constraints (e.g., Solidity * Eccentricity)
        steps.append(
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))
        )
        steps.append(("pt_post_poly", PowerTransformer(method="yeo-johnson")))

    elif pipeline_type == "component_poly":
        # Group C: Component-Wise Polynomial Experts
        # Topology: PT -> PCA(0.95) -> Poly(2) -> PT -> [Estimator]
        # Densifies sparse/high-dim component features before polynomial expansion
        steps.append(
            ("pca_dense", PCA(n_components=pca_variance, random_state=RANDOM_SEED))
        )
        steps.append(
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))
        )
        steps.append(("pt_post_poly", PowerTransformer(method="yeo-johnson")))

    elif pipeline_type == "subspace_poly":
        # Group D: Discriminative-Subspace Expert
        # Topology: PT -> LDA(15) -> Poly(2) -> PT -> [Estimator]
        # Projects global features to class-discriminative subspace before expansion
        # Note: This pipeline requires 'y' during fit()
        steps.append(
            ("lda_reduce", LinearDiscriminantAnalysis(n_components=lda_components))
        )
        steps.append(
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))
        )
        steps.append(("pt_post_poly", PowerTransformer(method="yeo-johnson")))

    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")

    return Pipeline(steps)
