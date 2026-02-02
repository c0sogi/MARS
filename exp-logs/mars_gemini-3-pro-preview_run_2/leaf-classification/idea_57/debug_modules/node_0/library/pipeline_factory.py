import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def build_global_linear(shrinkage="auto", solver="lsqr"):
    """
    Constructs the Global Linear Anchor pipeline.
    Topology: PowerTransformer -> LDA

    Args:
        shrinkage (str or float): Regularization parameter for LDA. 'auto' uses Ledoit-Wolf.
        solver (str): LDA solver ('lsqr' or 'eigen' required for shrinkage).

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    steps = [
        ("scaler", PowerTransformer(method="yeo-johnson")),
        ("classifier", LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)),
    ]
    return Pipeline(steps)


def build_rotational(shrinkage="auto", solver="lsqr", n_pca_components=None):
    """
    Constructs the Rotational Expert pipeline.
    Topology: PowerTransformer -> PCA(whiten=False) -> PowerTransformer -> LDA

    Args:
        shrinkage (str or float): Regularization parameter for LDA.
        solver (str): LDA solver.
        n_pca_components (int or float, optional): Number of components for PCA.
                                                   If None, keeps all components.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    steps = [
        ("pre_scaler", PowerTransformer(method="yeo-johnson")),
        ("rotator", PCA(n_components=n_pca_components, whiten=False)),
        ("post_scaler", PowerTransformer(method="yeo-johnson")),
        ("classifier", LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)),
    ]
    return Pipeline(steps)


def build_robust(shrinkage="auto", solver="lsqr", n_quantiles=50):
    """
    Constructs the Robust Expert pipeline.
    Topology: QuantileTransformer -> LDA

    Args:
        shrinkage (str or float): Regularization parameter for LDA.
        solver (str): LDA solver.
        n_quantiles (int): Number of quantiles for the transformer.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    steps = [
        (
            "scaler",
            QuantileTransformer(
                output_distribution="normal", n_quantiles=n_quantiles, random_state=42
            ),
        ),
        ("classifier", LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)),
    ]
    return Pipeline(steps)


def build_polynomial(shrinkage="auto", solver="lsqr", degree=2):
    """
    Constructs the Physical Polynomial Expert pipeline.
    Topology: PowerTransformer -> PolynomialFeatures -> PowerTransformer -> LDA

    Args:
        shrinkage (str or float): Regularization parameter for LDA.
        solver (str): LDA solver.
        degree (int): Degree of polynomial features.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    steps = [
        ("pre_scaler", PowerTransformer(method="yeo-johnson")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("post_scaler", PowerTransformer(method="yeo-johnson")),
        ("classifier", LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)),
    ]
    return Pipeline(steps)


def build_bottleneck_interaction(
    shrinkage="auto", solver="lsqr", n_bottleneck=10, degree=2
):
    """
    Constructs the Interaction Expert pipeline (Intra- or Inter-Component).
    Topology: PowerTransformer -> LDA(Transformer) -> PolynomialFeatures -> PowerTransformer -> LDA(Classifier)

    Args:
        shrinkage (str or float): Regularization parameter for the final LDA classifier.
        solver (str): LDA solver for the final classifier.
        n_bottleneck (int): Number of discriminative components to project onto.
        degree (int): Degree of polynomial interactions to model in the bottleneck space.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Note: The first LDA acts as a transformer (dimensionality reduction).
    # We use solver='svd' for the transformer part as it is generally stable for reduction
    # unless shrinkage is specifically required for the bottleneck itself.
    # However, to be consistent with "Discriminative Axes", we can use the same solver/shrinkage
    # or default to standard LDA. Here we use standard SVD for the projection to pure discriminative subspace.

    steps = [
        ("pre_scaler", PowerTransformer(method="yeo-johnson")),
        ("bottleneck", LinearDiscriminantAnalysis(n_components=n_bottleneck)),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("post_scaler", PowerTransformer(method="yeo-johnson")),
        ("classifier", LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)),
    ]
    return Pipeline(steps)
