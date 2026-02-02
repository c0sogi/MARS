import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def build_pipeline(
    topology: str, shrinkage: float = None, n_components_lda: int = 25
) -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline for a specific MTPGE topology.

    Topologies:
    - A: Marginal Parametric Anchors (PowerTransformer -> LDA)
    - B: Rotational Parametric Experts (PowerTransformer -> PCA -> PowerTransformer -> LDA)
    - C: Constrained Non-Parametric Experts (QuantileTransformer -> LDA)
    - D: Discriminative-Interaction Experts (PowerTransformer -> LDA_Transform -> Poly -> PowerTransformer -> LDA)
    - E: Polynomial Physical Experts (PowerTransformer -> Poly -> PowerTransformer -> LDA_LedoitWolf)

    Args:
        topology (str): The topology identifier ('A', 'B', 'C', 'D', 'E').
        shrinkage (float, optional): The shrinkage parameter for the final LDA classifier.
                                     Required for topologies A, B, C, D.
                                     Ignored for topology E (uses 'auto' for Ledoit-Wolf).
        n_components_lda (int, optional): Number of components for the intermediate LDA transformer
                                          in Topology D. Defaults to 25.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    topology = topology.upper()

    # Configure the final classifier
    # Topologies A, B, C, D use a fixed shrinkage value provided by the hyperparameter search/selection.
    # Topology E uses Ledoit-Wolf shrinkage ('auto').
    if topology in ["A", "B", "C", "D"]:
        if shrinkage is None:
            raise ValueError(
                f"Shrinkage parameter is required for Topology {topology}."
            )
        # solver='lsqr' is required to use shrinkage
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
    elif topology == "E":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(
            f"Unknown topology: {topology}. Valid options are A, B, C, D, E."
        )

    # Construct the pipeline steps based on the topology
    if topology == "A":
        # Topology A: Marginal Parametric Anchors
        # Preserves the state-of-the-art baseline using robust covariance estimation on global features.
        steps = [
            ("scaler", PowerTransformer(method="yeo-johnson")),
            ("clf", classifier),
        ]

    elif topology == "B":
        # Topology B: Rotational Parametric Experts
        # Aligns data with principal axes of variation without whitening noise amplification.
        steps = [
            ("pre_pca_scaler", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(whiten=False)),  # Keeps all components by default
            ("post_pca_scaler", PowerTransformer(method="yeo-johnson")),
            ("clf", classifier),
        ]

    elif topology == "C":
        # Topology C: Constrained Non-Parametric Experts
        # Uses rank-based normalization to handle skewed distributions, constrained to prevent overfitting.
        steps = [
            (
                "scaler",
                QuantileTransformer(output_distribution="normal", n_quantiles=50),
            ),
            ("clf", classifier),
        ]

    elif topology == "D":
        # Topology D: Discriminative-Interaction Experts
        # Models quadratic interactions in a class-discriminative subspace.

        # Intermediate LDA for dimensionality reduction (Transformer)
        # Uses default solver (svd) for stability in projection
        lda_transformer = LinearDiscriminantAnalysis(n_components=n_components_lda)

        steps = [
            ("pre_lda_scaler", PowerTransformer(method="yeo-johnson")),
            ("lda_transform", lda_transformer),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("post_poly_scaler", PowerTransformer(method="yeo-johnson")),
            ("clf", classifier),
        ]

    elif topology == "E":
        # Topology E: Polynomial Physical Experts
        # Captures non-linear physical constraints from morphometric features.
        steps = [
            ("pre_poly_scaler", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("post_poly_scaler", PowerTransformer(method="yeo-johnson")),
            ("clf", classifier),
        ]

    return Pipeline(steps)
