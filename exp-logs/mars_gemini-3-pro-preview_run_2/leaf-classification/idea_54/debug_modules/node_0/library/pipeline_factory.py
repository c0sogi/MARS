import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import (
    RANDOM_SEED,
    POWER_METHOD,
    QUANTILE_N_QUANTILES,
    QUANTILE_OUTPUT_DIST,
    POLY_DEGREE,
    BOTTLENECK_N_COMPONENTS,
)


def get_global_pipeline(topology):
    """
    Constructs a preprocessing pipeline for the Global Feature View (Group A).

    Args:
        topology (str): The specific topology type.
                        Options: 'marginal', 'rotational', 'robust'.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    if topology == "marginal":
        # Standard normalization to Gaussian-like distribution
        return Pipeline([("pt", PowerTransformer(method=POWER_METHOD))])

    elif topology == "rotational":
        # Aligns data with principal axes without noise amplification
        return Pipeline(
            [
                ("pt1", PowerTransformer(method=POWER_METHOD)),
                ("pca", PCA(whiten=False, random_state=RANDOM_SEED)),
                ("pt2", PowerTransformer(method=POWER_METHOD)),
            ]
        )

    elif topology == "robust":
        # Rank-based normalization to handle skew/outliers strictly
        return Pipeline(
            [
                (
                    "qt",
                    QuantileTransformer(
                        output_distribution=QUANTILE_OUTPUT_DIST,
                        n_quantiles=QUANTILE_N_QUANTILES,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        )

    else:
        raise ValueError(
            f"Unknown global topology: {topology}. "
            "Expected 'marginal', 'rotational', or 'robust'."
        )


def get_physical_pipeline():
    """
    Constructs the preprocessing pipeline for Physical/Morphometric Experts (Group B).

    Pipeline:
    1. Normalize inputs.
    2. Expand to polynomial features (degree 2) to capture physical constraints (e.g., Area * Solidity).
    3. Re-normalize.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    return Pipeline(
        [
            ("pt1", PowerTransformer(method=POWER_METHOD)),
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("pt2", PowerTransformer(method=POWER_METHOD)),
        ]
    )


def get_interaction_pipeline():
    """
    Constructs the preprocessing pipeline for Cross-Domain Interaction Experts (Group C).

    Implements the 'Discriminative Bottleneck':
    1. Normalize inputs.
    2. Project onto most discriminative axes using LDA (supervised).
    3. Expand interactions in this optimized subspace.
    4. Re-normalize.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    return Pipeline(
        [
            ("pt1", PowerTransformer(method=POWER_METHOD)),
            # LDA here acts as a supervised dimensionality reduction transformer
            (
                "lda_bottleneck",
                LinearDiscriminantAnalysis(n_components=BOTTLENECK_N_COMPONENTS),
            ),
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("pt2", PowerTransformer(method=POWER_METHOD)),
        ]
    )
