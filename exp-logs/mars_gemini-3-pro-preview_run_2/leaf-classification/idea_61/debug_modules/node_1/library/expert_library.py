import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from library.config import (
    LDA_SHRINKAGE_CANDIDATES,
    LDA_SOLVER,
    POLY_DEGREE,
    FACTORIZED_N_COMPONENTS,
    FLOAT_PRECISION,
)
from library.custom_transformers import FactorizedDiscriminantProjector


def get_expert_library():
    """
    Generates the library of expert pipelines for the Factorized-Discriminative
    Manifold Ensemble (FDME) strategy.

    The library consists of three groups:
    1. Group A: Global Statistical Anchors (Marginal and Rotational topologies).
    2. Group B: Physical Polynomial Experts (Morphometric domain).
    3. Group C: Factorized Discriminative-Interaction Experts (Semantic Factorization).

    Returns:
        dict: A dictionary where keys are unique expert identifiers (str) and
              values are dictionaries containing:
              - 'pipeline': The sklearn Pipeline object.
              - 'features': The required feature set ('global' or 'morphometrics').
              - 'description': Brief description of the topology.
    """
    library = {}

    # =========================================================================
    # Group A: Global Statistical Anchors
    # Input: Global View (192 features)
    # Role: Preserves the state-of-the-art baseline (Linear/Rotational)
    # =========================================================================

    for shrinkage in LDA_SHRINKAGE_CANDIDATES:
        shrink_str = str(shrinkage)

        # 1. Marginal Topology
        # Stabilizes variance feature-wise before Linear Discriminant Analysis
        marginal_name = f"GroupA_Marginal_LDA_{shrink_str}"
        marginal_pipe = Pipeline(
            [
                ("pt", PowerTransformer(method="yeo-johnson")),
                (
                    "lda",
                    LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=shrinkage),
                ),
            ]
        )

        library[marginal_name] = {
            "pipeline": marginal_pipe,
            "features": "global",
            "description": "Global features -> Yeo-Johnson -> LDA",
        }

        # 2. Rotational Topology
        # Aligns data with principal axes to remove correlation before re-stabilizing
        # PCA(whiten=False) is used purely for rotation, keeping all components
        rotational_name = f"GroupA_Rotational_LDA_{shrink_str}"
        rotational_pipe = Pipeline(
            [
                ("pt1", PowerTransformer(method="yeo-johnson")),
                ("pca", PCA(whiten=False, random_state=42)),
                ("pt2", PowerTransformer(method="yeo-johnson")),
                (
                    "lda",
                    LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=shrinkage),
                ),
            ]
        )

        library[rotational_name] = {
            "pipeline": rotational_pipe,
            "features": "global",
            "description": "Global features -> PT -> PCA(Rotation) -> PT -> LDA",
        }

    # =========================================================================
    # Group B: Physical Polynomial Experts
    # Input: Polarity-Corrected Morphometrics (11 features)
    # Role: Captures non-linear physical constraints (e.g., Solidity * Eccentricity)
    # =========================================================================

    # Uses Ledoit-Wolf shrinkage ('auto') for robust estimation on polynomial expansion
    physical_name = "GroupB_Physical_Poly_LDA_Auto"
    physical_pipe = Pipeline(
        [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("lda", LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage="auto")),
        ]
    )

    library[physical_name] = {
        "pipeline": physical_pipe,
        "features": "morphometrics",
        "description": "Morphometrics -> PT -> Poly(2) -> PT -> LDA(Ledoit-Wolf)",
    }

    # =========================================================================
    # Group C: Factorized Discriminative-Interaction Experts
    # Input: Global View (192 features)
    # Role: Models quadratic dependencies between biological domains (Margin, Shape, Texture)
    # =========================================================================

    # The FactorizedDiscriminantProjector handles the semantic splitting,
    # discriminative projection, and interaction synthesis internally.

    for shrinkage in LDA_SHRINKAGE_CANDIDATES:
        shrink_str = str(shrinkage)

        factorized_name = f"GroupC_Factorized_LDA_{shrink_str}"

        factorized_pipe = Pipeline(
            [
                (
                    "fdp",
                    FactorizedDiscriminantProjector(
                        n_components=FACTORIZED_N_COMPONENTS,
                        solver=LDA_SOLVER,
                        shrinkage="auto",  # Robust covariance for the internal projections
                    ),
                ),
                ("pt", PowerTransformer(method="yeo-johnson")),
                (
                    "lda",
                    LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=shrinkage),
                ),
            ]
        )

        library[factorized_name] = {
            "pipeline": factorized_pipe,
            "features": "global",
            "description": "Global -> Factorized Projector -> PT -> LDA",
        }

    return library
