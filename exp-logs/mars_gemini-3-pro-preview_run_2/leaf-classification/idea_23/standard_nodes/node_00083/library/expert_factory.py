import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import OAS
from library.config import SHRINKAGE_GRID, VAR_SMOOTHING_GRID


def get_expert_library():
    """
    Generates the library of diverse probabilistic experts based on the DCGL strategy.

    The library consists of:
    1. Group A: LDA with Ledoit-Wolf shrinkage (Global & Combined views).
    2. Group B: LDA with OAS covariance estimation (Global & Combined views).
    3. Group C: LDA with fixed shrinkage values (Global view).
    4. Group D: Gaussian Naive Bayes with varying smoothing (Global & Combined views).

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'name': Unique string identifier for the expert.
              - 'model': The instantiated scikit-learn estimator.
              - 'view': The data view to use ('Global' or 'Combined').
    """
    experts = []

    # =========================================================================
    # Group A: Ledoit-Wolf Experts (Baseline)
    # Algorithm: LDA with shrinkage='auto' (Ledoit-Wolf)
    # Rationale: Proven baseline balancing bias and variance.
    # =========================================================================
    for view in ["Global", "Combined"]:
        experts.append(
            {
                "name": f"LDA_LW_{view}",
                "model": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                "view": view,
            }
        )

    # =========================================================================
    # Group B: OAS Experts (Innovation)
    # Algorithm: LDA with Oracle Approximating Shrinkage (OAS)
    # Rationale: Alternative covariance estimation often superior for Gaussian data.
    # =========================================================================
    for view in ["Global", "Combined"]:
        # covariance_estimator requires solver='lsqr' or 'eigen'
        experts.append(
            {
                "name": f"LDA_OAS_{view}",
                "model": LinearDiscriminantAnalysis(
                    solver="lsqr", covariance_estimator=OAS()
                ),
                "view": view,
            }
        )

    # =========================================================================
    # Group C: Fixed-Shrinkage Experts (Exploitation)
    # Algorithm: LDA with fixed shrinkage values
    # Rationale: Optimizing log loss directly via grid search on shrinkage.
    # Constraint: Applied to Global view only as per design.
    # =========================================================================
    for shrinkage in SHRINKAGE_GRID:
        experts.append(
            {
                "name": f"LDA_Fixed_{shrinkage}_Global",
                "model": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage),
                "view": "Global",
            }
        )

    # =========================================================================
    # Group D: Diagonal Covariance Experts (Boundary Case)
    # Algorithm: Gaussian Naive Bayes with var_smoothing
    # Rationale: High-bias regularizer, assumes feature independence.
    # =========================================================================
    for var_smoothing in VAR_SMOOTHING_GRID:
        # We apply to both views. The rationale mentions Morphometrics (Combined),
        # and Global is a standard baseline.
        for view in ["Global", "Combined"]:
            experts.append(
                {
                    "name": f"GNB_{var_smoothing}_{view}",
                    "model": GaussianNB(var_smoothing=var_smoothing),
                    "view": view,
                }
            )

    return experts
