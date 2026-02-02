import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from library.config import (
    LDA_SHRINKAGE_OPTIONS,
    MACRO_LDA_SOLVER,
    MACRO_LDA_SHRINKAGE,
)


def get_expert_library():
    """
    Constructs the library of probabilistic experts based on the Stereoscopic Architecture.

    The library consists of:
    1. Global Parametric Experts: LDA on Yeo-Johnson transformed global features.
    2. Global Rank Experts: LDA on Regularized Quantile transformed global features.
    3. Macro Expert: LDA on Morphometric features.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
                    - 'name': str, Unique identifier for the expert.
                    - 'view': str, The data view to use ('global_parametric', 'global_rank', 'macro').
                    - 'model': sklearn estimator instance.
    """
    experts = []

    # =========================================================================
    # 1. Global View Experts (Parametric & Rank-Based Pipelines)
    # =========================================================================
    # We apply the same set of LDA solvers to both the Parametric (Yeo-Johnson)
    # and Regularized Rank (Quantile n=30) views.

    # Define the views corresponding to the preprocessing pipelines
    views = ["global_parametric", "global_rank"]

    for view in views:
        for shrinkage_opt in LDA_SHRINKAGE_OPTIONS:

            # Determine Model Configuration based on shrinkage option
            if shrinkage_opt == "oas":
                # Use Oracle Approximating Shrinkage for Covariance Estimation
                # We use solver='lsqr' which supports custom covariance estimators
                model = LinearDiscriminantAnalysis(
                    solver="lsqr", covariance_estimator=OAS()
                )
                suffix = "oas"
            else:
                # Use Fixed Shrinkage (float)
                # shrinkage_opt is expected to be a float (e.g., 0.001, 0.01)
                model = LinearDiscriminantAnalysis(
                    solver="lsqr", shrinkage=shrinkage_opt
                )
                suffix = str(shrinkage_opt)

            # Create Expert Entry
            experts.append(
                {"name": f"{view}_lda_{suffix}", "view": view, "model": model}
            )

    # =========================================================================
    # 2. Macro View Expert (Orthogonal Morphometrics)
    # =========================================================================
    # Uses Ledoit-Wolf shrinkage ('auto') as defined in config to handle
    # the low-dimensional but potentially correlated morphometric features.

    macro_model = LinearDiscriminantAnalysis(
        solver=MACRO_LDA_SOLVER, shrinkage=MACRO_LDA_SHRINKAGE
    )

    experts.append({"name": "macro_lda", "view": "macro", "model": macro_model})

    return experts
