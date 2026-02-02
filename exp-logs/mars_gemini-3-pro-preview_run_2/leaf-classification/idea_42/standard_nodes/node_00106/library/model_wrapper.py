import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from library.config import Config


def get_lda_model(shrinkage):
    """
    Factory function to create a Linear Discriminant Analysis (LDA) estimator
    configured with the specified shrinkage strategy.

    According to the IGCME strategy:
    - 'auto' maps to the OAS (Oracle Approximating Shrinkage) estimator to handle
      covariance estimation robustly in the small-sample regime.
    - Float values (e.g., 0.01, 0.1) map to fixed shrinkage with the LSQR solver.

    Args:
        shrinkage (str or float): The shrinkage parameter.
            - If 'auto', uses sklearn.covariance.OAS as the covariance_estimator.
            - If float, uses this value as the fixed shrinkage coefficient.

    Returns:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis: The configured estimator.
    """
    if shrinkage == "auto":
        # Strategy: Use OAS for robust covariance estimation
        # When covariance_estimator is provided, the 'shrinkage' parameter of LDA
        # is ignored, but solver must be 'lsqr' or 'eigen'.
        model = LinearDiscriminantAnalysis(
            solver="lsqr", covariance_estimator=OAS(), store_covariance=True
        )
    else:
        # Strategy: Use Fixed Shrinkage
        # Ensure the value is a float
        try:
            shrinkage_val = float(shrinkage)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid shrinkage value: {shrinkage}. " f"Expected 'auto' or a float."
            )

        model = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=shrinkage_val, store_covariance=True
        )

    return model
