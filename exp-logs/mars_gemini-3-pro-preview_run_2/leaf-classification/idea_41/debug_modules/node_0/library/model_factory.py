import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_expert_model(shrinkage):
    """
    Factory function to create a Linear Discriminant Analysis (LDA) expert.

    The solution strategy requires experts with varying covariance estimation
    properties. We use the 'lsqr' solver which supports shrinkage, allowing
    for regularized covariance estimation in high-dimensional spaces (or
    when features are highly correlated).

    Args:
        shrinkage (str or float):
            - 'auto': Uses the Ledoit-Wolf lemma to automatically determine
              the optimal shrinkage parameter. This serves as the "OAS-like"
              estimator mentioned in the strategy.
            - float: A fixed shrinkage parameter between 0 and 1.

    Returns:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
            Configured LDA model instance.
    """
    # The 'lsqr' solver is required to use shrinkage.
    # 'eigen' also supports it, but 'lsqr' is generally preferred for classification.
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)


def get_shrinkage_candidates():
    """
    Returns the list of shrinkage hyperparameters defined in the solution strategy.

    The ensemble includes:
    1. 'auto': Automatic shrinkage (Ledoit-Wolf).
    2. Fixed values: [0.001, 0.01, 0.1] to capture different degrees of regularization.

    Returns:
        list: A list containing 'auto' and float values.
    """
    return ["auto", 0.001, 0.01, 0.1]
