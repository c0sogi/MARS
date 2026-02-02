import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_lda_expert(shrinkage_param):
    """
    Factory function to initialize a Linear Discriminant Analysis (LDA) estimator
    with specific shrinkage configurations.

    This expert serves as the base learner for the Dynamic Generative Ensemble.
    It uses the 'lsqr' solver to support shrinkage, which improves estimation
    in high-dimensional, low-sample regimes (the 'Small N, Large P' problem).

    Args:
        shrinkage_param (float or str):
            - If 'auto': Uses the Ledoit-Wolf lemma to automatically estimate
              the optimal shrinkage (equivalent to OAS in this implementation context).
            - If float (0.0 to 1.0): Uses a fixed shrinkage coefficient.
              0.0 = No shrinkage (Empirical Covariance).
              1.0 = Full shrinkage (Diagonal Covariance).

    Returns:
        LinearDiscriminantAnalysis: An instantiated sklearn LDA estimator.
    """
    # Solver 'lsqr' is required to support shrinkage.
    # store_covariance=True allows for potential downstream inspection of the
    # estimated covariance structure, though not strictly required for predict_proba.
    clf = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage=shrinkage_param, store_covariance=True
    )
    return clf


def postprocess_probabilities(probas):
    """
    Applies the specific post-processing steps required by the competition metric
    (Multi-class Log Loss).

    Steps:
    1. Normalize probabilities to sum to 1 (Row-wise).
    2. Clip probabilities to the range [1e-15, 1 - 1e-15] to avoid infinite
       log loss penalties.

    Args:
        probas (np.ndarray): Raw probability matrix of shape (n_samples, n_classes).

    Returns:
        np.ndarray: Processed probability matrix ready for scoring or submission.
    """
    # Ensure input is a numpy array
    probas = np.array(probas)

    # 1. Normalize (Row-wise sum to 1)
    # Handle potential division by zero if a row sums to 0 (unlikely with LDA softmax)
    row_sums = probas.sum(axis=1, keepdims=True)

    # Identify rows that sum to zero (to assign uniform probability later)
    zero_sum_mask = (row_sums == 0).flatten()

    row_sums[zero_sum_mask] = 1.0
    probas = probas / row_sums

    # Assign uniform probabilities to rows that were originally all zeros
    if np.any(zero_sum_mask):
        n_classes = probas.shape[1]
        probas[zero_sum_mask] = 1.0 / n_classes

    # 2. Clip to avoid log(0) extremes
    # Metric definition: max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    probas = np.clip(probas, epsilon, 1 - epsilon)

    return probas
