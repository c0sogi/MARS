import numpy as np
from scipy.stats import spearmanr
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Wraps the implementation from library.config.

    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)


def compute_spearman_metric(predictions, targets):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    This metric is used to evaluate the performance of the model on the 30
    subjective target labels. It handles edge cases where columns might be
    constant (zero variance) by assigning a correlation of 0.0.

    Args:
        predictions (np.ndarray): Predicted probabilities with shape (N, num_labels).
                                  Values should be in range [0, 1].
        targets (np.ndarray): Ground truth labels with shape (N, num_labels).
                              Values are in range [0, 1].

    Returns:
        float: The mean Spearman's correlation coefficient across all columns.
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    num_labels = predictions.shape[1]
    corrs = []

    for i in range(num_labels):
        pred_col = predictions[:, i]
        target_col = targets[:, i]

        # Check for constant values (zero variance) which make correlation undefined
        # We check std dev < epsilon or equality of all elements
        if np.all(pred_col == pred_col[0]) or np.all(target_col == target_col[0]):
            corr = 0.0
        else:
            # Calculate Spearman correlation
            # scipy.stats.spearmanr returns an object with a 'statistic' attribute in newer versions
            # or a tuple (correlation, pvalue) in older versions.
            res = spearmanr(pred_col, target_col)

            if hasattr(res, "statistic"):
                corr = res.statistic
            else:
                corr = res[0]

        # Handle NaN results if they still occur
        if np.isnan(corr):
            corr = 0.0

        corrs.append(corr)

    return np.mean(corrs)
