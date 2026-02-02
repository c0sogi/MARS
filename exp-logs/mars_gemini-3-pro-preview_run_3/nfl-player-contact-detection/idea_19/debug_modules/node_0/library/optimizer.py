import numpy as np
from library.utils import calc_mcc


class ThresholdOptimizer:
    """
    Handles post-processing optimization of prediction thresholds.
    Provides methods to find the optimal probability threshold that maximizes MCC
    and to apply specific thresholds to probability arrays.
    """

    @staticmethod
    def optimize_thresholds(y_true, y_prob, num_steps=99):
        """
        Performs a linear search to find the probability threshold that maximizes
        the Matthews Correlation Coefficient (MCC).

        Args:
            y_true (np.array or pd.Series): Ground truth binary labels.
            y_prob (np.array or pd.Series): Predicted probabilities.
            num_steps (int): Number of steps to search between 0.01 and 0.99.
                             Default is 99 (step size ~0.01).

        Returns:
            float: The threshold value that yielded the highest MCC.
            float: The maximum MCC score achieved.
        """
        # Ensure inputs are numpy arrays for consistent indexing and performance
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        best_threshold = 0.5
        best_mcc = -1.0

        # Define search space: 0.01 to 0.99
        # We avoid 0.0 and 1.0 to prevent potential edge case issues with all-class predictions
        thresholds = np.linspace(0.01, 0.99, num_steps)

        for thresh in thresholds:
            # Apply threshold
            y_pred = (y_prob >= thresh).astype(int)

            # Calculate metric
            score = calc_mcc(y_true, y_pred)

            # Update best
            if score > best_mcc:
                best_mcc = score
                best_threshold = thresh

        # Print full precision as requested
        print(
            f"Optimization Complete. Best Threshold: {best_threshold}, Best MCC: {best_mcc}"
        )

        return best_threshold, best_mcc

    @staticmethod
    def apply_thresholds(y_prob, threshold):
        """
        Converts predicted probabilities into binary class labels based on a threshold.

        Args:
            y_prob (np.array or pd.Series): Predicted probabilities.
            threshold (float): The decision threshold (values >= threshold become 1).

        Returns:
            np.array: Binary predictions (0 or 1).
        """
        y_prob = np.array(y_prob)
        return (y_prob >= threshold).astype(int)
