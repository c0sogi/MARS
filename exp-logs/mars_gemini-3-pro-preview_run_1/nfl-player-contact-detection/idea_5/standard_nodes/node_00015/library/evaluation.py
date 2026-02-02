import numpy as np
from sklearn.metrics import matthews_corrcoef


def calculate_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient between true labels and predictions.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


class ThresholdOptimizer:
    """
    Optimizes the binary classification threshold to maximize MCC.
    """

    def __init__(self, start=0.01, end=0.99, step=0.01):
        """
        Initializes the optimizer with a range of thresholds.

        Args:
            start (float): Starting threshold value.
            end (float): Ending threshold value.
            step (float): Step size for threshold iteration.
        """
        # Generate thresholds including the end value
        self.thresholds = np.arange(start, end + step / 10.0, step)

    def optimize(self, y_true, y_probs):
        """
        Finds the best threshold for the given probabilities and true labels.

        Args:
            y_true (array-like): Ground truth binary labels.
            y_probs (array-like): Predicted probabilities (between 0 and 1).

        Returns:
            tuple: (best_threshold, best_mcc)
        """
        # Ensure inputs are numpy arrays for efficient vectorized operations
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        best_mcc = -1.0
        best_threshold = 0.5

        # Iterate through defined thresholds to find the optimum
        for thresh in self.thresholds:
            # Generate binary predictions for the current threshold
            y_pred = (y_probs >= thresh).astype(int)

            # Calculate metric
            mcc = calculate_mcc(y_true, y_pred)

            # Update best parameters if current MCC is higher
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        # Print full precision results as requested
        print(
            f"Optimization Result - Best Threshold: {best_threshold}, Best MCC: {best_mcc}"
        )

        return best_threshold, best_mcc
